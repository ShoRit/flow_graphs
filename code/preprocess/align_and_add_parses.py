import json
import pickle
import random
from collections import defaultdict as ddict

import fire
from nltk.stem import WordNetLemmatizer
import spacy
import stanza
import torch
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

from amr.amr_utils import is_valid_amr
from amr.annotate_datasets import format_sentence
from amr.graph_construction import construct_amr_data
from amr.realignment_heuristics import add_heuristic_alignments
from dataloading_utils import load_amr_rel2id, load_deprels
from preprocess.intervals import IntervalMapping
from utils import dump_dill


def create_datafield(
    data_dir,
    splits,
    bert_model="bert-base-uncased",
    text_tokenizer="scispacy",
):

    # STUFF LOADING
    ##################################################
    amr_count = 0
    invalid_amr_count = 0
    arg1_missing = 0
    arg2_missing = 0
    both_missing = 0

    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model)

    stanza_nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")
    deprel_dict = load_deprels(enhanced=False)

    lemmatizer = WordNetLemmatizer()

    data = ddict(lambda: ddict(list))
    labels = {}

    if text_tokenizer == "scispacy":
        nlp = spacy.load("en_core_sci_md")
    elif text_tokenizer == "spacy":
        nlp = spacy.load("en_core_web_sm")

    bad_sents = []
    for split in splits:
        docs = json.load(open(f"{data_dir}/{split}.json"))
        amr_relation_encoding = load_amr_rel2id()

        with open(f"{data_dir}/amr_{split}.pkl", "rb") as f:
            parsed_amrs = pickle.load(f)
        # docs 		= json.load(open(f'{data_dir}/parses_{split}.json'))
        for count, (doc, amr_content) in tqdm(enumerate(zip(docs, parsed_amrs)), total=len(docs)):

            for instance in amr_content:
                if not is_valid_amr(instance["graph"]):
                    continue
                add_heuristic_alignments(instance["graph"], instance["text"], lemmatizer)

            # if count < 119:
            #     continue
            rel_map = {}
            lbl_cnt = ddict(int)
            interdict = ddict(list)
            sents = [x for x in nlp(doc["text"]).sents]

            # CREATING A SPAN-RELATION MAP
            ##################################################
            for rel in doc["rels"]:
                start_span = min(rel["arg1_start"], rel["arg2_start"])
                end_span = max(rel["arg1_end"], rel["arg2_end"])
                rel_map[(start_span, end_span)] = (
                    rel["arg1_start"],
                    rel["arg1_end"],
                    rel["arg1_word"],
                    rel["arg1_label"],
                    rel["arg2_start"],
                    rel["arg2_end"],
                    rel["arg2_word"],
                    rel["arg2_label"],
                    rel["arg_label"],
                )

            # CREATING A SPAN-SENTENCE MAP
            ##################################################

            # sent_idxs: token indices of where each sentence begins
            sent_idxs = [0] + [max([tok.idx for tok in sent]) for sent in sents]
            # sent_ints: token intervals of sentences in the documents
            sent_ints = [(sent_idxs[i], sent_idxs[i + 1]) for i in range(0, len(sent_idxs) - 1)]
            rel_ints = sorted(rel_map)

            for rel_start, rel_end in rel_ints:
                for sent_cnt, sent in enumerate(sent_ints):
                    sent_start, sent_end = sent_ints[sent_cnt]
                    if rel_start > sent_end:
                        continue
                    if rel_end < sent_start:
                        break
                    interdict[(rel_start, rel_end)].append((sent_cnt, sents[sent_cnt]))

            ### creates each separate instance for each relation.

            # WITH EACH SPAN/SENTENCE/RELATION
            # MAPPING ENTITIES/SENTENCES TO TOKENS
            ##################################################
            for rel_int in interdict:
                (
                    arg1_start,
                    arg1_end,
                    arg1_word,
                    arg1_label,
                    arg2_start,
                    arg2_end,
                    arg2_word,
                    arg2_label,
                    arg_lbl,
                ) = rel_map[rel_int]

                # if arg_lbl in omit_rels: continue

                tokens, tok_range, org_toks, arg1_tokens, arg2_tokens = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                arg1_ann_map, arg2_ann_map, bw_arg_ann_map = (
                    IntervalMapping(),
                    IntervalMapping(),
                    IntervalMapping(),
                )

                sent_str = ""
                sent_start = None
                sent_end = None
                for _, sent in interdict[rel_int]:
                    if sent_start is None:
                        sent_start = sent.start_char
                    sent_end = sent.end_char
                sent_str = doc["text"][sent_start:sent_end]

                start_sentence_idx = min([sent_idx for (sent_idx, _) in interdict[rel_int]])
                end_sentence_idx = max([sent_idx for (sent_idx, _) in interdict[rel_int]]) + 1
                relation_amrs = amr_content[start_sentence_idx:end_sentence_idx]

                all_sentences_text = [format_sentence(sent.text) for sent in sents]
                assert all([instance["text"] in all_sentences_text for instance in amr_content])
                assert [
                    instance["text"]
                    for instance in amr_content[start_sentence_idx:end_sentence_idx]
                ] == all_sentences_text[start_sentence_idx:end_sentence_idx]

                ## obtained the sentence boundary for the two relations in question.
                assert sent_start <= min(arg1_start, arg2_start) and sent_end >= max(
                    arg2_end, arg1_end
                )

                e1_start, e1_end, e2_start, e2_end = (
                    arg1_start - sent_start,
                    arg1_end - sent_start,
                    arg2_start - sent_start,
                    arg2_end - sent_start,
                )
                arg1_ann_map[e1_start:e1_end] = (e1_start, e1_end, arg1_label)
                arg2_ann_map[e2_start:e2_end] = (e2_start, e2_end, arg2_label)

                sent_toks = tokenizer(sent_str, return_offsets_mapping=True, max_length=512)
                bert_toks = sent_toks["input_ids"]
                tok_range = sent_toks["offset_mapping"]

                e1_toks = (
                    [0]
                    + [
                        1 if arg1_ann_map.contains(elem[0], elem[1]) else 0
                        for elem in tok_range[1:-1]
                    ]
                    + [0]
                )
                e2_toks = (
                    [0]
                    + [
                        1 if arg2_ann_map.contains(elem[0], elem[1]) else 0
                        for elem in tok_range[1:-1]
                    ]
                    + [0]
                )

                assert 1 in e1_toks
                assert 1 in e2_toks
                # spans for e1_toks, e2_toks

                node_dict = {}
                node_idx_dict = {}
                node_mask_dict = {}
                edge_arr = []
                dep_arr = []

                # DEPENDENCY PARSE, MAP WORDS TO ENTITIES
                ##################################################

                # Specifically for the root that is attached to the main verb
                # STAR NODE
                node_dict[(-1, -1)] = 0
                node_idx_dict[(-1, -1)] = (1, len(bert_toks) - 1)
                node_mask_dict[(-1, -1)] = 0

                dep_doc = stanza_nlp(sent_str)
                num_sents = len(dep_doc.sentences)

                for sent_cnt, dep_sent in enumerate(dep_doc.sentences):
                    for word in dep_sent.words:
                        if arg1_ann_map.contains(
                            word.start_char, word.end_char
                        ) and arg2_ann_map.contains(word.start_char, word.end_char):
                            dep_val = 3
                        elif arg2_ann_map.contains(word.start_char, word.end_char):
                            dep_val = 2
                        elif arg1_ann_map.contains(word.start_char, word.end_char):
                            dep_val = 1
                        else:
                            dep_val = 0

                        # if 		word.start_char	>= e1_start  and word.end_char <= e1_end : dep_val = 1
                        # elif	word.start_char >= e2_start  and word.end_char <= e2_end : dep_val = 2
                        # else:   dep_val = 0
                        dep_arr.append(
                            (
                                (sent_cnt, word.id),
                                word.text,
                                (sent_cnt, word.head),
                                word.deprel,
                                word.start_char,
                                word.end_char,
                                dep_val,
                            )
                        )

                last_six, last_eix = 1, 1

                # CONSTRUCT GRAPH FROM DEPENDENCIES
                ##################################################

                for elem in dep_arr:
                    (
                        start_idx,
                        word,
                        end_idx,
                        deprel,
                        start_char,
                        end_char,
                        mask_val,
                    ) = elem
                    curr_map = IntervalMapping()
                    curr_map[start_char:end_char] = 1
                    if start_idx not in node_dict:
                        node_dict[start_idx] = len(node_dict)
                    if end_idx not in node_dict:
                        node_dict[end_idx] = len(node_dict)

                    start_flag = False
                    start_tok_idx = 1
                    end_tok_idx = 1
                    for idx in range(start_tok_idx, len(tok_range) - 1):
                        curr_start, curr_end = tok_range[idx][0], tok_range[idx][1]
                        if curr_map.contains(curr_start, curr_end):
                            if start_flag is False:
                                start_tok_idx = idx
                                start_flag = True
                            end_tok_idx = idx + 1

                        # if 		curr_end 	== 0  and idx ==len(tok_range)-1 : end_tok_idx =idx; break
                        # elif 	curr_end 	<= start_char	: start_tok_idx = idx +1; continue
                        # elif 	curr_end	<= end_char		: continue
                        # elif	curr_start 	>= end_char		: end_tok_idx = idx; break

                    # if 	idx == len(tok_range) -2		: end_tok_idx = idx+1
                    # if 	idx == len(tok_range) -1		: end_tok_idx = idx+1

                    if start_tok_idx == 1 and end_tok_idx == 1:
                        start_tok_idx, end_tok_idx = last_six, last_eix
                    node_idx_dict[start_idx] = (start_tok_idx, end_tok_idx)

                    if ":" in deprel:
                        deprel = deprel.split(":")[0]
                    # edge_index[0].append(start_idx)
                    # edge_index[1].append(end_idx)
                    # edge_type.append(deprel_dict[deprel])
                    edge_arr.append((start_idx, end_idx, deprel_dict[deprel]))
                    node_mask_dict[start_idx] = mask_val
                    last_six, last_eix = start_tok_idx, end_tok_idx

                for sent_num in range(num_sents):
                    tok_idxs = [
                        node_idx_dict[elem] for elem in node_idx_dict if elem[0] == sent_num
                    ]
                    min_tok_idx = min([tok_idx[0] for tok_idx in tok_idxs])
                    max_tok_idx = max([tok_idx[1] for tok_idx in tok_idxs])

                    node_idx_dict[(sent_num, 0)] = (min_tok_idx, max_tok_idx)
                    node_mask_dict[(sent_num, 0)] = 0

                ## Setting up masks for each node??
                x, edge_index, edge_type, n1_mask, n2_mask = [], [[], []], [], [], []
                for node in node_dict:
                    six, eix = node_idx_dict[node]
                    temp_ones = torch.ones((512,)) * -torch.inf

                    try:
                        assert six < eix
                    except Exception as e:
                        import pdb

                        pdb.set_trace()
                    temp_ones[six:eix] = 0
                    x.append(temp_ones)

                    mask = node_mask_dict[node]
                    if mask == 0:
                        n1_mask.append(0)
                        n2_mask.append(0)
                    if mask == 1:
                        n1_mask.append(1)
                        n2_mask.append(0)
                    if mask == 2:
                        n1_mask.append(0)
                        n2_mask.append(1)
                    if mask == 3:
                        n1_mask.append(1)
                        n2_mask.append(1)

                ## Setting up the edge arrays
                for edge in edge_arr:
                    n1, n2, rel_idx = edge
                    edge_index[0].append(node_dict[n1])
                    edge_index[1].append(node_dict[n2])
                    edge_type.append(rel_idx)

                ## Add the top node in
                for sent_num in range(num_sents):
                    edge_index[0].append(node_dict[(sent_num, 0)])
                    edge_index[1].append(node_dict[(-1, -1)])
                    edge_type.append(deprel_dict["STAR"])

                try:
                    assert sum(n1_mask) > 0 and sum(n2_mask) > 0
                except Exception as e:
                    import pdb

                    pdb.set_trace()

                ## Set up the Data instance for the relation
                try:
                    x, edge_index, edge_type, n1_mask, n2_mask = (
                        torch.stack(x, dim=0),
                        torch.LongTensor(edge_index),
                        torch.LongTensor(edge_type),
                        torch.LongTensor(n1_mask),
                        torch.LongTensor(n2_mask),
                    )
                    dep_data = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        n1_mask=n1_mask,
                        n2_mask=n2_mask,
                    )
                except Exception as e:
                    import pdb

                    pdb.set_trace()

                # import pdb; pdb.set_trace()

                # FIX AMR ALIGNMENT AND CONSTRUCT AMR DATA
                ##################################################

                amr_data_dict = construct_amr_data(
                    relation_amrs,
                    sent_str,
                    amr_relation_encoding,
                    sent_toks,
                    e1_toks,
                    e2_toks,
                    {"tokenizer": tokenizer, "arg1_word": arg1_word, "arg2_word": arg2_word},
                )

                amr_data = amr_data_dict["amr_data"]

                amr_count += 1
                invalid_amr_count += amr_data_dict["invalid"]
                arg1_missing += amr_data_dict["arg1_missing"]
                arg2_missing += amr_data_dict["arg2_missing"]
                both_missing += amr_data_dict["both_missing"]

                data[split]["rels"].append(
                    {
                        "tokens": bert_toks,
                        "file": doc["_id"],
                        "tok_range": tok_range,
                        "org_toks": org_toks,
                        "arg1_ids": e1_toks,
                        "arg2_ids": e2_toks,
                        "span_info": rel_map[rel_int],
                        "label": arg_lbl,
                        "sent": sent_str,
                        "sent_start": sent_start,
                        "dep_data": dep_data,
                        "amr_data": amr_data,
                    }
                )

        if "japflow" in data_dir:
            rels_data = list(data[split]["rels"])
            random.shuffle(rels_data)
            data["train"]["rels"] = rels_data[0 : int(len(rels_data) * 0.8)]
            data["dev"]["rels"] = rels_data[int(len(rels_data) * 0.8) : int(len(rels_data) * 0.9)]
            data["test"]["rels"] = rels_data[int(len(rels_data) * 0.9) :]

    print(f"Invalid AMR count: {invalid_amr_count}/{amr_count}")
    print(f"Missing arg1: {arg1_missing}")
    print(f"Missing arg2: {arg2_missing}")
    print(f"Both missing: {both_missing}")
    return data


def align_and_add_parses(dataset):
    splits = {
        "risec": ["train", "dev", "test"],
        "japflow": ["train", "test"],
        "chemu": ["train", "dev", "test"],
        "mscorpus": ["train", "dev", "test"],
    }[dataset]

    preprocessed_dataset = create_datafield(
        f"/home/sgururaj/src/flow_graphs/data/{dataset}",
        splits,
        bert_model="bert-base-uncased",
        text_tokenizer="scispacy",
    )
    dump_dill(preprocessed_dataset, f"/projects/flow_graphs/data/{dataset}/data_amr.dill")


if __name__ == "__main__":
    fire.Fire(align_and_add_parses)

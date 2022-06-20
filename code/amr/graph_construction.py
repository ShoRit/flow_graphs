import re

import penman
import torch
from torch_geometric.data import Data

from amr.annotate_datasets import align_tokens_to_sentence
from amr.create_amr_rel2id import UNKNOWN_RELATION
from amr.indexing_utils import (
    compute_token_overlap_range,
    get_overlapping_sentences_and_amrs,
    get_sentence_offsets,
)


def construct_amr_data(amr_content, sent_str, amr_relation_encoding, sent_toks, e1_toks, e2_toks):

    split_sentences, aligned_amrs = get_overlapping_sentences_and_amrs(amr_content, sent_str)

    bert_toks = sent_toks["input_ids"]

    aligned_tokens = [
        align_tokens_to_sentence(
            [token for token in re.split("\s", sentence) if token.strip()], sentence
        )
        for sentence in split_sentences
    ]

    # annotations are computed relative to the individual sentence, but tokens relative
    #  to all the sentences together
    sentence_offsets = get_sentence_offsets(split_sentences)
    amr_node_dict = {}
    amr_node_idx_dict = {}
    amr_node_mask_dict = {}
    amr_edge_arr = []
    amr_edge_types = []

    amr_node_dict[(-1, -1)] = 0
    amr_node_idx_dict[(-1, -1)] = (1, len(bert_toks) - 1)
    amr_node_mask_dict[(-1, -1)] = 0

    for sentence_idx, (amr_graph, aligned_tokens, sentence_offset) in enumerate(
        zip(aligned_amrs, aligned_tokens, sentence_offsets)
    ):
        if amr_graph is None:
            continue
        if sentence_offset >= len(sent_str):
            # this condition accounts for where there is a duplicate sentence that passes the crude `in sent_str` filter above,
            # but is not actually in `sent_str`. Example: repeated sentences.
            continue
        alignments = penman.surface.alignments(amr_graph)
        for triple in amr_graph.triples:
            s, r, t = triple

            ## Add nodes to the node map
            if r == ":instance":
                # if this is an instance node, only add the source as a node, bc this defines what the node "is"
                if s not in amr_node_dict:
                    amr_node_dict[(sentence_idx, s)] = len(amr_node_dict)
            else:
                # if it's not an instance node, add both the source and the target
                if s not in amr_node_dict:
                    amr_node_dict[(sentence_idx, s)] = len(amr_node_dict)
                if t not in amr_node_dict:
                    amr_node_dict[(sentence_idx, t)] = len(amr_node_dict)

                ## Add the corresponding edges
                amr_edge_arr.append(
                    (amr_node_dict[(sentence_idx, s)], amr_node_dict[(sentence_idx, t)])
                )
                amr_edge_types.append(
                    amr_relation_encoding.get(r.lower(), amr_relation_encoding[UNKNOWN_RELATION])
                )

            ## Map the nodes of the graph to BERT tokens and relations
            if triple in alignments:
                alignment = alignments[triple]

                # set up the node-berttoken map
                for token_idx in alignment.indices:
                    token = aligned_tokens[token_idx]
                    overlapping_offset_range = compute_token_overlap_range(
                        token, sentence_offset, sent_toks["offset_mapping"]
                    )
                    node_start_token, node_end_token = overlapping_offset_range

                    if r == ":instance":
                        amr_node_idx_dict[(sentence_idx, s)] = overlapping_offset_range
                        if any(e1_toks[node_start_token:node_end_token]) and any(
                            e2_toks[node_start_token:node_end_token]
                        ):
                            amr_node_mask_dict[(sentence_idx, s)] = 3
                        elif any(e1_toks[node_start_token:node_end_token]):
                            amr_node_mask_dict[(sentence_idx, s)] = 1
                        elif any(e2_toks[node_start_token:node_end_token]):
                            amr_node_mask_dict[(sentence_idx, s)] = 2
                        else:
                            amr_node_mask_dict[(sentence_idx, s)] = 0
                    else:
                        amr_node_idx_dict[(sentence_idx, t)] = overlapping_offset_range
                        if any(e1_toks[node_start_token:node_end_token]):
                            amr_node_mask_dict[(sentence_idx, t)] = 1
                        elif any(e2_toks[node_start_token:node_end_token]):
                            amr_node_mask_dict[(sentence_idx, t)] = 2
                        else:
                            amr_node_mask_dict[(sentence_idx, t)] = 0

    # set up sentence nodes

    for sentence_idx in range(len(split_sentences)):
        prev_sentence_start = 0 if sentence_idx == 0 else sentence_offsets[sentence_idx - 1]
        sentence_token_idxs = [
            i
            for i, (token_start, token_end) in enumerate(sent_toks["offset_mapping"])
            if token_start >= prev_sentence_start
            and token_end <= sentence_offsets[sentence_idx]
            and i not in {1, len(bert_toks)}
        ]

        min_tok_idx = min(sentence_token_idxs)
        max_tok_idx = max(sentence_token_idxs)

        amr_node_idx_dict[(sentence_idx, 0)] = (min_tok_idx, max_tok_idx)
        amr_node_mask_dict[(sentence_idx, 0)] = 0

    ## Setting up masks for each node??
    amr_x, amr_edge_index, amr_n1_mask, amr_n2_mask = (
        [],
        [[], []],
        [],
        [],
    )
    for node in amr_node_dict:
        if node in amr_node_idx_dict:
            six, eix = amr_node_idx_dict[node]
        else:
            six = 0
            eix = 0
        temp_ones = torch.ones((512,)) * -torch.inf

        try:
            assert six <= eix
        except Exception as e:
            import pdb

            pdb.set_trace()

        temp_ones[six:eix] = 0
        amr_x.append(temp_ones)

        mask = amr_node_mask_dict.get(node, 0)
        if mask == 0:
            amr_n1_mask.append(0)
            amr_n2_mask.append(0)
        if mask == 1:
            amr_n1_mask.append(1)
            amr_n2_mask.append(0)
        if mask == 2:
            amr_n1_mask.append(0)
            amr_n2_mask.append(1)
        if mask == 3:
            amr_n1_mask.append(1)
            amr_n2_mask.append(1)

    ## Setting up the edge arrays
    for edge in amr_edge_arr:
        n1, n2 = edge
        amr_edge_index[0].append(n1)
        amr_edge_index[1].append(n2)

    ## Add the top node in
    for sentence_idx in range(len(split_sentences)):
        if aligned_amrs[sentence_idx] is None:
            continue
        if sentence_offsets[sentence_idx] >= len(sent_str):
            # this condition accounts for where there is a duplicate sentence that passes the crude `in sent_str` filter above,
            # but is not actually in `sent_str`. Example: repeated sentences.
            continue
        amr_edge_index[0].append(amr_node_dict[(sentence_idx, "z1")])
        amr_edge_index[1].append(amr_node_dict[(-1, -1)])
        amr_edge_types.append(amr_relation_encoding["STAR"])

    try:
        # amr_count += 1
        assert sum(amr_n1_mask) > 0 and sum(amr_n2_mask) > 0
    except Exception as e:
        pass
        # invalid_amr_count += 1
        # if sum(amr_n1_mask) == 0:
        #     arg1_missing += 1
        # if sum(amr_n2_mask) == 0:
        #     arg2_missing += 1
        # if sum(amr_n1_mask) == 0 and sum(amr_n2_mask) == 0:
        #     both_missing += 1

    amr_x, amr_edge_index, amr_edge_type, amr_n1_mask, amr_n2_mask = (
        torch.stack(amr_x, dim=0),
        torch.LongTensor(amr_edge_index),
        torch.LongTensor(amr_edge_types),
        torch.LongTensor(amr_n1_mask),
        torch.LongTensor(amr_n2_mask),
    )
    amr_data = Data(
        x=amr_x,
        edge_index=amr_edge_index,
        edge_type=amr_edge_type,
        n1_mask=amr_n1_mask,
        n2_mask=amr_n2_mask,
    )
    return amr_data

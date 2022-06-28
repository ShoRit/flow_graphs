import re

import penman
import torch
from torch_geometric.data import Data

from amr.create_amr_rel2id import UNKNOWN_RELATION
from amr.indexing_utils import (
    align_tokens_to_sentence,
    compute_token_overlap_range,
    get_sentence_offsets,
)
from amr.amr_utils import aligner_tokenize


"""
STRING FOR INSPECTION
This prints a coherence check when stopped at the end of this file, where we construct the AMR data.
Paste this into a debugger console. 
We can check that the words from arg1 and arg2 correspond to the correct nodes in the graph. Process:
- look at the nodes that are referred to by the n1 and n2 mask
- check against the printed graphs and the nodes in them: do the arg1 and arg2 words match the nodes in the AMR?

print(debug_content["arg1_word"], debug_content["arg2_word"])
inverse_node_dict = {v:k for k,v in node_dict.items()}
print([inverse_node_dict[idx] for idx in torch.where(n1_mask)[0].numpy()])
print([inverse_node_dict[idx] for idx in torch.where(n2_mask)[0].numpy()])
for graph in aligned_amrs:
    print(penman.encode(graph))
"""


def add_top_node(aligned_amrs, node_dict, edge_index, edge_types, amr_relation_encoding):
    for sentence_idx in range(len(aligned_amrs)):
        if aligned_amrs[sentence_idx] is None:
            continue
        edge_index[0].append(node_dict[(sentence_idx, "z1")])
        edge_index[1].append(node_dict[(-1, -1)])
        edge_types.append(amr_relation_encoding["STAR"])


def construct_entity_masks(node_dict, node_idx_dict, node_mask_dict):
    x = []
    n1_mask = []
    n2_mask = []

    for node in node_dict:
        if node in node_idx_dict:
            six, eix = node_idx_dict[node]
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
        x.append(temp_ones)

        mask = node_mask_dict.get(node, 0)
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
    return x, n1_mask, n2_mask


def construct_amr_data(
    amr_content, sent_str, amr_relation_encoding, sent_toks, e1_toks, e2_toks, debug_content
):

    split_sentences = [instance["text"] for instance in amr_content]
    aligned_amrs = [instance["graph"] for instance in amr_content]

    bert_toks = sent_toks["input_ids"]

    aligned_tokens = [
        align_tokens_to_sentence(aligner_tokenize(sentence), sentence)
        for sentence in split_sentences
    ]

    # annotations are computed relative to the individual sentence, but tokens relative
    #  to all the sentences together
    sentence_offsets = get_sentence_offsets(split_sentences)
    node_dict = {}
    node_idx_dict = {}
    node_mask_dict = {}
    edge_index = [[], []]
    edge_types = []

    node_dict[(-1, -1)] = 0
    node_idx_dict[(-1, -1)] = (1, len(bert_toks) - 1)
    node_mask_dict[(-1, -1)] = 0

    ## Build the node/edge/edge type data structures
    for sentence_idx, (amr_graph, aligned_tokens, sentence_offset) in enumerate(
        zip(aligned_amrs, aligned_tokens, sentence_offsets)
    ):
        if amr_graph is None:
            continue
        alignments = penman.surface.alignments(amr_graph)
        for triple in amr_graph.triples:
            s, r, t = triple

            ## Add nodes to the node map
            if r == ":instance":
                # if this is an instance node, only add the source as a node, bc this defines what the node "is"
                if (sentence_idx, s) not in node_dict:
                    node_dict[(sentence_idx, s)] = len(node_dict)
            else:
                # if it's not an instance node, add both the source and the target
                if (sentence_idx, s) not in node_dict:
                    node_dict[(sentence_idx, s)] = len(node_dict)
                if (sentence_idx, t) not in node_dict:
                    node_dict[(sentence_idx, t)] = len(node_dict)

                ## Add the corresponding edges
                edge_index[0].append((node_dict[(sentence_idx, s)]))
                edge_index[1].append(node_dict[(sentence_idx, t)])
                edge_types.append(
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
                        node_idx_dict[(sentence_idx, s)] = overlapping_offset_range
                        if any(e1_toks[node_start_token:node_end_token]) and any(
                            e2_toks[node_start_token:node_end_token]
                        ):
                            node_mask_dict[(sentence_idx, s)] = 3
                        elif any(e1_toks[node_start_token:node_end_token]):
                            node_mask_dict[(sentence_idx, s)] = 1
                        elif any(e2_toks[node_start_token:node_end_token]):
                            node_mask_dict[(sentence_idx, s)] = 2
                        else:
                            node_mask_dict[(sentence_idx, s)] = 0
                    else:
                        node_idx_dict[(sentence_idx, t)] = overlapping_offset_range
                        if any(e1_toks[node_start_token:node_end_token]):
                            node_mask_dict[(sentence_idx, t)] = 1
                        elif any(e2_toks[node_start_token:node_end_token]):
                            node_mask_dict[(sentence_idx, t)] = 2
                        else:
                            node_mask_dict[(sentence_idx, t)] = 0

    ## set up sentence nodes
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

        node_idx_dict[(sentence_idx, 0)] = (min_tok_idx, max_tok_idx)
        node_mask_dict[(sentence_idx, 0)] = 0

    ## Add the top node in
    add_top_node(aligned_amrs, node_dict, edge_index, edge_types, amr_relation_encoding)

    ## Setting up masks for each node
    x, n1_mask, n2_mask = construct_entity_masks(node_dict, node_idx_dict, node_mask_dict)

    invalid = 0
    arg1_missing = 0
    arg2_missing = 0
    both_missing = 0

    try:
        assert sum(n1_mask) > 0 and sum(n2_mask) > 0
    except Exception as e:
        invalid += 1
        if sum(n1_mask) == 0:
            arg1_missing += 1
        if sum(n2_mask) == 0:
            arg2_missing += 1
        if sum(n1_mask) == 0 and sum(n2_mask) == 0:
            both_missing += 1

    try:
        if edge_index[0]:
            assert torch.tensor(edge_index).max() in node_dict.values()
    except Exception:
        pass

    x, edge_index, edge_type, n1_mask, n2_mask = (
        torch.stack(x, dim=0),
        torch.LongTensor(edge_index),
        torch.LongTensor(edge_types),
        torch.LongTensor(n1_mask),
        torch.LongTensor(n2_mask),
    )
    amr_data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        n1_mask=n1_mask,
        n2_mask=n2_mask,
    )
    return {
        "amr_data": amr_data,
        "invalid": invalid,
        "arg1_missing": arg1_missing,
        "arg2_missing": arg2_missing,
        "both_missing": both_missing,
    }

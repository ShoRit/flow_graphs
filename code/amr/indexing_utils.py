from typing import List, Tuple

import penman

from amr.amr_utils import PenmanToken, aligner_tokenize


def compute_token_overlap_range(
    token: PenmanToken, sentence_offset: int, offset_mapping: List[Tuple]
) -> List[int]:
    overlapping_offset_indices = []
    for i, (offset_start, offset_end) in enumerate(offset_mapping):
        token_start = sentence_offset + token.start_idx
        token_end = sentence_offset + token.end_idx
        if (token_start >= offset_start and token_start <= offset_end) or (
            token_end >= offset_start and token_end <= offset_end
        ):
            overlapping_offset_indices.append(i)
    overlap_range_start = overlapping_offset_indices[0]
    # add 1 in both of these cases because range end indices are exclusive
    if overlapping_offset_indices[0] == overlapping_offset_indices[-1]:
        overlap_range_end = overlap_range_start + 1
    else:
        overlap_range_end = overlapping_offset_indices[-1] + 1
    return (overlap_range_start, overlap_range_end)


def get_sentence_offsets(split_sentences):
    sentence_offsets = [0]
    acc = 0
    for sentence in split_sentences[:-1]:
        # making an assumption here: that sentences are space-separated in the actual text
        acc = acc + len(sentence) + 1
        sentence_offsets.append(acc)
    return sentence_offsets


def align_tokens_to_sentence(token_list, sentence):
    cur_ptr = 0
    aligned_tokens = []
    for token in token_list:
        while not sentence[cur_ptr:].startswith(token):
            cur_ptr += 1
        if not sentence[cur_ptr:]:
            raise AssertionError(f'Failed to align sentence "{sentence}" on token "{token}"')
        aligned_tokens.append(PenmanToken(token, cur_ptr, cur_ptr + len(token)))
        cur_ptr += len(token)
    return aligned_tokens


def get_unaligned_tokens(graph, sentence):
    aligned_tokens = align_tokens_to_sentence(aligner_tokenize(sentence), sentence)

    is_aligned = [False] * len(aligned_tokens)
    for alignment in penman.surface.alignments(graph).values():
        for index in alignment.indices:
            is_aligned[index] = True
    return [(i, token) for i ,(token_aligned, token) in enumerate(zip(is_aligned, aligned_tokens)) if not token_aligned]

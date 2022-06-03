from code.amr.annotate_datasets import PenmanToken


from typing import List, Tuple

from amr.annotate_datasets import PenmanToken


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
    if overlapping_offset_indices[0] == overlapping_offset_indices[-1]:
        overlap_range_end = overlap_range_start + 1
    else:
        overlap_range_end = overlapping_offset_indices[-1]
    return (overlap_range_start, overlap_range_end)

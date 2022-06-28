from dataclasses import dataclass
import re

@dataclass
class PenmanToken:
    token_str: str
    start_idx: int
    end_idx: int

def is_valid_amr(amr_graph):
    return not amr_graph is None and not amr_graph.triples[0][0] is None


def aligner_tokenize(sentence):
    return [token for token in re.split("\s", sentence) if token.strip()]

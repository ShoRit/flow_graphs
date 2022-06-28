import os
import re
import string

from nltk.stem import WordNetLemmatizer
import penman
from penman.surface import Alignment

from amr.indexing_utils import get_unaligned_tokens

node_pattern = re.compile(r"z\d+")
pbf_pattern = re.compile(r"(\w+-)+(\d){2}")


with open(os.path.join(os.path.dirname(__file__), "amrlib_additions.txt")) as f:
    additions = f.read().split("\n")

with open(os.path.join(os.path.dirname(__file__), "amr_keywords.txt")) as f:
    amr_keywords = [
        word.strip() for word in f.readlines() if word.strip() and not word.startswith("#")
    ]


def is_unit_instance(triple, graph):
    s, r, t = triple
    is_instance = r == ":instance"
    is_unit_instance = any(
        [
            parent_triple[1] in {":unit", ":scale"} and parent_triple[2] == s
            for parent_triple in graph.triples
        ]
    )
    return is_instance and is_unit_instance


def should_be_unaligned(triple, graph):
    # if this is a node like z1 --> z2, or z1 --> temperature_quantity
    is_node2node = bool(node_pattern.fullmatch(triple[0])) and bool(node_pattern.match(triple[2]))
    # if this is an AMR specifier node
    is_amr_keyword = (
        triple[1] == ":instance" and triple[2] in additions or triple[2] in amr_keywords
    )
    # if this is an intervening node triple
    is_name_triple = bool(node_pattern.fullmatch(triple[0])) and triple[2] == "name"
    # if this is a "you" truple
    is_imperative = triple[2] == "you" or triple[1] == ":mode" and triple[2] == "imperative"

    return (
        is_node2node
        or is_amr_keyword
        or is_name_triple
        or is_unit_instance(triple, graph)
        or is_imperative
    )


def parse_node(node, lemmatizer):
    pos = "n"
    # propbank frames
    if pbf_pattern.match(node):
        segments = node.split("-")
        # if we have more than 2 segments to a propbank frame, it is probably a reification
        # these are unlikely to align.
        if len(segments) > 2:
            return None, None
        else:
            nodes = [segments[0]]
            word_lemmas = [lemmatizer.lemmatize(node.replace('"', ""), pos="v") for node in nodes]
            pos = "v"
    # hyphenated words that don't match a propbank frame
    elif "-" in node:
        nodes = node.split("-")
        word_lemmas = [lemmatizer.lemmatize(node.strip('"')) for node in nodes]
    # regular words
    else:
        nodes = [node]
        word_lemmas = [lemmatizer.lemmatize(node.strip('"')) for node in nodes]
    return word_lemmas, pos


def add_heuristic_alignments(amr_graph, sentence, lemmatizer=None):
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    alignments = penman.surface.alignments(amr_graph)

    unaligned_tokens = get_unaligned_tokens(amr_graph, sentence)
    for triple in amr_graph.triples:
        if triple in alignments or should_be_unaligned(triple, amr_graph):
            continue

        node = triple[2]
        node_lemmas, pos = parse_node(node, lemmatizer)
        if node_lemmas is None:
            continue

        node_indices = []
        for node_lemma in node_lemmas:
            for token_idx, unaligned_token in unaligned_tokens:
                token_lemma = lemmatizer.lemmatize(
                    unaligned_token.token_str.lower().strip(string.punctuation), pos=pos
                )
                if node_lemma == token_lemma:
                    node_indices.append(token_idx)
                    # each node lemma should be related to only one token, for our sanity
                    break
        new_alignment = Alignment(indices=tuple(sorted(node_indices)), prefix="e.")
        alignments[triple] = new_alignment

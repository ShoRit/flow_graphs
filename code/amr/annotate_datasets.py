import ast
from dataclasses import dataclass
import json
import os
import pickle
from typing import Any, List, Tuple

from amrlib import load_stog_model
from amrlib.graph_processing.annotator import add_lemmas
from amrlib.alignments.faa_aligner import FAA_Aligner
from fire import Fire
import penman
import spacy
from tqdm.auto import tqdm


DATASET_BASE_PATH = "/projects/flow_graphs/data"
DATASETS = ["risec", "japflow", "chemu", "mscorpus"]

default_spacy_fn = spacy.load("en_core_sci_md", disable=["lemmatizer", "ner"])

@dataclass
class PenmanToken:
    token_str: str
    start_idx: int
    end_idx: int

def load_corpus(corpus_name):
    with open(os.path.join(DATASET_BASE_PATH, corpus_name, "train.json")) as f:
        data = json.load(f)
    return [item["text"] for item in data]

def align_graph(graph, sentence, aligner):
    if 'amr-empty' in graph.triples[0]:
        return None
    aligned_graphs, _ = aligner.align_sents([sentence], [penman.encode(graph)])
    return aligned_graphs[0]


def sentencizer(text, spacy_fn=None):
    if spacy_fn is None:
        spacy_fn = default_spacy_fn
    return [sent.text.replace("\n", " ") for sent in spacy_fn(text).sents]


def align_tokens_to_sentence(token_list, sentence):
    cur_ptr = 0
    aligned_tokens = []
    for token in token_list:
        while not sentence[cur_ptr:].startswith(token):
            cur_ptr += 1
        if not sentence[cur_ptr:]:
            raise AssertionError(f"Failed to align sentence \"{sentence}\" on token \"{token}\"")
        aligned_tokens.append(PenmanToken(token, cur_ptr, cur_ptr + len(token)))
        cur_ptr += len(token)
    return aligned_tokens

def annotate_sentences(sentences: str, amr_model: Any, spacy_fn: spacy.Language) -> Tuple[List[str], List[penman.Graph], List[List[str]]]:
    sentences_split = sentencizer(sentences, spacy_fn)
    aligner = FAA_Aligner()

    graphs = amr_model.parse_sents(sentences_split, disable_progress=False, return_penman=True)
    aligned_graphs = [align_graph(graph, text, aligner) for graph, text in zip(graphs, sentences_split)]
    tokenized_sentences = [sentence.split(" ") for sentence in sentences_split]

    aligned_tokens = []
    for token_list, sentence in zip(tokenized_sentences, sentences_split):
        aligned_tokens.append(align_tokens_to_sentence(token_list, sentence))

    return aligned_graphs, sentences_split, aligned_tokens


def annotate_corpus(corpus_name, amr_model):
    instances = load_corpus(corpus_name)
    segmented = [sentencizer(instance) for instance in instances]
    all_graphs = []
    all_texts = []

    aligner = FAA_Aligner()

    for segmented_batch in segmented:
        graphs = amr_model.parse_sents(segmented_batch, disable_progress=False, return_penman=True)
        lemma_graphs = [add_lemmas(penman.encode(graph), snt_key="snt") for graph in graphs]
        aligned_graphs = [align_graph(graph, text, aligner) for graph, text in zip(graphs, segmented_batch)]
        all_graphs.append([{
            "graph": graph,
            "tokens": ast.literal_eval(lemma_graph.metadata["tokens"]) if graph is not None else None,
            "text": text,
            } 
            for text, graph, lemma_graph in zip(segmented_batch, aligned_graphs, lemma_graphs)])
        all_texts.append([penman.encode(graph, top=graph.top) for graph in aligned_graphs])
    with open(os.path.join(DATASET_BASE_PATH, corpus_name, "amr.pkl"), "wb") as f:
        pickle.dump(all_graphs, f)
    with open(os.path.join(DATASET_BASE_PATH, corpus_name, "amr.json"), "w") as f:
        json.dump(all_texts, f, indent=4)



def annotate_corpora(model_path, device=None, batch_size=4):
    amr_model = load_stog_model(model_path, device=device, batch_size=batch_size)
    for dataset in DATASETS:
        print(f"Annotating {dataset}")
        # try:
        annotate_corpus(dataset, amr_model)
        # except:
        #     print(f"Failed to annotate {dataset}")
        #     pass
    

if __name__ == "__main__":
    Fire(annotate_corpora)
import ast
from dataclasses import dataclass
import json
import os
import pickle
from typing import Any, List, Optional, Tuple

from amrlib import load_stog_model
from amrlib.graph_processing.annotator import add_lemmas
from amrlib.alignments.faa_aligner import FAA_Aligner
from fire import Fire
import penman
import spacy
from tqdm.auto import tqdm

from amr.amr_utils import PenmanToken
from amr.indexing_utils import align_tokens_to_sentence

DATASET_BASE_PATH = "/projects/flow_graphs/data"
DATASETS = ["risec", "japflow", "chemu", "mscorpus"]

default_spacy_fn = spacy.load("en_core_sci_md", disable=["lemmatizer", "ner"])



def load_corpus(corpus_name, split):
    with open(os.path.join(DATASET_BASE_PATH, corpus_name, f"{split}.json")) as f:
        data = json.load(f)
    return [item["text"] for item in data]


def align_graph(graph, sentence, aligner):
    if "amr-empty" in graph.triples[0] or None in graph.triples[0]:
        return None
    try:
        aligned_graphs, _ = aligner.align_sents([sentence], [penman.encode(graph)])
    except AttributeError as e:
        print(f"\nFailed to align sentence:\n\n{sentence}\n\nError: {e}")
        return None
    return penman.decode(aligned_graphs[0])


def format_sentence(text):
    return text.replace("\n", " ")


def sentencizer(text, spacy_fn=None):
    if spacy_fn is None:
        spacy_fn = default_spacy_fn
    return [format_sentence(sent.text) for sent in spacy_fn(text).sents]

def annotate_sentences(
    sentences: str, amr_model: Any, spacy_fn: spacy.Language
) -> Tuple[List[str], List[penman.Graph], List[List[str]]]:
    sentences_split = sentencizer(sentences, spacy_fn)
    aligner = FAA_Aligner()

    graphs = amr_model.parse_sents(sentences_split, disable_progress=False, return_penman=True)
    aligned_graphs = [
        align_graph(graph, text, aligner) for graph, text in zip(graphs, sentences_split)
    ]
    tokenized_sentences = [sentence.split(" ") for sentence in sentences_split]

    aligned_tokens = []
    for token_list, sentence in zip(tokenized_sentences, sentences_split):
        aligned_tokens.append(align_tokens_to_sentence(token_list, sentence))

    return aligned_graphs, sentences_split, aligned_tokens


def annotate_corpus(corpus_name, split, amr_model, output_base_path):
    instances = load_corpus(corpus_name, split)
    segmented = [sentencizer(instance) for instance in instances]
    all_graphs = []

    aligner = FAA_Aligner()

    for segmented_batch in tqdm(segmented):
        graphs = amr_model.parse_sents(segmented_batch, disable_progress=False, return_penman=True)
        aligned_graphs = [
            align_graph(graph, text, aligner) if graph is not None else None
            for graph, text in zip(graphs, segmented_batch)
        ]
        all_graphs.append(
            [
                {
                    "graph": graph,
                    "text": text,
                }
                for text, graph in zip(segmented_batch, aligned_graphs)
            ]
        )
    with open(os.path.join(output_base_path, corpus_name, f"amr_{split}.pkl"), "wb") as f:
        pickle.dump(all_graphs, f)


def annotate_corpora(
    model_path,
    datasets: Optional[List[str]] = None,
    splits="all",
    device=None,
    batch_size=4,
    output_base_path=DATASET_BASE_PATH,
):
    amr_model = load_stog_model(model_path, device=device, batch_size=batch_size)
    if datasets is None:
        datasets = DATASETS
    if splits == "all":
        splits = ["train", "dev", "test"]
    for dataset in datasets:
        for split in splits:
            print(f"Annotating {dataset}/{split}")
            annotate_corpus(dataset, split, amr_model, output_base_path)


if __name__ == "__main__":
    Fire(annotate_corpora)

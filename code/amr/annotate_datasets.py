import json
import os
import pickle

from amrlib import load_stog_model
from fire import Fire
import penman
import spacy
from tqdm.auto import tqdm


DATASET_BASE_PATH = "/projects/flow_graphs/data"
DATASETS = ["risec", "japflow", "chemu", "mscorpus"]

spacy_fn = spacy.load("en_core_sci_md", disable=["lemmatizer", "ner"])

def load_corpus(corpus_name):
    with open(os.path.join(DATASET_BASE_PATH, corpus_name, "train.json")) as f:
        data = json.load(f)
    return [item["text"] for item in data]


def sentencizer(text):
    return [sent.text for sent in spacy_fn(text).sents]

def annotate_corpus(corpus_name, amr_model):
    instances = load_corpus(corpus_name)
    segmented = [sentencizer(instance) for instance in instances]
    all_graphs = []
    all_texts = []

    for segmented_batch in segmented:
        graphs = amr_model.parse_sents(segmented_batch, disable_progress=False, return_penman=True)
        all_graphs.append(graphs)
        all_texts.append([penman.encode(graph, top=graph.top) for graph in graphs])
    with open(os.path.join(DATASET_BASE_PATH, corpus_name, "amr.pkl"), "wb") as f:
        pickle.dump(all_graphs, f)
    with open(os.path.join(DATASET_BASE_PATH, corpus_name, "amr.json"), "w") as f:
        json.dump(all_texts, f, indent=4)



def annotate_corpora(model_path, device=None, batch_size=4):
    amr_model = load_stog_model(model_path, device=device, batch_size=batch_size)
    for dataset in DATASETS:
        print(f"Annotating {dataset}")
        try:
            annotate_corpus(dataset, amr_model)
        except:
            pass
    

if __name__ == "__main__":
    Fire(annotate_corpora)
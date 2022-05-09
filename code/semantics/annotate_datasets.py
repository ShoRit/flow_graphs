import json
import os

from fire import Fire
from nltk.tokenize import sent_tokenize
import requests
import spacy
from tqdm.auto import tqdm


ANNOTATION_SERVICE_URL = "http://localhost:{}/predict/semantics/"

DATASET_BASE_PATH = "/projects/flow_graphs/data"

DATASETS = ["wsj", "risec", "japflow", "chemu", "mscorpus" ]


def annotate_text(text, service_url, max_retries=5):
    response = requests.get(service_url, params={"utterance": text})
    retries = 0
    while response.status_code != 200 and retries < max_retries:
        response = requests.get(service_url, params={"utterance": text})
        retries += 1
    if retries >= max_retries:
        print(text)
        return None
    return response.json()


def load_corpus(corpus_name):
    with open(os.path.join(DATASET_BASE_PATH, corpus_name, "train.json")) as f:
        data = json.load(f)
    return [item["text"] for item in data]


def annotate_corpus(corpus_name, annotation_service_port):
    
    if corpus_name in {"chemu", "mscorpus"}:
        spacy_fn = spacy.load("en_core_sci_md", disable=["lemmatizer", "ner"])

    sentence_tokenizer = {
        "wsj": lambda x: x.split("\n"),
        "risec": lambda x: x.split("\n"),
        "japflow": sent_tokenize,
        "chemu": lambda x: [sent.text for sent in spacy_fn(x).sents],
        "mscorpus": lambda x: [sent.text for sent in spacy_fn(x).sents]
    }[corpus_name]

    corpus = load_corpus(corpus_name)
    annotations = []

    for item in tqdm(corpus):
        for sentence in sentence_tokenizer(item):
            sentence_annotation = annotate_text(sentence, ANNOTATION_SERVICE_URL.format(annotation_service_port))
            annotations.append({
                "text": sentence,
                "annotations": sentence_annotation
            })

    with open(f"{corpus_name}_semantics.json", "w") as f:
        json.dump(annotations, f, indent=4)


def annotate_corpora(annotation_service_port):
    for dataset in DATASETS:
        print(f"Annotating {dataset}")
        annotate_corpus(dataset, annotation_service_port)
    

if __name__ == "__main__":
    Fire(annotate_corpora)

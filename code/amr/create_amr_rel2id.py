from collections import Counter
import json
import os
import pickle
from typing import Dict

import fire

DATASET_BASE_PATH = "/projects/flow_graphs/data"
DATASETS = ["risec", "japflow", "chemu", "mscorpus"]

UNKNOWN_RELATION = "UNK"


def create_amr_rel2id(dataset_paths, frequency_cutoff: 0) -> Dict[str, int]:
    all_relations = ["STAR", UNKNOWN_RELATION]

    for dataset in dataset_paths:
        with open(os.path.join(dataset, "amr.pkl"), "rb") as f:
            amrs = pickle.load(f)
        for document in amrs:
            for sentence in document:
                amr_graph = sentence["graph"]
                if amr_graph is None:
                    continue
                relations = [triple[1].lower() for triple in amr_graph.triples]
                all_relations.extend(relations)

    relation_counter = Counter(all_relations)
    relation_counter = Counter(
        {
            rel: count
            for rel, count in relation_counter.items()
            if count >= frequency_cutoff or rel in {"STAR", UNKNOWN_RELATION}
        }
    )

    amr_rel2id = {}
    for relation in relation_counter:
        amr_rel2id[relation] = len(amr_rel2id)

    return amr_rel2id


def create_amr_rel2id_wrapper(
    datasets="all", frequency_cutoff=0, output_filename="amr_rel2id.json"
):
    if datasets == "all":
        dataset_paths = [os.path.join(DATASET_BASE_PATH, dataset) for dataset in DATASETS]
    else:
        dataset_paths = [os.path.join(DATASET_BASE_PATH, dataset) for dataset in datasets]

    rel2id = create_amr_rel2id(dataset_paths, frequency_cutoff)
    with open(output_filename, "w") as f:
        json.dump(rel2id, f, indent=4)


if __name__ == "__main__":
    fire.Fire(create_amr_rel2id_wrapper)

import json

import dill
from tqdm import tqdm

from utils import check_file

DATASET_PATH_TEMPLATE = "{base_path}/data/{dataset_name}/data_amr.dill"


def load_glove(path="/data/glove_vector/glove.6B.100d.txt"):
    glove_dict = {}
    lines = open(path).readlines()
    for line in tqdm(lines):
        data = line.strip().split()
        glove_dict[data[0]] = [float(data[i]) for i in range(1, len(data))]
    return glove_dict


def load_deprels(path="/projects/flow_graphs/data/enh_dep_rel.txt", enhanced=False):
    dep_dict = {}
    lines = open(path).readlines()
    for line in tqdm(lines):
        data = line.strip().split()
        rel_name = data[0]
        if enhanced:
            if rel_name.endswith(":"):
                rel_name = rel_name[:-1]
        else:
            rel_name = rel_name.split(":")[0]
        if rel_name not in dep_dict:
            dep_dict[rel_name] = len(dep_dict)

    if "STAR" not in dep_dict:
        dep_dict["STAR"] = len(dep_dict)

    return dep_dict


def load_amr_rel2id(path="/projects/flow_graphs/data/amr_rel2id.json"):
    with open(path) as f:
        return json.load(f)


def load_dataset(base_path: str, dataset_name: str):
    dataset_path = DATASET_PATH_TEMPLATE.format(base_path=base_path, dataset_name=dataset_name)
    check_file(dataset_path)
    with open(dataset_path, "rb") as f:
        dataset = dill.load(f)

    return dataset

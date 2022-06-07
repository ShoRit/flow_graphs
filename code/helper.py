from glob import glob
import argparse
from collections import defaultdict as ddict, Counter
import pandas as pd, numpy as np
from bs4 import BeautifulSoup as bs
import ast

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import json, pickle, time, os, random, pathlib, dill
import regex as re

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_glove(file="/data/glove_vector/glove.6B.100d.txt"):
    glove_dict = {}
    lines = open(file).readlines()
    for line in tqdm(lines):
        data = line.strip().split()
        glove_dict[data[0]] = [float(data[i]) for i in range(1, len(data))]

    return glove_dict


def load_deprels(file="/projects/flow_graphs/data/enh_dep_rel.txt", enhanced=False):
    dep_dict = {}
    lines = open(file).readlines()
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


def load_pickle(filename):
    with open(filename, "rb") as handle:
        data = pickle.load(handle)
    return data


def dump_pickle(obj, filename):
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle)


def dump_dill(obj, fname):
    dill.dump(obj, open(fname, "wb"))
    print("Pickled Dumped {}".format(fname))


def load_dill(fname):
    return dill.load(open(fname, "rb"))


punct_dict = {
    "<": "_lt_",
    ">": "_gt_",
    "+": "_plus_",
    "?": "_question_",
    "&": "_amp_",
    ":": "_colon_",
    ".": "_period_",
    "!": "_exclamation_",
    ",": "_comma",
}


def check_file(filename):
    return pathlib.Path(filename).is_file()

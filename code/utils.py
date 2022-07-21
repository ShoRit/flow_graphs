import pathlib
import pickle
import random

import dill
import numpy as np
import torch


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device(gpu_index: int = 0):
    return f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"


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


def check_file(filename):
    return pathlib.Path(filename).is_file()
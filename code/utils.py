import random

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

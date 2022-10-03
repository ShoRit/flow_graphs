from collections import defaultdict as ddict
import os
from typing import Optional

import fire
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from transformers import AutoTokenizer, BertConfig

from dataloader import get_data_loaders
from dataloading_utils import load_dataset
from evaluation import format_evaluation_df, get_eval_df
from experiment_configs import model_configurations
from modeling.bert import CONNECTION_TYPE_TO_CLASS
from modeling.metadata_utils import (
    get_case,
    get_transfer_checkpoint_filename,
    load_model_from_file,
    parse_indomain_checkpoint_filename,
)
from utils import get_device


def evaluate_indomain_model(
    model, data, device, graph_data_source, max_seq_len, batch_size, **kwargs
):
    train_data = data["train"]["rels"]
    dev_data = data["dev"]["rels"]
    test_data = data["test"]["rels"]

    train_labels = [data["label"] for data in train_data]
    labels = sorted(list(set(train_labels)))
    lbl2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2lbl = {idx: lbl for (lbl, idx) in lbl2id.items()}

    train_loader, dev_loader, test_loader = get_data_loaders(
        train_data,
        dev_data,
        test_data,
        lbl2id,
        graph_data_source,
        max_seq_len,
        batch_size,
        shuffle_train=False,
    )

    model.eval()
    print("Evaluating model on train set...")
    train_df = get_eval_df(model, dev_data, dev_loader, device, id2lbl, "Train")

    print("Evaluating model on dev set...")
    dev_df = get_eval_df(model, dev_data, dev_loader, device, id2lbl, "Dev")

    print("Evaluating model on test set...")
    test_df = get_eval_df(model, test_data, test_loader, device, id2lbl, "Test")

    return train_df, dev_df, test_df


def eval_indomain_model_wrapper(
    model_filename: str, experiment_config: str, gpu: Optional[int] = 0
):
    device = get_device(gpu)
    experiment_config = model_configurations[experiment_config]
    model_config = parse_indomain_checkpoint_filename(model_filename)

    print("Loading data from disk")
    dataset_loaded = load_dataset(experiment_config["base_path"], model_config["dataset_name"])
    print("Done loading data.")

    print("Loading model checkpoint")
    labels = set([data["label"] for data in dataset_loaded["train"]["rels"]])
    case = get_case(**experiment_config)

    model = load_model_from_file(
        model_filename, experiment_config, device=device, n_labels=len(labels)
    )

    train_df, dev_df, test_df = evaluate_indomain_model(
        model, dataset_loaded, device, **experiment_config
    )

    tokenizer = AutoTokenizer.from_pretrained(experiment_config["bert_model"])

    train_df = format_evaluation_df(train_df, tokenizer)
    dev_df = format_evaluation_df(dev_df, tokenizer)
    test_df = format_evaluation_df(test_df, tokenizer)

    train_df.to_csv(
        os.path.join(
            experiment_config["base_path"],
            "results",
            f"indomain_results_train_{model_config['dataset_name']}_seed-{model_config['seed']}_{case}.csv",
        )
    )
    dev_df.to_csv(
        os.path.join(
            experiment_config["base_path"],
            "results",
            f"indomain_results_dev_{model_config['dataset_name']}_seed-{model_config['seed']}_{case}.csv",
        )
    )
    test_df.to_csv(
        os.path.join(
            experiment_config["base_path"],
            "results",
            f"indomain_results_test_{model_config['dataset_name']}_seed-{model_config['seed']}_{case}.csv",
        )
    )


if __name__ == "__main__":
    fire.Fire(eval_indomain_model_wrapper)

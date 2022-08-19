import os
from typing import Dict, Optional

import fire
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import torch
from tqdm.auto import tqdm
from transformers import BertConfig

from dataloader import get_data_loaders
from dataloading_utils import load_dataset, load_deprels
from evaluation import get_labels_and_model_predictions
from experiment_configs import model_configurations
from modeling.bert import CONNECTION_TYPE_TO_CLASS
from modeling.metadata_utils import get_transfer_checkpoint_filename, get_case
from utils import get_device, seed_everything, check_file
from validation import graph_data_not_equal, validate_graph_data_source
import wandb


def load_model_from_config(configuration, fewshot, seed, n_labels, device):
    model_checkpoint_file = os.path.join(
        configuration["checkpoint_path"],
        get_transfer_checkpoint_filename(
            **configuration, case=get_case(**configuration), fewshot=fewshot, seed=seed
        ),
    )
    check_file(model_checkpoint_file)

    bertconfig = BertConfig.from_pretrained(configuration["bert_model"], num_labels=n_labels)
    if "bert-large" in configuration["bert_model"]:
        bertconfig.relation_emb_dim = 1024
    elif "bert-base" in configuration["bert_model"]:
        bertconfig.relation_emb_dim = 768

    deprel_dict = load_deprels(
        path=os.path.join(configuration["base_path"], "data", "enh_dep_rel.txt"), enhanced=False
    )

    bertconfig.node_emb_dim = configuration["node_emb_dim"]
    bertconfig.dep_rels = len(deprel_dict)
    bertconfig.gnn_depth = configuration["gnn_depth"]
    bertconfig.gnn = configuration["gnn"]

    use_graph_data = configuration["graph_data_source"] is not None
    model_class = CONNECTION_TYPE_TO_CLASS[configuration["graph_connection_type"]]

    model = model_class.from_pretrained(
        configuration["bert_model"], config=bertconfig, use_graph_data=use_graph_data
    )

    model.load_state_dict(torch.load(model_checkpoint_file, map_location=device))

    return model


def evaluate_transfer_model(
    model, src_data, tgt_data, device, graph_data_source, max_seq_len, batch_size, **kwargs
):
    train_data = src_data["train"]["rels"]
    dev_data = tgt_data["dev"]["rels"]
    test_data = tgt_data["test"]["rels"]

    train_labels = [data["label"] for data in train_data]
    labels = sorted(list(set(train_labels)))
    lbl2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2lbl = {idx: lbl for (lbl, idx) in lbl2id.items()}

    _, dev_loader, test_loader = get_data_loaders(
        train_data,
        dev_data,
        test_data,
        lbl2id,
        graph_data_source,
        max_seq_len,
        batch_size,
    )

    model.eval()
    print("Evaluating model on dev set...")

    dev_df = pd.DataFrame(dev_data)

    with torch.no_grad():
        dev_labels, dev_predictions = get_labels_and_model_predictions(model, dev_loader, device)
        dev_df["label_idxs"] = dev_labels
        dev_df["prediction_idxs"] = dev_predictions
        dev_df["predictions"] = [id2lbl[pred] for pred in dev_predictions]
        dp, dr, df1, _ = precision_recall_fscore_support(
            dev_labels, dev_predictions, average="macro"
        )
        print(f"Dev\tF1: {df1}\tPrecision: {dp}\tRecall: {dr}")

    print("Evaluating model on test set...")
    test_df = pd.DataFrame(test_loader)

    with torch.no_grad():
        test_labels, test_predictions = get_labels_and_model_predictions(model, test_loader, device)
        test_df["label_idxs"] = test_labels
        test_df["prediction_idxs"] = test_predictions
        test_df["predictions"] = [id2lbl[pred] for pred in test_predictions]
        tp, tr, tf1, _ = precision_recall_fscore_support(
            test_labels, test_predictions, average="macro"
        )
        print(f"Dev\tF1: {tf1}\tPrecision: {tp}\tRecall: {tr}")

    return dev_df, test_df


def eval_transfer_model_wrapper(
    src_dataset: str,
    tgt_dataset: str,
    fewshot: float,
    seed: int,
    experiment_config: str,
    gpu: Optional[int] = 0,
):
    device = get_device(gpu)
    configuration = model_configurations[experiment_config]
    print("Loading data from disk...")
    print("Loading data from disk...")
    src_dataset_loaded = load_dataset(configuration["base_path"], src_dataset)
    tgt_dataset_loaded = load_dataset(configuration["base_path"], tgt_dataset)
    print("Done loading data.")

    labels = set([data["label"] for data in tgt_dataset_loaded["train"]["rels"]])

    print("Loading model checkpoint")

    model = load_model_from_config(
        configuration,
        fewshot=fewshot,
        seed=seed,
    )

    dev_df, test_df = evaluate_transfer_model(
        model=model,
        src_data=src_dataset_loaded,
        tgt_data=tgt_dataset_loaded,
        device=device,
        **configuration,
    )

    dev_df.


if __name__ == "__main__":
    fire.Fire(eval_transfer_model_wrapper)

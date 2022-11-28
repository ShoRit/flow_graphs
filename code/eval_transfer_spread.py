import os
from collections import defaultdict as ddict
from typing import Optional

import fire
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer

from dataloader import get_data_loaders
from dataloading_utils import load_dataset
from evaluation import get_eval_df, format_evaluation_df, get_labels_and_model_predictions, get_relation_embeddings
from experiment_configs import model_configurations
from modeling.bert import CONNECTION_TYPE_TO_CLASS
from modeling.metadata_utils import get_case, get_transfer_checkpoint_filename, load_model_from_file
from utils import check_file, get_device


def evaluate_transfer_spread(
    model, src_data, tgt_data, device, graph_data_source, max_seq_len, batch_size, **kwargs
    ):
    
    src_train_data = src_data["train"]["rels"]
    tgt_train_data = tgt_data["train"]["rels"]
    dev_data = tgt_data["dev"]["rels"]
    test_data = tgt_data["test"]["rels"]

    train_labels = [data["label"] for data in tgt_train_data]
    labels = sorted(list(set(train_labels)))
    lbl2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2lbl = {idx: lbl for (lbl, idx) in lbl2id.items()}

    train_loader, dev_loader, test_loader = get_data_loaders(
        tgt_train_data,
        dev_data,
        test_data,
        lbl2id,
        graph_data_source,
        max_seq_len,
        batch_size,
    )

    model.eval()
    test_embeddings, y_test     = get_relation_embeddings(model, test_loader, device)
    train_embeddings, y_train   = get_relation_embeddings(model, train_loader, device)
    
    import pdb; pdb.set_trace()
    
    


def eval_transfer_spread_wrapper(
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
    src_dataset_loaded = load_dataset(configuration["base_path"], src_dataset)
    tgt_dataset_loaded = load_dataset(configuration["base_path"], tgt_dataset)
    print("Done loading data.")

    labels = set([data["label"] for data in tgt_dataset_loaded["train"]["rels"]])

    print("Loading model checkpoint")
    case = get_case(**configuration)

    model_checkpoint_file = os.path.join(
        configuration["checkpoint_folder"],
        get_transfer_checkpoint_filename(
            **configuration,
            src_dataset_name=src_dataset,
            tgt_dataset_name=tgt_dataset,
            case=case,
            fewshot=fewshot,
            seed=seed,
        ),
    )

    model = load_model_from_file(
        model_checkpoint_file,
        configuration,
        device=device,
        n_labels=len(labels),
    )
    
    evaluate_transfer_spread(
        model=model,
        src_data=src_dataset_loaded,
        tgt_data=tgt_dataset_loaded,
        device=device,
        **configuration,
    )

    # tokenizer = AutoTokenizer.from_pretrained(configuration["bert_model"])

    # dev_df = format_evaluation_df(dev_df, tokenizer)
    # test_df = format_evaluation_df(test_df, tokenizer)

    # dev_df.to_csv(
    #     os.path.join(
    #         configuration["base_path"],
    #         "results",
    #         f"transfer_results_dev_{src_dataset}_{tgt_dataset}_{fewshot}_{seed}_{case}.csv",
    #     )
    # )
    # test_df.to_csv(
    #     os.path.join(
    #         configuration["base_path"],
    #         "results",
    #         f"transfer_results_test_{src_dataset}_{tgt_dataset}_{fewshot}_{seed}_{case}.csv",
    #     )
    # )


if __name__ == "__main__":
    fire.Fire(eval_transfer_spread_wrapper)

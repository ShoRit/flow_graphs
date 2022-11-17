import os
from typing import Optional

import fire
from transformers import AutoTokenizer

from dataloader import get_data_loaders
from dataloading_utils import load_dataset
from evaluation import (
    add_evaluation_context,
    evaluate_model_add_context,
    get_indomain_eval_filename,
)
from experiment_configs import model_configurations
from modeling.bert import CONNECTION_TYPE_TO_CLASS
from modeling.metadata_utils import (
    get_case,
    get_experiment_config_from_filename,
    load_model_from_file,
    parse_indomain_checkpoint_filename,
)
from utils import get_device


def evaluate_indomain_model(
    model, data, tokenizer, device, graph_data_source, max_seq_len, batch_size, **kwargs
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
    _, _, _, train_df = evaluate_model_add_context(
        model, train_data, train_loader, device, id2lbl, "Train"
    )

    print("Evaluating model on dev set...")
    _, _, _, dev_df = evaluate_model_add_context(model, dev_data, dev_loader, device, id2lbl, "Dev")

    print("Evaluating model on test set...")
    _, _, _, test_df = evaluate_model_add_context(
        model, test_data, test_loader, device, id2lbl, "Test"
    )

    return train_df, dev_df, test_df


def eval_indomain_model_wrapper(model_filename: str, gpu: Optional[int] = 0):
    device = get_device(gpu)
    experiment_config = model_configurations[get_experiment_config_from_filename(model_filename)]
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
        model, dataset_loaded, tokenizer, device, **experiment_config
    )

    tokenizer = AutoTokenizer.from_pretrained(experiment_config["bert_model"])

    train_df = add_evaluation_context(train_df, tokenizer)
    dev_df = add_evaluation_context(dev_df, tokenizer)
    test_df = add_evaluation_context(test_df, tokenizer)

    train_df.to_csv(
        os.path.join(
            experiment_config["base_path"],
            "results",
            get_indomain_eval_filename(
                dataset_name=model_config["dataset_name"],
                split="train",
                seed=model_config["seed"],
                case=case,
            ),
        )
    )
    dev_df.to_csv(
        os.path.join(
            experiment_config["base_path"],
            "results",
            get_indomain_eval_filename(
                dataset_name=model_config["dataset_name"],
                split="dev",
                seed=model_config["seed"],
                case=case,
            ),
        )
    )
    test_df.to_csv(
        os.path.join(
            experiment_config["base_path"],
            "results",
            get_indomain_eval_filename(
                dataset_name=model_config["dataset_name"],
                split="test",
                seed=model_config["seed"],
                case=case,
            ),
        )
    )


if __name__ == "__main__":
    fire.Fire(eval_indomain_model_wrapper)

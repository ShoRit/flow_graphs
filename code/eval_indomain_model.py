from typing import Optional

import fire
from transformers import AutoTokenizer

from dataloader import get_data_loaders
from dataloading_utils import load_dataset
from evaluation import eval_model_add_context, save_indomain_eval_df
from experiment_configs import model_configurations
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
    _, _, _, train_df = eval_model_add_context(
        model=model,
        data=train_data,
        dataloader=train_loader,
        tokenizer=tokenizer,
        device=device,
        id2lbl=id2lbl,
        split_name="Train",
    )

    print("Evaluating model on dev set...")
    _, _, _, dev_df = eval_model_add_context(
        model=model,
        data=dev_data,
        dataloader=dev_loader,
        tokenizer=tokenizer,
        device=device,
        id2lbl=id2lbl,
        split_name="Dev",
    )

    print("Evaluating model on test set...")
    _, _, _, test_df = eval_model_add_context(
        model=model,
        data=test_data,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
        id2lbl=id2lbl,
        split_name="Test",
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

    tokenizer = AutoTokenizer.from_pretrained(experiment_config["bert_model"])
    train_df, dev_df, test_df = evaluate_indomain_model(
        model, dataset_loaded, tokenizer, device, **experiment_config
    )

    save_indomain_eval_df(
        train_df,
        model_config["dataset_name"],
        split="train",
        seed=model_config["seed"],
        case=case,
        base_path=experiment_config["base_path"],
    )
    save_indomain_eval_df(
        dev_df,
        model_config["dataset_name"],
        split="dev",
        seed=model_config["seed"],
        case=case,
        base_path=experiment_config["base_path"],
    )
    save_indomain_eval_df(
        test_df,
        model_config["dataset_name"],
        split="test",
        seed=model_config["seed"],
        case=case,
        base_path=experiment_config["base_path"],
    )


if __name__ == "__main__":
    fire.Fire(eval_indomain_model_wrapper)

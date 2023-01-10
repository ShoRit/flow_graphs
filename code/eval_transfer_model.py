import os
from typing import Optional

import fire
from transformers import AutoTokenizer

from dataloader import get_data_loaders
from dataloading_utils import load_dataset
from evaluation import eval_model_add_context, save_transfer_eval_df
from experiment_configs import model_configurations
from modeling.metadata_utils import get_case, get_transfer_checkpoint_filename, load_model_from_file
from utils import get_device
from validation import ABLATIONS


def evaluate_transfer_model(
    model,
    tgt_data,
    tokenizer,
    device,
    graph_data_source,
    max_seq_len,
    batch_size,
    ablation=None,
    **kwargs,
):
    tgt_train_data = tgt_data["train"]["rels"]
    dev_data = tgt_data["dev"]["rels"]
    test_data = tgt_data["test"]["rels"]

    train_labels = [data["label"] for data in tgt_train_data]
    labels = sorted(list(set(train_labels)))
    lbl2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2lbl = {idx: lbl for (lbl, idx) in lbl2id.items()}

    _, dev_loader, test_loader = get_data_loaders(
        tgt_train_data,
        dev_data,
        test_data,
        lbl2id,
        graph_data_source,
        max_seq_len,
        batch_size,
        shuffle_train=False,
        ablation=ablation,
    )

    model.eval()

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

    return dev_df, test_df


def eval_transfer_model_wrapper(
    src_dataset: str,
    tgt_dataset: str,
    fewshot: float,
    seed: int,
    experiment_config: str,
    ablation: str = None,
    gpu: Optional[int] = 0,
):
    device = get_device(gpu)
    configuration = model_configurations[experiment_config]

    if ablation not in ABLATIONS:
        raise AssertionError(
            f"Specified ablation not in allowed list. Provided: {ablation}; allowed: {ABLATIONS}"
        )

    print("Loading data from disk...")
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

    tokenizer = AutoTokenizer.from_pretrained(configuration["bert_model"])
    model = load_model_from_file(
        model_checkpoint_file,
        configuration,
        device=device,
        n_labels=len(labels),
    )

    dev_df, test_df = evaluate_transfer_model(
        model=model,
        tgt_data=tgt_dataset_loaded,
        tokenizer=tokenizer,
        ablation=ablation,
        device=device,
        **configuration,
    )

    save_transfer_eval_df(
        dev_df,
        src_dataset,
        tgt_dataset,
        fewshot,
        "dev",
        seed,
        case,
        configuration["base_path"],
        ablation,
    )
    save_transfer_eval_df(
        test_df,
        src_dataset,
        tgt_dataset,
        fewshot,
        "test",
        seed,
        case,
        configuration["base_path"],
        ablation,
    )


if __name__ == "__main__":
    fire.Fire(eval_transfer_model_wrapper)

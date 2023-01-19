import os
from typing import Optional

import pandas as pd
from transformers import AutoTokenizer

from dataloader import get_data_loaders
from dataloading_utils import load_dataset
from eval_transfer_model import evaluate_transfer_model
from evaluation import eval_model_add_context, save_transfer_eval_df
from experiment_configs import model_configurations, _base_config
from modeling.metadata_utils import get_case, get_transfer_checkpoint_filename, load_model_from_file
from utils import get_device, check_file
from validation import ABLATIONS


DATASETS = ["risec", "japflow", "mscorpus"]
SEEDS = [0, 1, 2]
CASES = ["amr_residual", "dep_residual"]
FEWSHOT = 10


if __name__ == "__main__":

    device = get_device(0)
    ablation_metrics = []

    for case in CASES:
        configuration = model_configurations[case]
        configuration["batch_size"] = 32
        for tgt_dataset in DATASETS:
            print(f"Loading dataset {tgt_dataset}")
            tgt_dataset_loaded = load_dataset(configuration["base_path"], tgt_dataset)
            labels = set([data["label"] for data in tgt_dataset_loaded["train"]["rels"]])
            print("Done.")

            for src_dataset in DATASETS:
                if src_dataset == tgt_dataset:
                    continue
                for seed in SEEDS:
                    model_checkpoint_file = os.path.join(
                        configuration["checkpoint_folder"],
                        get_transfer_checkpoint_filename(
                            **configuration,
                            src_dataset_name=src_dataset,
                            tgt_dataset_name=tgt_dataset,
                            case=case,
                            fewshot=FEWSHOT,
                            seed=seed,
                        ),
                    )

                    if not check_file(model_checkpoint_file):
                        print(f"Model filename {model_checkpoint_file} not found, continuing...")
                        continue
                    print(f"Loading model checkpoint file {model_checkpoint_file}")
                    tokenizer = AutoTokenizer.from_pretrained(configuration["bert_model"])
                    model = load_model_from_file(
                        model_checkpoint_file,
                        configuration,
                        device=device,
                        n_labels=len(labels),
                    )

                    for ablation in ABLATIONS:
                        print(f"Running ablation: {ablation}")
                        dev_df, test_df, metric_summary = evaluate_transfer_model(
                            model=model,
                            tgt_data=tgt_dataset_loaded,
                            tokenizer=tokenizer,
                            ablation=ablation,
                            device=device,
                            **configuration,
                        )

                        row_dict = {
                            "src_dataset": src_dataset,
                            "tgt_dataset": tgt_dataset,
                            "fewshot": FEWSHOT,
                            "case": case,
                            "seed": seed,
                            "ablation": ablation,
                            **metric_summary,
                        }

                        ablation_metrics.append(row_dict)

    metrics_df = pd.DataFrame(ablation_metrics)
    metrics_df.to_csv("ablation_metrics_df.csv")

import os
from typing import Dict, List, Optional

import fire
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch
from tqdm.auto import tqdm
from transformers import BertConfig
import wandb

from dataloader import get_data_loaders
from dataloading_utils import load_deprels, load_dataset
from evaluation import seen_eval
from experiment_configs import model_configurations
from modeling.bert import BertRGCNRelationClassifier
from utils import seed_everything, get_device
from validation import validate_graph_data_source


def train_model_in_domain(
    dataset_name: str,
    dataset: Dict,
    bert_model: str,
    node_emb_dim: int,
    gnn: str,
    gnn_depth: int,
    graph_data_source: str,
    lr: float,
    seed: int,
    batch_size: int,
    epochs: int,
    patience: int,
    max_seq_len: int,
    device: str,
    checkpoint_folder: str,
    wandb_entity: str,
    wandb_project: str,
    conf_blob: dict,
    **kwargs,
):
    seed_everything(seed)
    validate_graph_data_source(graph_data_source)
    checkpoint_file = os.path.join(
        checkpoint_folder,
        f"indomain-{dataset_name}"
        f"-{graph_data_source}-{gnn}-depth_"
        f"{gnn_depth}-seed_{seed}-lr_{lr}",
    )

    #######################################
    # wandb init                          #
    #######################################
    wandb.login()
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f'{checkpoint_file.split("/")[-1]}',
    )
    wandb.config.update(conf_blob)

    #######################################
    # LOAD DATA                           #
    #######################################
    train_data = dataset["train"]["rels"]
    dev_data = dataset["dev"]["rels"]
    test_data = dataset["test"]["rels"]

    deprel_dict = load_deprels(enhanced=False)

    print(
        "train size: {}, dev size {}, test size: {}".format(
            len(train_data), len(dev_data), len(test_data)
        )
    )

    train_labels = [data["label"] for data in train_data]
    labels = sorted(list(set(train_labels)))
    lbl2id = {lbl: idx for idx, lbl in enumerate(labels)}

    train_loader, dev_loader, test_loader = get_data_loaders(
        train_data,
        dev_data,
        test_data,
        lbl2id,
        graph_data_source,
        max_seq_len,
        batch_size,
    )

    print("Data is successfully loaded")

    #######################################
    # LOAD MODELS                         #
    #######################################

    print("Loading model...")
    train_label = [data["label"] for data in train_data]
    bertconfig = BertConfig.from_pretrained(bert_model, num_labels=len(set(train_label)))

    if "bert-large" in bert_model:
        bertconfig.relation_emb_dim = 1024
    elif "bert-base" in bert_model:
        bertconfig.relation_emb_dim = 768

    bertconfig.node_emb_dim = node_emb_dim
    bertconfig.dep_rels = len(deprel_dict)
    bertconfig.gnn_depth = gnn_depth
    bertconfig.gnn = gnn

    use_graph_data = graph_data_source is not None

    model = BertRGCNRelationClassifier.from_pretrained(
        bert_model, config=bertconfig, use_graph_data=use_graph_data
    )
    model.to(device)
    print("Done loading model.")

    #######################################
    # TRAINING LOOP                       #
    #######################################
    print("Setting up training loop...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_p, best_r, best_f1 = 0, 0, 0
    kill_cnt = 0

    for epoch in range(epochs):
        print(f"============== TRAINING ON EPOCH {epoch} ==============")
        running_loss, correct, total = 0.0, 0, 0

        model.train()

        for data in tqdm(train_loader):
            tokens_tensors = data["tokens_tensors"].to(device)
            segments_tensors = data["segments_tensors"].to(device)
            e1_mask = data["e1_mask"].to(device)
            e2_mask = data["e2_mask"].to(device)
            masks_tensors = data["masks_tensors"].to(device)
            labels = data["label_ids"].to(device)

            graph_data = data["graph_data"].to(device)
            dependency_tensors = data["dependency_data"].to(device)
            amr_tensors = data["amr_data"].to(device)

            optimizer.zero_grad()

            output_dict = model(
                input_ids=tokens_tensors,
                token_type_ids=segments_tensors,
                e1_mask=e1_mask,
                e2_mask=e2_mask,
                attention_mask=masks_tensors,
                labels=labels,
                graph_data=graph_data,
                # can optionally pass in dependency/amr tensors if we want to use both.
            )
            loss, logits = output_dict["loss"], output_dict["logits"]

            total += labels.size(0)
            _, pred = torch.max(logits, 1)
            correct += (pred == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            wandb.log({"batch_loss": loss.item()})
        wandb.log({"loss": running_loss})

        print("============== EVALUATION ON DEV DATA ==============")

        p_train, r_train, f1_train = seen_eval(model, train_loader, device=device)
        print(f"Train data F1: {f1_train} \t Precision: {p_train} \t Recall: {r_train}")
        wandb.log({"train_f1": f1_train})

        p_dev, r_dev, f1_dev = seen_eval(model, dev_loader, device=device)
        wandb.log({"dev_f1": f1_dev})
        print(f"Eval data F1: {f1_dev} \t Precision: {p_dev} \t Recall: {r_dev}")

        if f1_dev > best_f1:
            kill_cnt = 0

            best_p, best_r, best_f1 = p_dev, r_dev, f1_dev
            best_model = model
            torch.save(best_model.state_dict(), checkpoint_file)
        else:
            kill_cnt += 1
            if kill_cnt >= patience:
                torch.save(best_model.state_dict(), checkpoint_file)
                break

        wandb.log({"running_best_f1": best_f1})
        print(f"[best val] precision: {best_p:.4f}, recall: {best_r:.4f}, f1 score: {best_f1:.4f}")
    wandb.log({"best_f1": best_f1, "best_precision": best_p, "best_recall": best_r})
    torch.save(best_model.state_dict(), checkpoint_file)

    print("============== EVALUATION ON TEST DATA ==============")
    best_model.to(device)
    best_model.eval()
    pt, rt, test_f1 = seen_eval(best_model, test_loader, device=device)
    wandb.log({"test_f1": test_f1})
    wandb.run.finish()


def train_model_indomain_wrapper(
    dataset: str,
    seed: int,
    experiment_config: str,
    gpu: Optional[int] = 0,
):
    device = get_device(gpu)
    configuration = model_configurations[experiment_config]
    loaded_dataset = load_dataset(configuration["base_path"], dataset)

    train_model_in_domain(
        dataset_name=dataset,
        dataset=loaded_dataset,
        device=device,
        seed=seed,
        conf_blob=configuration,
        **configuration,
    )


if __name__ == "__main__":
    fire.Fire(train_model_indomain_wrapper)
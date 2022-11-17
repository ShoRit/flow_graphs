import os
from typing import Dict, Optional, Union

import fire
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertConfig
import wandb


from dataloader import get_data_loaders
from dataloading_utils import load_dataset, load_deprels
from evaluation import eval_model_add_context, save_transfer_eval_df
from experiment_configs import model_configurations
from modeling.bert import CONNECTION_TYPE_TO_CLASS
from modeling.metadata_utils import (
    get_case,
    get_indomain_checkpoint_filename,
    get_transfer_checkpoint_filename,
)
from utils import get_device, seed_everything
from validation import graph_data_not_equal, validate_graph_data_source


def train_transfer_model(
    src_dataset_name: str,
    src_dataset: Dict,
    tgt_dataset_name: str,
    tgt_dataset: Dict,
    fewshot: float,
    bert_model: str,
    node_emb_dim: int,
    gnn: str,
    gnn_depth: int,
    graph_data_source: Optional[str],
    graph_connection_type: str,
    lr: float,
    seed: int,
    batch_size: int,
    grad_accumulation_steps: int,
    epochs: int,
    patience: int,
    max_seq_len: int,
    device: str,
    checkpoint_folder: str,
    base_path: str,
    wandb_entity: str,
    wandb_project: str,
    conf_blob: dict,
    **kwargs,
):
    seed_everything(seed)
    validate_graph_data_source(graph_data_source)
    case = get_case(**conf_blob)

    src_checkpoint_file = os.path.join(
        checkpoint_folder,
        get_indomain_checkpoint_filename(
            **conf_blob, case=case, dataset_name=src_dataset_name, seed=seed
        ),
    )

    tgt_checkpoint_file = os.path.join(
        checkpoint_folder,
        get_transfer_checkpoint_filename(
            **conf_blob,
            case=case,
            src_dataset_name=src_dataset_name,
            tgt_dataset_name=tgt_dataset_name,
            fewshot=fewshot,
            seed=seed,
        ),
    )

    #######################################
    # wandb init                          #
    #######################################
    wandb.login()
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f'{tgt_checkpoint_file.split("/")[-1]}',
    )
    wandb.config.update(conf_blob)
    wandb.config.case = case

    #######################################
    # LOAD DATA                           #
    #######################################
    print("Setting up dataloaders...")
    train_data = tgt_dataset["train"]["rels"]
    dev_data = tgt_dataset["dev"]["rels"]
    test_data = tgt_dataset["test"]["rels"]

    deprel_dict = load_deprels(
        path=os.path.join(base_path, "data", "enh_dep_rel.txt"), enhanced=False
    )

    print(
        "train size: {}/{}, dev size {}, test size: {}".format(
            int(len(train_data) * fewshot), len(train_data), len(dev_data), len(test_data)
        )
    )

    src_labels = set([data["label"] for data in src_dataset["train"]["rels"]])
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
        fewshot=fewshot,
    )

    wandb.config.train_dataset_instance_hash = train_loader.dataset.sampled_index_hash
    wandb.config.train_dataset_instances = train_loader.dataset.sampled_indices

    print(f"Sampled instance hash: {train_loader.dataset.sampled_index_hash}")
    print(f"Sampled instance indices: {train_loader.dataset.sampled_indices}")

    print("Data is successfully loaded")

    #######################################
    # LOAD MODELS                         #
    #######################################

    print("Loading models...")
    src_bertconfig = BertConfig.from_pretrained(bert_model, num_labels=len(src_labels))
    tgt_bertconfig = BertConfig.from_pretrained(bert_model, num_labels=len(labels))

    for bertconfig in [src_bertconfig, tgt_bertconfig]:
        if "bert-large" in bert_model:
            bertconfig.relation_emb_dim = 1024
        elif "bert-base" in bert_model:
            bertconfig.relation_emb_dim = 768

        bertconfig.node_emb_dim = node_emb_dim
        bertconfig.dep_rels = len(deprel_dict)
        bertconfig.gnn_depth = gnn_depth
        bertconfig.gnn = gnn

    use_graph_data = graph_data_source is not None
    wandb.config.use_graph_data = use_graph_data

    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model_class = CONNECTION_TYPE_TO_CLASS[graph_connection_type]

    src_model = model_class.from_pretrained(
        bert_model, config=src_bertconfig, use_graph_data=use_graph_data
    )

    src_model.load_state_dict(torch.load(src_checkpoint_file, map_location="cpu"))
    model = model_class.from_pretrained(
        bert_model, config=tgt_bertconfig, use_graph_data=use_graph_data
    )

    src_model_dict = src_model.state_dict()
    model_dict = model.state_dict()

    for k, v in src_model_dict.items():
        if k in ["rel_classifier.weight", "rel_classifier.bias"]:
            continue
        model_dict[k] = v
    model.load_state_dict(model_dict)

    wandb.config.model_class = type(model)
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
        running_loss = 0.0

        model.train()

        for i, data in enumerate(tqdm(train_loader)):
            tokens_tensors = data["tokens_tensors"].to(device)
            segments_tensors = data["segments_tensors"].to(device)
            e1_mask = data["e1_mask"].to(device)
            e2_mask = data["e2_mask"].to(device)
            masks_tensors = data["masks_tensors"].to(device)
            labels = data["label_ids"].to(device)

            if data["graph_data"] is not None:
                graph_data = data["graph_data"].to(device)
            else:
                graph_data = None

            # we don't currently need to load these, until we start using both AMRs and dependencies.
            # we do want to check that the correct graph data is being used, but can be ommitted
            # with python -o
            if __debug__:
                dependency_tensors = data["dependency_data"].to(device)
                amr_tensors = data["amr_data"].to(device)
                assert graph_data_not_equal(dependency_tensors.x, amr_tensors.x)
                if case.startswith("amr"):
                    assert (graph_data.x == amr_tensors.x).all()
                elif case.startswith("dep"):
                    assert (graph_data.x == dependency_tensors.x).all()
                else:
                    assert graph_data is None

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

            loss.backward()
            running_loss += loss.item()
            if (i + 1) % grad_accumulation_steps == 0:
                optimizer.step()

        # final gradient step if we hadn't handled it already
        if (i + 1) % grad_accumulation_steps != 0:
            optimizer.step()

            wandb.log({"batch_loss": loss.item()})
        wandb.log({"loss": running_loss})

        print("============== EVALUATION ==============")

        p_train, r_train, f1_train, _ = eval_model_add_context(
            model=model,
            data=train_data,
            dataloader=train_loader,
            tokenizer=tokenizer,
            device=device,
            id2lbl=id2lbl,
            split_name="train",
        )
        print(f"Train data F1: {f1_train} \t Precision: {p_train} \t Recall: {r_train}")
        wandb.log({"train_f1": f1_train})

        p_dev, r_dev, f1_dev, dev_eval_df = eval_model_add_context(
            model=model,
            data=dev_data,
            dataloader=dev_loader,
            tokenizer=tokenizer,
            device=device,
            id2lbl=id2lbl,
            split_name="dev",
        )
        wandb.log({"dev_f1": f1_dev})
        print(f"Eval data F1: {f1_dev} \t Precision: {p_dev} \t Recall: {r_dev}")

        if f1_dev > best_f1:
            kill_cnt = 0

            best_p, best_r, best_f1 = p_dev, r_dev, f1_dev
            best_model = model
            best_eval_df = dev_eval_df
            torch.save(best_model.state_dict(), tgt_checkpoint_file)
        else:
            kill_cnt += 1
            if kill_cnt >= patience:
                torch.save(best_model.state_dict(), tgt_checkpoint_file)
                save_transfer_eval_df(
                    best_eval_df,
                    src_dataset_name,
                    tgt_dataset_name,
                    fewshot,
                    "dev",
                    seed,
                    case,
                    base_path,
                )
                break

        wandb.log({"running_best_f1": best_f1})
        print(f"[best val] precision: {best_p:.4f}, recall: {best_r:.4f}, f1 score: {best_f1:.4f}")
    wandb.log({"best_f1": best_f1, "best_precision": best_p, "best_recall": best_r})
    torch.save(best_model.state_dict(), tgt_checkpoint_file)
    save_transfer_eval_df(
        best_eval_df, src_dataset_name, tgt_dataset_name, fewshot, "dev", seed, case, base_path
    )

    print("============== EVALUATION ON TEST DATA ==============")
    best_model.to(device)
    best_model.eval()
    pt, rt, test_f1, test_eval_df = eval_model_add_context(
        model=best_model,
        data=test_data,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
        id2lbl=id2lbl,
        split_name="test",
    )
    save_transfer_eval_df(
        test_eval_df, src_dataset_name, tgt_dataset_name, fewshot, "test", seed, case, base_path
    )
    wandb.log({"test_f1": test_f1, "test_precision": pt, "test_recall": rt})
    wandb.run.finish()


def train_transfer_model_wrapper(
    src_dataset: str,
    tgt_dataset: str,
    fewshot: Union[int, float],
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

    train_transfer_model(
        src_dataset_name=src_dataset,
        src_dataset=src_dataset_loaded,
        tgt_dataset_name=tgt_dataset,
        tgt_dataset=tgt_dataset_loaded,
        fewshot=fewshot,
        device=device,
        seed=seed,
        conf_blob=configuration,
        **configuration,
    )


if __name__ == "__main__":
    fire.Fire(train_transfer_model_wrapper)

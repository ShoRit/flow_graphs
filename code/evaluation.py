from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import torch
from tqdm.auto import tqdm


def get_labels_and_model_predictions(model, loader, device):
    y_true, y_pred = [], []

    for data in tqdm(loader):
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

        dependency_tensors = data["dependency_data"].to(device)
        amr_tensors = data["amr_data"].to(device)

        with torch.no_grad():
            outputs_dict = model(
                input_ids=tokens_tensors,
                token_type_ids=segments_tensors,
                e1_mask=e1_mask,
                e2_mask=e2_mask,
                attention_mask=masks_tensors,
                labels=labels,
                graph_data=graph_data,
            )
            logits = outputs_dict["logits"]

        _, pred = torch.max(logits, 1)
        y_pred.extend(list(np.array(pred.cpu().detach())))
        y_true.extend(list(np.array(labels.cpu().detach())))

    return y_true, y_pred


def add_evaluation_context(df, tokenizer):
    # n1_mask, n2_mask --> Do not have any entity
    # Compute readability statistics
    # Error analysis based on label_class and dataset, generate a confusion matrix
    processed_dict = defaultdict(list)
    for index, row in df.iterrows():
        arg1_mask = torch.tensor(row["arg1_ids"]).type(torch.bool)
        arg2_mask = torch.tensor(row["arg2_ids"]).type(torch.bool)
        tokens = torch.tensor(row["tokens"])
        tok_locs = torch.tensor(row["tok_range"])
        ent1_locs = np.array(tok_locs[arg1_mask])
        ent2_locs = np.array(tok_locs[arg2_mask])
        ent1_start, ent1_end = ent1_locs[0][0], ent1_locs[-1][1]
        ent2_start, ent2_end = ent2_locs[0][0], ent2_locs[-1][1]
        ent1_name = tokenizer.decode(tokens[arg1_mask])
        ent2_name = tokenizer.decode(tokens[arg2_mask])

        processed_dict["ent1_name"].append(ent1_name)
        processed_dict["ent2_name"].append(ent2_name)
        processed_dict["ent1_start"].append(ent1_start)
        processed_dict["ent2_start"].append(ent2_start)
        processed_dict["ent1_end"].append(ent1_end)
        processed_dict["ent2_end"].append(ent2_end)

        processed_dict["ent1_amr"].append(row.amr_data.n1_mask.sum().item())
        processed_dict["ent2_amr"].append(row.amr_data.n2_mask.sum().item())

        processed_dict["sent"].append(row["sent"])
        processed_dict["labels"].append(row["label"])
        processed_dict["predictions"].append(row["predictions"])

    processed_df = pd.DataFrame(processed_dict)
    return processed_df


def seen_eval(model, loader, device):
    model.eval()
    y_true, y_pred = get_labels_and_model_predictions(model, loader, device)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return p, r, f1


def evaluate_model(model, data, dataloader, device, id2lbl, split_name):
    eval_df = pd.DataFrame(data)

    with torch.no_grad():
        labels, predictions = get_labels_and_model_predictions(model, dataloader, device)
        eval_df["label_idxs"] = labels
        eval_df["prediction_idxs"] = predictions
        eval_df["predictions"] = [id2lbl[pred] for pred in predictions]
        p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")
        print(f"{split_name}\tF1: {f1}\tPrecision: {p}\tRecall: {r}")

    return p, r, f1, eval_df


def eval_model_add_context(model, data, dataloader, tokenizer, device, id2lbl, split_name):
    p, r, f1, eval_df = evaluate_model(model, data, dataloader, device, id2lbl, split_name)
    contextualized_df = add_evaluation_context(eval_df, tokenizer)
    return p, r, f1, contextualized_df


def get_indomain_eval_filename(dataset_name, split, seed, case):
    return f"indomain-results-{split}-{dataset_name}-seed_{seed}-{case}.csv"

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from tqdm.auto import tqdm


def seen_eval(model, loader, device, use_amr):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []
    for data in tqdm(loader):
        tokens_tensors = data["tokens_tensors"].to(device)
        segments_tensors = data["segments_tensors"].to(device)
        e1_mask = data["e1_mask"].to(device)
        e2_mask = data["e2_mask"].to(device)
        masks_tensors = data["masks_tensors"].to(device)
        labels = data["label_ids"].to(device)
        dependency_tensors = data["dependency_data"].to(device)
        amr_tensors = data["amr_data"].to(device)

        graph_data = amr_tensors if use_amr else dependency_tensors

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

        total += labels.size(0)
        _, pred = torch.max(logits, 1)
        correct += (pred == labels).sum().item()
        y_pred.extend(list(np.array(pred.cpu().detach())))
        y_true.extend(list(np.array(labels.cpu().detach())))

    f1 = f1_score(y_true, y_pred, average="macro")
    p1 = precision_score(y_true, y_pred, average="macro")
    r1 = recall_score(y_true, y_pred, average="macro")

    return p1, r1, f1

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from tqdm.auto import tqdm


def seen_eval(model, loader, device):
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

        total += labels.size(0)
        _, pred = torch.max(logits, 1)
        correct += (pred == labels).sum().item()
        y_pred.extend(list(np.array(pred.cpu().detach())))
        y_true.extend(list(np.array(labels.cpu().detach())))

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    return p, r, f1

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

# from sklearn.metrics import f1_score, precision_recall_fscore_support
# from sklearn.metrics import precision_score, recall_score,
from sklearn.metrics import classification_report, f1_score
from helper import *


def compute_macro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
    """
    This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
    code borrowed from the following link:
    https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
    """
    if i == -1:
        i = len(predicted_idx)

    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = predicted_idx[:i] == r
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1


def extract_relation_emb(model, testloader, device, use_amr):
    out_relation_embs = None
    model.eval()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for data in tqdm(testloader):

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
                device=device,
            )
            logits = outputs_dict["logits"]

        # if out_relation_embs is None:
        #     out_relation_embs = out_relation_emb
        # else:
        #     out_relation_embs = torch.cat((out_relation_embs, out_relation_emb))
        if out_relation_embs is None:
            out_relation_embs = outputs_dict["relation_embeddings"]
        else:
            out_relation_embs = torch.cat((out_relation_embs, outputs_dict["relation_embeddings"]))

    return out_relation_embs


# def evaluate(preds, y_attr, y_label, idxmap, num_train_y, dist_func='inner'):
#     assert dist_func in ['inner', 'euclidian', 'cosine']
#     if dist_func == 'inner':
#         tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=lambda a, b: -(a@b))
#     elif dist_func == 'euclidian':
#         tree = NearestNeighbors(n_neighbors=1)
#     elif dist_func == 'cosine':
#         tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=lambda a, b: -((a@b) / (( (a@a) **.5) * ( (b@b) ** .5) )))
#     tree.fit(y_attr)
#     predictions = tree.kneighbors(preds, 1, return_distance=False).flatten() + num_train_y
#     p_macro, r_macro, f_macro = compute_macro_PRF(predictions, y_label)
#     return p_macro, r_macro, f_macro


def get_hits(y_label, tree, preds, k):
    predictions = tree.kneighbors(preds, k, return_distance=False)
    hits = 0
    for y_true, y_pred in zip(y_label, predictions):
        if y_true in y_pred:
            hits += 1

    return hits / len(y_label)


def evaluate(preds, y_attr, y_label, idxmap, dist_func="inner", lbl2id=None, num_neighbors=1):
    assert dist_func in ["inner", "euclidian", "cosine"]
    if dist_func == "inner":
        tree = NearestNeighbors(
            n_neighbors=num_neighbors,
            algorithm="ball_tree",
            metric=lambda a, b: -(a @ b),
        )
    elif dist_func == "euclidian":
        tree = NearestNeighbors(n_neighbors=num_neighbors)
    elif dist_func == "cosine":
        tree = NearestNeighbors(
            n_neighbors=num_neighbors,
            algorithm="ball_tree",
            metric=lambda a, b: -((a @ b) / (((a @ a) ** 0.5) * ((b @ b) ** 0.5))),
        )

    tree.fit(y_attr)
    predictions = tree.kneighbors(preds, 1, return_distance=False).flatten()
    p_macro, r_macro, f_macro = compute_macro_PRF(predictions, y_label)

    H_K = get_hits(y_label, tree, preds, num_neighbors)

    if lbl2id is not None:
        print(classification_report(y_label, predictions, target_names=[i for i in lbl2id]))

    H_1 = get_hits(y_label, tree, preds, 1)
    H_2 = get_hits(y_label, tree, preds, 2)
    H_3 = get_hits(y_label, tree, preds, 3)

    print(f"H@1 {round(H_1,2)}, H@2 {round(H_2,2)} H@3 {round(H_3,2)}")

    return p_macro, r_macro, f_macro, H_K

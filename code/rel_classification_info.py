import argparse

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    BertConfig,
)

from dataloader import GraphyRelationsDataset, create_mini_batch
from helper import check_file, load_deprels, load_dill
from modeling.bert import BertRGCNRelationClassifier
from modeling.zsbert import ZSBert_RGCN
from preprocess import generate_reldesc
from utils import get_device, seed_everything
import wandb
from zsbert_evaluate import evaluate, extract_relation_emb


# wandb.login()
# wandb.init(project="test-project", entity="flow-graphs-cmu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dataset", help="choice of src_dataset", type=str, default="risec")
    parser.add_argument("--tgt_dataset", help="choice of tgt_dataset", type=str, default="japflow")
    parser.add_argument("--mode", help="choice of operation", type=str, default="eval")
    parser.add_argument(
        "--domain", help="seen or unseen domain", type=str, default="src"
    )  # src -->

    parser.add_argument(
        "--bert_model",
        help="choice of bert model  ",
        type=str,
        default="bert-base-uncased",
    )
    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument("--gpu", help="choice of device", type=str, default="0")

    parser.add_argument("--node_emb_dim", help="number of unseen classes", type=int, default=768)
    parser.add_argument("--dep", help="dependency_parsing", type=str, default="1")
    parser.add_argument("--amr", help="amr_parsing", type=str, default="0")
    parser.add_argument("--gnn", help="Choice of GNN used", type=str, default="rgcn")
    parser.add_argument("--gnn_depth", help="Depth of GNN used", type=int, default=2)

    parser.add_argument("--n_unseen", help="number of unseen classes", type=int, default=10)
    parser.add_argument("--gamma", help="margin factor gamma", type=float, default=7.5)
    parser.add_argument("--alpha", help="balance coefficient alpha", type=float, default=0.5)
    parser.add_argument(
        "--dist_func", help="distance computing function", type=str, default="cosine"
    )

    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_neighbors", type=int, default=2)
    parser.add_argument("--omit_rels", type=str, default="")

    args = parser.parse_args()
    return args


def get_lbl_features(data, rel2desc_emb):

    labels = [elem["label"] for elem in data]
    lbl2id = {}
    for elem in labels:
        if elem not in lbl2id:
            lbl2id[elem] = len(lbl2id)

    test_y_attr, test_y = [], []
    test_idxmap = {}

    for i, elem in enumerate(data):
        lbl = lbl2id[elem["label"]]
        test_y.append(lbl)
        test_idxmap[i] = lbl

    test_y_attr = [rel2desc_emb[i] for i in lbl2id]
    test_y_attr = np.array(test_y_attr)
    test_y = np.array(test_y)

    return test_y, test_idxmap, labels, test_y_attr, lbl2id


def get_known_lbl_features(data, rel2desc_emb, lbl2id):
    labels = [elem["label"] for elem in data]

    test_y_attr, test_y = [], []
    test_idxmap = {}

    for i, elem in enumerate(data):
        lbl = lbl2id[elem["label"]]
        test_y.append(lbl)
        test_idxmap[i] = lbl

    test_y_attr = [rel2desc_emb[i] for i in lbl2id]
    test_y_attr = np.array(test_y_attr)
    test_y = np.array(test_y)

    return test_y, test_idxmap, labels, test_y_attr, lbl2id


def seen_eval(model, loader, device):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []
    for data in tqdm(loader):
        (
            tokens_tensors,
            segments_tensors,
            marked_e1,
            marked_e2,
            masks_tensors,
            relation_emb,
            labels,
            graph_data,
        ) = [t.to(device) for t in data]

        with torch.no_grad():
            outputs_dict = model(
                input_ids=tokens_tensors,
                token_type_ids=segments_tensors,
                e1_mask=marked_e1,
                e2_mask=marked_e2,
                attention_mask=masks_tensors,
                input_relation_emb=relation_emb,
                labels=labels,
                graph_data=graph_data,
                device=device,
            )
            logits = outputs_dict["logits"]

        total += labels.size(0)
        _, pred = torch.max(logits, 1)
        correct += (pred == labels).sum().item()
        y_pred.extend(list(np.array(pred.cpu().detach())))
        y_true.extend(list(np.array(labels.cpu().detach())))

    f1 = f1_score(y_true, y_pred, average="macro")
    p1, r1 = precision_score(y_true, y_pred, average="macro"), recall_score(
        y_true, y_pred, average="macro"
    )
    # print(f'val acc: {correct/total}, f1 : {f1_score(y_true,y_pred, average="macro")}')

    return p1, r1, f1


def main(args):
    seed_everything(args.seed)
    device = get_device()
    src_dir = f"/projects/flow_graphs/data/{args.src_dataset}"
    src_file = f"{src_dir}/data_amr.dill"
    tgt_dir = f"/projects/flow_graphs/data/{args.tgt_dataset}"
    tgt_file = f"{tgt_dir}/data_amr.dill"

    deprel_dict = load_deprels(enhanced=False)

    if check_file(src_file):
        src_dataset = load_dill(src_file)
    else:
        raise FileNotFoundError("SRC FILE IS NOT CREATED")

    if check_file(tgt_file):
        tgt_dataset = load_dill(tgt_file)
    else:
        raise FileNotFoundError("TGT FILE IS NOT CREATED")

    if args.src_dataset == args.tgt_dataset:
        train_data, dev_data, test_data = (
            src_dataset["train"]["rels"],
            src_dataset["dev"]["rels"],
            src_dataset["test"]["rels"],
        )
    else:
        train_data, dev_data, test_data = (
            src_dataset["train"]["rels"],
            tgt_dataset["dev"]["rels"],
            tgt_dataset["test"]["rels"],
        )

    rel2desc, all_rel2id, id2all_rel, rel2desc_emb = generate_reldesc()

    print(
        "train size: {}, dev size {}, test size: {}".format(
            len(train_data), len(dev_data), len(test_data)
        )
    )
    print("Data is successfully loaded")
    train_label = [data["label"] for data in train_data]

    bertconfig = BertConfig.from_pretrained(args.bert_model, num_labels=len(set(train_label)))

    if "bert-large" in args.bert_model:
        bertconfig.relation_emb_dim = 1024
    elif "bert-base" in args.bert_model:
        bertconfig.relation_emb_dim = 768

    bertconfig.margin = args.gamma
    bertconfig.alpha = args.alpha
    bertconfig.dist_func = args.dist_func

    bertconfig.node_emb_dim = args.node_emb_dim
    bertconfig.dep_rels = len(deprel_dict)
    bertconfig.dep = args.dep
    bertconfig.gnn_depth = args.gnn_depth
    bertconfig.amr = args.amr
    bertconfig.gnn = args.gnn

    use_graph_data = bool(args.amr) or bool(args.dep)

    model = BertRGCNRelationClassifier.from_pretrained(
        args.bert_model, config=bertconfig, use_graph_data=use_graph_data
    )
    model = model.to(device)

    if args.domain == "src":
        labels = sorted(list(set(train_label)))
        lbl2id = {lbl: idx for idx, lbl in enumerate(labels)}
        (
            train_y,
            train_idxmap,
            train_labels,
            train_y_attr,
            train_lbl2id,
        ) = get_known_lbl_features(train_data, rel2desc_emb, lbl2id)
    else:
        (
            train_y,
            train_idxmap,
            train_labels,
            train_y_attr,
            train_lbl2id,
        ) = get_lbl_features(train_data, rel2desc_emb)

    trainset = GraphyRelationsDataset(train_data, train_lbl2id)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle=True
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_p, best_r, best_f1 = 0, 0, 0

    checkpoint_file = (
        f"/scratch/sgururaj/flow_graphs/checkpoints/{args.src_dataset}-"
        f"{args.tgt_dataset}-{args.domain}-dep_{args.dep}-amr_{args.amr}-gnn_{args.gnn}-gnn-depth_{args.gnn_depth}-seed_{args.seed}-lr_{args.lr}.pt"
    )

    if args.mode == "train":
        wandb.login()
        wandb.init(
            project="narrative-flow-staging",
            entity="flow-graphs-cmu",
            name=f'{checkpoint_file.split("/")[-1]}',
        )

        wandb.config.update(vars(args))

        if args.domain == "src":
            (
                dev_y,
                dev_idxmap,
                dev_labels,
                dev_y_attr,
                dev_lbl2id,
            ) = get_known_lbl_features(dev_data, rel2desc_emb, lbl2id)
        else:
            dev_y, dev_idxmap, dev_labels, dev_y_attr, dev_lbl2id = get_lbl_features(
                dev_data, rel2desc_emb
            )

        devset = GraphyRelationsDataset(dev_data, dev_lbl2id)
        devloader = DataLoader(devset, batch_size=args.batch_size, collate_fn=create_mini_batch)
        kill_cnt = 0

        for epoch in range(args.epochs):
            print(f"============== TRAIN ON THE {epoch+1}-th EPOCH ==============")
            running_loss, correct, total = 0.0, 0, 0

            model.train()
            y_true, y_pred = [], []

            for data in tqdm(trainloader):
                tokens_tensors = data["tokens_tensors"].to(device)
                segments_tensors = data["segments_tensors"].to(device)
                e1_mask = data["e1_mask"].to(device)
                e2_mask = data["e2_mask"].to(device)
                masks_tensors = data["masks_tensors"].to(device)
                labels = data["label_ids"].to(device)
                dependency_tensors = data["dependency_data"].to(device)
                amr_tensors = data["amr_data"].to(device)

                graph_data = amr_tensors if args.amr else dependency_tensors

                optimizer.zero_grad()
                output_dict = model(
                    input_ids=tokens_tensors,
                    token_type_ids=segments_tensors,
                    e1_mask=e1_mask,
                    e2_mask=e2_mask,
                    attention_mask=masks_tensors,
                    labels=labels,
                    graph_data=graph_data,
                    device=device,
                )
                loss, logits = output_dict["loss"], output_dict["logits"]

                total += labels.size(0)
                _, pred = torch.max(logits, 1)
                correct += (pred == labels).sum().item()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                wandb.log({"batch_loss": loss.item()})
                y_pred.extend(list(np.array(pred.cpu().detach())))
                y_true.extend(list(np.array(labels.cpu().detach())))

            print(f'train acc: {correct/total}, f1 : {f1_score(y_true,y_pred, average="macro")}')

            print("============== EVALUATION ON DEV DATA ==============")

            wandb.log({"loss": running_loss})

            if args.domain == "src":
                pt, rt, f1t = seen_eval(model, trainloader, device=device)
                print(f"Train data {f1t} \t Prec {pt} \t Rec {rt}")
                wandb.log({"train_f1": f1t})
                pt, rt, f1t = seen_eval(model, devloader, device=device)
                wandb.log({"dev_f1": f1t})
                print(f"Eval data {f1t} \t Prec {pt} \t Rec {rt}")

            else:
                preds = (
                    extract_relation_emb(model, devloader, device=device, use_amr=args.amr)
                    .cpu()
                    .numpy()
                )
                pt, rt, f1t, h_K = evaluate(preds, dev_y_attr, dev_y, dev_idxmap, args.dist_func)
                print(
                    f"[val] f1 score: {f1t:.4f}, precision: {pt:.4f}, recall: {rt:.4f}, H@K :{h_K}"
                )

            if f1t > best_f1:
                best_p, best_r, best_f1 = pt, rt, f1t
                best_model = model
                kill_cnt = 0
                torch.save(best_model.state_dict(), checkpoint_file)
            else:
                kill_cnt += 1
                if kill_cnt >= args.patience:
                    torch.save(best_model.state_dict(), checkpoint_file)
                    break

            wandb.log({"running_best_f1": best_f1})
            print(
                f"[best val] precision: {best_p:.4f}, recall: {best_r:.4f}, f1 score: {best_f1:.4f}"
            )
        wandb.log({"best_f1": best_f1, "best_precision": best_p, "best_recall": best_r})
        torch.save(best_model.state_dict(), checkpoint_file)

        if args.domain == "src":
            (
                test_y,
                test_idxmap,
                test_labels,
                test_y_attr,
                test_lbl2id,
            ) = get_known_lbl_features(test_data, rel2desc_emb, lbl2id)
        else:
            (
                test_y,
                test_idxmap,
                test_labels,
                test_y_attr,
                test_lbl2id,
            ) = get_lbl_features(test_data, rel2desc_emb)

        testset = GraphyRelationsDataset(test_data, test_lbl2id)
        testloader = DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)
        best_model = best_model.to(device)
        best_model.eval()
        pt, rt, test_f1 = seen_eval(best_model, testloader, device=device)
        wandb.log({"test_f1": test_f1})

    if args.mode == "eval":
        if not check_file(checkpoint_file):
            print("MODEL CANNOT BE LOADED BY THESE SPECIFICATIONS")
            return
        if args.domain == "src":
            (
                test_y,
                test_idxmap,
                test_labels,
                test_y_attr,
                test_lbl2id,
            ) = get_known_lbl_features(test_data, rel2desc_emb, lbl2id)
        else:
            (
                test_y,
                test_idxmap,
                test_labels,
                test_y_attr,
                test_lbl2id,
            ) = get_lbl_features(test_data, rel2desc_emb)

        testset = GraphyRelationsDataset(test_data, test_lbl2id)
        testloader = DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)

        model = ZSBert_RGCN.from_pretrained(args.bert_model, config=bertconfig)
        model = model.to(device)
        model.load_state_dict(torch.load(checkpoint_file))
        model.eval()

        if args.domain == "src":
            pt, rt, f1t = seen_eval(model, testloader, device=device)
            print(f"Train data {f1t} \t Prec {pt} \t Rec {rt}")
        else:
            preds = (
                extract_relation_emb(model, testloader, device=device, use_amr=args.amr)
                .cpu()
                .numpy()
            )
            pt, rt, f1t, h_K = evaluate(
                preds,
                test_y_attr,
                test_y,
                test_idxmap,
                args.dist_func,
                test_lbl2id,
                num_neighbors=args.num_neighbors,
            )

            print(
                f"[best test] precision: {pt:.4f}, recall: {rt:.4f}, f1 score: {f1t:.4f}, H@K : {h_K}"
            )

    if args.mode == "batch_eval":

        if args.domain == "src":
            (
                test_y,
                test_idxmap,
                test_labels,
                test_y_attr,
                test_lbl2id,
            ) = get_known_lbl_features(test_data, rel2desc_emb, lbl2id)
        else:
            (
                test_y,
                test_idxmap,
                test_labels,
                test_y_attr,
                test_lbl2id,
            ) = get_lbl_features(test_data, rel2desc_emb)

        testset = GraphyRelationsDataset(test_data, test_lbl2id)
        testloader = DataLoader(testset, batch_size=args.batch_size, collate_fn=create_mini_batch)

        model = ZSBert_RGCN.from_pretrained(args.bert_model, config=bertconfig)
        model = model.to(device)

        f1_arr, prec_arr, rec_arr, hits_arr = [], [], [], []

        for seed in range(0, 3):
            checkpoint_file = f"/projects/flow_graphs/checkpoints/{args.src_dataset}-{args.src_dataset}-src-dep_{args.dep}-amr_{args.amr}-gnn_{args.gnn}-gnn-depth_{args.gnn_depth}-alpha_{args.alpha}-seed_{seed}-lr_{args.lr}.pt"
            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()

            if args.domain == "src":
                pt, rt, f1t = seen_eval(model, testloader, device=device)
                f1_arr.append(f1t)
                prec_arr.append(pt)
                rec_arr.append(rt)
            else:
                preds = (
                    extract_relation_emb(model, testloader, device=device, use_amr=args.amr)
                    .cpu()
                    .numpy()
                )
                pt, rt, f1t, h_K = evaluate(
                    preds,
                    test_y_attr,
                    test_y,
                    test_idxmap,
                    args.dist_func,
                    test_lbl2id,
                    num_neighbors=args.num_neighbors,
                )
                # print(f'[best test] precision: {pt:.4f}, recall: {rt:.4f}, f1 score: {f1t:.4f}, H@K : {h_K}')
                f1_arr.append(f1t)
                prec_arr.append(pt)
                rec_arr.append(rt)
                hits_arr.append(h_K)

        wandb.login()
        wandb.init(
            project="narrative-flow-eval",
            entity="flow-graphs-cmu",
            name=f'{checkpoint_file.split("/")[-1]}',
        )
        wandb.log({"f1": np.mean(f1_arr)})
        wandb.log({"std_f1": np.std(f1_arr)})
        wandb.log({"precision": np.mean(prec_arr)})
        wandb.log({"recall": np.mean(rec_arr)})


if __name__ == "__main__":
    args = get_args()
    main(args)

import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as geo_DataLoader


def create_mini_batch(samples):

    tokens_tensors = [s["tokens"] for s in samples]
    segments_tensors = [s["segments"] for s in samples]
    e1_mask = [s["e1_mask"] for s in samples]
    e2_mask = [s["e2_mask"] for s in samples]

    if samples[0]["label"] is not None:
        label_ids = torch.stack([s["label"] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    e1_mask = pad_sequence(e1_mask, batch_first=True)
    e2_mask = pad_sequence(e2_mask, batch_first=True)
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    dependency_data = [s["dep_data"] for s in samples]
    dependency_loader = geo_DataLoader(dependency_data, batch_size=len(dependency_data))
    dependency_tensors = [e for e in dependency_loader][0]

    amr_data = [s["amr_data"] for s in samples]
    amr_loader = geo_DataLoader(amr_data, batch_size=len(amr_data))
    amr_tensors = [e for e in amr_loader][0]

    return {
        "tokens_tensors": tokens_tensors,
        "segments_tensors": segments_tensors,
        "e1_mask": e1_mask,
        "e2_mask": e2_mask,
        "masks_tensors": masks_tensors,
        "label_ids": label_ids,
        "dependency_data": dependency_tensors,
        "amr_data": amr_tensors,
    }


class ZSBertRelDataset(Dataset):
    def __init__(self, dataset, rel2id, tokenizer, params, data_idx=0, domain="src"):
        self.dataset = dataset
        self.rel2id = rel2id
        self.p = params
        self.tokenizer = tokenizer
        self.data_idx = data_idx
        self.cls_tok = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("[CLS]"))
        self.sep_tok = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("[SEP]"))
        self.domain = domain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele = self.dataset[idx]

        tokens = torch.tensor(self.cls_tok + ele["tokens"] + self.sep_tok)
        marked_1 = torch.tensor([0] + ele["arg1_ids"] + [0])
        marked_2 = torch.tensor([0] + ele["arg2_ids"] + [0])
        segments = torch.tensor([0] * len(tokens))
        desc_emb = ele["desc_emb"]

        if self.domain == "src":
            label = torch.tensor(self.rel2id[ele["label"]])
        else:
            label = None

        return (tokens, segments, marked_1, marked_2, desc_emb, label)


class GraphyRelationsDataset(Dataset):
    def __init__(self, dataset, rel2id, tokenizer, params, fewshot=1.0, skip_hashing=False):
        if fewshot == 1.0:
            self.dataset = tuple(dataset)
        else:
            self.dataset = tuple(
                random.sample(
                    dataset,
                    int(fewshot * len(dataset)),
                )
            )

        self.rel2id = rel2id

        if not skip_hashing:
            self.dataset_hash = hash(self.dataset)
        else:
            self.dataset_hash = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        instance = self.dataset[idx]
        return {
            "tokens": torch.tensor(instance["tokens"]),
            "segments": torch.tensor([0] * len(instance["tokens"])),
            "e1_mask": torch.tensor(instance["arg1_ids"]),
            "e2_mask": torch.tensor(instance["arg2_ids"]),
            "label": torch.tensor(self.rel2id[instance["label"]]),
            "dep_data": instance["dep_data"],
            "amr_data": instance["dep_data"],
        }

from collections import defaultdict
import random
from typing import Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as geo_DataLoader

from validation import GRAPH_DATA_KEYS


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

    graph_data = [s["graph_data"] for s in samples]
    if any([graph is None for graph in graph_data]):
        graph_tensors = None
    else:
        graph_loader = geo_DataLoader(graph_data, batch_size=len(graph_data))
        graph_tensors = [e for e in graph_loader][0]

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
        "graph_data": graph_tensors,
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


def sample_fraction(dataset, fraction):
    sampled_instances = random.sample(list(enumerate(dataset)), int(fraction * len(dataset)))

    sampled_indices, sampled_dataset = zip(*sampled_instances)
    return sampled_indices, sampled_dataset


def stratified_sample(dataset, n_per_class):
    instances_by_label = defaultdict(list)
    for index, instance in enumerate(dataset):
        instances_by_label[instance["label"]].append((index, instance))

    all_sampled_instances = []
    for label, instance_list in instances_by_label.items():
        if n_per_class > len(instance_list):
            print(
                f"Requested more labels of class {label} ({n_per_class}) than exist ({len(instance_list)}). Using all examples."
            )
            all_sampled_instances.extend(instance_list)
        else:
            sampled_instances = random.sample(instance_list, n_per_class)
            all_sampled_instances.extend(sampled_instances)

    random.shuffle(all_sampled_instances)
    sampled_indices, sampled_dataset = zip(*all_sampled_instances)
    return sampled_indices, sampled_dataset


class GraphyRelationsDataset(Dataset):
    def __init__(
        self,
        dataset: dict,
        rel2id: dict,
        graph_data_source: str,
        max_seq_len: int,
        fewshot: Union[float, int] = 1.0,
    ):
        if fewshot == 1.0:
            self.dataset = tuple(dataset)
        elif fewshot < 1.0:
            sampled_indices, sampled_dataset = sample_fraction(dataset, fewshot)
            self.dataset = sampled_dataset
            self.sampled_indices = tuple(sampled_indices)
            self.sampled_index_hash = hash(sampled_indices)
        elif isinstance(fewshot, int):
            sampled_indices, sampled_dataset = stratified_sample(dataset, n_per_class=fewshot)
            self.dataset = sampled_dataset
            self.sampled_indices = tuple(sampled_indices)
            self.sampled_index_hash = hash(sampled_indices)
        else:
            raise AssertionError(
                f"Unexpected value for parameter 'fewshot': {fewshot}. Parameter should be a float in the range (0, 1.0] or an int > 0"
            )

        self.rel2id = rel2id
        self.max_seq_len = max_seq_len
        self.graph_data_key = GRAPH_DATA_KEYS[graph_data_source]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        instance = self.dataset[idx]

        tokens = torch.tensor(
            instance["tokens"] + [0] * (self.max_seq_len - len(instance["tokens"]))
        )
        e1_mask = torch.tensor(
            instance["arg1_ids"] + [0] * (self.max_seq_len - len(instance["tokens"]))
        )
        e2_mask = torch.tensor(
            instance["arg2_ids"] + [0] * (self.max_seq_len - len(instance["tokens"]))
        )
        segments = torch.tensor([0] * len(tokens))

        if self.graph_data_key is not None:
            graph_data = instance[self.graph_data_key]
        else:
            graph_data = None

        return {
            "tokens": tokens,
            "segments": segments,
            "e1_mask": e1_mask,
            "e2_mask": e2_mask,
            "label": torch.tensor(self.rel2id[instance["label"]]),
            "graph_data": graph_data,
            "dep_data": instance["dep_data"],
            "amr_data": instance["amr_data"],
        }


def get_data_loaders(
    train_data, dev_data, test_data, lbl2id, graph_data_source, max_seq_len, batch_size, fewshot=1.0
):
    train_set = GraphyRelationsDataset(
        train_data, lbl2id, graph_data_source, max_seq_len, fewshot=fewshot
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=True
    )

    dev_set = GraphyRelationsDataset(dev_data, lbl2id, graph_data_source, max_seq_len)
    dev_loader = DataLoader(
        dev_set, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=False
    )

    test_set = GraphyRelationsDataset(test_data, lbl2id, graph_data_source, max_seq_len)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=create_mini_batch, shuffle=False
    )

    return train_loader, dev_loader, test_loader

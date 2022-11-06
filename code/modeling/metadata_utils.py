import os
import re

from transformers import BertConfig
import torch

from dataloading_utils import load_deprels
from modeling.bert import CONNECTION_TYPE_TO_CLASS
from utils import check_file

INDOMAIN_CHECKPOINT_RE = re.compile(
    "indomain-(?P<dataset_name>.*?)-(?P<case>.*?)-(?P<gnn>.*?)-depth_(?P<gnn_depth>.*?)-seed_(?P<seed>.*?)-lr_(?P<lr>.*?).pt"
)


def get_case(graph_data_source, graph_connection_type, **kwargs):
    return (
        "plaintext" if graph_data_source is None else f"{graph_data_source}_{graph_connection_type}"
    )


def get_experiment_config_from_filename(filename):
    match = INDOMAIN_CHECKPOINT_RE.fullmatch(os.path.basename(filename))
    if match is None:
        raise AssertionError(
            f"Passed filename does not validate as an indomain filename!\nPassed filename: {filename}"
        )

    case = match.group("case")

    if "amr_residual" in case:
        return "amr_residual"
    elif "dep_residual" in case:
        return "dep_residual"
    elif "amr" in case:
        return "amr"
    elif "dep" in case:
        return "dep"
    elif "plaintext" in case:
        return "baseline"
    else:
        raise AssertionError("Unable to identify experiment config from passed file name!")


def get_transfer_checkpoint_filename(
    src_dataset_name,
    tgt_dataset_name,
    fewshot,
    case,
    graph_connection_type,
    gnn,
    gnn_depth,
    seed,
    lr,
    **kwargs,
):
    return (
        f"transfer-{src_dataset_name}-{tgt_dataset_name}-fewshot_{fewshot}-{case}"
        f"-{graph_connection_type}"
        f"-{gnn}-depth"
        f"_{gnn_depth}-seed"
        f"_{seed}-lr_{lr}.pt"
    )


def get_indomain_checkpoint_filename(dataset_name, case, gnn, gnn_depth, seed, lr, **kwargs):
    filename = f"indomain-{dataset_name}-{case}-{gnn}-depth_{gnn_depth}-seed_{seed}-lr_{lr}.pt"
    assert INDOMAIN_CHECKPOINT_RE.fullmatch(filename)
    return filename


def parse_indomain_checkpoint_filename(filename):
    match = INDOMAIN_CHECKPOINT_RE.fullmatch(os.path.basename(filename))
    if match is None:
        raise AssertionError(
            f"Passed filename does not validate as an indomain filename!\nPassed filename: {filename}"
        )

    return match.groupdict()


def load_model_from_file(model_checkpoint_file, configuration, device, n_labels):
    check_file(model_checkpoint_file)

    bertconfig = BertConfig.from_pretrained(configuration["bert_model"], num_labels=n_labels)
    if "bert-large" in configuration["bert_model"]:
        bertconfig.relation_emb_dim = 1024
    elif "bert-base" in configuration["bert_model"]:
        bertconfig.relation_emb_dim = 768

    deprel_dict = load_deprels(
        path=os.path.join(configuration["base_path"], "data", "enh_dep_rel.txt"), enhanced=False
    )

    bertconfig.node_emb_dim = configuration["node_emb_dim"]
    bertconfig.dep_rels = len(deprel_dict)
    bertconfig.gnn_depth = configuration["gnn_depth"]
    bertconfig.gnn = configuration["gnn"]

    use_graph_data = configuration["graph_data_source"] is not None
    model_class = CONNECTION_TYPE_TO_CLASS[configuration["graph_connection_type"]]

    model = model_class.from_pretrained(
        configuration["bert_model"], config=bertconfig, use_graph_data=use_graph_data
    )

    model.load_state_dict(torch.load(model_checkpoint_file, map_location=device))
    model.to(device)
    return model

def get_case(graph_data_source, graph_connection_type, **kwargs):
    return (
        "plaintext" if graph_data_source is None else f"{graph_data_source}_{graph_connection_type}"
    )


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
    return f"indomain-{dataset_name}-{case}-{gnn}-depth_{gnn_depth}-seed_{seed}-lr_{lr}.pt"

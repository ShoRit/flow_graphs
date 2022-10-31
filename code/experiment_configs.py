__all__ = ["model_configurations"]

_base_config = {
    "base_path": "/projects/flow_graphs/",
    "checkpoint_folder": "/projects/flow_graphs/checkpoints",
    "bert_model": "bert-base-uncased",
    "node_emb_dim": 768,
    "gnn": "rgcn",
    "gnn_depth": 4,
    "max_seq_len": 512,
    "lr": 2e-5,
    "batch_size": 4,
    "grad_accumulation_steps": 1,
    "epochs": 30,
    "patience": 5,
    "wandb_entity": "flow-graphs-cmu",
    "wandb_project": "narrative-flow-simplified",
}

baseline_config = dict(
    **_base_config, **{"graph_data_source": None, "graph_connection_type": "concat"}
)

dep_config = dict(**_base_config, **{"graph_data_source": "dep", "graph_connection_type": "concat"})

amr_config = dict(**_base_config, **{"graph_data_source": "amr", "graph_connection_type": "concat"})

dep_residual = dict(
    **_base_config, **{"graph_data_source": "dep", "graph_connection_type": "residual"}
)

amr_residual = dict(
    **_base_config, **{"graph_data_source": "amr", "graph_connection_type": "residual"}
)


model_configurations = {
    "baseline": baseline_config,
    "dep": dep_config,
    "amr": amr_config,
    "dep_residual": dep_residual,
    "amr_residual": amr_residual,
}

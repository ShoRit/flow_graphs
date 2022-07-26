__all__ = ["model_configurations"]

_base_config = {
    "base_directory": "/projects/flow_graphs",
    "bert_model": "bert-base-uncased",
    "node_emb_dim": 768,
    "gnn": "rgcn",
    "gnn_depth": 2,
    "max_seq_len": 512,
    "lr": 2e-5,
    "batch_size": 16,
    "epochs": 30,
    "patience": 5,
}

baseline_config = dict(**_base_config, **{"dep": 0, "amr": 0})

dep_config = dict(**_base_config, **{"dep": 1, "amr": 0})

amr_config = dict(**_base_config, **{"dep": 1, "amr": 1})


model_configurations = {
    "baseline": baseline_config,
    "dep": dep_config,
    "amr": amr_config,
}

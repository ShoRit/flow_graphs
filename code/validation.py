import torch

GRAPH_DATA_SOURCES = (None, "dep", "amr")

# map from graph sources to keys in the preprocessed data for the graph-formatted data
GRAPH_DATA_KEYS = {None: None, "dep": "dep_data", "amr": "amr_data"}

assert set(GRAPH_DATA_KEYS.keys()) == set(GRAPH_DATA_SOURCES)


def graph_data_not_equal(data1: torch.Tensor, data2: torch.Tensor):
    try:
        is_equal = (data1 == data2).all()
        return not is_equal
    except RuntimeError as e:
        return True


def validate_graph_data_source(graph_data_source):
    assert graph_data_source in GRAPH_DATA_SOURCES

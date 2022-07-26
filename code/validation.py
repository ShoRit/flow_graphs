GRAPH_DATA_SOURCES = ("plaintext", "dep", "amr")


def validate_graph_data_source(graph_data_source):
    assert graph_data_source in GRAPH_DATA_SOURCES

import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx


def flip(tup):
    a, b = tup
    return (b, a)


def get_string_for_indices(tokens, node_indices, graph_data, tokenizer):
    tokens = []
    for index in node_indices:
        token = tokenizer.decode(np.array(tokens)[(graph_data.x[1] != -np.inf)[: len(tokens)]])
        tokens.append(token)
    return " ".join(tokens)


def get_path_edge_types(shortest_path, graph_data, amr_id2rel):
    path_edge_tuples = [
        (shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)
    ]

    path_edge_indices = []
    path_edge_types = []

    for path_edge_tuple in path_edge_tuples:
        for i, edge in enumerate(graph_data.edge_index.T):
            edge_tuple = tuple(edge.tolist())
            if edge_tuple == path_edge_tuple or edge_tuple == flip(path_edge_tuple):
                path_edge_indices.append(i)
                path_edge_types.append(amr_id2rel[graph_data.edge_type[i].item()])
    return path_edge_types


def get_type_paths_for_relation(graph_data, amr_id2rel):
    nx_graph = to_networkx(graph_data).to_undirected()
    n1_indices = np.nonzero(graph_data.n1_mask).flatten().tolist()
    n2_indices = np.nonzero(graph_data.n2_mask).flatten().tolist()

    type_paths = []
    for n1_index in n1_indices:
        for n2_index in n2_indices:
            shortest_path = nx.shortest_path(nx_graph, n1_index, n2_index)
            path_edge_types = get_path_edge_types(shortest_path, graph_data, amr_id2rel)
            type_paths.append(path_edge_types)
    return type_paths

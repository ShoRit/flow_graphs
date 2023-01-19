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


def kl_divergence(src_distro_counter, tgt_distro_counter):
    src_norm = sum(src_distro_counter.values())
    tgt_norm = sum(tgt_distro_counter.values())

    src_smoothing_factor = len(
        [path for path in tgt_distro_counter if path not in src_distro_counter]
    )
    src_norm += src_smoothing_factor

    tgt_smoothing_factor = len(
        [path for path in src_distro_counter if path not in tgt_distro_counter]
    )
    tgt_norm += tgt_smoothing_factor

    all_keys = set(src_distro_counter.keys())  # .union(set(tgt_distro_counter.keys()))

    divergence = 0
    for key in all_keys:
        p_x = src_distro_counter.get(key, 1) / src_norm
        q_x = tgt_distro_counter.get(key, 1) / tgt_norm

        if p_x == 0 or q_x == 0:
            continue

        divergence += p_x * np.log(p_x / q_x)

    return divergence


def gini_coefficient(distro_counter):
    n = len(distro_counter)
    mean = sum(distro_counter.values()) / n
    rma_diff = 0
    for key_i, count_i in distro_counter.items():
        for key_j, count_j in distro_counter.items():
            rma_diff += np.abs(count_i - count_j)

    gini_coef = rma_diff / (2 * n**2 * mean)
    return gini_coef

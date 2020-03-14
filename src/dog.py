import networkx as nx
import pandas as pd


def build_graph_with_attributes(node_path, edge_path):
    g = nx.read_edgelist(edge_path,
                         delimiter=',',
                         create_using=nx.DiGraph(),
                         nodetype=int, data=(('size', float),))
    nodes = pd.read_csv(node_path, sep=',')
    data = nodes.set_index('node').to_dict('index').items()
    g.add_nodes_from(data)
    return g
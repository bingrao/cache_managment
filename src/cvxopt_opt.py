import itertools
import networkx as nx
import logging
from cvxopt import spmatrix, matrix
from dog import build_graph_with_attributes


def cvxopt_optimization():
    capacity = 1024
    edge_path = '../data/edges.csv'
    node_path = '../data/nodes.csv'
    graph = build_graph_with_attributes(node_path, edge_path)
    stages = [[0, 1, 2], [0, 5, 6], [3, 4], [7, 8], [9], [10, 11], [12]]
    dataset = set(graph.nodes)
    # exe_path =
    # [[0, 1, 2],
    #  [0, 1, 2, 3, 4],
    #  [0, 5, 6],
    #  [0, 1, 2, 5, 6, 7, 8],
    #  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #  [0, 5, 6, 10, 11],
    #  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    exe_path = []
    for i in range(len(stages)):
        # list(paths)
        # Out[98]: [[1, 2, 3, 4, 9], [1, 2, 7, 8, 9]]
        # paths = nx.all_simple_paths(graph, 1, 9)
        paths = list(nx.all_simple_paths(graph, 0, stages[i][-1]))
        universe = list(set(itertools.chain(*paths)))
        exe_path.append(universe)
    cache_map = [{2}, {2, 6}, {2, 4, 6}, {4, 6, 8}, {6, 9}, {9, 11}, set()]
    print(exe_path)
    m = len(dataset)
    n = len(stages)
    number_of_placement_variables = n * m

    def position(stage, data):
        return stage * m + data

    A = []
    b = []
    row = 0

    logging.debug("Set all unached datasets with value 0 in Matrix X ...")
    # For each stage, there are some datasets should not
    # in memory
    for x in range(n):
        uncached = dataset - cache_map[i]
        if len(uncached) > 0:
            for d in uncached:
                A += [(1.0, row, position(x, d))]
                b += [0.0]
                row += 1

    total_equality_constraints = row

    G = []
    h = []
    logging.debug('Creating capacity constaints...')
    # Capacity constraints
    G += [(1.0 * graph.nodes[d]['mem_size'], s, position(s, d)) for s in range(n) for d in range(m)]
    h += [capacity for s in range(n)]
    logging.debug('...done at %d rows' % len(h))

    row = n
    t = number_of_placement_variables

    # y's less than 1 and bigger than 0
    logging.debug('Creating y ll one gg 0 constraints...')
    x = 0
    while x < number_of_placement_variables:
        G += [(1.0, row, x)]
        h += [1.0]
        row += 1
        G += [(-1.0, row, x)]
        h += [0.0]
        row += 1
        x += 1
    logging.debug('...done at %d rows' % row)

    total_inequality_constraints = n

    val, I, J = zip(*A)
    A = spmatrix(val, I, J, size=(total_equality_constraints, number_of_placement_variables))
    b = matrix(b)

    val, I, J = zip(*G)
    G = spmatrix(val, I, J, size=(total_inequality_constraints, number_of_placement_variables))
    h = matrix(h)

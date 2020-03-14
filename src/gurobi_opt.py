import itertools
import networkx as nx
import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from dog import build_graph_with_attributes


def gurobi_optimization():
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
    cache_propabiltiy = [{2}, {2, 6}, {2, 4, 6}, {4, 6, 8}, {6, 9}, {9, 11}, set()]

    logging.debug(exe_path)

    start = time.time()
    model = gp.Model()
    nums_stages = len(stages)
    nums_data = len(dataset)

    # add optimization variables
    cache_map = model.addVars(nums_stages, nums_data, vtype=GRB.BINARY, name="cache_map")
    model.update()

    # Set 0 to these datases which are not needed to be cached
    model.addConstrs(
        (cache_map[i, j] == 0 for i in range(nums_stages) for j in (dataset - cache_propabiltiy[i]))
    )

    # Cache Capacity constraits
    for i in range(nums_stages):
        model.addConstr(gp.quicksum(cache_map[i, j]*graph.nodes[j]['mem_size'] for j in range(nums_data)) <= capacity)

    # Set up model objective
    def expect_stage_cost(stageID):
        stage_op = exe_path[stageID]
        stage_sink = stage_op[-1]
        cost = 0
        for op in stage_op:
            prop_op = 0
            for path in nx.all_simple_paths(graph, op, stage_sink):
                prop_path_sum = sum([cache_map[stageID, p] for p in path])
                logging.info(prop_path_sum)
                prop_path = min(1, prop_path_sum)
                # prop_path = np.prod([(1 - cache_map[stageID, p]) for p in path])
                prop_op = prop_op + prop_path
            cost_op = graph.nodes[op]['exe_time'] * prop_op
            cost = cost + cost_op
        return cost

    model.setObjective(np.sum([expect_stage_cost(i) for i in range(nums_stages)]), GRB.MINIMIZE)

    model.setParam('OutputFlag', 0)
    model.optimize()
    end = time.time()

    logging.info('----- Output -----')
    logging.info('  Running time : %s seconds' % float(end - start))
    logging.info('  Optimal coverage points: %g' % model.objVal)

    logging.info(np.array([cache_map[i, j].X for i in range(nums_stages)
                    for j in range(nums_data)],
                   dtype=int).reshape(nums_stages, nums_data))


if __name__ == "__main__":
    logging.basicConfig(filename='log_gurobic.txt',level=logging.DEBUG)
    gurobi_optimization()
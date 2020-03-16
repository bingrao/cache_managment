import itertools
import networkx as nx
import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from dog import build_graph_with_attributes
from math import ceil


def gurobi_optimization():
    capacity = 1024000
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
        paths = list(nx.all_simple_paths(G=graph, source=0, target=stages[i][-1]))
        universe = list(set(itertools.chain(*paths)))
        exe_path.append(universe)
    cache_propabiltiy = [{2}, {2, 6}, {2, 4, 6}, {4, 6, 8}, {6, 9}, {9, 11}, set()]

    start = time.time()
    model = gp.Model()
    nums_stages = len(stages)
    nums_data = len(dataset)

    # add optimization variables
    cache_map = model.addVars(nums_stages, nums_data, vtype=GRB.BINARY, name="cache_map")

    path_sum_var = []
    idx = 0
    map_idx = {}

    def set_map_idx(key, value):
        map_idx.update({key: value})

    def get_map_idx(key):
        return map_idx[key]

    for stageID in range(nums_stages):
        stage_op = exe_path[stageID]
        stage_sink = stage_op[-1]
        for op in stage_op:
            all_paths = list(nx.all_simple_paths(graph, op, stage_sink))
            if op == stage_sink:
                all_paths.append([stage_sink])
            for path in all_paths:
                key = str(stageID) + "," + str(op) + "," + str(path)
                path_sum_var.append(model.addVar(vtype=GRB.BINARY, name=key))

                # model.addConstr((path_sum_var[idx] == 1) >> (gp.quicksum(cache_map[stageID, p] for p in path) >= 1))
                # model.addConstr((path_sum_var[idx] == 0) >> (gp.quicksum(cache_map[stageID, p] for p in path) == 0))

                model.addConstr((path_sum_var[idx] == 0) >> (len(path) - gp.quicksum(1 - cache_map[stageID, p] for p in path) >= 1))
                model.addConstr((path_sum_var[idx] == 1) >> (len(path) - gp.quicksum(1 - cache_map[stageID, p] for p in path) == 0))

                set_map_idx(key, idx)
                idx = idx + 1

    model.update()

    # When conduct an operation, the inputs of the operations
    # should be in cache at the same time
    cache_before = [set(), set(), {2}, {2, 6}, {4, 8}, {6}, {9, 11}]
    for i in range(nums_stages):
        if i >= 1:
            if len(cache_before[i]) != 0:
                for j in cache_before[i]:
                    model.addConstr(cache_map[i-1, j] == 1)


    # Cache Capacity constraits
    for i in range(nums_stages):
        model.addConstr(lhs=gp.quicksum(cache_map[i, j] * graph.nodes[j]['mem_size'] for j in range(nums_data)),
                        sense=GRB.LESS_EQUAL,
                        rhs=capacity,
                        name="s%d cache_capacity" % i)

    # for i in range(nums_stages):
    #     for j in range(nums_data):
    #         if i >= 1:
    #             model.addConstr((cache_map[i, j] == 1) >> (cache_map[i-1, j] == 1))

    # Set 0 to these datases which are not needed to be cached
    model.addConstrs(
        (cache_map[i, j] == 0 for i in range(nums_stages) for j in (dataset - cache_propabiltiy[i]))
    )

    # Set up model objective
    def expect_stage_cost(stageID):
        stage_op = exe_path[stageID]
        stage = stages[stageID]
        stage_sink = stage_op[-1]
        cost = 0
        for op in stage_op:
            prop_op = 0
            all_paths = list(nx.all_simple_paths(graph, op, stage_sink))
            if op == stage_sink:
                all_paths.append([stage_sink])
            logging.debug("Stage%d-%s, Operation[%d], All Paths: %s", stageID, stage, op, all_paths)
            for path in all_paths:
                prop_path = gp.LinExpr()
                key = str(stageID) + "," + str(op) + "," + str(path)
                sum_var = path_sum_var[get_map_idx(key)]
                prop_path += sum_var
                logging.debug("Path %s, Sum Key: %s", path, key)
                prop_op = prop_op + prop_path
            cost_op = prop_op * graph.nodes[op]['exe_time']
            cost = cost + cost_op
        logging.debug("#######################################################################\n")
        return cost

    model.setObjective(np.sum([expect_stage_cost(i) for i in range(nums_stages)]), GRB.MINIMIZE)

    model.setParam('OutputFlag', 0)
    model.optimize()
    end = time.time()

    logging.info('----- Output -----')
    logging.info('  Running time : %s seconds' % float(end - start))
    logging.info('  Optimal coverage points: %g' % model.objVal)
    solution = np.array([cache_map[i, j].X for i in range(nums_stages)
                    for j in range(nums_data)],
                   dtype=int).reshape(nums_stages, nums_data)
    logging.info("  The Optimal Solution:\n%s\n", solution)
    logging.info("  The Optimal Sum of Path:")
    for sum_var in path_sum_var:
        logging.info("Name: %s, Value %s", sum_var.varName, sum_var.X)


if __name__ == "__main__":
    # logging.basicConfig(filename='log_gurobic.txt',level=logging.DEBUG)
    # logging.basicConfig(format='%(asctime)s %(message)s',
    #                     datefmt='%m/%d/%Y %I:%M:%S %p',
    #                     filename='log_gurobic.txt',
    #                     filemode='w+',
    #                     level=logging.DEBUG)
    logging.basicConfig(filename='log_gurobic.txt',
                        filemode='w+',
                        level=logging.DEBUG)
    gurobi_optimization()

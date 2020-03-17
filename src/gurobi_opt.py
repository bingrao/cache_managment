import itertools
import networkx as nx
import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from dog import build_graph_with_attributes


def gurobi_optimization():

    # **********************************Model input parameters**********************************

    # Cache capacity limitation for each node
    capacity = 500

    # Stage info which is indexed from 0 to len(stages)
    stages = [[0, 1, 2], [3, 4], [0, 5, 6], [7, 8], [9], [10, 11], [12]]

    # Execution order of stages, the content is the index of an stage.
    stages_exe_order = [0, 2, 1, 3, 4, 5, 6]

    # During execution, the needed dataset should be in cached for a stage is executed
    stages_exe_cache_before = [set(), set(), {2}, {2, 6}, {4, 8}, {6}, {9, 11}]

    # During execution, the suggested datasets should be in cached after a stage is executed
    stages_exe_cache_propabiltiy = [{2}, {2, 6}, {2, 4, 6}, {4, 6, 8}, {6, 9}, {9, 11}, {12}]

    # the input path of edges
    edge_path = '../data/edges.csv'
    # the input path of nodes
    node_path = '../data/nodes.csv'

    # Build a Dog graph to represent the application
    # where each node represents an operation, and there is bonding attributes
    # for each node, such as input and output datasets.
    graph = build_graph_with_attributes(node_path, edge_path)
    dataset = set(graph.nodes)
    # exe_path =
    # [[0, 1, 2],
    #  [0, 1, 2, 3, 4],
    #  [0, 5, 6],
    #  [0, 1, 2, 5, 6, 7, 8],
    #  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #  [0, 5, 6, 10, 11],
    #  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    stages_exe_path = []
    for i in range(len(stages)):
        # list(paths)
        # Out[98]: [[1, 2, 3, 4, 9], [1, 2, 7, 8, 9]]
        # paths = nx.all_simple_paths(graph, 1, 9)
        paths = list(nx.all_simple_paths(G=graph, source=0, target=stages[i][-1]))
        universe = list(set(itertools.chain(*paths)))
        stages_exe_path.append(universe)
    nums_stages = len(stages)
    nums_data = len(dataset)

    logging.debug(stages_exe_path)

    start = time.time()
    model = gp.Model()

    # add optimization variables, this matrix is based on execution order, rather than stages index
    # that means x = {d0, d1, d2, ..., dn} where y = stages_exe_order
    cache_map = model.addVars(nums_stages, nums_data, vtype=GRB.BINARY, name="cache_map")

    path_sum_var = []
    idx = 0
    map_idx = {}
    for exeID in range(nums_stages):
        stageID = stages_exe_order[exeID]
        stage_op = stages_exe_path[stageID]
        stage = stages[stageID]
        stage_sink = stage_op[-1]
        for op in stage_op:
            all_paths = list(nx.all_simple_paths(graph, op, stage_sink))
            if op == stage_sink:
                all_paths.append([stage_sink])
            logging.debug("Execution[%d], Stage[%d]-%s, Operation[%d], All Paths: %s", exeID, stageID, stage, op, all_paths)
            for path in all_paths:
                key = str(exeID) + "," + str(stageID) + "," + str(op) + "," + str(path)
                path_sum_var.append(model.addVar(vtype=GRB.BINARY, name=key))

                # min(1, gp.quicksum(cache_map[stageID, p] for p in path))
                # sum_cache = len(path) - gp.quicksum(1 - cache_map[stageID, p] for p in path)
                # model.addConstr((path_sum_var[idx] == 0) >> (sum_cache >= 1))
                # model.addConstr((path_sum_var[idx] == 1) >> (sum_cache == 0))

                if exeID == 0:
                    model.addConstr(path_sum_var[idx] == 1)
                else:
                    # sum_cache = len(path) - gp.quicksum(1 - cache_map[exeID - 1, p] for p in path)
                    sum_cache = gp.quicksum(cache_map[exeID-1, p] for p in path)
                    model.addConstr((path_sum_var[idx] == 0) >> (sum_cache >= 1))
                    model.addConstr((path_sum_var[idx] == 1) >> (sum_cache == 0))
                logging.debug("MIN, Previous Execution[%d], Current Execution[%d] Current Stage: %d, Key: %s",
                              exeID - 1, exeID, stageID, key)
                map_idx.update({key: idx})
                idx = idx + 1

        logging.debug("#######################################################################\n")

    # After adding variables then update model
    model.update()

    # [[constraits 1]]
    # When conduct an operation, the inputs of the operations
    # should be in cache at the same time
    # for i in range(nums_stages):
    #     if i >= 1:
    #         if len(stages_exe_cache_before[i]) != 0:
    #             for j in stages_exe_cache_before[i]:
    #                 model.addConstr(cache_map[i-1, j] == 1)

    # [[constraits 2]]
    # Cache Capacity constraits
    for i in range(nums_stages):
        model.addConstr(lhs=gp.quicksum(cache_map[i, j] * graph.nodes[j]['mem_size'] for j in range(nums_data)),
                        sense=GRB.LESS_EQUAL,
                        rhs=capacity,
                        name="s%d cache_capacity" % i)

    # [[constraits 3]]
    # Set 0 to these datases which are not needed to be cached
    model.addConstrs(
        (cache_map[i, j] == 0 for i in range(nums_stages) for j in (dataset - stages_exe_cache_propabiltiy[i]))
    )

    # Set up model objective
    def expect_stage_cost(exeID):
        stageID = stages_exe_order[exeID]
        stage_op = stages_exe_path[stageID]
        stage_sink = stage_op[-1]
        cost = 0
        for op in stage_op:
            prop_op = 0
            all_paths = list(nx.all_simple_paths(graph, op, stage_sink))
            if op == stage_sink:
                all_paths.append([stage_sink])
            for path in all_paths:
                prop_path = gp.LinExpr()
                key = str(exeID) + "," + str(stageID) + "," + str(op) + "," + str(path)
                sum_var = path_sum_var[map_idx[key]]
                prop_path += sum_var
                prop_op = prop_op + prop_path
            cost_op = prop_op * graph.nodes[op]['exe_time']
            cost = cost + cost_op
        return cost

    # Set up model Primary objective function (Two Methods)
    model_objective = gp.LinExpr()
    model_objective += np.sum([expect_stage_cost(exe_id) for exe_id in range(nums_stages)])
    model.setObjective(model_objective, GRB.MINIMIZE)
    # model.setObjective(np.sum([expect_stage_cost(exe_id) for exe_id in range(nums_stages)]), GRB.MINIMIZE)

    # Set up multiple object
    for i in range(nums_stages):
        model.setObjectiveN(gp.quicksum(-cache_map[i, j] for j in range(nums_data)), index=i+1)

    model.setParam('OutputFlag', 0)
    model.optimize()
    end = time.time()

    logging.info('----- Output -----')
    logging.debug("The number of objectives: %d", model.getAttr("NumObj"))
    logging.info('  Running time : %s seconds' % float(end - start))
    logging.info('  Optimal coverage points[Primary OBJ]: %g' % model.objVal)

    solution = np.array([cache_map[i, j].X for i in range(nums_stages)
                    for j in range(nums_data)],
                   dtype=int).reshape(nums_stages, nums_data)

    logging.info("  The Gurobi Optimal Solution:\n%s\n", solution)
    logging.debug("  The Optimal Sum of Path:")
    for sum_var in path_sum_var:
        logging.debug("Name: %s, Value %s", sum_var.varName, sum_var.X)

    # model.Params.ObjNumber = 5
    # logging.info('  Optimal coverage points: %S' % GRB.Param.objN)


if __name__ == "__main__":
    logging.basicConfig(filename='log_gurobic.txt',
                        filemode='w+',
                        level=logging.DEBUG)
    gurobi_optimization()

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

    # Feature 1: When invoking an operation, we need to make sure all
    # needed datasets in memory
    F_ALL_IN = False

    # Feature 2: Max using an node memory
    F_MAX_USE = False

    # Cache capacity limitation for each node
    capacity = 600

    # Stage info which is indexed from 0 to len(stages)
    stages = [[0, 1, 2], [3, 4], [0, 5, 6], [7, 8], [9], [10, 11], [12]]

    # Execution order of stages, the content is the index of an stage.
    stages_exe_order = [0, 2, 1, 3, 4, 5, 6]

    # During execution, the needed dataset should be in cached for a stage is executed
    stages_exe_cache_before = [set(), set(), {2}, {2, 6}, {4, 8}, {6}, {9, 11}]

    # During execution, the suggested datasets should be in cached after a stage is executed
    stages_exe_cache_keep = [{2}, {2, 6}, {2, 4, 6}, {4, 6, 8}, {6, 9}, {9, 11}, {12}]

    # Identify which execution stage generate which dataset
    # for example 5:1 means dataset 5 is generated in execution stage 1
    stages_exe_dataset_gen = {0: 0, 1: 0, 2: 0,
                            5: 1, 6: 1,
                            3: 2, 4: 2,
                            7: 3, 8: 3,
                            9: 4,
                            10: 5, 11: 5,
                            12: 6}

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
                if exeID == 0:
                    model.addConstr(path_sum_var[idx] == 1, name=key+str(1))
                else:
                    # sum_cache = len(path) - gp.quicksum(1 - cache_map[exeID - 1, p] for p in path)
                    # https://www.gurobi.com/documentation/9.0/refman/py_tempconstr.html#pythonclass:TempConstr
                    sum_cache = gp.quicksum(cache_map[exeID-1, p] for p in path)
                    model.addConstr((path_sum_var[idx] == 0) >> (sum_cache >= 1), name=key+str(0)) # if >= 1, then 0
                    model.addConstr((path_sum_var[idx] == 1) >> (sum_cache == 0), name=key+str(1)) # if == 0, then 1
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
    if F_ALL_IN:
        for i in range(nums_stages):
            if i >= 1:
                if len(stages_exe_cache_before[i]) != 0:
                    for j in stages_exe_cache_before[i]:
                        model.addConstr(cache_map[i-1, j] == 1)

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
        (cache_map[i, j] == 0 for i in range(nums_stages) for j in (dataset - stages_exe_cache_keep[i]))
    )

    # [[constraits 4]]
    # If a dataset is cached after some stages, we need to make sure that
    # it should put in memory when it is created, after that it should be
    # in memory until it is evictd from memory
    for j in range(nums_data):
        gen_stage = stages_exe_dataset_gen[j]
        for i in range(nums_stages - 1, gen_stage-1, -1):
            logging.debug("The data[%d], Stage[%d]", j, i)
            if i-1 >= gen_stage:
                # No matter when cache_map[i, j] == 1, then cache_map[i-1, j] == 1
                # Must be hold
                model.addConstr((cache_map[i, j] == 1) >> (cache_map[i-1, j] == 1))

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

    # # Set up multiple object
    if F_MAX_USE:
        for i in range(nums_stages):
            model.setObjectiveN(gp.quicksum(-cache_map[i, j] for j in range(nums_data)), index=i+1)

    model.setParam('OutputFlag', 0)
    model.optimize()
    end = time.time()

    # Ensure status is optimal
    assert model.Status == GRB.Status.OPTIMAL

    logging.info('------------------------- Output -------------------------')
    logging.info('Running time : %s seconds' % float(end - start))
    # Query number of multiple objectives, and number of solutions
    nSolutions = model.SolCount
    nObjectives = model.NumObj
    logging.info("Problem has %s objectives", nObjectives)
    logging.info('Found %d solutions', nSolutions)

    # For each solution, print value of first three variables, and
    # value for each objective function
    for s in range(nSolutions):

        # Set which solution we will query from now on
        model.params.SolutionNumber = s
        if nObjectives == 1:
            # Primary objective (indexed by 0) can its return value is model.objVal
            logging.info('Solution %d, Optimal Obj %d, Value %g', s, 0, model.ObjVal)
        else:
            # Print objective value of this solution in each objective
            for o in range(nObjectives):
                # Set which objective we will query
                model.params.ObjNumber = o
                # Query the o-th objective value
                logging.info('Solution %d, Optimal Obj %d, Value %g', s, o, model.ObjNVal)

    sol_cache = np.array([cache_map[i, j].X for i in range(nums_stages)
                    for j in range(nums_data)],
                   dtype=int).reshape(nums_stages, nums_data)

    logging.info("The Gurobi Optimal Cache Map for each execution Stage Solution:\n%s\n", sol_cache)
    logging.debug("The Optimal Sum of Path:")
    for sum_var in path_sum_var:
        logging.debug("Name: %s, Value %s", sum_var.varName, sum_var.X)


if __name__ == "__main__":
    logging.basicConfig(filename='log_gurobic.txt',
                        filemode='w+',
                        level=logging.INFO)
    gurobi_optimization()

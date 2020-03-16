# This example formulates and solves the following simple QP model:
#
#    minimize    x + y + x^2 + x*y + y^2 + y*z + z^2
#    subject to  x + 2 y + 3 z >= 4
#                x +   y       >= 1
#                x, y, z non-negative
#
# The example illustrates the use of dense matrices to store A and Q
# (and dense vectors for the other relevant data).  We don't recommend
# that you use dense matrices, but this example may be helpful if you
# already have your data in this format.

import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp


def dense_optimize():
    # Put model data into dense matrices
    c = [1, 1, 0]
    Q = [[1, 1, 0], [0, 1, 1], [0, 0, 1]]
    A = [[1, 2, 3], [1, 1, 0]]
    sense = [GRB.GREATER_EQUAL, GRB.GREATER_EQUAL]
    rhs = [4, 1]
    lb = [0, 0, 0]
    ub = [GRB.INFINITY, GRB.INFINITY, GRB.INFINITY]
    vtype = [GRB.CONTINUOUS, GRB.CONTINUOUS, GRB.CONTINUOUS]
    solution = [0] * 3
    rows = 2
    cols = 3

    # Optimize
    model = gp.Model()

    # Add variables to model
    vars = []
    for j in range(cols):
        vars.append(model.addVar(lb=lb[j], ub=ub[j], vtype=vtype[j]))

    # Populate A matrix
    for i in range(rows):
        expr = gp.LinExpr()
        for j in range(cols):
            if A[i][j] != 0:
                expr += A[i][j]*vars[j]
        model.addConstr(expr, sense[i], rhs[i])

    # Populate objective
    obj = gp.QuadExpr()
    for i in range(cols):
        for j in range(cols):
            if Q[i][j] != 0:
                obj += Q[i][j]*vars[i]*vars[j]
    for j in range(cols):
        if c[j] != 0:
            obj += c[j]*vars[j]
    model.setObjective(obj)

    # Solve
    model.optimize()

    # Write model to a file
    # model.write('dense.lp')

    if model.status == GRB.OPTIMAL:
        x = model.getAttr('x', vars)
        for i in range(cols):
            solution[i] = x[i]
        return True, solution
    else:
        return False, solution


def dense_optimize_v2():
    solution = [0] * 3
    # Optimize
    model = gp.Model()

    xyz = model.addMVar(shape=3, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="xyz")
    x = xyz.vararr[0]
    y = xyz.vararr[1]
    z = xyz.vararr[2]

    # Build (sparse) constraint matrix
    data = np.array([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    row = np.array([0, 0, 0, 1, 1, 2, 3, 4])
    col = np.array([0, 1, 2, 0, 1, 0, 1, 2])
    A = sp.csr_matrix((data, (row, col)), shape=(5, 3))

    # Build rhs vector
    rhs = np.array([4.0, 1.0, 0.0, 0.0, 0.0])

    # Add constraints
    model.addConstr(A @ xyz >= rhs, name="c")

    # Populate objective
    obj = gp.QuadExpr()
    obj += x + y + x * x + y * y + y * z + z * z
    model.setObjective(obj)

    # Solve
    model.optimize()

    # Write model to a file
    # model.write('dense.lp')

    if model.status == GRB.OPTIMAL:
        x = model.getAttr('x', vars)
        for i in range(3):
            solution[i] = x[i]
        return True, solution
    else:
        return False, solution


if __name__ == "__main__":

    success, sol = dense_optimize()
    if success:
        print('x: %g, y: %g, z: %g' % (sol[0], sol[1], sol[2]))

    # print("*********************************************************")
    # success, sol = dense_optimize_v2()
    # if success:
    #     print('x: %g, y: %g, z: %g' % (sol[0], sol[1], sol[2]))
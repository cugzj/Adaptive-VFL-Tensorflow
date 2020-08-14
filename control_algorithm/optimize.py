from gurobipy import *
import math

# params.NonConvex = 2
K = 5
# R = {0: 10, 1: 20, 2: 30, 3: 50, 4: 100}  # imbalanced
R = {0: 10, 1: 13, 2: 20, 3: 30, 4: 50}  # imbalanced
# R = {0: 42, 1: 42, 2: 42, 3: 42, 4: 42}  # balanced
R_sum = sum(R.values())
# print('sum of R:', R_sum)
p = {0: 2, 1: 1.5, 2: 1.2, 3: 2, 4: 1}
c0 = 20
d0 = 0.2
N = 30000
t_2 = 0.5
epsilon = 0.01
eta = 0.5
D = 200
# G = 1
# L_max = 1
# gamma = c0 * d0 * D / R_sum
# alpha = eta * eta * G
# beta = eta * D * L_max * math.sqrt(G)
# lamda = eta * eta * epsilon * epsilon


# try:

# 最优解
def solve_model(L, G):
    # initialization
    L_max = max(L.values())
    gamma = c0 * d0 * N / R_sum
    alpha = eta * eta * G
    beta = eta * D * L_max * math.sqrt(G)
    lamda = eta * eta * epsilon * epsilon

    print('L_max:', L_max)
    print('c0: d0', c0, d0)

    # Create a new model
    m = Model("mip1")

    # Create variables
    T = m.addVar(vtype=GRB.INTEGER, name="T")
    E = {}
    for k in range(K):
        E[k] = m.addVar(vtype=GRB.INTEGER, name="E")

    t_max = m.addVar(vtype=GRB.CONTINUOUS, name="t_max")
    a = m.addVar(vtype=GRB.CONTINUOUS, name="R_1")
    b = m.addVar(vtype=GRB.CONTINUOUS, name="R_2")
    c, mul, pow = {}, {}, {}
    for k in range(K):
        c[k] = m.addVar(vtype=GRB.CONTINUOUS, name="R_3")
        mul[k] = m.addVar(vtype=GRB.CONTINUOUS, name="R_4")
        pow[k] = m.addVar(vtype=GRB.CONTINUOUS, name="R_5")
    d = m.addVar(vtype=GRB.CONTINUOUS, name="R_6")
    E_min = m.addVar(vtype=GRB.INTEGER, name="E_min")
    E_min_2 = m.addVar(vtype=GRB.INTEGER, name="E_min_2")
    s = m.addVar(vtype=GRB.CONTINUOUS, name="R_sum")

    # Set objective
    m.setObjective(T * (t_max + t_2), GRB.MINIMIZE)

    # Add constraint: T_max >= E_k * R_k
    for k in range(K):
        m.addConstr(t_max * p[k] >= gamma * R[k] * E[k], "c0")

    # Add constraint: a == \sum E_k * E_k * R_k
    m.addConstr(a == quicksum(E[k] * E[k] * R[k] for k in range(K)))

    for k in range(K):
        m.addConstr(mul[k] == E[k] * (E[k] - 1))
        m.addConstr(c[k] * c[k] == R[k])
        m.addConstr(pow[k] == mul[k] * c[k])
        # m.addConstr(E[k] >=1)
        m.addConstr(E_min <= E[k])

    m.addConstr(b == quicksum((pow[k] * pow[k]) for k in range(K)))
    m.addConstr(d * d == b)
    m.addConstr(s == alpha * a + D * D + beta * d)
    m.addConstr(E_min_2 == E_min * E_min)

    # Add constraint: x + y >= 1 R_e <= epsilon
    m.addConstr(s * s <= E_min_2 * T * 1, "c1")

    def printSolution():
        if m.status == GRB.OPTIMAL:
            print('\nTotal Training Time: %g' % m.objVal)
            print('-----------Solution------------')
            Ex = m.getAttr('x', E)
            print('t_max=%d \t E_min:%d \t T=%d' % (t_max.x, E_min.x, T.x))
            print('a=%d \t b:%f \t d=%f' % (a.x, b.x, d.x))
            print('lamda=%f \t s=%f ' % (lamda, s.x))
            for k in range(K):
                if E[k].x > 0.0001:
                    print('E[%d] = %d' % (k, Ex[k]))
        else:
            print('No solution')

    m.params.NonConvex = 2
    # m.setParam("OutputFlag", 0)
    m.optimize()
    local_epoch = []
    if m.status == GRB.OPTIMAL:
        Ex = m.getAttr('x', E)
        for k in range(K):
            if E[k].x > 0.0001:
                local_epoch.append(Ex[k])
    else:
        local_epoch = [1 for i in range(K)]
    print('Optimization solution:', local_epoch)
    return local_epoch


L = {0:0.9, 1:1.48, 2:0.89, 3:1.2, 4:1.1}
G = 0.23
solve_model(L,G)

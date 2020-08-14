from gurobipy import *
import math

# params.NonConvex = 2
# K = 5
# R = {0: 1000, 1: 5000, 2: 10000, 3: 20000, 4: 69354}  # imbalanced

# R = {0: 50, 1: 100, 2: 150, 3: 84, 4: 400}  # imbalanced

# R = {0: 100, 1: 200, 2: 484}  # imbalanced

# R = {0: 10, 1: 20, 2: 30, 3: 50, 4: 100}  # imbalanced
# R = {0: 10, 1: 13, 2: 20, 3: 30, 4: 50}  # imbalanced
# R = {0: 42, 1: 42, 2: 42, 3: 42, 4: 42}  # balanced
# R_sum = sum(R.values())
# print('sum of R:', R_sum)
N = 897
# B = 100
# R = {0: 10, 1: 23, 2: 90}
# R_sum = sum(R.values())
# K = len(R)
# p = {0: 3, 1: 1.5, 2: 1.2, 3: 2, 4: 1, 5: 1.3, 6: 1, 7: 2, 8: 3, 9: 2}
p = {0: 3, 1: 1.5, 2: 1.2, 3: 2, 4: 1}

t_2 = 0
epsilon = 0.01
D = 1000
c0 = 20
d0 = 0.2
eta = 0.5
# L_max = 0.4561413560946449
# G = 0.02
# G = 0.08
# L_max = 0.09
# L_max = 1.167833240579852
# G = 0.49404299383897043

def solve_model(K, R, N, L_max, G):
    print('parameters==| k=%d \t |R=%s \t |N=%d \t |eta=%f \t |L_max=%f \t |G=%f'% (K, R, N, eta, L_max, G))
    R_sum = sum(R.values())
    gamma = c0 * d0 * 1 / R_sum
    alpha = eta * eta * G
    beta = eta * D * L_max * math.sqrt(G)
    lamda = eta * eta * epsilon * epsilon
    print('parameters==| d0=%f \t |gamma=%s \t |alpha=%f \t |beta=%f \t |lamda=%f'% (d0, gamma, alpha, beta, lamda))

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

    m.params.NonConvex = 2
    m.setParam("OutputFlag", 0)
    m.params.TimeLimit = 40  # 限制求解时间为 100s
    m.optimize()
    local_epoch = []
    if m.status == GRB.OPTIMAL:
        print('solved!')
        Ex = m.getAttr('x', E)
        for k in range(K):
            if E[k].x > 0.0001:
                local_epoch.append(int(Ex[k]))
    else:
        local_epoch = [1 for i in range(K)]
    print('Optimization solution:', local_epoch)
    return local_epoch

# K = 5
# G = 0.23
# L_max = 1.48
# R = {0: 10, 1: 13, 2: 20, 3: 30, 4: 50}  # imbalanced
# N = 30000
# eta = 0.5
# print('parameters:', K, R, N, eta, L_max, G)
# ll = solve_model(K, R, N, L_max, G)

import numpy as np
# from control_algorithm.optimize import solve_model
from control_algorithm.opt import solve_model

class ControlAdaptiveE:
    def __init__(self, args, R, temp_contro_params, current_learning_rate, batch_count):
        self.n_party = args.n_party
        self.R = R
        self.N = args.n_sample
        self.lr = current_learning_rate
        self.batch_count = batch_count
        self.last_weight = [0] * self.n_party
        self.current_weight = [0] * self.n_party
        self.last_gradient = [[0] * self.batch_count] * self.n_party
        self.current_gradient = [[0] * self.batch_count] * self.n_party
        for i in range(self.n_party):
            # print('i', i)
            # print(len(temp_contro_params[i]))
            self.last_weight[i] = temp_contro_params[i][0][0]
            self.current_weight[i] = temp_contro_params[i][1][0]
            self.last_gradient[i] = temp_contro_params[i][0][1]
            self.current_gradient[i] = temp_contro_params[i][1][1]

    def compute_new_E(self, com_time, cmp_time):
        # compute L
        L = {}
        L_max = 0
        for k in range(self.n_party):
            le = len(self.last_gradient[k][0])
            last_grad_mean = np.array([0.0] * le)
            curr_grad_mean = np.array([0.0] * le)

            for last_grad_index in range(len(self.last_gradient[k])):
                # print(len(self.last_gradient[k]))
                last_grad_mean += np.array(self.last_gradient[k][last_grad_index])
            last_grad_mean = last_grad_mean / [self.batch_count for _ in range(le)]

            for curr_grad_index in range(len(self.current_gradient[k])):
                curr_grad_mean += np.array(self.current_gradient[k][curr_grad_index])
            curr_grad_mean = curr_grad_mean / [self.batch_count for _ in range(le)]

            c1 = np.linalg.norm(curr_grad_mean-last_grad_mean, ord=1)
            c2 = np.linalg.norm(self.current_weight[k]-self.last_weight[k], ord=1)
            L[k] = c1 / c2
        L_max = max(L.values())
        print('L:', L)
        print('L_max:', max(L.values()))
        if L_max >= 5:
            L_max = 1

        # compute G
        G = 0
        for k in range(self.n_party):
            d = np.linalg.norm(self.current_gradient[k])
            G += d * d / self.R[k]
        G = G / self.n_party
        if G >= 5:
            G = 2
        if G <= 1e-05:
            G = 0.02
        print('G:', G)

        # choose phi
        # epsilon = 0.01
        # D = 100
        t_com = com_time
        t_cmp = cmp_time
        # phi =  D / epsilon
        # print('com_time:%f \t cmp_time:%f'%(t_com, t_cmp))

        # compute E
        # local_epoch = solve_model(self.n_party, self.R, L, G, D, epsilon, self.lr, t_com, t_cmp)
        local_epoch = solve_model(self.n_party, self.R, self.N, L_max, G)
        return local_epoch

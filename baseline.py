from option import args_parser
import numpy as np
import time
from utils.utils import get_dataset, csr_to_indices, eval_pred, get_weight, get_gradient
from utils.initial import initial
import model.NN as nn
import tensorflow as tf  # tf v1
import pandas as pd
from control_algorithm.adaptive_E import ControlAdaptiveE

# R = {0: 100, 1: 200, 2: 484}  # imbalanced MNIST
R = {0: 1000, 1: 5000, 2: 10000, 3: 20000, 4: 69354}  # imbalanced
# R = {0: 50, 1: 100, 2: 150, 3: 84, 4: 400}  # imbalanced
# R = {0: 10, 1: 50, 2: 100, 3: 200, 4: 100}  # imbalanced
# R = {0: 10, 1: 13, 2: 20, 3: 30, 4: 50}  # imbalanced
# R = {0: 10, 1: 23, 2: 90}  # imbalanced
R_sum = sum(R.values())
train_dimensions = []
for k in range(len(R)):
    train_dimensions.append(R[k])
# train_dimensions = R.values()
local_epoch = [10 for _ in range(len(R))]
# local_epoch = [6,6,1,3,1]

if __name__ == '__main__':
    args = args_parser()
    # Hyper-parameters
    data_home = args.data_home
    # data_home = "./data"
    # data_home = "./data-mnist/"
    if args.dataset == "a9a":
        n_sample = 30000  # a9a 30000 mnist-binary 12000
        dimension = 123  # 123
    elif args.dataset == "Citeseer":
        n_sample = 897  # a9a 30000 mnist-binary 12000
        dimension = 105354  # 123
    else:  # mnist-binary 12000
        n_sample = 12000
        dimension = 784
    n_party = args.n_party

    num_epochs = args.num_epochs  # 600
    batch_size = args.batch_size
    # com_B = 100
    learning_rate = args.learining_rate
    l2_reg = args.l2_reg
    print('number of party:', args.n_party)
    print('dataset:', args.dataset)
    # print('fix_epochs:', args.fix_epochs)
    # train_dimensions = [10, 50, 100, 200, 424]
    # train_dimensions = []

    # 初始化
    _, feature, label, index = initial(args, dimension, data_home, n_party, train_dimensions)

    start = time.time()
    # Initialization server
    local_value_cache_server = np.zeros([n_sample, n_party])  # 服务器端局部数值缓存
    aggregate_value_caceh_server = np.zeros([n_sample, 1])  # 服务器端全局累加缓存

    # --worker --
    local_value_cache_worker = [np.zeros([n_sample, 1])] * args.n_party  # worker全局累加量缓存,定义一个list保存
    local_other_cache_worker = [np.zeros([n_sample, 1])] * args.n_party  # worker全局累加量减去自己的局部预测值，用来辅助标准化tf的训练模块过程
    feature_input_indices_worker = [tf.placeholder(tf.int64, [None, 2])] * args.n_party
    feature_input_value_worker = [tf.placeholder(tf.float32, [None])] * args.n_party
    feature_input_worker = []
    for i in range(n_party):
        feature_input_worker.append(tf.SparseTensor(feature_input_indices_worker[i],
                                                    feature_input_value_worker[i],
                                                    [batch_size, train_dimensions[i]]))  #
    print("feature input", feature_input_worker[0], feature_input_worker[0].shape, type(feature_input_worker[0]))
    other_sum_worker = [tf.placeholder(tf.float32, [None, 1])] * args.n_party
    label_worker = [tf.placeholder(tf.float32, [None, 1])] * args.n_party
    learning_rate_worker = tf.placeholder(tf.float32, 1)  # 用于动态递减学习率
    # current_learning_rate = [tf.placeholder(tf.float32, 1)]*args.n_party

    pred_worker = [tf.placeholder(tf.float32, [batch_size, 1])] * args.n_party
    print("pred_worker[0]=", pred_worker[0])
    loss_worker = [tf.placeholder(tf.float32, [batch_size, 1])] * args.n_party
    all_weight = [tf.placeholder(tf.float32, [None, 1])] * args.n_party
    local_pred_worker = [tf.placeholder(tf.float32, [batch_size, 1])] * args.n_party

    optimizer_worker = [None] * args.n_party

    # Build Model
    if args.model == "logistic":
        for i in range(args.n_party):
            scopes = "worker_" + str(i)
            pred_worker[i], loss_worker[i], local_pred_worker[i], all_weight[i] = nn.partial_logistic_regression(
                feature_input_worker[i], other_sum_worker[i], label_worker[i], train_dimensions[i],
                scope=scopes, sparse_input=True, l2_reg=l2_reg)
            loss_worker[i] = loss_worker[i] + tf.add_n(
                tf.get_collection("worker_" + str(i) + "_reg_losses"))  # attach the regulation loss
            optimizer_worker[i] = tf.train.GradientDescentOptimizer(learning_rate_worker[0]).minimize(loss_worker[i])

    else:
        for i in range(args.n_party):
            scopes = "worker_" + str(i)
            pred_worker[i], loss_worker[i], local_pred_worker[i], all_weight[i] = nn.partial_multilayer(
                feature_input_worker[i], other_sum_worker[i], label_worker[i], train_dimensions[i],
                scope=scopes, sparse_input=True, l2_reg=l2_reg)
            loss_worker[i] = loss_worker[i] + tf.add_n(
                tf.get_collection("worker_" + str(i) + "_reg_losses"))  # attach the regulation loss
            optimizer_worker[i] = tf.train.GradientDescentOptimizer(learning_rate_worker[0]).minimize(loss_worker[i])

    ##============================
    # 正式迭代过程
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.get_variable_scope().reuse_variables()
    batch_count = int(n_sample / batch_size)

    # Start Training
    # cost = [[0]*num_epochs] * n_party
    # auc = [[0]*num_epochs] * n_party
    cost = [[0 for _ in range(num_epochs)] for _ in range(n_party)]
    auc = [[0 for _ in range(num_epochs)] for _ in range(n_party)]
    # exec_time = [[]] * n_party
    exec_time = [0] * num_epochs
    for epoch in range(num_epochs):
        print('-------------------------epoch:%d----------------------' % epoch)
        current_learning_rate = learning_rate / np.sqrt(1 + epoch / 30.)  # 每个worker的learning rate都相同
        avg_cost_worker = [0] * args.n_party
        avg_acc_worker = [0] * args.n_party
        avg_time = [0] * args.n_party
        current_cost = [0] * args.n_party
        com_time = 0
        # local_epoch = []

        # 预处理，计算梯度
        for batch in range(batch_count):  # batch_count 按照batch size 执行多少次，使所有样本数据遍历一次
            # 预处理
            start_idx, end_idx = batch_size * batch, batch_size * (batch + 1)
            total_batch_index = epoch * batch_size + batch
            # current_learning_rate = learning_rate / np.sqrt(1 + epoch / 30.) # 每个worker的learning rate都相同

            # 获取每个worker当前batch的local results
            for worker in range(n_party):
                # 获取当前batch的local results
                # --worker--
                local_value_cache_worker[worker][start_idx:end_idx] = sess.run(
                    local_pred_worker[worker], feed_dict={
                        feature_input_indices_worker[worker]: csr_to_indices(feature[worker][start_idx:end_idx]),
                        feature_input_value_worker[worker]: feature[worker][start_idx:end_idx].data})

                # 局部预测值向服务器同步，服务器累加
                # Push local values from --worker 0--
                for i in np.arange(start_idx, end_idx):
                    local_value_cache_server[i, worker] = local_value_cache_worker[worker][i]

            # --server--累加值
            aggregate_value_caceh_server[start_idx:end_idx] = np.sum(
                local_value_cache_server[start_idx:end_idx, :], axis=1).reshape(batch_size, 1)

            # 工作节点拉取全局数据，预处理后更新
            # Pull global aggregation --worker--
            start_com = time.time()
            for worker in range(n_party):  #
                local_other_cache_worker[worker][start_idx:end_idx] = aggregate_value_caceh_server[start_idx:end_idx] \
                                                                      - local_value_cache_worker[worker][
                                                                        start_idx:end_idx]

        # ---------------+++++++++++++++++++++Start Training+++++++++++++++++++++-------------------#
        # for batch in range(batch_count):  # batch_count 按照batch size 执行多少次，使所有样本数据遍历一次
        #     # 预处理
        #     start_idx, end_idx = batch_size * batch, batch_size * (batch + 1)
        #     total_batch_index = epoch * batch_size + batch

            for worker in range(n_party):  # 每个worker都进行更新操作
                start_epoch = time.time()

                for i in range(local_epoch[worker]):
                    # 本地更新参数
                    start_cmp = time.time()
                    current_cost[worker], _, pred_vals = sess.run([loss_worker[worker],
                                                                   optimizer_worker[worker],
                                                                   pred_worker[worker]],feed_dict={
                        feature_input_indices_worker[worker]: csr_to_indices(feature[worker][start_idx:end_idx]),
                        feature_input_value_worker[worker]: feature[worker][start_idx:end_idx].data,
                        other_sum_worker[worker]:local_other_cache_worker[worker][start_idx:end_idx],
                        label_worker[worker]:label[worker][start_idx:end_idx].reshape(batch_size, 1),
                        learning_rate_worker: [current_learning_rate]})
                    # print('weight:', all_weight[worker])
                    end_cmp = time.time()
                    cmp_time = end_cmp - start_cmp

                    # print("current_cost=",current_cost)
                avg_cost_worker[worker] += current_cost[worker] / batch_count
                test_auc, test_loss = eval_pred(label[worker][start_idx:end_idx], pred_vals)
                avg_acc_worker[worker] += test_auc / batch_count
                # print("---| Woker ",'%d' % (worker + 1)," Batch:", '%04d' % (batch + 1), "cost=", "{:.9f}".format(avg_cost_worker[worker]))
                end_epoch = time.time()
                avg_time[worker] += (end_epoch - start_epoch) / batch_count

        # 每个worker更新完毕
        for i in range(n_party):  # 每个cost都进行更新操作
            cost[i][epoch] = avg_cost_worker[i]
            auc[i][epoch] = avg_acc_worker[i]
            # exec_time[worker].append(max(avg_time_0, avg_time_1))
            # 显示每个epoch训练信息
            print("Woker", '%d' % i, " Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost_worker[i]))
        exec_time[epoch] = max(avg_time)
        print("\n")

    e = 0
    final_time = []
    for index in range(len(exec_time)):
        e += exec_time[index]
        final_time.append(e)
    print('execution time:', final_time)

    tep = []
    for i in range(len(cost[0])):
        tep.append([cost[0][i], auc[0][i], final_time[i]])
    tep = pd.DataFrame(tep, columns=["loss", "auc", "time"])
    tep.to_excel("./log/baseline/" + str(args.model) + "_" + str(args.dataset) + "_E[" + str(local_epoch[0]) + "]_N[" + str(
        args.n_party) + "].xlsx", index=False)
    # tep.to_excel("./log[" + str(train_dimensions[0]) + "]_" + "B[" + str(batch_size) + "]_" + "E["
    #             + str(batch_epoch_0) + "+" + str(batch_epoch_1) + "]_" + str(num_epochs) + ".xlsx",index = False)

    # 画图
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(range(len(cost[0])), cost[0], label='LR')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(range(len(auc[0])), auc[0], 'r-.', label='LR')
    plt.xlabel('Round')
    plt.ylabel('AUC')
    plt.legend()

    # show函数展示出这个图，如果没有这行代码，则程序完成绘图，但看不到
    plt.show()

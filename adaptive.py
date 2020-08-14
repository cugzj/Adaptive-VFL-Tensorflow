from option import args_parser
import numpy as np
import time
from utils.utils import get_dataset, csr_to_indices, eval_pred, get_weight, get_gradient
from utils.initial import initial
import model.NN as nn
import tensorflow as tf # tf v1
import pandas as pd
from control_algorithm.adaptive_E import ControlAdaptiveE

# def args_parser():
#     parser = argparse.ArgumentParser()
#     # distributed features  arguments
#     parser.add_argument('--n_party', type=int, default=5, help="number of parties: MAX 50")
#     parser.add_argument('--num_epochs', type=int, default=100, help="number of epochs")
#     parser.add_argument('--fix_epochs', type=int, default=5, help="fix number of epochs")
#     parser.add_argument('--num_features', type=int, default=123, help="total number of features")
#     parser.add_argument('--batch_size', type=int, default=100, help="batch size")
#     parser.add_argument('--learining_rate', type=float, default=1.0, help="learning rate")
#     parser.add_argument('--l2_reg', type=float, default=0.0001, help="l2_reg")
#     parser.add_argument('--dimensions', type=str, default='Non-AVG', help="train_dimensions(average or not)")
#
#     # model arguments
#     parser.add_argument('--model', type=str, default='logistic', help='model name')
#
#     # dataset arguments
#     parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
#     parser.add_argument('--input_path', type=str, default='./data-mnist/mnist-binary', help='path of input')
#     args = parser.parse_args()
#     return args

# def split(n_party,milestones):
#     # change this for specific purpose
#     # input_path = "./data/a9a" #mnist-binary
#     input_path = args.input_path
#     for i in range(n_party-1):
#         milestones[i+1] += milestones[i]
#     print("milestones=", milestones)
#     # milestones: [0, 66, 123] feature分为两个部分66+57
#
#     [feature_raw, label] = load_svmlight_file(input_path)
#     print('feature_raw_shape:', feature_raw.shape)
#     print('label:', label, label.shape)
#     [num_row, num_col] = feature_raw.shape
#     milestones.insert(0, 0)
#     milestones.append(num_col)
#     intervals = []
#     for i in range(len(milestones) - 1):
#         intervals.append(np.arange(milestones[i], milestones[i + 1]))
#     # print('intervals:', intervals)
#     feature_coo = sp.sparse.coo_matrix(feature_raw)
#     # print('feature_coo:', feature_coo)
#
#     for file in glob.glob(input_path + "-*"):
#         os.remove(file)
#
#     # generating and saving the parts
#     query_id = np.arange(num_row)  # 32561
#     num_parts = len(intervals)   # 2
#     for i in range(num_parts-1):
#         # split the matrix
#         print(feature_coo.col, len(feature_coo.col))
#         index_merge = (feature_coo.col == intervals[i][0])
#         # print('index_merge:', index_merge, len(index_merge))
#         for j in intervals[i]:
#             index_merge = index_merge + (feature_coo.col == j)
#         cur_feature_row = feature_coo.row[index_merge]
#         cur_feature_col = feature_coo.col[index_merge]
#         # print("debug info:")
#         # print("col min", np.min(cur_feature_col))
#         # print("intervals min", np.min(intervals[i]))
#         cur_feature_col = cur_feature_col - np.min(intervals[i])
#         cur_feature_data = feature_coo.data[index_merge]
#         cur_feature = sp.sparse.coo_matrix((cur_feature_data, (cur_feature_row, cur_feature_col)),
#                                            shape=(num_row, intervals[i].size))
#         cur_output_path = input_path + "-part" + str(i)
#         dump_svmlight_file(cur_feature, label, cur_output_path, query_id=query_id)
#         cur_output_meta_path = cur_output_path + ".meta"
#         with open(cur_output_meta_path, "w") as fout:
#             fout.write(str(np.max(intervals[i]) - np.min(intervals[i]) + 1))
#
#     # dump a normalized version
#     dump_svmlight_file(feature_raw, label, input_path, query_id=query_id)
#
#     # generating the server parts
#     command = ["cut", "-d", "\" \"", "-f1,2", input_path + "-part0", ">", input_path + "-server"]
#     command = " ".join(command)
#     print(command)
#     os.system(command)
#
#     with open(input_path + "-server.meta", "w") as fout:
#         fout.write("1")
# R = {0: 10, 1: 20, 2: 30, 3: 50, 4: 100}  # imbalanced
# R = {0: 100, 1: 200, 2: 484}  # imbalanced
# R = {0: 50, 1: 100, 2: 150, 3: 84, 4: 400}  # imbalanced
# R = {0: 10, 1: 23, 2: 90}
# R = {0: 10, 1: 13, 2: 20, 3: 30, 4: 50}  # imbalanced
# R = {0: 50, 1: 100, 2: 150, 3: 84, 4: 400}  # imbalanced
#
R = {0: 1000, 1: 5000, 2: 10000, 3: 20000, 4: 69354}  # imbalanced

R_sum = sum(R.values())
train_dimensions = []
for k in range(len(R)):
    train_dimensions.append(R[k])

if __name__ == '__main__':
    args = args_parser()
    # Hyper-parameters
    data_home = args.data_home
    # data_home = "./data"
    # data_home = "./data-mnist/"
    if args.dataset == "a9a":
        n_sample = 30000 # a9a 30000 mnist-binary 12000
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
    # print('fix_epochs:', args.fix_epochs)
    # train_dimensions = [10, 50, 100, 200, 424]
    # train_dimensions = [10, 13, 20, 30, 50]
    # train_dimensions = [13, 4, 4, 5, 3, 21, 20, 19, 2, 32]

    # if train_dimensions:
    #     if args.dimensions == "AVG":
    #         local_epoch = [1] * n_party
    #     else:
    #         local_epoch = []
    #         for i in range(n_party):  # 浮点数转为int, 同时随机每个client训练的local_epoch数
    #             # local_epoch.append(np.random.randint(1, 10))  # 每一个client随机训练1-10次
    #             local_epoch = [args.fix_epochs] * n_party
    #             # local_epoch = [4, 4, 3, 2, 1]
    #
    #         print("local_epoch", local_epoch)
    #
    #         # train_dimensions = [10,113]
    #     print("train_dimensions=", train_dimensions)
    #     train_dimensions_split = train_dimensions.copy()  # 拷贝，防止后面get_dataset传引用而改变
    #     # dataset
    #     feature, label, index = get_dataset(n_party, data_home, train_dimensions_split)  # 此处传参为引用类型
    # else:
    #
    #     if args.dimensions == "AVG":
    #         last = dimension % n_party  # 均分之后剩余的feature数
    #         avg = dimension // n_party # 均分每个client分到的features数
    #         train_dimensions = [avg] * n_party
    #         train_dimensions[n_party-1] = avg+last
    #         local_epoch = [1] * n_party
    #     else:
    #         local_epoch = []
    #         low = dimension // n_party -10 # 随机每一个client上的feature数量，最少
    #         high = dimension // n_party + 10 # 随机每一个client上的feature数量，最多
    #         y0 = np.random.randint(low,high, size= n_party-1)
    #         ratio = sum(y0)/dimension
    #         if n_party > 10:
    #             train_dimensions = y0//ratio
    #         else:
    #             train_dimensions = y0
    #         train_dimensions = train_dimensions.tolist()
    #         train_dimensions.append(dimension-sum(train_dimensions))
    #         for i in range(n_party): # 浮点数转为int, 同时随机每个client训练的local_epoch数
    #             train_dimensions[i] = int(train_dimensions[i])
    #             local_epoch.append(np.random.randint(1,10)) # 每一个client随机训练1-10次
    #             # local_epoch = []
    #         print("local_epoch",local_epoch)
    #
    #     # train_dimensions = [10,113]
    #     print("train_dimensions=",train_dimensions)
    #     train_dimensions_split = train_dimensions.copy() #拷贝，防止后面get_dataset传引用而改变
    #     # dataset
    #     feature, label, index = get_dataset(n_party, data_home, train_dimensions_split) #此处传参为引用类型

    # 初始化

    _, feature, label, index = initial(args, dimension, data_home, n_party, train_dimensions)

    start = time.time()
    # Initialization server
    local_value_cache_server = np.zeros([n_sample, n_party])  # 服务器端局部数值缓存
    aggregate_value_caceh_server = np.zeros([n_sample, 1])  # 服务器端全局累加缓存

    # temp_gradient = np.zeros([n_party, 2])  # 保存相邻两次同步的梯度，用于计算 L_k
    temp_gradient = [[] for i in range(n_party)]
    temp_weight = [[] for i in range(n_party)]  # 保存相邻两次同步的梯度对应的参数值，用于计算 L_k
    temp_control_params = [[] for i in range(args.n_party)]   # [(g0,w0),(g1, w1)]

    # --worker --
    local_value_cache_worker = [np.zeros([n_sample, 1])]*args.n_party  # worker全局累加量缓存,定义一个list保存
    local_other_cache_worker = [np.zeros([n_sample, 1])]*args.n_party  # worker全局累加量减去自己的局部预测值，用来辅助标准化tf的训练模块过程
    feature_input_indices_worker = [tf.placeholder(tf.int64, [None, 2])]*args.n_party
    feature_input_value_worker = [tf.placeholder(tf.float32, [None])]*args.n_party
    feature_input_worker = []
    for i in range(n_party):
        feature_input_worker.append(tf.SparseTensor(feature_input_indices_worker[i],
                                                feature_input_value_worker[i], [batch_size, train_dimensions[i]])) #
    print("feature input",feature_input_worker[0], feature_input_worker[0].shape, type(feature_input_worker[0]))
    other_sum_worker = [tf.placeholder(tf.float32, [None, 1])]*args.n_party
    label_worker = [tf.placeholder(tf.float32, [None, 1])]*args.n_party
    learning_rate_worker = tf.placeholder(tf.float32, 1)  # 用于动态递减学习率
    # current_learning_rate = [tf.placeholder(tf.float32, 1)]*args.n_party

    pred_worker = [tf.placeholder(tf.float32, [batch_size, 1])]*args.n_party
    print("pred_worker[0]=",pred_worker[0])
    loss_worker = [tf.placeholder(tf.float32, [batch_size, 1])]*args.n_party
    all_weight = [tf.placeholder(tf.float32, [None, 1])]*args.n_party
    local_pred_worker = [tf.placeholder(tf.float32, [batch_size, 1])]*args.n_party

    # pred_worker = [None]*args.n_party
    # # print("pred_worker[0]=",pred_worker[0])
    # loss_worker = [None]*args.n_party
    # local_pred_worker = [None]*args.n_party    
    optimizer_worker = [None]*args.n_party
    gradient_worker = [None]*args.n_party

    # Build Model
    if args.model == "logistic":
        for i in range(args.n_party):
            scopes = "worker_" + str(i)
            pred_worker[i], loss_worker[i], local_pred_worker[i], all_weight[i] = nn.partial_logistic_regression(
                feature_input_worker[i], other_sum_worker[i], label_worker[i], train_dimensions[i],
                scope=scopes, sparse_input=True, l2_reg=l2_reg)
            loss_worker[i] = loss_worker[i] + tf.add_n(tf.get_collection("worker_"+str(i)+"_reg_losses"))  # attach the regulation loss
            optimizer_worker[i] = tf.train.GradientDescentOptimizer(learning_rate_worker[0]).minimize(loss_worker[i])
            gradient_worker[i] = tf.train.GradientDescentOptimizer(learning_rate_worker[0]).compute_gradients(loss_worker[i],var_list=all_weight[i])
    else:
        for i in range(args.n_party):
            scopes = "worker_" + str(i)
            pred_worker[i], loss_worker[i], local_pred_worker[i], all_weight[i] = nn.partial_multilayer(
                feature_input_worker[i], other_sum_worker[i], label_worker[i], train_dimensions[i],
                scope=scopes, sparse_input=True, l2_reg=l2_reg)
            loss_worker[i] = loss_worker[i] + tf.add_n(tf.get_collection("worker_"+str(i)+"_reg_losses"))  # attach the regulation loss
            optimizer_worker[i] = tf.train.GradientDescentOptimizer(learning_rate_worker[0]).minimize(loss_worker[i])
            gradient_worker[i] = tf.train.GradientDescentOptimizer(learning_rate_worker[0]).compute_gradients(loss_worker[i],var_list=all_weight[i])

    ##============================
    # 正式迭代过程
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # print('initialized parameters:', sess.run(all_weight[0]))
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
        avg_cost_worker = [0]*args.n_party
        avg_acc_worker = [0]*args.n_party
        avg_time = [0]*args.n_party
        current_cost = [0]*args.n_party
        # current_gradient = [[0] * batch_count]*args.n_party  # 以这种方式创建的多维列表，其中一个子列表改变了，另外的子列表值也会随之改变
        current_gradient = [[0 for _ in range(batch_count)] for _ in range(args.n_party)]
        current_weight = [0]*args.n_party
        com_time = 0
        local_epoch = []

        # 获取当前weight
        weight_flatten_array = get_weight(args, sess, all_weight)
        for i in range(n_party):
            current_weight[i] = weight_flatten_array[i]
            # temp_weight[i] = current_weight[i]

        # 预处理，计算梯度
        for batch in range(batch_count): # batch_count 按照batch size 执行多少次，使所有样本数据遍历一次
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

            # 工作节点拉取全局数据
            # Pull global aggregation --worker--
            start_com = time.time()
            for worker in range(n_party): #
                local_other_cache_worker[worker][start_idx:end_idx] = aggregate_value_caceh_server[start_idx:end_idx] \
                                                        - local_value_cache_worker[worker][start_idx:end_idx]
            end_com = time.time()
            com_time = end_com - start_com

            # print('batch, start_idx:end_idx', batch, start_idx, end_idx)

            # 先计算当前模型在每个mini-batch的梯度
            for worker in range(n_party):
                # print('current gradient:', current_gradient[worker])
                # print('gradient worker:', gradient_worker[worker])
                # print(worker, label[worker][start_idx:end_idx], len(label[worker][start_idx:end_idx]))
                grad_var_list = sess.run([grad for grad in gradient_worker[worker]], feed_dict={
                    feature_input_indices_worker[worker]: csr_to_indices(feature[worker][start_idx:end_idx]),
                    feature_input_value_worker[worker]: feature[worker][start_idx:end_idx].data,
                    other_sum_worker[worker]: local_other_cache_worker[worker][start_idx:end_idx],
                    label_worker[worker]: label[worker][start_idx:end_idx].reshape(batch_size, 1)})

                # reshape the gradient
                current_gradient[worker][batch] = get_gradient(grad_var_list)
                # temp_gradient[worker] = current_gradient[worker]
                # print('temp_control_params[worker]', worker, temp_control_params[worker], len(temp_control_params))

        # ----------------保存当前每个worker对应的（w_t,g_t）对--------------------------
        for worker in range(n_party):
            # print('worker', worker, len(current_gradient[worker]), len(current_weight[worker]))
            temp_control_params[worker].append((current_weight[worker], current_gradient[worker]))
            # print(len(temp_control_params[worker]))
            if len(temp_control_params[worker]) > 2:
                del temp_control_params[worker][0]
            # else:
            #     continue

        # ----------------------------------------------------------------------
        # 估计当前的worker对应的local_epoch值
        # print('temp_control_params:',len(temp_control_params[0]))
        cmp_time = 0.001359
        if len(temp_control_params[0]) < 2:
            local_epoch = [1 for i in range(n_party)]
        else:
            local_epoch = ControlAdaptiveE(args, R, temp_control_params, current_learning_rate, batch_count).compute_new_E(com_time, cmp_time)

        # ---------------+++++++++++++++++++++Start Training+++++++++++++++++++++-------------------#
        for batch in range(batch_count):  # batch_count 按照batch size 执行多少次，使所有样本数据遍历一次
            # 预处理
            start_idx, end_idx = batch_size * batch, batch_size * (batch + 1)
            total_batch_index = epoch * batch_size + batch_size

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

            for worker in range(n_party):  # 每个worker都进行更新操作
                start_epoch = time.time()

                # Do the training
                # print('local_epoch:', local_epoch[worker])
                for i in range(int(local_epoch[worker])):
                    # 本地更新参数
                    start_cmp = time.time()
                    current_cost[worker], _, pred_vals, current_weight[worker] = sess.run([loss_worker[worker],
                                                                                           optimizer_worker[worker],
                                                                                           pred_worker[worker],
                                                                                           all_weight[worker]],
                                                                                          feed_dict={
                        feature_input_indices_worker[worker]: csr_to_indices(feature[worker][start_idx:end_idx]),
                        feature_input_value_worker[worker]: feature[worker][start_idx:end_idx].data,
                        other_sum_worker[worker]: local_other_cache_worker[worker][start_idx:end_idx],
                        label_worker[worker]: label[worker][start_idx:end_idx].reshape(batch_size, 1),
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
                avg_time[worker] += (end_epoch - start_epoch)/batch_count

            # if batch == 0:
            # 保存当前round开始时的梯度
            # grad_flatten_array = get_gradient(args, current_gradient)
            # temp_gradient[worker].append(grad_flatten_array[worker])

            # 每个worker更新完毕
        for i in range(n_party): # 每个cost都进行更新操作
            cost[i][epoch] = avg_cost_worker[i]
            auc[i][epoch] = avg_acc_worker[i]
            # exec_time[worker].append(max(avg_time_0, avg_time_1))
            # 显示每个epoch训练信息
            print("Woker", '%d' %i," Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost_worker[i]))
        exec_time[epoch] = max(avg_time)
        print("\n")

    e = 0
    final_time = []
    for index in range(len(exec_time)):
        e += exec_time[index]
        final_time.append(e)
    print('execution time:', final_time)

    tep = []
    for i in range(len(cost[-1])):
        tep.append([cost[-1][i], auc[-1][i], final_time[i]])
    tep = pd.DataFrame(tep, columns=["loss", "auc", "time"])
    tep.to_excel("./log/adaptive/" + str(args.model) + "_"+ str(args.dataset)+"_N["+str(args.n_party)+"].xlsx",index = False)
    # tep.to_excel("./log[" + str(train_dimensions[0]) + "]_" + "B[" + str(batch_size) + "]_" + "E["
    #             + str(batch_epoch_0) + "+" + str(batch_epoch_1) + "]_" + str(num_epochs) + ".xlsx",index = False)

    # 画图
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(len(cost[-1])), cost[-1], label = 'LR')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(range(len(auc[-1])), auc[-1], 'r-.', label = 'LR')
    plt.xlabel('Round')
    plt.ylabel('AUC')
    plt.legend()

    # show函数展示出这个图，如果没有这行代码，则程序完成绘图，但看不到
    plt.show()
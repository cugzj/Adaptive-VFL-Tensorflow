import numpy as np
import os
from sklearn.datasets import load_svmlight_file
import tensorflow as tf
from sklearn import metrics
from utils.split import split


def get_dataset(args, n_party, data_home, train_dimensions):
    split(n_party,train_dimensions) # 切分数据集
    feature = []
    label = []
    index = []

    for i in np.arange(int(n_party)):
        if args.dataset == "a9a":
            current_path = os.path.join(data_home, 'a9a-part'+str(i))
        elif args.dataset == "Citeseer":
            current_path = os.path.join(data_home, 'Citeseer-part'+str(i))
        else:
            current_path = os.path.join(data_home, 'mnist-binary-part' + str(i))
        # print('current_path:', current_path)
        [temp_feature, temp_label, temp_index] = load_svmlight_file(current_path, query_id = True)
        feature.append(temp_feature)
        label.append(temp_label>0)
        index.append(temp_index)
    # print('feature:', feature)
    # print('label:', label)
    # print('index:', index)
    return feature, label, index


def csr_to_indices(X):
    coo = X.tocoo()
    return np.mat([coo.row, coo.col]).transpose()


def csr_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def eval_pred(true_label, pred_prob):
    fpr, tpr, thresholds = metrics.roc_curve(true_label, pred_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    nll = metrics.log_loss(true_label, pred_prob, eps=1e-7)
    return auc, nll


def acc_pred(true_lable, pred_prob):
    acc, acc_op = tf.metrics.accuracy(labels=true_lable, predictions=pred_prob, name="acc_op")
    return acc_op


def get_weight(args, sess, all_weight):
    weight_flatten_list = {}
    weight_flatten_array = {}
    for i in range(args.n_party):
        weight_flatten_list[i] = []
        for weight in all_weight[i]:
            weight_var = sess.run(weight)
            weight_flatten_list[i].append(np.reshape(weight_var, weight_var.size))

        weight_flatten_array[i] = np.hstack(weight_flatten_list[i])
    # print('weight:', weight_flatten_array)

    # weight_flatten_list = []
    # for weight in self.all_weights:
    #     weight_var = self.session.run(weight)
    #     weight_flatten_list.append(np.reshape(weight_var, weight_var.size))
    #
    # weight_flatten_array = np.hstack(weight_flatten_list)

    return weight_flatten_array


def get_gradient(current_gradient_worker):
    grad_flatten_list = []
    for l in current_gradient_worker:
        grad_flatten_list.append(np.reshape(l[0], l[0].size))

    grad_flatten_array = np.hstack(grad_flatten_list)

    del current_gradient_worker
    del grad_flatten_list

    return grad_flatten_array

def get_control_params(args):
    pass
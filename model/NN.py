#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np 

def _linear_layer(all_weight, feature_input, size, scope = 'default',sparse_input = True, l2_reg = 0.01, is_with_bias = False):
    # print(scope)
    with tf.variable_scope(scope):
        W = tf.get_variable("W", [size, 1], initializer = tf.random_normal_initializer(stddev = 0.1), trainable = True)
        all_weight.append(W)
        if sparse_input == True:
            output = tf.sparse_tensor_dense_matmul(feature_input, W)
        else:
            output = tf.matmul(feature_input, W)
        if is_with_bias == True:
            B = tf.get_variable("B", initializer=tf.constant(0.), trainable = True)
            output = output + B
            all_weight.append(B)

        tf.add_to_collection(scope+"_reg_losses", tf.contrib.layers.l2_regularizer(l2_reg)(W))
        return output, all_weight

def _FM_layer_no_linear(feature_input, size, scope = 'default',sparse_input = True, l2_reg = 0.01, dim = 4, is_without_self_cross = True):
    with tf.variable_scope(scope):
        # size = 66
        W = tf.get_variable("Hidden_Vec", [size, dim], initializer = tf.random_normal_initializer(stddev = 0.1/np.sqrt(float(dim))), trainable = True)
        if sparse_input == True:
            if is_without_self_cross == False:
                output = 0.5 * tf.reduce_sum(tf.square(tf.sparse_tensor_dense_matmul(feature_input, W)), 1, keepdims = True) 
            else:
                output = 0.5 * tf.reduce_sum((tf.square(tf.sparse_tensor_dense_matmul(feature_input, W)) - tf.sparse_tensor_dense_matmul(tf.square(feature_input), tf.square(W))), 1, keepdims = True)
        else:
            if is_without_self_cross == False:
                output = 0.5 * tf.reduce_sum(tf.square(tf.matmul(feature_input, W)), 1, keepdims = True) 
            else:
                output = 0.5 * tf.reduce_sum((tf.square(tf.matmul(feature_input, W)) - tf.matmul(tf.square(feature_input), tf.square(W))), 1, keepdims = True)

        tf.add_to_collection(scope+"_reg_losses", tf.contrib.layers.l2_regularizer(l2_reg/np.sqrt(float(dim)))(W))
        return output

def _full_connect_activated_dense_input(x, weight_shape, bias_shape, activation = 'relu', l2_reg = 0.001, scope = 'default', name_prefix = 'default', is_train = True):
    with tf.variable_scope(scope):
        W =  tf.get_variable(name_prefix+"_W", weight_shape, initializer = tf.random_normal_initializer(stddev = 0.1), trainable = is_train)
        output = tf.matmul(x, W )
        if bias_shape != None:
            B = tf.get_variable(name_prefix+"_B", initializer = tf.constant(np.zeros(bias_shape).astype(np.float32)))
            output = output + B

        if activation == 'relu':
            output = tf.nn.relu(output)
        elif activation == 'tanh':
            output = tf.nn.tanh(output)
        else:
            print ("Undefined activation.")
            raise TypeError
        return output

def _full_connect_activated_sparse_input(all_weight, x, weight_shape, bias_shape, activation = 'relu', l2_reg = 0.001, scope = 'default', name_prefix = 'default', is_train = True):
    with tf.variable_scope(scope):
        W =  tf.get_variable(name_prefix+"_W", weight_shape, initializer = tf.random_normal_initializer(stddev = 0.1), trainable = is_train)
        output = tf.sparse_tensor_dense_matmul(x, W )
        all_weight.append(W)
        if bias_shape != None:
            B = tf.get_variable(name_prefix+"_B", initializer = tf.constant(np.zeros(bias_shape).astype(np.float32)))
            output = output + B
            all_weight.append(B)

        if activation == 'relu':
            output = tf.nn.relu(output)
        elif activation == 'tanh':
            output = tf.nn.tanh(output)
        else:
            print ("Undefined activation.")
            raise TypeError
        return output, all_weight

def partial_logistic_regression(feature_input, other_sum, label, size, scope = 'default', sparse_input = True, l2_reg = 0.01):
    '''
    并行计算逻辑回归模块
        feature_input [none * size]
        other_sum [none * 1]
    '''
    all_weight = []
    with tf.variable_scope(scope):
        # W = tf.get_variable("W", [size, 1], initializer=tf.random_normal_initializer())
        local, all_weight = _linear_layer(all_weight, feature_input, size, scope, sparse_input, l2_reg, is_with_bias = False)
        
        if scope == "worker_0":
            print ("Woker 0: initialized a bias term. ")
            B = tf.get_variable("B", initializer = tf.constant(0.))
            local = local + B
            all_weight.append(B)

        logit = tf.add(local, other_sum)
        output = tf.nn.sigmoid(logit)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits = logit))
        return output, loss, local, all_weight

def partial_FM(feature_input, other_sum, label, size, scope = 'default', sparse_input = True, l2_reg = 0.01, dim = 4):
    '''
    并行计算FM
    '''
    with tf.variable_scope(scope):
        # 线性部分
        local = _linear_layer(feature_input, size, scope, sparse_input, l2_reg, is_with_bias = False)
        # 交叉部分
        # local = _FM_layer_no_linear(feature_input, size, scope,sparse_input, l2_reg, dim = 4, is_without_self_cross = True)
        local = tf.add(local, _FM_layer_no_linear(feature_input, size, scope,sparse_input, l2_reg, dim = 4, is_without_self_cross = True))
        
        if scope == "worker_0":
            print ("Woker 0: initialized a bias term. ")
            B = tf.get_variable("B", initializer = tf.constant(0.))
            local = local + B

        logit = tf.add(local, other_sum)
        output = tf.nn.sigmoid(logit)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits = logit))
        # loss = tf.reduce_mean(tf.square(logit - label))
        return output, loss, local

def partial_multilayer(feature_input, other_sum, label, size, scope = 'default', sparse_input = True, l2_reg = 0.001, hidden_size = [20, 5, 5]):
    '''
    并行计算逻辑回归模块
        feature_input [none * size]
        other_sum [none * 1]
    '''
    all_weight = []
    num_hidden_layers = len(hidden_size)
    with tf.variable_scope(scope):
        if sparse_input == True:
            local, all_weight = _full_connect_activated_sparse_input(all_weight, feature_input, [size, hidden_size[0]], hidden_size[0], activation = 'relu', l2_reg = l2_reg, scope = scope, name_prefix = '0', is_train = True)
        else:
            local = _full_connect_activated_dense_input(feature_input, [size, hidden_size[0]], hidden_size[0], activation = 'relu', l2_reg = l2_reg, scope = scope, name_prefix = '0', is_train = True)
        if num_hidden_layers > 1:
            for layer in range(1, num_hidden_layers):
                local = _full_connect_activated_dense_input(local, [hidden_size[layer-1], hidden_size[layer]], hidden_size[layer], activation = 'relu', l2_reg = l2_reg, scope = scope, name_prefix = str(layer), is_train = True)
        local, all_weight = _linear_layer(all_weight, local, hidden_size[num_hidden_layers-1], scope, False, l2_reg, is_with_bias = False)
        if scope == "worker_0":
            print ("Woker 0: initialized a bias term. ")
            B = tf.get_variable("B", initializer = tf.constant(0.))
            local = local + B
            all_weight.append(B)
        print('local result:', local)
        logit = tf.add(local, other_sum)   # server端聚合
        output = tf.nn.sigmoid(logit)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits = logit))
        return output, loss, local, all_weight

def logistic_regression(feature_input, label, size, scope = 'default', sparse_input = True, l2_reg = 0.01):
    '''
    并行计算逻辑回归模块
        feature_input [none * size]
    '''
    with tf.variable_scope(scope):
        # W = tf.get_variable("W", [size, 1], initializer=tf.random_normal_initializer())
        logit = _linear_layer(feature_input, size, scope, sparse_input, l2_reg, is_with_bias = True)
        output = tf.nn.sigmoid(logit)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits = logit))
        return output, loss

def FM(feature_input, label, size, scope = 'default', sparse_input = True, l2_reg = 0.01, dim = 4):
    '''
    FM
    '''
    with tf.variable_scope(scope):
        # 线性部分
        logit = _linear_layer(feature_input, size, scope, sparse_input, l2_reg, is_with_bias = True)
        # 交叉部分
        logit = tf.add(logit, _FM_layer_no_linear(feature_input, size, scope,sparse_input, l2_reg, dim = 4, is_without_self_cross = True))

        output = tf.nn.sigmoid(logit)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits = logit))
        # loss = tf.reduce_mean(tf.square(logit - label))
        return output, loss

def Multilayer(feature_input, label, size, scope = 'default', sparse_input = True, l2_reg = 0.001, hidden_size = [20, 5, 5]):
    '''
    并行计算逻辑回归模块
        feature_input [none * size]
        other_sum [none * 1]
    '''
    num_hidden_layers = len(hidden_size)
    with tf.variable_scope(scope):
        if sparse_input == True:
            # local = _full_connect_activated_sparse_input(feature_input, [size, hidden_size[0]], hidden_size[0], activation = 'relu', l2_reg = l2_reg, scope = scope, name_prefix = '0', is_train = True)
            logit = _full_connect_activated_sparse_input(feature_input, [size, hidden_size[0]], hidden_size[0], activation = 'relu', l2_reg = l2_reg, scope = scope, name_prefix = '0', is_train = True)
        else:
            logit = _full_connect_activated_dense_input(feature_input, [size, hidden_size[0]], hidden_size[0], activation = 'relu', l2_reg = l2_reg, scope = scope, name_prefix = '0', is_train = True)
        if num_hidden_layers > 1:
            for layer in range(1, num_hidden_layers):
                logit = _full_connect_activated_dense_input(logit, [hidden_size[layer-1], hidden_size[layer]], hidden_size[layer], activation = 'relu', l2_reg = l2_reg, scope = scope, name_prefix = str(layer), is_train = True)
        logit = _linear_layer(logit, hidden_size[num_hidden_layers-1], scope, False, l2_reg, is_with_bias = True)
        output = tf.nn.sigmoid(logit)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits = logit))
        return output, loss
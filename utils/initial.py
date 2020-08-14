from option import args_parser
import numpy as np
import time
from utils.utils import get_dataset
import model.NN as nn
import tensorflow as tf # tf v1

# 初始化
def initial(args, dimension, data_home, n_party, train_dimensions):
    if train_dimensions:
        if args.dimensions == "AVG":
            local_epoch = [1] * n_party
        else:
            local_epoch = []
            for i in range(n_party):  # 浮点数转为int, 同时随机每个client训练的local_epoch数
                # local_epoch.append(np.random.randint(1, 10))  # 每一个client随机训练1-10次
                local_epoch = [args.fix_epochs] * n_party
                # local_epoch = [4, 4, 3, 2, 1]

            print("local_epoch", local_epoch)

            # train_dimensions = [10,113]
        print("train_dimensions=", train_dimensions)
        train_dimensions_split = train_dimensions.copy()  # 拷贝，防止后面get_dataset传引用而改变
        # dataset
        print('data_home:', data_home)
        feature, label, index = get_dataset(args, n_party, data_home, train_dimensions_split)  # 此处传参为引用类型
    else:

        if args.dimensions == "AVG":
            last = dimension % n_party  # 均分之后剩余的feature数
            avg = dimension // n_party # 均分每个client分到的features数
            train_dimensions = [avg] * n_party
            train_dimensions[n_party -1] = avg +last
            local_epoch = [1] * n_party
        else:
            local_epoch = []
            low = dimension // n_party -10  # 随机每一个client上的feature数量，最少
            high = dimension // n_party + 10  # 随机每一个client上的feature数量，最多
            y0 = np.random.randint(low, high, size=n_party - 1)
            ratio = sum(y0) / dimension
            if n_party > 10:
                train_dimensions = y0 // ratio
            else:
                train_dimensions = y0
            train_dimensions = train_dimensions.tolist()
            train_dimensions.append(dimension - sum(train_dimensions))
            for i in range(n_party):  # 浮点数转为int, 同时随机每个client训练的local_epoch数
                train_dimensions[i] = int(train_dimensions[i])
                local_epoch.append(np.random.randint(1, 10))  # 每一个client随机训练1-10次
                # local_epoch = []
            print("local_epoch", local_epoch)

        # train_dimensions = [10,113]
        print("train_dimensions=", train_dimensions)
        train_dimensions_split = train_dimensions.copy()  # 拷贝，防止后面get_dataset传引用而改变
        # dataset
        feature, label, index = get_dataset(args, n_party, data_home, train_dimensions_split)  # 此处传参为引用类型

    return local_epoch, feature, label, index

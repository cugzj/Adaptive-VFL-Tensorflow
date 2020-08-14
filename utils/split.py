"""
 Transform the original libsvm file into several seperate data format
"""
from option import args_parser
import scipy as sp
import numpy as np
import glob
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import os

def split(n_party,milestones):
    # change this for specific purpose
    # input_path = "./data/a9a" #mnist-binary
    args = args_parser()
    input_path = args.input_path
    for i in range(n_party-1):
        milestones[i+1] += milestones[i]
    print("milestones=", milestones)
    # milestones: [0, 66, 123] feature分为两个部分66+57
    n_features = 0
    if args.input_path == './data/a9a':
        n_features = 123
    elif args.input_path == './data-mnist/mnist-binary':
        n_features = 784
    else:
        n_features = 105354
    [feature_raw, label] = load_svmlight_file(input_path, n_features)
    print('feature_raw_shape:', feature_raw.shape)
    print('label:', label, label.shape)
    [num_row, num_col] = feature_raw.shape
    milestones.insert(0, 0)
    milestones.append(num_col)
    intervals = []
    for i in range(len(milestones) - 1):
        intervals.append(np.arange(milestones[i], milestones[i + 1]))
    # print('intervals:', intervals)
    feature_coo = sp.sparse.coo_matrix(feature_raw)
    # print('feature_coo:', feature_coo)

    for file in glob.glob(input_path + "-*"):
        os.remove(file)

    # generating and saving the parts
    query_id = np.arange(num_row)  # 32561
    num_parts = len(intervals)   # 2
    for i in range(num_parts-1):
        # split the matrix
        print(feature_coo.col, len(feature_coo.col))
        index_merge = (feature_coo.col == intervals[i][0])
        # print('index_merge:', index_merge, len(index_merge))
        for j in intervals[i]:
            index_merge = index_merge + (feature_coo.col == j)
        cur_feature_row = feature_coo.row[index_merge]
        cur_feature_col = feature_coo.col[index_merge]
        # print("debug info:")
        # print("col min", np.min(cur_feature_col))
        # print("intervals min", np.min(intervals[i]))
        cur_feature_col = cur_feature_col - np.min(intervals[i])
        cur_feature_data = feature_coo.data[index_merge]
        cur_feature = sp.sparse.coo_matrix((cur_feature_data, (cur_feature_row, cur_feature_col)),
                                           shape=(num_row, intervals[i].size))
        cur_output_path = input_path + "-part" + str(i)
        dump_svmlight_file(cur_feature, label, cur_output_path, query_id=query_id)
        cur_output_meta_path = cur_output_path + ".meta"
        with open(cur_output_meta_path, "w") as fout:
            fout.write(str(np.max(intervals[i]) - np.min(intervals[i]) + 1))

    # dump a normalized version
    dump_svmlight_file(feature_raw, label, input_path, query_id=query_id)

    # generating the server parts
    command = ["cut", "-d", "\" \"", "-f1,2", input_path + "-part0", ">", input_path + "-server"]
    command = " ".join(command)
    print(command)
    os.system(command)

    with open(input_path + "-server.meta", "w") as fout:
        fout.write("1")

if __name__ == "__main__":
    # change this for specific purpose
    input_path = "data_unequal_[93_30]/a9a"
    # if not os.path.exists(input_path):
    # milestones: [0, 66, 123] feature分为两个部分66+57
    milestones = [93]

    [feature_raw, label] = load_svmlight_file(input_path)
    print('feature_raw_shape:', feature_raw.shape)
    print('label:', label, label.shape)
    [num_row, num_col] = feature_raw.shape
    milestones.insert(0, 0)
    milestones.append(num_col)
    print('milestones:', milestones)
    intervals = []
    for i in range(len(milestones) - 1):
        intervals.append(np.arange(milestones[i], milestones[i + 1]))
    print('intervals:', intervals)
    feature_coo = sp.sparse.coo_matrix(feature_raw)
    print('feature_coo:', feature_coo)

    for file in glob.glob(input_path + "-*"):
        os.remove(file)

    # generating and saving the parts
    query_id = np.arange(num_row)  # 32561
    num_parts = len(intervals)   # 2
    for i in range(num_parts):
        # split the matrix
        print(feature_coo.col, len(feature_coo.col))
        index_merge = (feature_coo.col == intervals[i][0])
        # print('index_merge:', index_merge, len(index_merge))
        for j in intervals[i]:
            index_merge = index_merge + (feature_coo.col == j)
        cur_feature_row = feature_coo.row[index_merge]
        cur_feature_col = feature_coo.col[index_merge]
        print("debug info:")
        print("col min", np.min(cur_feature_col))
        print("intervals min", np.min(intervals[i]))
        cur_feature_col = cur_feature_col - np.min(intervals[i])
        cur_feature_data = feature_coo.data[index_merge]
        cur_feature = sp.sparse.coo_matrix((cur_feature_data, (cur_feature_row, cur_feature_col)),
                                           shape=(num_row, intervals[i].size))
        cur_output_path = input_path + "-part" + str(i)
        dump_svmlight_file(cur_feature, label, cur_output_path, query_id=query_id)
        cur_output_meta_path = cur_output_path + ".meta"
        with open(cur_output_meta_path, "w") as fout:
            fout.write(str(np.max(intervals[i]) - np.min(intervals[i]) + 1))

    # dump a normalized version
    dump_svmlight_file(feature_raw, label, input_path, query_id=query_id)

    # generating the server parts
    command = ["cut", "-d", "\" \"", "-f1,2", input_path + "-part0", ">", input_path + "-server"]
    command = " ".join(command)
    print(command)
    os.system(command)

    with open(input_path + "-server.meta", "w") as fout:
        fout.write("1")

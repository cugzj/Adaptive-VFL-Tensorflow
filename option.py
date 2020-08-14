import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # distributed features  arguments
    parser.add_argument('--n_party', type=int, default=5, help="number of parties: MAX 50")
    parser.add_argument('--num_epochs', type=int, default=200, help="number of epochs")
    parser.add_argument('--fix_epochs', type=int, default=5, help="fix number of epochs")
    parser.add_argument('--batch_size', type=int, default=100, help="batch size")
    parser.add_argument('--learining_rate', type=float, default=0.1, help="learning rate")
    parser.add_argument('--l2_reg', type=float, default=0.0001, help="l2_reg")
    parser.add_argument('--dimensions', type=str, default='Non-AVG', help="train_dimensions(average or not)")

    # model arguments
    parser.add_argument('--model', type=str, default='NN', help='model name')

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--data_home', type=str, default='./data-cite', help='path of input')
    parser.add_argument('--input_path', type=str, default='./data-cite/Citeseer', help='path of input')
    parser.add_argument('--num_features', type=int, default=105354, help="total number of features")
    parser.add_argument('--n_sample', type=int, default=879, help="total number of samples")
    args = parser.parse_args()
    return args
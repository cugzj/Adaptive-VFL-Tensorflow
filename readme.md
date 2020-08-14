## Distributed Features

### 参数说明

```shell
--n_party 参与的party个数,默认为2
--num_epochs 训练轮数,默认为100
--batch_size batch size大小,默认为100
--learining_rate learining rate 默认为1
--l2_reg  l2_reg 默认为1.0
--dimensions feature是否均分给client，默认为AVG,表示均分，且每个client的local_epoch相同。若为Non-AVG则表示随机分，每个client的local_epoch随机1-10次
--model 训练的模型，默认为logistic, 可选择mltilayer
```

### 使用举例
#### Case 1. Baseline(local_epoch提前给定)

1. feature均分，client上的local_epoch均为1

```shell
python baseline.py # feature均分，每个client的local_epoch都为1
```

2. feature不均分， client上的local_epoch不同

```shell
python baseline.py --n_party 3 --num_epochs 50 --model logistic --dimensions Non-AVG
```


#### Case 2. Adaptive(local_epoch 由算法给定)

1. feature均分，client上的local_epoch均为1

```shell
python adaptive.py # feature均分，每个client的local_epoch都为1
```

2. feature不均分， client上的local_epoch不同

```shell
python adaptive.py --n_party 3 --num_epochs 50 --model logistic --dimensions Non-AVG

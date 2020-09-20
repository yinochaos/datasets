# datasets

![image](https://img.shields.io/pypi/v/datasets.svg%0A%20%20%20%20%20:target:%20https://pypi.python.org/pypi/datasets)

![image](https://img.shields.io/travis/yinochaos/datasets.svg%0A%20%20%20%20%20:target:%20https://travis-ci.com/yinochaos/datasets)

![image](https://readthedocs.org/projects/datasets/badge/?version=latest%0A%20%20%20%20%20:target:%20https://datasets.readthedocs.io/en/latest/?badge=latest%0A%20%20%20%20%20:alt:%20Documentation%20Status)

-   Free software: Apache Software License 2.0
-   Documentation: <https://ml-dataset.readthedocs.io>.

datasets for easy machine learning use
该项目的目的是提供一个简洁方便的dataset封装，使用少量代码即可实现的dataset，用以喂到模型中进行训练:
- 同时支持tensorflow和pytorch
- 支持从本地、HDFS、以及其他网络接口获取数据
- 支持文本、lmdb、tfrecord等数据文件格式【可以大大提高训练时GPU利用率】
- 支持文本、图像、语音等数据的基本处理
- 支持多种数据增强(data agumentation)方法 【TODO 未完成】
- 可以支持直接从输入数据文件，自动检查格式并生成适用于该数据的代码【TODO 未完成】


## Datasets 重要数据结构
--------
- DataSchema : 用于描述数据schema的结构体
```
name : 数据名称[每个数据名称不要重复]
processor : 对于该列数据需要使用的数据处理函数,具体参见data_processor_dicts, 包括对于数组、文本、图像、语音等的处理函数
type : 只对于TFDataset有用, 后续考虑去除掉该依赖
dtype : numpy的数据类型
shape : 处理后的数据shape
token_dict_name : 所需的词典【e.g. 】
is_with_len : 对于变长序列，是否需要产出数组变长维度的大小
max_len : 对于定长数组的最大长度设置
```
- TFDataset、PTDataset
    - 1
- Parser : 数据解析器
  - TextlineParser


### data_processor_dicts
data_processor_dicts是数据处理函数的集合词典，这里面包含了很多针对不同数据类型（e.g. 文本、语音、图像、数值等）进行特征提取、数据转换等处理，最终转换成Dataset,用以喂到模型进行训练预测等操作。

## example data
--------
- tests/data/raw_datasets/query_float.input format:id\tlabel\tquery\tfloats
```
1	1	面 对 疫 情	0.12 0.34 0.87 0.28
2	0	球 王 马 拉 多 纳 目 前 处 在 隔 离 之 中	0.12 0.34 0.87 0.28
```
- tests/data/raw_datasets/varnum.input format:id\tlabel\tnums
```
1	2	2 3 4 6 8 23 435 234 12 234 234
1	2	2 3 4 6 8 23 4 2 9 4 5 6 2 4
1	2	2 3 4 6 8 23 45 24 12 234 234
```
- tests/data/pairwise_datasets/simple_pair.input format: id\titem_feat1\titem_feat2\tscore1\tfeat1\tfeat2\tscore2\tfeat1\tfeat2\tscore3\tfeat1\tfeat2
```

```
- tests/data/seq_datasets/simple_seq.input


## TODO
--------
- 1
- 2


# datasets

![image](https://img.shields.io/pypi/v/ml-dataset.svg%0A%20%20%20%20%20:target:%20https://pypi.python.org/pypi/ml-dataset)
[![Build Status](https://travis-ci.com/yinochaos/datasets.svg?branch=master)](https://travis-ci.com/yinochaos/datasets)
[![Documentation Status](https://readthedocs.org/projects/ml-dataset/badge/?version=latest)](https://ml-dataset.readthedocs.io/en/latest/?badge=latest)

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
- TFDataset、PTDataset: dataset数据结构
    - generate_dataset() : 用于产出dataset的接口
- Parser : 数据解析器
  - TextlineParser

### data_processor_dicts
data_processor_dicts是数据处理函数的集合词典，这里面包含了很多针对不同数据类型（e.g. 文本、语音、图像、数值等）进行特征提取、数据转换等处理，最终转换成Dataset,用以喂到模型进行训练预测等操作。

## example data
--------
- tests/data/raw_datasets/query_float.input format:id\tlabel\tquery\tfloats
```
1  1  面 对 疫 情  0.12 0.34 0.87 0.28
2  0  球 王 马 拉 多 纳 目 前 处 在 隔 离 之 中  0.12 0.34 0.87 0.28
```
针对该数据集，需要新建feature_schema_list和label_schema , 完整代码可参考[code](https://github.com/yinochaos/datasets/blob/master/tests/test_tf_datasets.py#L82)
```python
token_dicts = TokenDicts('tests/data/dicts', {'query': 0})
data_field_list = []
# param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
data_field_list.append(DataSchema(name='query', processor='to_tokenid',
                                    dtype='int32', shape=(None,), is_with_len=True, token_dict_name='query'))
"""
这里的DataSchema描述数据和处理逻辑如下:
- 数据名称为query
- 使用to_tokenid的func进行数据处理，处理完成后，数据shape为(None,),数据type为int32，词典名称是query
- is_with_len=True意味着对于变长数据，会产出变长维度的具体大小
"""
data_field_list.append(DataSchema(
    name='width', processor='to_np', dtype='float32', shape=(4)))
"""
这里的DataSchema描述数据和处理逻辑如下:
- 数据名称为label
- 使用to_np的func进行数据处理，处理完成后，数据shape为(,),数据type为int32，词典名称是query
"""
label_field = DataSchema(name='label', processor='to_np', dtype='float32', shape=(1,))
"""
这里的DataSchema描述数据和处理逻辑如下:
- 数据名称为label
- 使用to_np的func进行数据处理，处理完成后，数据shape为(,),数据type为int32，词典名称是query
"""
#新建一个parser,该parser负责解析处理一般单行的数据输入
parser = TextlineParser(token_dicts, data_field_list, label_field)
# 新建generator，处理file_path下面，文件后缀是file_suffix的数据
generator = TFDataset(parser=parser, file_path='tests/data/raw_datasets', file_suffix='query_float.input')
# 产生dataset
dataset = generator.generate_dataset(
    batch_size=12, num_epochs=1, is_shuffle=False)
# 遍历dataset 
for _ in enumerate(dataset):
    pass

```
- tests/data/raw_datasets/varnum.input format:id\tlabel\tnums
```
1  2  2 3 4 6 8 23 435 234 12 234 234
1  2  2 3 4 6 8 23 4 2 9 4 5 6 2 4
1  2  2 3 4 6 8 23 45 24 12 234 234
```
针对该数据集，需要新建feature_schema_list和label_schema , 完整代码可以参见 [code](https://github.com/yinochaos/datasets/blob/master/tests/test_tf_datasets.py#L100)
```python
token_dicts = None
data_field_list = []
data_field_list.append(DataSchema(name='query', processor='to_np',
                                    dtype='int32', shape=(None,), is_with_len=True))
label_field = DataSchema(name='label', processor='to_np', dtype='float32', shape=(1,), is_with_len=False)
parser = TextlineParser(token_dicts, data_field_list, label_field)
generator = TFDataset(parser=parser, file_path='tests/data/raw_datasets', file_suffix='varnum.input')
# 产生dataset
dataset = generator.generate_dataset(
    batch_size=12, num_epochs=1, is_shuffle=False)
# 遍历dataset 
for _ in enumerate(dataset):
    pass
```
dataset遍历,可以参考[pass_dataset](https://github.com/yinochaos/datasets/blob/dfacaca19a04dccf43575aadfe85c2001e88047a/tests/test_tf_datasets.py#L36)
```python
def pass_dataset(self, is_training, weight_fn, dataset):
    if weight_fn:
        if is_training:
            for batch_num, (x, label, weight) in enumerate(dataset):
                print('x', x)
                print('weight', weight)
                for d in x:
                    print('d.shape', d.shape)
                print('label.shape', label.shape)
                print('batch_num', batch_num)
                break
    else:
        if is_training:
            for batch_num, (x, label) in enumerate(dataset):
                print('x', x)
                for d in x:
                    print('d.shape', d.shape)
                print('label.shape', label.shape)
                print('batch_num', batch_num)
                break
        else:
            for batch_num, (info, x, label) in enumerate(dataset):
                print('info', info)
                print('x', x)
                for d in x:
                    print('d.shape', d.shape)
                print('label.shape', label.shape)
                print('batch_num', batch_num)
                break
```


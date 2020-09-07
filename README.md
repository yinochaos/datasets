# datasets

![image](https://img.shields.io/pypi/v/datasets.svg%0A%20%20%20%20%20:target:%20https://pypi.python.org/pypi/datasets)

![image](https://img.shields.io/travis/yinochaos/datasets.svg%0A%20%20%20%20%20:target:%20https://travis-ci.com/yinochaos/datasets)

![image](https://readthedocs.org/projects/datasets/badge/?version=latest%0A%20%20%20%20%20:target:%20https://datasets.readthedocs.io/en/latest/?badge=latest%0A%20%20%20%20%20:alt:%20Documentation%20Status)

datasets for easy machine learning use

-   Free software: Apache Software License 2.0
-   Documentation: <https://ml-dataset.readthedocs.io>.

## Datasets API
--------
- DataSchema
- generate_dataset()

### raw_dataset
RawDataset是针对一行的数据内容进行解析处理，转换得到Dataset的
<details>
<summary>展开查看具体举例说明</summary>
<pre><code>
token_dicts = None
data_filed_list = []
data_filed_list.append(DataSchema(name='query', processor='to_np', type=tf.int32,
                                    dtype='int32', shape=(None,), is_with_len=True))
label_field = DataSchema(name='label', processor='to_np',
                            type=tf.float32, dtype='float32', shape=(1,), is_with_len=False)
generator = RawDataset(file_path="tests/data/raw_datasets", token_dicts=token_dicts,
                        data_field_list=data_filed_list, label_field=label_field, file_suffix='varnum.input')
dataset = generator.generate_dataset(batch_size=4, is_training=True)
for batch_num, (x, label) in enumerate(dataset):
    pass
</code></pre>
</details>

### seq_dataset
SeqDataset是针对序列按照行来排列的形式解析解析处理，并转换得到Dataset的。

<details>
<summary>展开查看具体举例说明</summary>
<pre><code>
#@TODO
</code></pre>
</details>


### pairwise_dataset

<details>
<summary>展开查看具体举例说明</summary>
<pre><code>
#@TODO
</code></pre>
</details>

### listwise_dataset

<details>
<summary>展开查看具体举例说明</summary>
<pre><code>
#@TODO
</code></pre>
</details>

### data_processor_dicts
data_processor_dicts是数据处理函数的集合词典，这里面包含了很多针对不同数据类型（e.g. 文本、语音、图像、数值等）进行特征提取、数据转换等处理，最终转换成Dataset,用以喂到模型进行训练预测等操作。

<details>
<summary>展开查看具体举例说明</summary>
<pre><code>
System.out.println("Hello to see U!"); # aaadf
</code></pre>
hello
</details>

## example data
--------
- tests/data/raw_datasets/query_float.input format:id\tlabel\tquery\tfloats
- tests/data/raw_datasets/varnum.input format:id\tlabel\tnums
- tests/data/pairwise_datasets/simple_pair.input
- tests/data/seq_datasets/simple_seq.input


## TODO
--------
- 1
- 2
- 3

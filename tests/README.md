

设计模块
- 对应不同的数据组织的通用的datasets的封装 
- 针对不同开源数据集的python API获取接口以及文档
- 针对任务需求语义的自动datasets生成建议

```
tests/data/
├── dicts
│   └── query.vocab  //format: word,id
├── pairwise_datasets
│   └── simple_pair.input  // format: id,qid,queue_info,item_info,score
├── raw_datasets
│   └── query_float.input  // format: id,words_seqs,
└── seq_datasets
    └── simple_seq.input   // format: 
```

```
├── test_data_processor.py # 测试 data_processor
├── test_pt_datasets.py         # 测试PTDataset
├── test_textline_parser.py  # 测试textline解析器
├── test_tf_datasets.py          # 测试TFDataset
└── test_tokendicts.py           # 测试tokendicts
```
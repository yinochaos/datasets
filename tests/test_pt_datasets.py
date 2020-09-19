#!/usr/bin/env python

"""Tests for `datasets` package.
    ref : https://docs.python.org/zh-cn/3/library/unittest.html
"""

import unittest
import torch
from datasets import TextlineParser
from datasets import PTDataset
from datasets.utils import TokenDicts, DataSchema


class TestDatasets(unittest.TestCase):
    """Tests for `datasets` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_raw_query_float_dataset(self):
        # init token_dicts
        token_dicts = TokenDicts('tests/data/dicts', {'query': 0})
        data_field_list = []
        # param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        data_field_list.append(DataSchema(name='query', processor='to_tokenid', dtype='int32',
                                          shape=(None,), is_with_len=True, token_dict_name='query'))
        data_field_list.append(DataSchema(name='width', processor='to_np', dtype='float32', shape=(4)))
        label_field = DataSchema(name='label', processor='to_np', dtype='float32', shape=(1,))
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        dataset = PTDataset(parser=parser, file_path='tests/data/raw_datasets', file_suffix='query_float.input')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        for _ in enumerate(dataloader):
            pass

    """
    def test_raw_dataset_varnum(self):
        token_dicts = None
        data_field_list = []
        data_field_list.append(DataSchema(name='query', processor='to_np', type=tf.int32,
                                          dtype='int32', shape=(None,), is_with_len=True))
        label_field = DataSchema(name='label', processor='to_np',
                                 type=tf.float32, dtype='float32', shape=(1,), is_with_len=False)
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path='tests/data/raw_datasets', file_suffix='varnum.input')
        self.pass_allway_dataset(generator, 4)

    def test_text_seq2seq_model(self):
        # init token_dicts
        token_dicts = TokenDicts('tests/data/dicts', {'query': 0})
        data_field_list = []
        # param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        data_field_list.append(DataSchema(name='query', processor='to_tokenid', type=tf.int32,
                                          dtype='int32', shape=(None,), is_with_len=False, token_dict_name='query'))
        label_field = DataSchema(
            name='label', processor='to_tokenid', type=tf.int32, dtype='int32', shape=(None,), is_with_len=False, token_dict_name='query')
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path='tests/data/raw_datasets', file_suffix='text_seq2seq.input')
        dataset = generator.generate_dataset(
            batch_size=12, num_epochs=1, is_shuffle=False)
        for (batchs, (inputs, targets)) in enumerate(dataset):
            print('bacths', batchs, 'inputs', inputs, 'targets', targets)

    def test_tfrecord_dataset_varnum_writer_and_reader(self):
        token_dicts = None
        data_field_list = []
        data_field_list.append(DataSchema(name='query', processor='to_np', type=tf.int32,
                                          dtype='int32', shape=(None,), is_with_len=True))
        label_field = DataSchema(name='label', processor='to_np',
                                 type=tf.float32, dtype='float32', shape=(1,), is_with_len=False)
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path='tests/data/raw_datasets', file_suffix='varnum.input')
        generator.to_tfrecords('outputs/file.tfrecord')
        generator = TFDataset(parser=parser, file_path='outputs', file_suffix='.tfrecord', file_system='tfrecord')
        dataset = generator.generate_dataset(batch_size=1, num_epochs=1)
        for d in dataset.take(4):
            print(d)
    """


if __name__ == '__main__':
    unittest.main()

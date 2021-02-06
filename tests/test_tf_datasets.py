#!/usr/bin/env python

"""Tests for `datasets` package.
    ref : https://docs.python.org/zh-cn/3/library/unittest.html
"""

import unittest
import os
import tensorflow as tf

from datasets import TextlineParser, SeqParser
from datasets import TFDataset
from datasets.utils import TokenDicts, DataSchema


class TestDatasets(unittest.TestCase):
    """Tests for `datasets` package."""

    def setUp(self):
        if tf.__version__.startswith('1.'):
            tf.compat.v1.enable_eager_execution()
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def debug_tfrecord_format(self, tfrecord_file):
        filenames = [tfrecord_file]
        raw_dataset = tf.data.TFRecordDataset(filenames)
        for raw_record in raw_dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            print(example)

    def pass_allway_dataset(self, generator, batch_size=4):
        print('is_training = True ', 'weight_fn=1')

        def weight_fn(x):
            return 1
        generator.parser.set_weight_fn(weight_fn)
        dataset = generator.generate_dataset(batch_size, 1, is_training=True)
        self.pass_dataset(True, True, dataset)
        print('is_training = False ', 'No weight_fn')
        generator.parser.set_weight_fn(None)
        dataset = generator.generate_dataset(batch_size, 1, is_training=False)
        self.pass_dataset(False, False, dataset)
        print('is_training = True ', 'No weight_fn')
        dataset = generator.generate_dataset(batch_size, 1, is_training=True)
        self.pass_dataset(True, False, dataset)

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

    def test_session_parser_dataset(self):
        # init token_dicts
        token_dicts = TokenDicts('tests/data/dicts', {'query': 0})
        data_field_list = []
        # param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        data_field_list.append(DataSchema(name='query', processor='to_tokenid',
                                          dtype='int32', shape=(None, None,), is_with_len=False, token_dict_name='query', has_session=True))
        data_field_list.append(DataSchema(name='width', processor='to_np', dtype='float32', shape=(None, 4,), has_session=True))
        label_field = DataSchema(name='label', processor='to_np', dtype='float32', shape=(1,))
        parser = SeqParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path='tests/data/seq_datasets', file_suffix='simple_seq.input')
        print('Shapes', generator._get_shapes(is_training=True))
        dataset = generator.generate_dataset(
            batch_size=12, num_epochs=1, is_shuffle=False)
        for _ in enumerate(dataset):
            pass

    def test_raw_query_float_dataset(self):
        # init token_dicts
        token_dicts = TokenDicts('tests/data/dicts', {'query': 0})
        data_field_list = []
        # param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        data_field_list.append(DataSchema(name='query', processor='to_tokenid',
                                          dtype='int32', shape=(None,), is_with_len=True, token_dict_name='query'))
        data_field_list.append(DataSchema(name='width', processor='to_np', dtype='float32', shape=(4)))
        label_field = DataSchema(name='label', processor='to_np', dtype='float32', shape=(1,))
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path='tests/data/raw_datasets', file_suffix='query_float.input')
        dataset = generator.generate_dataset(
            batch_size=12, num_epochs=1, is_shuffle=False)
        for _ in enumerate(dataset):
            pass

    def test_raw_dataset_varnum(self):
        token_dicts = None
        data_field_list = []
        data_field_list.append(DataSchema(name='query', processor='to_np',
                                          dtype='int32', shape=(None,), is_with_len=True))
        label_field = DataSchema(name='label', processor='to_np', dtype='float32', shape=(1,), is_with_len=False)
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path='tests/data/raw_datasets', file_suffix='varnum.input')
        self.pass_allway_dataset(generator, 4)

    def test_text_seq2seq_model(self):
        # init token_dicts
        token_dicts = TokenDicts('tests/data/dicts', {'query': 0})
        data_field_list = []
        # param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        data_field_list.append(DataSchema(name='query', processor='to_tokenid',
                                          dtype='int32', shape=(None,), is_with_len=False, token_dict_name='query'))
        label_field = DataSchema(name='label', processor='to_tokenid', dtype='int32', shape=(None,), is_with_len=True, token_dict_name='query')
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path='tests/data/raw_datasets', file_suffix='text_seq2seq.input')
        dataset = generator.generate_dataset(batch_size=12, num_epochs=1, is_shuffle=False)
        for (batchs, (inputs, targets)) in enumerate(dataset):
            print('bacths', batchs, 'inputs', inputs, 'targets', targets)

    def test_tfrecord_dataset_varnum_writer_and_reader(self):
        token_dicts = None
        data_field_list = []
        data_field_list.append(DataSchema(name='query', processor='to_np', dtype='int32', shape=(None,), is_with_len=True))
        label_field = DataSchema(name='label', processor='to_np', dtype='float32', shape=(1,), is_with_len=False)
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path='tests/data/raw_datasets', file_suffix='varnum.input')
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        generator.to_tfrecords('outputs/file.tfrecord')
        generator = TFDataset(parser=parser, file_path='outputs', file_suffix='file.tfrecord', file_system='tfrecord')
        dataset = generator.generate_dataset(batch_size=1, num_epochs=1)
        for d in dataset.take(4):
            print(d)

        def weight(_):
            return 1.2
        parser = TextlineParser(token_dicts, data_field_list, label_field, weight_fn=weight)
        generator = TFDataset(parser=parser, file_path='tests/data/raw_datasets', file_suffix='varnum.input')
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        generator.to_tfrecords('outputs/file_weight.tfrecord')
        generator = TFDataset(parser=parser, file_path='outputs', file_suffix='file_weight.tfrecord', file_system='tfrecord')
        dataset = generator.generate_dataset(batch_size=1, num_epochs=1)
        for d in dataset.take(4):
            print(d)


if __name__ == '__main__':
    unittest.main()

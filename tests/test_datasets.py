#!/usr/bin/env python

"""Tests for `datasets` package."""

import unittest
import tensorflow as tf

from datasets.raw_dataset import RawDataset
from datasets.utils import TokenDicts, DataSchema

class TestDatasets(unittest.TestCase):
    """Tests for `datasets` package."""

    def setUp(self):
        if tf.__version__.startswith('1.'):
            tf.enable_eager_execution()
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def pass_allway_dataset(self, generator, batch_size=4):
        print('is_training = True ', 'weight_fn=1')
        weight_fn = lambda x:1
        generator.set_weight_fn(weight_fn)
        dataset = generator.generate_dataset(batch_size, 1, is_training=True)
        self.pass_dataset(True, True, dataset)
        print('is_training = False ', 'No weight_fn')
        generator.set_weight_fn(None)
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
                        print('d.shape',d.shape)
                    print('label.shape', label.shape)
                    print('batch_num', batch_num)
                    break
        else:
            if is_training:
                for batch_num, (x, label) in enumerate(dataset):
                    print('x', x)
                    for d in x:
                        print('d.shape',d.shape)
                    print('label.shape', label.shape)
                    print('batch_num', batch_num)
                    break
            else:
                for batch_num, (info, x, label) in enumerate(dataset):
                    print('info', info)
                    print('x', x)
                    for d in x:
                        print('d.shape',d.shape)
                    print('label.shape', label.shape)
                    print('batch_num', batch_num)
                    break

    def test_raw_query_float_dataset(self):
        """Test something."""
        # init token_dicts
        token_dicts = TokenDicts('tests/data/dicts', {'query':0})
        data_field_list = []
        #param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        #data_field_list.append(DataSchema(name='query', processor='to_np', type=tf.int32, dtype='int32', shape=(None,) ))
        #label_field = DataSchema(name='label', processor='to_np', type=tf.int32, dtype='int32', shape=() )
    #def __init__(self, file_path, token_dicts, data_field_list, label_field,  \
    #             file_system='local', file_suffix=None,  weight_fn=None):
        #generator = RawDataset(file_path='tests/data', token_dicts=token_dicts, data_field_list=data_field_list, label_field=label_field, file_suffix='.input')
        #dataset = generator.generate_dataset(batch_size=128, num_epochs=1, is_shuffle=False)
        #for _ in enumerate(dataset):
        #    pass

    def test_raw_dataset_varnum(self):
        token_dicts = None
        data_filed_list = []
        data_filed_list.append(DataSchema('query', 'to_np',tf.int32, 'int32', (None,), None, True))
        label_field = DataSchema('label','to_np',tf.float32, 'float32', (1,), None, False)

        generator = RawDataset(file_path="tests/data/raw_datasets", token_dicts=token_dicts, data_field_list=data_filed_list, label_field=label_field, file_suffix='varnum.input')
        self.pass_allway_dataset(generator, 4)


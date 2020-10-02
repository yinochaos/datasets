#!/usr/bin/env python

"""Tests for `datasets` package.
    ref : https://docs.python.org/zh-cn/3/library/unittest.html
    ref : assert断言 https://www.jianshu.com/p/eea0b0e432da
"""

import unittest
import numpy as np
import tensorflow as tf

from datasets.utils import DataSchema
from datasets import TextlineParser


class TestDatasets(unittest.TestCase):
    """Tests for `datasets` package."""

    def setUp(self):
        if tf.__version__.startswith('1.'):
            tf.compat.v1.enable_eager_execution()
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_simple_no_label_parser(self):
        # init token_dicts
        token_dicts = None
        feature_field_list = []
        # param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        feature_field_list.append(DataSchema(name='query', processor='to_np',
                                             dtype='int32', shape=(None,), is_with_len=True))
        parser = TextlineParser(token_dicts, feature_field_list)
        line = '12\t1 2 3 4 5'
        label, addinfo, features, _ = parser.parse(line)
        assert label is None
        self.assertListEqual(addinfo, ['12'])
        np.testing.assert_array_equal(features[0], np.asarray([1, 2, 3, 4, 5]))

    def test_simple_label_parser(self):
        # init token_dicts
        token_dicts = None
        feature_field_list = []
        # param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        feature_field_list.append(DataSchema(name='query', processor='to_np',
                                             dtype='int32', shape=(None,), is_with_len=False))
        label_field = DataSchema(name='label', processor='to_np', dtype='float32', shape=(1,))
        parser = TextlineParser(token_dicts, feature_field_list, label_field)
        line = '12\t2\t1 2 3 4 5'
        label, addinfo, features, _ = parser.parse(line)
        np.testing.assert_array_equal(label, np.asarray([2.0]))
        self.assertListEqual(addinfo, ['12'])
        np.testing.assert_array_equal(features, np.asarray([1, 2, 3, 4, 5]))

    def test_multi_feat_single_label_parser(self):
        # init token_dicts
        token_dicts = None
        feature_field_list = []
        # param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        feature_field_list.append(DataSchema(name='query', processor='to_np', dtype='int32', shape=(None,), is_with_len=True))
        feature_field_list.append(DataSchema(name='width', processor='to_np', dtype='int32', shape=(4)))
        label_field = DataSchema(name='label', processor='to_np', dtype='float32', shape=(1,))
        parser = TextlineParser(token_dicts, feature_field_list, label_field)
        line = '12\t2\t1 2 3 4 5\t1 2 3 4'
        label, addinfo, features, _ = parser.parse(line)
        # print('label',label, features)
        np.testing.assert_array_equal(label, np.asarray([2.0]))
        self.assertListEqual(addinfo, ['12'])
        np.testing.assert_array_equal(features[0], np.asarray([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(features[2], np.asarray([1, 2, 3, 4]))

    def test_multi_feat_multi_label_parser(self):
        # init token_dicts
        token_dicts = None
        feature_field_list = []
        # param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        feature_field_list.append(DataSchema(name='query', processor='to_np', dtype='int32', shape=(None,), is_with_len=True))
        feature_field_list.append(DataSchema(
            name='width', processor='to_np', dtype='int32', shape=(4)))
        label_field = [DataSchema(name='label1', processor='to_np', dtype='float32', shape=(1,)),
                       DataSchema(name='label2', processor='to_np', dtype='int32', shape=(None,), is_with_len=True)]

        parser = TextlineParser(token_dicts, feature_field_list, label_field, additive_schema=['id', 'session'])
        line = '12\tcuid\t2\t2 3 4\t1 2 3 4 5\t1 2 3 4'
        label, addinfo, features, _ = parser.parse(line)
        # print('label',label, features)
        assert len(label) == 3
        np.testing.assert_array_equal(label[0], np.asarray([2.0]))
        np.testing.assert_array_equal(label[1], np.asarray([2, 3, 4]))
        self.assertListEqual(addinfo, ['12', 'cuid'])
        assert len(features) == 3
        np.testing.assert_array_equal(features[0], np.asarray([1, 2, 3, 4, 5]))
        assert features[1] == 5
        np.testing.assert_array_equal(features[2], np.asarray([1, 2, 3, 4]))

    def test_multi_feat_none_multi_label_parser(self):
        # init token_dicts
        token_dicts = None
        feature_field_list = []
        # param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        feature_field_list.append(DataSchema(name='query', processor='to_np', dtype='int32', shape=(None,), is_with_len=True))
        feature_field_list.append(DataSchema(name='width', processor=None, dtype='int32', shape=(4)))
        label_field = [DataSchema(name='label1', processor='to_np', dtype='float32', shape=(1,)),
                       DataSchema(name='label2', processor='to_np', dtype='int32', shape=(None,), is_with_len=True)]

        parser = TextlineParser(token_dicts, feature_field_list, label_field, additive_schema=['id', 'session'])
        line = '12\tcuid\t2\t2 3 4\t1 2 3 4 5\t1 2 3 4'
        label, addinfo, features, _ = parser.parse(line)
        # print('label',label, features)
        assert len(label) == 3
        np.testing.assert_array_equal(label[0], np.asarray([2.0]))
        np.testing.assert_array_equal(label[1], np.asarray([2, 3, 4]))
        self.assertListEqual(addinfo, ['12', 'cuid'])
        assert len(features) == 2
        np.testing.assert_array_equal(features[0], np.asarray([1, 2, 3, 4, 5]))
        assert features[1] == 5

    def test_multi_feat_multi_label_none_parser(self):
        # init token_dicts
        token_dicts = None
        feature_field_list = []
        # param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        feature_field_list.append(DataSchema(name='query', processor='to_np', dtype='int32', shape=(None,), is_with_len=True))
        feature_field_list.append(DataSchema(name='width', processor='to_np', dtype='int32', shape=(4)))
        label_field = [DataSchema(name='label1', processor='to_np', dtype='float32', shape=(1,)),
                       DataSchema(name='label2', processor=None, dtype='int32', shape=(None,), is_with_len=True)]

        parser = TextlineParser(token_dicts, feature_field_list, label_field, additive_schema=['id', 'session'])
        line = '12\tcuid\t2\t2 3 4\t1 2 3 4 5\t1 2 3 4'
        label, addinfo, features, _ = parser.parse(line)
        # print('label',label, features)
        assert len(label) == 1
        np.testing.assert_array_equal(label, np.asarray([2.0]))
        self.assertListEqual(addinfo, ['12', 'cuid'])
        np.testing.assert_array_equal(features[0], np.asarray([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(features[2], np.asarray([1, 2, 3, 4]))

if __name__ == '__main__':
    unittest.main()

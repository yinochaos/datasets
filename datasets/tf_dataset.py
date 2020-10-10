#!/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2020 yinochaos <pspcxl@163.com>. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""raw multi-inputs dataset generator class
"""

from __future__ import absolute_import, division, print_function
import os
import subprocess
import tensorflow as tf
import codecs
import pickle

from datasets.utils.common_struct import data_schemas2shapes, data_schemas2types, type2tf_dict
from datasets.utils.common_struct import load_hdfs_filelist, load_local_filelist


class TFDataset(object):
    """从文件流创建的dataset
        利用tf.dataset支持多文件输入（本地和HDFS同时支持），对于大规模数据集十分友好；
        并通过token_dicts和datafields支持配置化的数据处理，灵活支持多输入数据集的处理
        text（local ， hdfs）：当前已经支持
        @TODO : 支持多种文件读取方式
        pickle：finish TODO test
        qiuckle:
        tfrecord：code&test finish
        LMDB（Lightning Memory-Mapped Database(快如闪电的内存映射数据库)）：TODO
        HDF5：TODO
    """
    ##
    # @brief
    #
    # @param file_path dataset文件路径
    # @param token_dicts 分词id词典
    # @param data_field_list dataset的schema数据
    # @param label_field label的schema数据
    # @param is_with_label dataset是否带有label
    # @param file_system
    #               local  : 从本地读取文件
    #               HDFS :  从HDFS读取文件
    # @param file_suffix dataset文件后缀,如果是None则使用所有文件
    # @param weight_fn 计算样本权重的函数 weight_fn(line)
    #
    # @return

    def __init__(self, parser, file_path, file_system='local', file_suffix=None, filenames=None):
        self.file_system = file_system
        self.parser = parser
        self.check_schemalist(parser.feature_field + parser.label_field)
        if self.file_system == 'HDFS':
            self.hadoop = os.environ.get('HADOOP')
            if self.hadoop is None:
                raise RuntimeError(
                    'os.environ[HADOOP] is None, should export HADOOP')
        if filenames is None:
            self.filenames = self.load_filelist(file_path, file_suffix, file_system)
            print('file names', self.filenames)
        else:
            self.filenames = filenames
        self.file_suffix = file_suffix

        if  len(data_schemas2types(self.parser.label_field)) > 1:
            self.is_multi_label = True
        else:
            self.is_multi_label = False

        if  len(data_schemas2types(self.parser.feature_field)) > 1:
            self.is_multi_features = True
        else:
            self.is_multi_features = False

    def check_schemalist(self, schema_list):
        for schema in schema_list:
            if schema.dtype not in type2tf_dict:
                raise ValueError("schema  %s :%s is not valid" % (schema.name, schema.dtype))

    ##
    # @brief
    #   载入目录所有模型输入文件
    # @param datadir 数据目录
    # @param file_suffix 文件后缀
    # @param file_system
    #
    # @return filenames 所有文件列表

    def load_filelist(self, datadir, file_suffix, file_system):
        filenames = []
        if file_system == 'HDFS':
            filenames = load_hdfs_filelist(datadir, file_suffix, self.hadoop)
        else:
            filenames = load_local_filelist(datadir, file_suffix)
        return filenames

    ##
    # @brief
    #       file的读取迭代器，支持本地文件和HDFS文件的读取
    #       TODO ： 可以考虑添加其他网络文件的读取支持
    #    pickle：
    #    tfrecord：
    #    LMDB（Lightning Memory-Mapped Database(快如闪电的内存映射数据库)）：
    #    HDF5：
    # @param filename
    #
    # @return line
    def read_input_lines(self, filename):
        if self.file_system == 'HDFS':
            cmd = self.hadoop + ' fs -cat ' + filename
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True) as pipe:
                for line in iter(pipe.stdout.readline, b''):
                    yield line.decode('utf8', 'ignore')
        elif self.file_system == 'local':
            with codecs.open(filename, encoding='utf8') as f:
                for line in f:
                    yield line
        elif self.file_system == 'pickle':
            with open(filename, 'rb') as f:
                while True:
                    try:
                        ex = pickle.load(f)
                        if isinstance(ex, list):
                            yield from ex
                        else:
                            yield ex
                    except EOFError:
                        break

    ##
    # @brief
    #       file的读取迭代器，支持本地文件和HDFS文件的读取
    #
    # @return dataset output

    def data_generator(self):
        """
        data_generator
        Returns:
            batch
        """
        for filename in self.filenames:
            for line in self.read_input_lines(filename):
                result = self.parser.parse(line)
                if result is None:
                    continue
                label, additive_info, features, _ = result
                if self.is_multi_label:
                    label = tuple(label)
                # output datas
                if self.parser.label_field is not None:
                    yield additive_info, tuple(features), label
                else:
                    yield additive_info, tuple(features)

    ##
    # @brief
    #       file的读取迭代器，支持本地文件和HDFS文件的读取
    #
    # @return dataset output
    def data_generator_training(self):
        """
        data_generator
        Returns:
            batch
        """
        for filename in self.filenames:
            for line in self.read_input_lines(filename):
                label, _, features, weight = self.parser.parse(line)
                if self.is_multi_label:
                    label = tuple(label)
                # output datas
                if weight is not None:
                    # 用于sample_weights的dataset: (inputs, targets, sample_weights)
                    if weight > 0:
                        yield tuple(features), label, weight
                else:
                    yield tuple(features), label

    ##
    # @brief
    #       file的读取迭代器，支持tfrecord
    #
    # @return dataset output

    def data_generator_tfrecord_flat(self):
        """
        data_generator
        Returns:
            batch
        """
        for filename in self.filenames:
            for line in self.read_input_lines(filename):
                label, _, features, weight = self.parser.parse(line)
                # output datas
                if weight is not None:
                    if weight > 0:
                        yield tuple(features + [label, weight])
                else:
                    yield tuple(features + [label])

    def _get_types(self, is_training, is_flat=False):
        types = []
        feature_types = data_schemas2types(self.parser.feature_field)
        label_types = None
        if self.parser.has_label:
            label_types = data_schemas2types(self.parser.label_field)
        if is_flat:
            types.extend(feature_types)
            if self.parser.has_label:
                types.extend(label_types)
            if self.parser.weight_fn is not None:
                types.append(tf.float32)
            return tuple(types)

        if len(feature_types) > 1:
            feature_types = tuple(feature_types)
        else:
            feature_types = feature_types[0]
        if len(label_types) > 1:
            label_types = tuple(label_types)
        else:
            label_types = label_types[0]
        if is_training:
            if self.parser.weight_fn is not None:
                types = tuple([feature_types, label_types, tf.float32])
            else:
                types = tuple([feature_types, label_types])
        else:
            if self.parser.has_label:
                types = tuple([tf.string, feature_types, label_types])
            else:
                types = tuple([tf.string, feature_types])
        return types

    def _get_shapes(self, is_training, is_flat=False):
        shapes = []
        feature_shapes = data_schemas2shapes(self.parser.feature_field)
        if self.parser.has_label:
            label_shapes = data_schemas2shapes(self.parser.label_field)
        else:
            label_shapes = None
        if is_flat:
            shapes.extend(feature_shapes)
            if self.parser.has_label:
                shapes.extend(label_shapes)
            if self.parser.weight_fn is not None:
                shapes.append(())
            return tuple(shapes)

        if len(feature_shapes) > 1:
            feature_shapes = tuple(feature_shapes)
        else:
            feature_shapes = feature_shapes[0]
        if len(label_shapes) > 1:
            label_shapes = tuple(label_shapes)
        else:
            label_shapes = label_shapes[0]
        if is_training:
            if self.parser.weight_fn is not None:
                shapes = tuple([feature_shapes, label_shapes, ()])
            else:
                shapes = tuple([feature_shapes, label_shapes])
        else:
            if self.parser.has_label:
                shapes = tuple([(len(self.parser.additive_schema)), feature_shapes, label_shapes])
            else:
                shapes = tuple([len(self.parser.additive_schema), feature_shapes])
        return shapes

    def _from_tfrecord(self):
        feature_description = {}
        for name in self.parser.features_dict:
            feature_description[name] = tf.io.FixedLenFeature([], tf.string)
            if self.parser.features_dict[name].is_with_len:
                feature_description[name + '_len'] = tf.io.FixedLenFeature([], tf.string)
        for name in self.parser.label_dict:
            feature_description[name] = tf.io.FixedLenFeature([], tf.string)
            if self.parser.label_dict[name].is_with_len:
                feature_description[name + '_len'] = tf.io.FixedLenFeature([], tf.string)
        if self.parser.weight_fn is not None:
            feature_description['weight'] = tf.io.FixedLenFeature([], tf.string)
        print('feature_des', feature_description)

        def _parse_function(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            data = tf.io.parse_single_example(example_proto, feature_description)
            features = []
            for field in self.parser.features_dict.values():
                features.append(tf.io.parse_tensor(data[field.name], type2tf_dict[field.dtype]))
                if field.is_with_len:
                    features.append(tf.io.parse_tensor(data[field.name + '_len'], tf.int32))
            if self.parser.has_label:
                labels = []
                for field in self.parser.label_dict.values():
                    labels.append(tf.io.parse_tensor(data[field.name], type2tf_dict[field.dtype]))
                    if field.is_with_len:
                        labels.append(tf.io.parse_tensor(data[field.name + '_len'], tf.int32))
                return tuple(features), tuple(labels)
            else:
                return tuple(features)
        print('files', self.filenames)
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(_parse_function)
        return dataset

    ##
    # @brief
    #       用from_generator的方式生成dataset
    # @param batch_size
    # @param num_epochs
    # @param is_shuffle
    # @param buffer_size
    #
    # @return dataset
    def generate_dataset(self, batch_size=128, num_epochs=1, is_training=True, is_shuffle=False, buffer_size=None):
        """
        生成dataset
        Args:
            batch_size:
            num_epochs:

        Returns:
            dataset
        """
        if self.file_system == 'tfrecord':
            dataset = self._from_tfrecord()
        else:
            if is_training:
                dataset = tf.data.Dataset.from_generator(self.data_generator_training,
                                                         output_shapes=self._get_shapes(is_training),
                                                         output_types=self._get_types(is_training))
            else:
                dataset = tf.data.Dataset.from_generator(self.data_generator,
                                                         output_shapes=self._get_shapes(is_training),
                                                         output_types=self._get_types(is_training))
        if is_shuffle:
            if not buffer_size:
                buffer_size = batch_size * 1000
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat(num_epochs)
        if self.parser.is_data_padding:
            padded_shapes = self._get_shapes(is_training)
            dataset = dataset.padded_batch(
                batch_size, padded_shapes=padded_shapes)
        else:
            dataset = dataset.batch(batch_size)
        return dataset

    def serialize_example(self, *features):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        def _encode_tensor(t):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(t).numpy()]))
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
        feature = {}
        for f, name in zip(features, self.parser.flat_feature_names):
            feature[name] = _encode_tensor(f)
        # Create a Features message using tf.train.Example
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def to_tfrecords(self, output_path, num=1, num_parallel_calls=8):
        def reduce_func(key, dataset):
            filename = tf.strings.join(
                [output_path, tf.strings.as_string(key)])
            writer = tf.data.experimental.TFRecordWriter(filename)
            writer.write(dataset.map(lambda _, x: x, num_parallel_calls=num_parallel_calls))
            return tf.data.Dataset.from_tensors(filename)

        def tf_serialize_example(*features):
            tf_string = tf.py_function(
                self.serialize_example,
                (features),  # pass these args to the above function.
                tf.string)      # the return type is `tf.string`.
            return tf.reshape(tf_string, ())  # The result is a scalar
        print('types', self._get_types(True, True))
        print('shapes', self._get_shapes(True, True))
        dataset = tf.data.Dataset.from_generator(self.data_generator_tfrecord_flat,
                                                 output_shapes=self._get_shapes(True, True),
                                                 output_types=self._get_types(True, True))
        dataset = dataset.map(tf_serialize_example)
        if num == 1:
            writer = tf.data.experimental.TFRecordWriter(output_path)
            writer.write(dataset)
        elif num > 1:
            dataset = dataset.enumerate()
            dataset = dataset.apply(tf.data.experimental.group_by_window(lambda i, _: i % num, reduce_func, tf.int64.max))
            for _ in enumerate(dataset):
                pass

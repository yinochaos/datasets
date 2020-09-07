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
#from collections import namedtuple
import logging

from datasets.utils import TokenDicts, DataSchema, data_processor_dicts

class Dataset(object):
    """从文件流创建的dataset
        利用tf.dataset支持多文件输入（本地和HDFS同时支持），对于大规模数据集十分友好；
        并通过token_dicts和datafields支持配置化的数据处理，灵活支持多输入数据集的处理
        text（local ， hdfs）：当前已经支持
        @TODO : 支持多种文件读取方式
        pickle：finish TODO test
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

    def __init__(self, file_path, token_dicts, data_field_list, label_field,
                 file_system='local', file_suffix=None, weight_fn=None, additive_schema=['id']):
        self.file_system = file_system
        if self.file_system == 'HDFS':
            self.hadoop = os.environ.get('HADOOP')
            if self.hadoop is None:
                raise RuntimeError(
                    'os.environ[HADOOP] is None, should export HADOOP')
        self.filenames = self.load_filelist(
            file_path, file_suffix, file_system)
        self.token_dicts = token_dicts
        self.data_field_list = data_field_list
        self.is_batch_padding = False
        for field in self.data_field_list:
            if self.get_batch_padding(field.shape):
                self.is_batch_padding = True
                break
        self.label_field = label_field
        self.is_with_label = label_field is not None
        self.file_suffix = file_suffix
        self.weight_fn = weight_fn
        self.additive_schema = additive_schema
        self.variable_shape_index_list = [
            self.get_variable_shape_index(x.shape) for x in data_field_list]
        print('variable_shape_index_list', self.variable_shape_index_list)
        self.flat_features = []
        for x in self.data_field_list:
            if x.processor is not None:
                self.flat_features.append(x)
                if x.is_with_len:
                    self.flat_features.append(DataSchema(
                        name=x.name+'len', type=tf.int32))
        if self.is_with_label:
            self.flat_features.append(label_field)
        if not self.is_batch_padding:
            for i in range(len(self.variable_shape_index_list)):
                if self.variable_shape_index_list[i] is None:
                    raise Exception("shape index", i,
                                    self.data_field_list[i].shape, "not valid")

    # 返回可变维度的下标的列表，用于生成可变维度的长度数据使用
    def get_variable_shape_index(self, shape):
        num = []
        if isinstance(shape, (tuple, list)):
            for i in range(len(shape)):
                if shape[i] is None:
                    num.append(i)
                elif not isinstance(shape[i], int):
                    return None
        else:
            if shape is None:
                return [0]
        return num

    def get_batch_padding(self, shape):
        if isinstance(shape, (tuple, list)):
            for sub_shape in shape:
                if self.get_batch_padding(sub_shape):
                    return True
        else:
            return shape is None
    ##
    # @brief
    #   载入目录所有模型输入文件
    # @param datadir 数据目录
    # @param file_suffix 文件后缀
    # @param file_system
    #
    # @return filenames 所有文件列表

    def load_filelist(self, datadir, file_suffix, file_system):
        if file_system == 'HDFS':
            filenames = []
            cmd = self.hadoop + ' fs -ls ' + datadir
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True) as pipe:
                for line in iter(pipe.stdout.readline, b''):
                    items = line.decode(
                        'utf8', 'ignore').strip('\n').split(' ')
                    if len(items) > 3:
                        if file_suffix is None or items[-1].split('/')[-1].endswith(file_suffix):
                            filenames.append(items[-1])
        elif file_system == 'local':
            filenames = os.listdir(datadir)
            filenames = map(lambda x: os.path.join(datadir, x), filenames)
            if file_suffix is None:
                filenames = [f for f in list(filenames)]
            else:
                filenames = [f for f in list(
                    filenames) if f.endswith(file_suffix)]
        else:
            pass
        return filenames

    ##
    # @brief
    #    根据data_field_list的token_dicts和processor来解析一行数据
    #    数据格式 : (label\t)sample_id\tadditive_info\tdata_field1\tdata_field2\tdata_field3\t...
    #    TODO : 增加对于sample_id的支持
    #    TODO : 增加对于sanple_weight的支持
    #    A tf.data dataset. Should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights)
    #
    # @param line
    #
    # @return parser results
    def _parse_fn(self, line):
        raise NotImplementedError('_parse not implement')

    ##
    # @brief
    #       根据label_field.processor来处理label的解析
    # @param label_text
    #
    # @return 解析结果
    def _parse_label(self, label_text):
        return data_processor_dicts[self.label_field.processor](label_text, self.label_field, self.token_dicts)

    ##
    # @brief
    #       file的读取迭代器，支持本地文件和HDFS文件的读取
    #       TODO ： 可以考虑添加其他网络文件的读取支持
    #    @TODO : 支持多种文件读取方式
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
            for line in codecs.open(filename, encoding='utf8'):
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

    def set_weight_fn(self, weigh_fn):
        self.weight_fn = weigh_fn
    ##
    # @brief
    #       file的读取迭代器，支持本地文件和HDFS文件的读取
    #       TODO ： 可以考虑添加其他网络文件的读取支持
    #
    # @return dataset output

    def data_generator(self):
        """
        data_generator
        Returns:
            batch
        """
        for filename in self.filenames:
            logging.info("deal file %s" % (filename))
            for line in self.read_input_lines(filename):
                label, additive_info, features = self._parse_fn(line)
                # output datas
                if self.is_with_label:
                    yield additive_info, tuple(features), label
                else:
                    yield additive_info, tuple(features)

    ##
    # @brief
    #       file的读取迭代器，支持本地文件和HDFS文件的读取
    #       TODO ： 可以考虑添加其他网络文件的读取支持
    #
    # @return dataset output
    def data_generator_training(self):
        """
        data_generator
        Returns:
            batch
        """
        for filename in self.filenames:
            logging.info("deal file %s" % (filename))
            for line in self.read_input_lines(filename):
                label, _, features = self._parse_fn(line)
                # output datas
                if self.weight_fn is not None:
                    # 用于sample_weights的dataset: (inputs, targets, sample_weights)
                    weight = self.weight_fn(line)
                    if weight > 0:
                        yield tuple(features), label, weight
                else:
                    yield tuple(features), label

    ##
    # @brief
    #       file的读取迭代器，支持本地文件和HDFS文件的读取
    #       TODO ： 可以考虑添加其他网络文件的读取支持
    #
    # @return dataset output

    def data_generator_tfrecord_flat(self):
        """
        data_generator
        Returns:
            batch
        """
        for filename in self.filenames:
            logging.info("deal file %s" % (filename))
            for line in self.read_input_lines(filename):
                label, _, features = self._parse_fn(line)
                # output datas
                if self.weight_fn is not None:
                    weight = self.weight_fn(line)
                    if weight > 0:
                        yield tuple(features+[label, weight])
                else:
                    yield tuple(features + [label])

    def _get_shapes(self, is_training, is_flat=False):
        shapes = []
        for field, var_list in zip(self.data_field_list, self.variable_shape_index_list):
            if field.processor is None:
                continue
            shapes.append(field.shape)
            if field.is_with_len:
                if len(var_list) == 1:
                    shapes.append(())
                else:
                    shapes.append((len(var_list)))
        if is_flat:
            if self.is_with_label:
                shapes.append(self.label_field.shape)
            if self.weight_fn is not None:
                shapes.append(())
            return tuple(shapes)

        shapes = tuple(shapes)
        if is_training:
            if self.weight_fn is None:
                shapes = tuple([shapes, self.label_field.shape])
            else:
                shapes = tuple([shapes, self.label_field.shape, ()])
        else:
            if self.is_with_label:
                shapes = tuple([(len(self.additive_schema)),
                                shapes, self.label_field.shape])
            else:
                shapes = tuple([len(self.additive_schema), shapes])
        return shapes

    def _get_types(self, is_training, is_flat=False):
        types = []
        for field in self.data_field_list:
            if field.processor is None:
                continue
            types.append(field.type)
            if field.is_with_len:
                types.append(tf.int32)
        if is_flat:
            if self.is_with_label:
                types.append(self.label_field.type)
            if self.weight_fn is not None:
                types.append(tf.float32)
            return tuple(types)
        types = tuple(types)
        if is_training:
            if self.weight_fn is None:
                types = tuple([types, self.label_field.type])
            else:
                types = tuple([types, self.label_field.type, tf.float32])
        else:
            if self.is_with_label:
                types = tuple([tf.string, types, self.label_field.type])
            else:
                types = tuple([tf.string, types])
        return types

    def generate_dataset_from_tfrecord(self, batch_size=128, num_epochs=1, is_training=True, is_shuffle=False, buffer_size=None):
        feature_description = {}
        for field in self.flat_features:
            feature_description[field.name] = tf.io.FixedLenFeature(
                [], tf.string)
        print('feature_des', feature_description)

        def _parse_function(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            data = tf.io.parse_single_example(
                example_proto, feature_description)
            features = []
            for field in self.flat_features:
                if not field.name.startswith('label'):
                    features.append(tf.io.parse_tensor(
                        data[field.name], field.type))
            return tuple(features), tf.io.parse_tensor(data['label'], self.flat_features[-1].type)

        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(_parse_function)

        if is_shuffle:
            if not buffer_size:
                buffer_size = batch_size * 1000
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat(num_epochs)
        if self.is_batch_padding:
            padded_shapes = self._get_shapes(is_training)
            dataset = dataset.padded_batch(
                batch_size, padded_shapes=padded_shapes)
        else:
            dataset = dataset.batch(batch_size)
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
        if is_training:
            dataset = tf.data.Dataset.from_generator(self.data_generator_training,
                                                     output_shapes=self._get_shapes(
                                                         is_training),
                                                     output_types=self._get_types(is_training))
        else:
            dataset = tf.data.Dataset.from_generator(self.data_generator,
                                                     output_shapes=self._get_shapes(
                                                         is_training),
                                                     output_types=self._get_types(is_training))
        if is_shuffle:
            if not buffer_size:
                buffer_size = batch_size * 1000
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat(num_epochs)
        if self.is_batch_padding:
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
        for f, field in zip(features, self.flat_features):
            feature[field.name] = _encode_tensor(f)
        # Create a Features message using tf.train.Example
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def to_tfrecords(self, output_path):
        def tf_serialize_example(*features):
            tf_string = tf.py_function(
                self.serialize_example,
                (features),  # pass these args to the above function.
                tf.string)      # the return type is `tf.string`.
            return tf.reshape(tf_string, ())  # The result is a scalar 
        #print('types', self._get_types(True,True), 'types', self._get_shapes(True,True))
        dataset = tf.data.Dataset.from_generator(self.data_generator_tfrecord_flat,
                                                 output_shapes=self._get_shapes(
                                                     True, True),
                                                 output_types=self._get_types(True, True))
        for d in dataset.take(2):
            print(d)
        dataset = dataset.map(tf_serialize_example)
        writer = tf.data.experimental.TFRecordWriter(output_path)
        writer.write(dataset)

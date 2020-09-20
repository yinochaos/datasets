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
import torch
import subprocess
import codecs
import pickle
# import quickle
# import lmdb

from datasets.utils.common_struct import load_hdfs_filelist, load_local_filelist


class PTDataset(torch.utils.data.IterableDataset):
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

    def __init__(self, parser, file_path, file_system='local', file_suffix=None, filenames=None, file_format='text', is_training=True):
        self.is_training = is_training
        self.file_system = file_system
        self.file_format = file_format
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
        self.parser = parser
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
        elif self.file_system == 'lmdb':
            pass
            """
            with lmdb.open(filename) as env:
                with env.begin() as txn:
                    cursor = txn.cursor()
                    for _, data in enumerate(cursor.iternext_dup()):
                        if isinstance(data, list):
                            for d in data:
                                yield d
                        else:
                            yield data
            """

    def __iter__(self):
        if self.file_format == 'text':
            for filename in self.filenames:
                for line in self.read_input_lines(filename):
                    label, add_info, features, weight = self.parser.parse(line)
                    if weight is not None and weight == 0:
                        continue
                    if self.is_training:
                        result = ((features, label) if weight is None else (features, label, weight))
                    else:
                        result = ((add_info, features) if label is None else (add_info, features, label))
                    yield result
        elif self.file_system == 'lmdb':
            for filename in self.filenames:
                for line in self.read_input_lines(filename):
                    pass

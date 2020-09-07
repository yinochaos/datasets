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
import sys
import os
import io
import tensorflow as tf
import numpy as np
import codecs
from collections import namedtuple
import logging

from datasets.utils import TokenDicts, DataSchema, data_processor_dicts
from datasets.dataset import Dataset


class RawDataset(Dataset):
    """从文件流创建的dataset
        利用tf.dataset支持多文件输入（本地和HDFS同时支持），对于大规模数据集十分友好；
        并通过token_dicts和datafields支持配置化的数据处理，灵活支持多输入数据集的处理
    """
    ##
    # @brief
    #    根据data_field_list的token_dicts和processor来解析一行数据
    #    数据格式 : additive_info(\tlabel)\tdata_field1\tdata_field2\tdata_field3\t...
    #    addtive_info e.g.      sample_id, type, .etc
    #    A tf.data dataset. Should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights)
    #
    # @param line
    #
    # @return parser results

    def _parse_fn(self, line):
        label = None
        additive_info = []
        additive_info_len = len(self.additive_schema)
        if additive_info_len > 0:
            items = line.strip('\n').split('\t', additive_info_len)
            for i in range(0, additive_info_len):
                additive_info.append(items[i])
            line = items[-1]
        if self.label_field is not None:
            label, rest_line = line.strip('\n').split('\t', 1)
            label = self._parse_label(label)
            line = rest_line
        data_fields = line.strip('\n').split('\t')
        data_num_fields = len(data_fields)
        if data_num_fields == 0:
            raise ValueError("data_fields with size 0")
        # 根据data_field_list的token_dicts和processor来解析一行数据，如processor为None,则丢弃该数据
        features = [data_processor_dicts[self.data_field_list[index].processor](
            data_fields[index], self.data_field_list[index], self.token_dicts) if self.data_field_list[index].processor is not None else None for index in range(0, data_num_fields)]
        if self.is_batch_padding:
            return_features = []
            for i in range(0, data_num_fields):
                if self.data_field_list[i].processor is None:
                    continue
                return_features.append(features[i])
                if self.data_field_list[i].is_with_len:
                    if len(self.variable_shape_index_list[i]) == 1:
                        return_features.append(
                            features[i].shape[self.variable_shape_index_list[i][0]])
                    elif len(self.variable_shape_index_list[i]) > 1:
                        return_features.append(np.asarray(
                            [features[i].shape[k] for k in self.variable_shape_index_list[i]], dtype='int32'))
            features = return_features
        else:
            features = [f for f in features if f is not None]
        # features = [data_processor_dicts[self.data_field_list[index].processor](data_fields[index],
        #                                                                   self.data_field_list[index],
        #                                                                   self.token_dicts)
        #            for index in range(0, data_num_fields) if self.data_field_list[index].processor is not None]
        return label, additive_info, features

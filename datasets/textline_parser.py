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
import numpy as np

from datasets.utils import data_processor_dicts
from datasets.parser import Parser


class TextlineParser(Parser):
    """ parser
    """
    # def __init__(self, data_field_list, label_field, additive_schema=['id']):
    #    self.

    def parse(self, line):
        label = None
        additive_info = []
        additive_info_len = len(self.additive_schema)
        if additive_info_len > 0:
            items = line.strip('\n').split('\t', additive_info_len)
            for i in range(0, additive_info_len):
                additive_info.append(items[i])
            line = items[-1]
        data_fields = line.strip('\n').split('\t')
        data_num_fields = len(data_fields)
        assert data_num_fields == len(self.data_field_list), 'line items:%d must equal fields num %d' % (data_num_fields, len(self.data_field_list))
        if data_num_fields == 0:
            raise ValueError("data_fields with size 0")
        # 根据data_field_list的token_dicts和processor来解析一行数据，如processor为None,则丢弃该数据
        datas = [data_processor_dicts[field.processor](text, field, self.token_dicts)
                 for text, field in zip(data_fields, self.data_field_list) if field.processor is not None]
        if self.is_data_padding:
            out_datas = []
            for feature, var_shape_index in zip(datas, self.var_shape_index_list):
                out_datas.append(feature)
                if var_shape_index is None:
                    continue
                if len(var_shape_index) == 1:
                    out_datas.append(
                        feature.shape[var_shape_index[0]])
                elif len(var_shape_index) > 1:
                    out_datas.append(np.asarray(
                        [feature.shape[k] for k in var_shape_index], dtype='int32'))
            datas = out_datas
        if self.label_range == 1:
            label = datas[0]
            features = datas[1:]
        elif self.label_range > 1:
            label = datas[:self.label_range]
            features = datas[self.label_range:]
        else:
            features = datas
        if len(features) == 1:
            features = features[0]
        if self.weight_fn is None:
            weight = None
        else:
            weight = self.weight_fn(line)
        return label, additive_info, features, weight

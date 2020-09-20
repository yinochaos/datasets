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
from datasets.utils.common_struct import get_variable_shape_index, is_batch_padding


class Parser(object):
    """ parser
    """

    def __init__(self, token_dicts, feature_field, label_field=None, additive_schema=['id'], weight_fn=None):
        self.token_dicts = token_dicts
        self.weight_fn = weight_fn
        assert feature_field is not None, 'feature_field must not none'
        if not isinstance(feature_field, list):
            feature_field = [feature_field]
        self.feature_field = feature_field
        if label_field is None:
            label_field = []
        elif not isinstance(label_field, list):
            label_field = [label_field]
        self.label_field = label_field
        self.additive_schema = additive_schema
        self.has_label = len(label_field) > 0
        self.data_field_list = []
        self.data_field_list.extend(label_field)
        self.data_field_list .extend(feature_field)
        # 储存边长shape的下标列表 e.g. [(12,None), (23,None,None)] 则 var_shape_index_list = [(1),(1,2)]
        self.var_shape_index_list = [get_variable_shape_index(
            x.shape) if x.is_with_len else None for x in self.data_field_list if x.processor is not None]
        self.label_range = 0
        if self.has_label:
            # print(label_field)
            for field in label_field:
                if field.processor is not None:
                    if field.is_with_len:
                        self.label_range += 2
                    else:
                        self.label_range += 1
                # print('range',self.label_range, field)
        self.flat_feature_names = []
        self.features_dict = {}
        for field in feature_field:
            if field.processor is not None:
                assert field.name != 'weight', 'weight is used, please change another field name'
                self.features_dict[field.name] = field
                self.flat_feature_names.append(field.name)
                if field.is_with_len:
                    self.flat_feature_names.append(field.name + '_len')
        self.label_dict = {}
        for field in label_field:
            if field.processor is not None:
                assert field.name != 'weight', 'weight is used, please change another field name'
                self.label_dict[field.name] = field
                self.flat_feature_names.append(field.name)
                if field.is_with_len:
                    self.flat_feature_names.append(field.name + '_len')
        if weight_fn is not None:
            self.flat_feature_names.append('weight')
        self.is_data_padding = self.is_padding(self.data_field_list)

    def set_weight_fn(self, weight_fn):
        self.weight_fn = weight_fn

    def is_padding(self, field_list):
        for field in field_list:
            if is_batch_padding(field.shape):
                return True
        return False

    def parse(self, line):
        raise NotImplementedError

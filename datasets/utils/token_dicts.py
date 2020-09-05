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
"""

"""
from __future__ import absolute_import, division, print_function
import sys
import os
import io
import codecs


class TokenDicts(object):
    """
        处理数据转换成embedding id的类，所有的padding zero的token都为0。例如：
        1）text to embedding id ，通过dict实现
        2）连续值(包括向量和标量)离散化 to embedding id,　通过SKLEARN的kmeans来实现
        token dict数据组织：
        dict_path/dict_name.vocab : for 1)
        dict_path/dict_name.pkl : for 2)
    """

    def __init__(self, dict_path, dicts_param, search_range=100):
        # dicts_param : {dict_name, min_freq}
        # min_freq > 1　则是直接设定min_freq;  <1 :则按照百分比决定min_freq
        # TODO : 统一 names_dicts min_freq_dicts到一个输入
        # TODO : 通过cutoff百分比来确定min_freq参数，即去除percent的freq占比，可以取0.05
        self.dicts = {}
        self.dict_size = {}
        self.unkown_emb_id = {}
        self.dict_min_freq = {}
        self.search_range = search_range
        for (dict_name, min_freq) in dicts_param.items():
            if isinstance(min_freq, int):
                self.dict_min_freq[dict_name] = float(min_freq)
            elif isinstance(min_freq, float):
                if min_freq >= 1.0:
                    self.dict_min_freq[dict_name] = min_freq
        for (dict_name, min_freq) in dicts_param.items():
            if min_freq == 'pkl':
                dict_file = dict_path + '/' + dict_name + '.pkl'
                self._vector_to_emb_id(dict_file, dict_name)
            elif isinstance(min_freq, int) or isinstance(min_freq, float):
                dict_file = dict_path + '/' + dict_name + '.vocab'
                if dict_name not in self.dict_min_freq:
                    self.dict_min_freq[dict_name] = self._compute_min_freq_by_percent(dict_file, min_freq)
                self._text_to_emb_id(dict_file, dict_name)

    def _compute_min_freq_by_percent(self, filename, percent):
        print('percent', percent)
        search_min_freqs = [0 for i in range(1, self.search_range)]
        sum_freqs = 0.0
        with codecs.open(filename, encoding='utf8') as f:
            for line in f:
                _, freq = line.strip('\n').split('\t')
                freq = float(freq)
                sum_freqs += freq
                for i in range(0, len(search_min_freqs)):
                    if freq > i+1:
                        search_min_freqs[i] += freq
        for i in range(0, len(search_min_freqs)):
            if search_min_freqs[i] < (1-percent) * sum_freqs:
                return i
        return self.search_range

    def _vector_to_emb_id(self, filename, dict_name):
        token_dict = {}
        token_id = 0
        for line in codecs.open(filename, encoding='utf8'):
            token, freq = line.strip('\n').split('\t')
            freq = int(freq)
            if freq >= self.dict_min_freq[dict_name]:
                token_id = token_id + 1
                token_dict[token] = token_id
        self.dicts[dict_name] = token_dict
        self.dict_size[dict_name] = token_id + 1
        self.unkown_emb_id[dict_name] = token_id + 1

    def _text_to_emb_id(self, filename, dict_name):
        token_dict = {}
        token_id = 0
        with codecs.open(filename, encoding='utf8') as f:
            for line in f:
                token, freq = line.strip('\n').split('\t')
                freq = float(freq)
                if freq >= self.dict_min_freq[dict_name]:
                    token_id = token_id + 1
                    token_dict[token] = token_id
        self.dicts[dict_name] = token_dict
        self.dict_size[dict_name] = token_id + 1
        self.unkown_emb_id[dict_name] = token_id + 1

    def dict_size_by_name(self, dict_name):
        """
        返回字典大小
        Args:
            dict_name: 名称

        Returns:
            int类型 dict大小
        """
        return self.dict_size[dict_name]

    def to_id(self, dict_name, token):
        """
        token to id
        Args:
            dict_name: 名称
            token: 分词

        Returns:
            int类型 token id
        """
        token_dict = self.dicts[dict_name]
        if token in token_dict:
            return token_dict[token]
        else:
            return self.unkown_emb_id[dict_name]

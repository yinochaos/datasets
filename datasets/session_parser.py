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
from datasets.utils.common_struct import to_array_pad
from datasets.textline_parser import TextlineParser


class SessionParser(TextlineParser):
    """ parser
    """

    def __init__(self, token_dicts, feature_field_list, label_field, additive_schema=['id', 'sid', 'time'], weight_fn=None, max_sessions=5):
        assert additive_schema[0] == 'id', 'additive_schema[0] must be id'
        assert additive_schema[1] == 'sid', 'additive_schema[1] must be sid'
        self.max_sessions = max_sessions
        super().__init__(token_dicts, feature_field_list, label_field, additive_schema, weight_fn)
        #self.session_field_list = [x for x in feature_field_list if x.has_session is True]
        self.feature_field_list = [x for x in feature_field_list if x.processor is not None]
        self.session_features = self.init_session()

    def init_session(self):
        session = [[] for field in self.feature_field_list]
        #session = [np.zeros(shape=field.shape, dtype=field.dtype) for field in self.feature_field_list if field.has_session is True]
        return session

    def update_session(self, features):
        for s in self.session_features:
            if len(s) == self.max_sessions:
                s.pop(0)
        #cur_append_features = [x for (x, f) in zip(features, self.feature_field_list)]
        cur_append_features = [x if f.has_session else None for (x, f) in zip(features, self.feature_field_list)]
        for s, x in zip(self.session_features, cur_append_features):
            if x is not None:
                s.append(x)
            else:
                s = None
        # self.session_features.append(cur_append_features)

    def parse(self, line):
        last_sid = None
        label, additive_info, features, weight = super().parse(line)
        sid = additive_info[1]
        if last_sid is not None and sid != last_sid:
            self.session_features = self.init_session()
        last_sid = sid
        #session_features = [to_array_pad(s + [f]) for s, f in zip(self.session_features, features)]
        session_features = [to_array_pad(s + [f]) if s else f for s, f in zip(self.session_features, features)]
        result = label, additive_info, session_features, weight
        self.update_session(features)
        return result

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
from collections import namedtuple

__all__ = ['DataSchema']

#
#shape : [12,10,12]
# max_len : [None, 10, None] means dim2 should be expand to 10 max_len [be the same dim with shapes]
param = ["name", "processor", "type", "dtype", "shape", "token_dict_name", "is_with_len", "max_len"]
DataSchema = namedtuple('DataSchema', field_names=param)
DataSchema.__new__.__defaults__ = tuple([None] * len(param))

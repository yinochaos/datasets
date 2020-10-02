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
import subprocess
import os
import tensorflow as tf

__all__ = ['DataSchema', 'load_local_filelist', 'type2tf_dict', 'load_hdfs_filelist', 'data_schemas2types',
           'data_schemas2shapes', 'is_batch_padding', 'get_variable_shape_index']

# shape : [12,10,12]
param = ["name", "processor", "dtype", "shape", "token_dict_name", "is_with_len", "max_len"]
DataSchema = namedtuple('DataSchema', field_names=param)
DataSchema.__new__.__defaults__ = tuple([None] * len(param))

type2tf_dict = {
    "int": tf.int32,
    "int32": tf.int32,
    "int64": tf.int64,
    "float": tf.float32,
    "float32": tf.float32,
    "float64": tf.float64
}


def load_hdfs_filelist(datadir, file_suffix, hadoop):
    filenames = []
    cmd = hadoop + ' fs -ls ' + datadir
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True) as pipe:
        for line in iter(pipe.stdout.readline, b''):
            items = line.decode(
                'utf8', 'ignore').strip('\n').split(' ')
            if len(items) > 3:
                if file_suffix is None or items[-1].split('/')[-1].endswith(file_suffix):
                    filenames.append(items[-1])
    return filenames


def load_local_filelist(datadir, file_suffix):
    filenames = os.listdir(datadir)
    filenames = map(lambda x: os.path.join(datadir, x), filenames)
    if file_suffix is None:
        filenames = [f for f in list(filenames)]
    else:
        filenames = [f for f in list(
            filenames) if f.endswith(file_suffix)]
    return filenames


def data_schemas2types(datafield_list):
    def data_schema2types(data_schema):
        types = type2tf_dict[data_schema.dtype]
        if data_schema.is_with_len:
            return [types, tf.int32]
        else:
            return types

    if isinstance(datafield_list, DataSchema):
        return data_schema2types(datafield_list)
    elif isinstance(datafield_list, list):
        types = []
        for field in datafield_list:
            if field.processor is None:
                continue
            type = data_schema2types(field)
            if isinstance(type, list):
                types.extend(type)
            else:
                types.append(type)
        return types


def data_schemas2shapes(datafield_list):
    def data_schema2shapes(data_schema):
        shapes = data_schema.shape
        if data_schema.is_with_len:
            var_list = get_variable_shape_index(field.shape)
            if len(var_list) == 1:
                return [shapes, ()]
            else:
                return [shapes, (len(var_list))]
        else:
            return shapes

    if isinstance(datafield_list, DataSchema):
        return data_schema2shapes(datafield_list)
    elif isinstance(datafield_list, list):
        shapes = []
        for field in datafield_list:
            if field.processor is None:
                continue
            shape = data_schema2shapes(field)
            if isinstance(shape, list):
                shapes.extend(shape)
            else:
                shapes.append(shape)
        return shapes

# 返回可变维度的下标的列表，用于生成可变维度的长度数据使用


def get_variable_shape_index(shape):
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

# 判断dataschemas是否存在变长情况


def is_batch_padding(shape):
    if isinstance(shape, (tuple, list)):
        for sub_shape in shape:
            if is_batch_padding(sub_shape):
                return True
    else:
        return shape is None

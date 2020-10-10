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
from __future__ import absolute_import, division, print_function
import numpy as np
import soundfile as sf
import python_speech_features

__all__ = ['data_processor_dicts']


def to_np(x, dtype):
    x_data = [i if i != 'None' else 0 for i in x.split(' ')]
    return np.asarray(x_data, dtype=dtype)


def to_tokenid(x, dict_name, token_dicts):
    x_data = [token_dicts.to_id(dict_name, token) for token in x.split(' ')]
    return np.asarray(x_data, dtype='int32')

# ref:https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
# @ TODO fix pad_width


def padding_seq(tokens, pad_width, pad_suffix=True, mode='constant', **kwargs):
    if len(tokens) >= pad_width:
        return tokens[0:pad_width]
    padding_num = pad_width - len(tokens)
    if pad_suffix:
        return np.pad(tokens, (0, padding_num), mode, **kwargs)
    else:
        return np.pad(tokens, (padding_num, 0), mode, **kwargs)


def to_fe_mfcc(x):
    """
    ref : https://python-speech-features.readthedocs.io/en/latest/
    Compute MFCC features from an audio signal.
    @param x: wav file
    return mfcc: shape[NUMFRAMES,numcep]
    """
    signal, samplerate = sf.read(x)
    mfcc = python_speech_features.base.mfcc(signal, samplerate=samplerate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nffft=512, lowfreq=0,
                                            highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hanmming)
    return mfcc


def to_fe_fbank(x):
    """
    Compute Mel-filterbank energy features from an audio signal.
    @param x: wav file
    return fbank: shape[NUMFRAMES, nfilt]
    """
    signal, samplerate = sf.read(x)
    fbank, _ = python_speech_features.base.fbank(signal, samplerate=samplerate, winlen=0.025, winstep=0.01, nfilt=26, nffft=512, lowfreq=0,
                                                 highfreq=None, preemph=0.97, winfunc=np.hanmming)
    return fbank


def to_fe_logfbank(x):
    """
    Compute log Mel-filterbank energy features from an audio signal.
    @param x: wav file
    return logfbank: shape[NUMFRAMES, nfilt]
    """
    signal, samplerate = sf.read(x)
    logfbank = python_speech_features.base.logfbank(signal, samplerate=samplerate, winlen=0.025, winstep=0.01, nfilt=26, nffft=512, lowfreq=0,
                                                    highfreq=None, preemph=0.97)
    return logfbank


def to_fe_ssc(x):
    """
    Compute Spectral Subband Centroid features from an audio signal
    @param x: wav file
    return logfbank: shape[NUMFRAMES, nfilt]
    """
    signal, samplerate = sf.read(x)
    logfbank = python_speech_features.base.logfbank(signal, samplerate=samplerate, winlen=0.025, winstep=0.01, nfilt=26, nffft=512, lowfreq=0,
                                                    highfreq=None, preemph=0.97)
    return logfbank


def token_process(token_id, vocab_size, token_mask_id, mask_prob=0.8, keep_prob=0.1):
    """以mask_prob的概率替换为[MASK]，以keep_prob的概率保持不变，否则替换为一个随机token
    """
    rand = np.random.random()
    if rand <= mask_prob:
        return token_mask_id
    elif rand <= mask_prob + keep_prob:
        return token_id
    else:
        return np.random.randint(1, vocab_size)  # 0 is padding id


def gen_causal_attention_mask(x):
    """
    下三角为1的矩阵
    e.g. x=np.ones([32,6,128])
    mask =
    array([[[[1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1]]]], dtype=int32)
    """
    seq_len = x.shape[1]
    idxs = np.arange(0, seq_len)
    mask = idxs[None, :] <= idxs[:, None]
    mask = np.asarray(mask, 'int32')
    return mask[None, None]


def gen_unilm_attention_mask(x):
    """
    (UNILM)Unified Language Model Pre-training for Natural Language Understanding and Generation
    论文地址：https://arxiv.org/abs/1905.03197
    论文代码：https://github.com/microsoft/unilm
    next sentence :
    >>> x=np.asarray([0,0,0,1,1,1])
    >>> y=gen_unilm_attention_mask(x[None,:])
    >>> y
    array([[[[1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1]]]], dtype=int32)
    """
    idxs = np.cumsum(x, axis=1)
    mask = idxs[:, None, :] <= idxs[:, :, None]
    mask = np.asarray(mask, 'int32')
    return mask[:, None]


data_processor_dicts = {
    'to_np': lambda x, schema, token_dicts: to_np(x, schema.dtype),
    'to_tokenid': lambda x, schema, token_dicts: to_tokenid(x, schema.token_dict_name, token_dicts),
    'to_tokenid_start': lambda x, schema, token_dicts: to_tokenid('<s> ' + x, schema.token_dict_name, token_dicts),
    'to_tokenid_end': lambda x, schema, token_dicts: to_tokenid(x + r' <\s>', schema.token_dict_name, token_dicts),
    'to_sentenceid': lambda x, schema, token_dicts: to_tokenid('<s> ' + x + r' <\s>', schema.token_dict_name, token_dicts),
    'to_fe_mfcc': lambda x, _, __: to_fe_mfcc(x),
    'to_fe_fbank': lambda x, _, __: to_fe_fbank(x),
    'to_fe_logfbank': lambda x, _, __: to_fe_logfbank(x),
    'to_fe_ssc': lambda x, _, __: to_fe_ssc(x),
    'None': None
}

#!/usr/bin/env python

"""Tests for `datasets` package."""

import unittest
import sklearn
from datasets.utils.token_dicts import TokenDicts


def make_pkl_dict():
    dict_path = 'tests/data/dict/time.pkl'


class TestDatasets(unittest.TestCase):
    """Tests for `datasets` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_vocab_dict_no_min_freq(self):
        """Test something."""
        # init token_dicts
        token_dicts = TokenDicts('tests/data/dicts', {'query': 0})
        print('min_freq no', token_dicts.dict_min_freq)
        print('unkown no', token_dicts.unkown_emb_id)
        id_find = token_dicts.to_id('query', '月')
        assert id_find != token_dicts.unkown_emb_id['query']
        print('find 月:', id_find)
        id_find = token_dicts.to_id('query', '在')
        assert id_find != token_dicts.unkown_emb_id['query']
        print('find 在:', id_find)
        id_find = token_dicts.to_id('query', '101')
        assert id_find == token_dicts.unkown_emb_id['query']
        print('find 101:', id_find)

    def test_vocab_dict_min_freq_float(self):
        """Test something."""
        # init token_dicts
        token_dicts = TokenDicts('tests/data/dicts', {'query': 0.5})
        print('min_freq float', token_dicts.dict_min_freq)

    def test_vocab_dict_min_freq_int(self):
        """Test something."""
        # init token_dicts
        token_dicts = TokenDicts('tests/data/dicts', {'query': 2})
        print('min_freq int', token_dicts.dict_min_freq)
        print('unkown int', token_dicts.unkown_emb_id)
        id_find = token_dicts.to_id('query', '月')
        assert id_find == token_dicts.unkown_emb_id['query']
        print('find 月:', id_find)
        id_find = token_dicts.to_id('query', '在')
        assert id_find != token_dicts.unkown_emb_id['query']
        print('find 在:', id_find)
        id_find = token_dicts.to_id('query', '101')
        assert id_find == token_dicts.unkown_emb_id['query']
        print('find 101:', id_find)


"""
    def test_pkl_dict(self):
        # init token_dicts
        token_dicts = TokenDicts('tests/data/dicts', {'time': 'pkl'})
"""

if __name__ == '__main__':
    unittest.main()
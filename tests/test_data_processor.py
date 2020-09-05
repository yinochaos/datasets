#!/usr/bin/env python

"""Tests for `datasets` package."""


import unittest

from datasets.dataset import Dataset
from datasets.utils.token_dicts import TokenDicts
import tensorflow as tf

class TestDatasets(unittest.TestCase):
    """Tests for `datasets` package."""

    def setUp(self):
        if tf.__version__.startswith('1.'):
            tf.enable_eager_execution()
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_raw_query_float_dataset(self):
        """Test something."""
        # init token_dicts
        token_dicts = TokenDicts('tests/data/dicts', {'query':0})
        
if __name__ == '__main__':
    unittest.main()
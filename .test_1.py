#!/usr/bin/env python3
from naive_bayes_filter import NaiveBayesFilter
import unittest
import numpy as np


class TestNaiveBayesFilter(unittest.TestCase):

    def test_basic_play_game_1(self):
        classifier = NaiveBayesFilter('test.csv')
        acc = classifier.classify_test()
        print("Your accuracy= {}%".format(acc))
        self.assertTrue(acc >= 75.)

unittest.main()
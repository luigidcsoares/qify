import doctest
import unittest

from qify.probab_dist import _core

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(_core))
    return tests

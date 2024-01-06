import doctest
import unittest

from qify.measure import bayes_vuln

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(bayes_vuln))
    return tests

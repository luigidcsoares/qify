import doctest

from qify.probab_dist import core

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(core))
    return tests

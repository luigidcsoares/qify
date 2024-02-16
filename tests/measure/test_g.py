import doctest

from qify.measure import g

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(g))
    return tests

import doctest

from qify.measure import bayes

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(bayes))
    return tests

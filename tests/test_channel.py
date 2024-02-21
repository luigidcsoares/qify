import doctest

from qify.channel import core

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(core))
    return tests

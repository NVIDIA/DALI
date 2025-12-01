import sys

# before running the test we add dali/test/python to the python path
import nose_utils  # noqa:F401  - for Python 3.10
from nose.core import run_exit
import inspect

if sys.version_info >= (3, 11):

    def legacy_getargspec(fun):
        args, varargs, varkw, defaults, *_ = inspect.getfullargspec(fun)
        return (args, varargs, varkw, defaults)

    inspect.getargspec = legacy_getargspec

if sys.argv[0].endswith("__main__.py"):
    sys.argv[0] = "%s -m nose_wrapper" % sys.executable

run_exit()

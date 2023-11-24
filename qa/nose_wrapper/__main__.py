import sys
from nose.core import run_exit
import collections
import nose.case
import nose.inspector
import nose.loader
import nose.suite
import nose.plugins.attrib
import inspect

if sys.version_info >= (3, 10) and not hasattr(collections, "Callable"):
    nose.case.collections = collections.abc
    nose.inspector.collections = collections.abc
    nose.loader.collections = collections.abc
    nose.suite.collections = collections.abc
    nose.plugins.attrib.collections = collections.abc

if sys.version_info >= (3, 11):

    def legacy_getargspec(fun):
        args, varargs, varkw, defaults, *_ = inspect.getfullargspec(fun)
        return (args, varargs, varkw, defaults)

    inspect.getargspec = legacy_getargspec

if sys.argv[0].endswith("__main__.py"):
    sys.argv[0] = "%s -m nose_wrapper" % sys.executable

run_exit()

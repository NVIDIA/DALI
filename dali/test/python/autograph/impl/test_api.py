# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for api module."""

import abc
import collections
import contextlib
import functools
import gc
import inspect
import os
import re
import sys
import textwrap
import types
import unittest
import unittest.mock

import numpy as np
import six

from operator import add
from functools import reduce

from nvidia.dali._autograph.core import ag_ctx
from nvidia.dali._autograph.core import converter
from nvidia.dali._autograph.core import converter_testing
from nvidia.dali._autograph.impl import api
from nvidia.dali._autograph.impl import conversion
from nvidia.dali._autograph.pyct import errors
from nvidia.dali._autograph.pyct import inspect_utils
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.utils import ag_logging

# from nvidia.dali._autograph.utils.all_utils import custom_constant


global_n = 2

DEFAULT_RECURSIVE = converter.ConversionOptions(recursive=True)


class TestResource(object):
    def __init__(self):
        self.x = 3


def custom_constant(val):
    return np.array(val)


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._transpiler_bkp = api._TRANSPILER
        cls._conversion_rules_bkp = api.config.CONVERSION_RULES
        api._TRANSPILER = None
        api.initialize_autograph()

    @classmethod
    def tearDownClass(cls):
        api._TRANSPILER = cls._transpiler_bkp
        api.config.CONVERSION_RULES = cls._conversion_rules_bkp

    def evaluate(self, x):
        return x

    def assertAllEqual(self, a, b):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            same = np.array(a) == np.array(b)
            self.assertTrue(np.all(same))
        else:
            self.assertEqual(a, b)

    @contextlib.contextmanager
    def assertPrints(self, expected, not_expected):
        try:
            out_capturer = six.StringIO()
            sys.stdout = out_capturer
            yield
            self.assertIn(expected, out_capturer.getvalue())
            self.assertNotIn(not_expected, out_capturer.getvalue())
        finally:
            sys.stdout = sys.__stdout__

    def assertNoMemoryLeaks(self, f):
        object_ids_before = {id(o) for o in gc.get_objects()}
        f()
        gc.collect()
        objects_after = tuple(o for o in gc.get_objects() if id(o) not in object_ids_before)
        self.assertEqual(tuple(o for o in objects_after if isinstance(o, TestResource)), ())

    def test_converted_call_kwonly_args(self):
        def test_fn(*, a):
            return a

        x = api.converted_call(test_fn, (), {"a": custom_constant(-1)}, options=DEFAULT_RECURSIVE)
        self.assertEqual(-1, self.evaluate(x))

    def test_super_with_no_arg(self):
        test_case_self = self

        class TestBase:
            def plus_three(self, x):
                return x + 3

        class TestSubclass(TestBase):
            def plus_three(self, x):
                test_case_self.fail("This should never be called.")

            def no_arg(self, x):
                return super().plus_three(x)

        tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)

        self.assertEqual(5, tc.no_arg(2))

    def test_converted_call_avoids_triggering_operators(self):
        test_self = self

        class Pair(collections.namedtuple("Pair", ["a", "b"])):
            def __call__(self):
                return self.a + self.b

            def __eq__(self, other):
                test_self.fail("Triggered operator")

        p = Pair(custom_constant(1), custom_constant(2))

        x = api.converted_call(p, (), {}, options=DEFAULT_RECURSIVE)
        self.assertIsNotNone(self.evaluate(x), 3)

    def test_decorator_recursive(self):
        class TestClass(object):
            def called_member(self, a):
                if a < 0:
                    a = -a
                return a

            @api.convert(recursive=True)
            def test_method(self, x, s, a):
                while reduce(add, x) > s:
                    x //= self.called_member(a)
                return x

        tc = TestClass()
        x = tc.test_method(custom_constant([2, 4]), custom_constant(1), custom_constant(-2))
        self.assertListEqual([0, 1], self.evaluate(x).tolist())

    def test_decorator_not_recursive(self):
        class TestClass(object):
            def called_member(self, a):
                return -a

            @api.convert(recursive=False)
            def test_method(self, x, s, a):
                while reduce(add, x) > s:
                    x //= self.called_member(a)
                return x

        tc = TestClass()
        x = tc.test_method(custom_constant([2, 4]), custom_constant(1), custom_constant(-2))
        self.assertListEqual([0, 1], self.evaluate(x).tolist())

    def test_convert_then_do_not_convert(self):
        class TestClass(object):
            @api.do_not_convert
            def called_member(self, a):
                return -a

            @api.convert(recursive=True)
            def test_method(self, x, s, a):
                while reduce(add, x) > s:
                    x //= self.called_member(a)
                return x

        tc = TestClass()
        x = tc.test_method(custom_constant((2, 4)), custom_constant(1), custom_constant(-2))
        self.assertAllEqual((0, 1), self.evaluate(x))

    def test_decorator_calls_decorated(self):
        class TestClass(object):
            @api.convert()
            def called_member(self, a):
                if a < 0:
                    a = -a
                return a

            @api.convert(recursive=True)
            def test_method(self, x, s, a):
                while reduce(add, x) > s:
                    x //= self.called_member(a)
                return x

        tc = TestClass()
        x = tc.test_method(custom_constant([2, 4]), custom_constant(1), custom_constant(-2))
        self.assertListEqual([0, 1], self.evaluate(x).tolist())

    # TODO(klecki): Argspec mismatches
    def _test_decorator_preserves_argspec(self):
        class TestClass(object):
            def test_method(self, a):
                if a < 0:
                    a = -a
                return a

            test_method_converted = api.convert()(test_method)

        tc = TestClass()
        self.assertListEqual(
            list(inspect.getfullargspec(tc.test_method)),
            list(inspect.getfullargspec(tc.test_method_converted)),
        )

    def test_do_not_convert_argspec(self):
        class TestClass(object):
            def test_method(self, x, y):
                z = x + y
                return z

            test_method_allowlisted = api.do_not_convert(test_method)

        tc = TestClass()
        self.assertTrue(inspect.ismethod(tc.test_method_allowlisted))
        # Because the wrapped function is not generated, we can't preserve its
        # arg spec.

    def test_do_not_convert_callable_object(self):
        class TestClass(object):
            def __call__(self):
                return 1

        tc = TestClass()
        self.assertEqual(1, api.do_not_convert(tc)())

    def test_convert_call_site_decorator(self):
        class TestClass(object):
            def called_member(self, a):
                if a < 0:
                    a = -a
                return a

            @api.convert(recursive=True)
            def test_method(self, x, s, a):
                while reduce(add, x) > s:
                    x //= api.converted_call(
                        self.called_member, (a,), None, options=DEFAULT_RECURSIVE
                    )
                return x

        tc = TestClass()
        x = tc.test_method(custom_constant([2, 4]), custom_constant(1), custom_constant(-2))
        self.assertListEqual([0, 1], self.evaluate(x).tolist())

    def test_converted_call_builtin(self):
        x = api.converted_call(range, (3,), None, options=DEFAULT_RECURSIVE)
        self.assertEqual((0, 1, 2), tuple(x))

        x = api.converted_call(
            re.compile, ("mnas_v4_a.*\\/.*(weights|kernel):0$",), None, options=DEFAULT_RECURSIVE
        )
        self.assertIsNotNone(x.match("mnas_v4_a/weights:0"))

    def test_converted_call_function(self):
        def test_fn(x):
            if x < 0:
                return -x
            return x

        x = api.converted_call(test_fn, (custom_constant(-1),), None, options=DEFAULT_RECURSIVE)
        self.assertEqual(1, self.evaluate(x))

    def test_converted_call_functools_partial(self):
        def test_fn(x, y, z):
            if x < 0:
                return -x, -y, -z
            return x, y, z

        x = api.converted_call(
            functools.partial(test_fn, custom_constant(-1), z=-3),
            (custom_constant(-2),),
            None,
            options=DEFAULT_RECURSIVE,
        )
        self.assertEqual((1, 2, 3), self.evaluate(x))

        x = api.converted_call(
            functools.partial(functools.partial(test_fn, custom_constant(-1)), z=-3),
            (custom_constant(-2),),
            None,
            options=DEFAULT_RECURSIVE,
        )
        self.assertEqual((1, 2, 3), self.evaluate(x))

    def test_converted_call_functools_partial_kwarg_mutation(self):
        def test_fn(x, y, z):
            if x < 0:
                return -x, -y, -z
            return x, y, z

        partial_fn = functools.partial(test_fn, custom_constant(-1), z=-3)
        # Call using kwargs to assign y first to ensure that partial_fn.keywords is
        # not mutated for subsequent calls (where y is assign through args).
        x = api.converted_call(
            partial_fn,
            args=(),
            kwargs={
                "y": custom_constant(-2),
            },
            options=DEFAULT_RECURSIVE,
        )
        self.assertEqual((1, 2, 3), self.evaluate(x))

        x = api.converted_call(
            partial_fn, args=(custom_constant(-4),), kwargs=None, options=DEFAULT_RECURSIVE
        )
        self.assertEqual((1, 4, 3), self.evaluate(x))

    def test_converted_call_method(self):
        class TestClass(object):
            def __init__(self, x):
                self.x = x

            def test_method(self):
                if self.x < 0:
                    return -self.x
                return self.x

        tc = TestClass(custom_constant(-1))
        x = api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
        self.assertEqual(1, self.evaluate(x))

    def test_converted_call_synthetic_method(self):
        class TestClass(object):
            def __init__(self, x):
                self.x = x

        def test_function(self):
            if self.x < 0:
                return -self.x
            return self.x

        tc = TestClass(custom_constant(-1))
        test_method = types.MethodType(test_function, tc)

        x = api.converted_call(test_method, (), None, options=DEFAULT_RECURSIVE)
        self.assertEqual(1, self.evaluate(x))

    def test_converted_call_method_wrapper(self):
        class TestClass(object):
            def foo(self):
                pass

        tc = TestClass()

        # `method.__get__()` returns a so-called method-wrapper.
        wrapper = api.converted_call(tc.foo.__get__, (tc,), None, options=DEFAULT_RECURSIVE)
        self.assertEqual(wrapper, tc.foo)

    def test_converted_call_method_as_object_attribute(self):
        class AnotherClass(object):
            def __init__(self):
                self.another_class_attr = custom_constant(1)

            def method(self):
                if self.another_class_attr > 0:
                    return self.another_class_attr + 1
                return self.another_class_attr + 10

        class TestClass(object):
            def __init__(self, another_obj_method):
                self.another_obj_method = another_obj_method

        obj = AnotherClass()
        tc = TestClass(obj.method)

        x = api.converted_call(tc.another_obj_method, (), None, options=DEFAULT_RECURSIVE)
        self.assertEqual(self.evaluate(x), 2)

    def test_converted_call_method_converts_recursively(self):
        class TestClass(object):
            def __init__(self, x):
                self.x = x

            def other_method(self):
                if self.x < 0:
                    return -self.x
                return self.x

            def test_method(self):
                return self.other_method()

        tc = TestClass(custom_constant(-1))
        x = api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
        self.assertEqual(1, self.evaluate(x))

    def test_converted_call_method_by_class(self):
        class TestClass(object):
            def __init__(self, x):
                self.x = x

            def test_method(self):
                if self.x < 0:
                    return -self.x
                return self.x

        tc = TestClass(custom_constant(-1))
        x = api.converted_call(TestClass.test_method, (tc,), None, options=DEFAULT_RECURSIVE)
        self.assertEqual(1, self.evaluate(x))

    def test_converted_call_callable_object(self):
        class TestClass(object):
            def __init__(self, x):
                self.x = x

            def __call__(self):
                if self.x < 0:
                    return -self.x
                return self.x

        tc = TestClass(custom_constant(-1))
        x = api.converted_call(tc, (), None, options=DEFAULT_RECURSIVE)
        self.assertEqual(1, self.evaluate(x))

    def test_converted_call_callable_metaclass(self):
        test_self = self

        class TestMetaclass(type):
            def __call__(cls):  # pylint: disable=method-hidden
                self.assertTrue(converter_testing.is_inside_generated_code())
                inst = object.__new__(cls)
                inst.__init__()

                def instance_call(unused_self):
                    test_self.fail(
                        "The class-bound __call__ should be called, not the instance" " bound one."
                    )

                inst.__call__ = instance_call
                return inst

        tmc = TestMetaclass("TestClass", (), {})
        tc = api.converted_call(tmc, (), None, options=DEFAULT_RECURSIVE)
        self.assertIsInstance(tc, tmc)

    def test_converted_call_callable_abc(self):
        test_self = self

        @six.add_metaclass(abc.ABCMeta)
        class TestBase(object):
            @abc.abstractmethod
            def __call__(self):
                test_self.fail("This should not be called")

        class TestSubclass(TestBase):
            def __init__(self):
                test_self.assertFalse(converter_testing.is_inside_generated_code())

            def __call__(self, expected):
                test_self.assertTrue(expected)
                test_self.assertTrue(converter_testing.is_inside_generated_code())

        tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)
        api.converted_call(tc, (True,), None, options=DEFAULT_RECURSIVE)

    def test_converted_call_constructor(self):
        test_self = self

        class TestClass(object):
            def __init__(self):
                test_self.assertFalse(converter_testing.is_inside_generated_code())

        tc = api.converted_call(TestClass, (), None, options=DEFAULT_RECURSIVE)
        self.assertIsInstance(tc, TestClass)

    def test_converted_call_mangled_properties(self):
        class TestClass(object):
            def __init__(self):
                self.__private = custom_constant(-1)

            def test_method(self):
                return self.__private

        tc = TestClass()
        with self.assertRaisesRegex(errors.UnsupportedLanguageElementError, "mangled names"):
            api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)

        # TODO(mdan): Refactor to avoid this use of global state.
        ag_logging.set_verbosity(0, True)
        os.environ["AUTOGRAPH_STRICT_CONVERSION"] = "0"
        with self.assertPrints("could not transform", "bug"):
            api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
        ag_logging.set_verbosity(0, False)
        os.environ["AUTOGRAPH_STRICT_CONVERSION"] = "1"

    def test_converted_call_partial_of_allowlisted_function(self):
        def test_fn(_):
            self.assertFalse(converter_testing.is_inside_generated_code())

        converter_testing.allowlist(test_fn)
        api.converted_call(functools.partial(test_fn, None), (), None, options=DEFAULT_RECURSIVE)

    def test_converted_call_already_converted(self):
        def f(x):
            return x == 0

        x = api.converted_call(f, (custom_constant(0),), None, options=DEFAULT_RECURSIVE)
        self.assertTrue(self.evaluate(x))

        converted_f = api.to_graph(f, experimental_optional_features=converter.Feature.ALL)
        x = api.converted_call(converted_f, (custom_constant(0),), None, options=DEFAULT_RECURSIVE)
        self.assertTrue(self.evaluate(x))

    def test_converted_call_then_already_converted_dynamic(self):
        @api.convert()
        def g(x):
            if x > 0:
                return x
            else:
                return -x

        def f(g, x):
            return g(x)

        x = api.converted_call(f, (g, custom_constant(1)), None, options=DEFAULT_RECURSIVE)
        self.assertEqual(self.evaluate(x), 1)

    def test_converted_call_forced_when_explicitly_allowlisted(self):
        @api.do_not_convert()
        def f(x):
            return x + 1

        opts = converter.ConversionOptions(recursive=True, user_requested=True)
        x = api.converted_call(f, (custom_constant(0),), None, options=opts)
        self.assertTrue(self.evaluate(x))

        converted_f = api.to_graph(f, experimental_optional_features=converter.Feature.ALL)
        x = api.converted_call(converted_f, (0,), None, options=DEFAULT_RECURSIVE)
        self.assertEqual(x, 1)

    def test_converted_call_no_kwargs_allowed(self):
        def f(*args):
            # Note: np.broadcast rejects any **kwargs, even *{}
            return np.broadcast(args[:1])

        opts = converter.ConversionOptions(internal_convert_user_code=False)
        self.assertIsNotNone(api.converted_call(f, (1, 2, 3, 4), None, options=opts))

    def test_converted_call_allowlisted_method(self):
        class TestClass(object):
            def method(self):
                return converter_testing.is_inside_generated_code()

        obj = TestClass()
        converter_testing.allowlist(obj.method.__func__)

        self.assertFalse(api.converted_call(obj.method, (), {}, options=DEFAULT_RECURSIVE))

    def test_converted_call_allowlisted_method_via_owner(self):
        class TestClass(object):
            def method(self):
                return converter_testing.is_inside_generated_code()

        converter_testing.allowlist(TestClass)

        obj = TestClass()
        self.assertFalse(api.converted_call(obj.method, (), {}, options=DEFAULT_RECURSIVE))

    def test_converted_call_numpy(self):
        x = api.converted_call(np.arange, (5,), None, options=DEFAULT_RECURSIVE)

        self.assertAllEqual(x, list(range(5)))

    def test_converted_call_tf_op_forced(self):
        # TODO(mdan): Add the missing level of support to LOGICAL_EXPRESSIONS.
        opts = converter.ConversionOptions(user_requested=True, optional_features=None)

        x = api.converted_call(add, (1, 1), None, options=opts)

        self.assertAllEqual(self.evaluate(x), 2)

    def test_converted_call_exec_generated_code(self):
        temp_mod = types.ModuleType("test_module")
        dynamic_code = """
      def foo(x):
        return x + 1
    """
        exec(textwrap.dedent(dynamic_code), temp_mod.__dict__)  # pylint:disable=exec-used
        opts = converter.ConversionOptions(optional_features=None)

        x = api.converted_call(temp_mod.foo, (1,), None, options=opts)

        self.assertAllEqual(x, 2)

    def test_converted_call_namedtuple(self):
        x = api.converted_call(
            collections.namedtuple, ("TestNamedtuple", ("a", "b")), None, options=DEFAULT_RECURSIVE
        )

        self.assertTrue(inspect_utils.isnamedtuple(x))

    def test_converted_call_namedtuple_via_collections(self):
        x = api.converted_call(
            collections.namedtuple, ("TestNamedtuple", ("a", "b")), None, options=DEFAULT_RECURSIVE
        )

        self.assertTrue(inspect_utils.isnamedtuple(x))

    def test_converted_call_namedtuple_subclass_bound_method(self):
        class TestClass(collections.namedtuple("TestNamedtuple", ("a", "b"))):
            def test_method(self, x):
                while reduce(add, x) > self.a:
                    x //= self.b
                return x

        obj = TestClass(5, 2)
        x = api.converted_call(
            obj.test_method, (custom_constant([2, 4]),), None, options=DEFAULT_RECURSIVE
        )

        self.assertAllEqual(self.evaluate(x), [1, 2])

    def test_converted_call_namedtuple_method(self):
        class TestClass(collections.namedtuple("TestNamedtuple", ("a", "b"))):
            pass

        obj = TestClass(5, 2)
        # _asdict is a documented method of namedtuple.
        x = api.converted_call(obj._asdict, (), None, options=DEFAULT_RECURSIVE)

        self.assertDictEqual(x, {"a": 5, "b": 2})

    def test_converted_call_namedtuple_subclass_unbound_method(self):
        class TestClass(collections.namedtuple("TestNamedtuple", ("a", "b"))):
            def test_method(self, x):
                while reduce(add, x) > self.a:
                    x //= self.b
                return x

        obj = TestClass(5, 2)
        x = api.converted_call(
            TestClass.test_method, (obj, custom_constant([2, 4])), None, options=DEFAULT_RECURSIVE
        )

        self.assertAllEqual(self.evaluate(x), [1, 2])

    def test_converted_call_lambda(self):
        l = lambda x: x == 0

        x = api.converted_call(l, (custom_constant(0),), None, options=DEFAULT_RECURSIVE)

        self.assertAllEqual(True, self.evaluate(x))

    def test_converted_call_native_binding(self):
        x = api.converted_call(np.power, (2, 2), None, options=DEFAULT_RECURSIVE)
        self.assertAllEqual(x, 4)

    def test_converted_call_native_binding_errorneous(self):
        class FaultyBinding(object):
            def __array__(self):
                raise ValueError("fault")

        bad_obj = FaultyBinding()

        def fail_if_warning(*_):
            self.fail("No warning should be issued")

        with unittest.mock.patch.object(ag_logging, "warning", fail_if_warning):
            with self.assertRaisesRegex(ValueError, "fault"):
                api.converted_call(np.power, (bad_obj, 2), None, options=DEFAULT_RECURSIVE)

    def test_converted_call_no_leaks_via_closure(self):
        def test_fn():
            res = TestResource()

            def f(y):
                return res.x + y

            api.converted_call(f, (1,), None, options=DEFAULT_RECURSIVE)

        self.assertNoMemoryLeaks(test_fn)

    def test_converted_call_no_leaks_via_inner_function_closure(self):
        def test_fn():
            res = TestResource()

            def f(y):
                def inner_f():
                    return res.x + y

                return inner_f

            api.converted_call(f, (1,), None, options=DEFAULT_RECURSIVE)()

        self.assertNoMemoryLeaks(test_fn)

    def test_converted_call_no_caching_on_abort(self):
        def test_fn(needs_autograph):
            if needs_autograph:
                if custom_constant(True):
                    x = custom_constant(1)
                else:
                    x = custom_constant(2)
            else:
                x = 3
            return x

        def call_in_disabled_context():
            with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.DISABLED):
                return api.converted_call(test_fn, (False,), None, options=DEFAULT_RECURSIVE)

        def call_in_default_context():
            with ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED):
                return api.converted_call(test_fn, (True,), None, options=DEFAULT_RECURSIVE)

        # Note: this is an invariant, not a test (see above).
        assert call_in_disabled_context() == 3

        # If api.convert placed test_fn in the unconverted cache, this second
        # invocation would fail.
        self.assertEqual(self.evaluate(call_in_default_context()), 1)

    def test_converted_call_caching_of_allowlisted_bound_methods(self):
        class TestClass(object):
            def __init__(self):
                self.__private = custom_constant(-1)

            def test_method(self):
                return self.__private

        # TODO(mdan): Refactor to avoid this use of global state.
        cache_size_before = len(conversion._ALLOWLIST_CACHE)

        # First invocation with fallback on, to allow recording it into cache.
        os.environ["AUTOGRAPH_STRICT_CONVERSION"] = "0"
        tc = TestClass()
        api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)
        os.environ["AUTOGRAPH_STRICT_CONVERSION"] = "1"

        # Entry should be added to the allowlist cache.
        self.assertEqual(len(conversion._ALLOWLIST_CACHE), cache_size_before + 1)

        # A second invocation should go through even with fallback off.
        tc = TestClass()
        api.converted_call(tc.test_method, (), None, options=DEFAULT_RECURSIVE)

        # No new entries should appear in the allowlist cache.
        self.assertEqual(len(conversion._ALLOWLIST_CACHE), cache_size_before + 1)

    def test_context_tracking_direct_calls(self):
        @api.do_not_convert()
        def unconverted_fn():
            self.assertEqual(ag_ctx.control_status_ctx().status, ag_ctx.Status.DISABLED)

        @api.convert()
        def converted_fn():
            self.assertEqual(ag_ctx.control_status_ctx().status, ag_ctx.Status.ENABLED)
            unconverted_fn()
            self.assertEqual(ag_ctx.control_status_ctx().status, ag_ctx.Status.ENABLED)

        self.assertEqual(ag_ctx.control_status_ctx().status, ag_ctx.Status.UNSPECIFIED)
        converted_fn()
        self.assertEqual(ag_ctx.control_status_ctx().status, ag_ctx.Status.UNSPECIFIED)

        @api.call_with_unspecified_conversion_status
        def unspecified_fn():
            self.assertEqual(ag_ctx.control_status_ctx().status, ag_ctx.Status.UNSPECIFIED)

        unspecified_fn()

    def test_to_graph_with_defaults(self):
        foo = 4

        def test_fn(x, s=foo):
            while reduce(add, x) > s:
                x //= 2
            return x

        compiled_fn = api.to_graph(test_fn)

        x = compiled_fn(custom_constant([4, 8]))
        self.assertListEqual([1, 2], self.evaluate(x).tolist())

    def test_to_graph_with_globals(self):
        def test_fn(x):
            global global_n
            global_n = x + global_n
            return global_n

        converted_fn = api.to_graph(test_fn)
        prev_val = global_n
        converted_fn(10)
        self.assertGreater(global_n, prev_val)

    def test_to_graph_with_kwargs_clashing_converted_call(self):
        def called_fn(**kwargs):
            return kwargs["f"] + kwargs["owner"]

        def test_fn():
            # These arg names intentionally match converted_call's
            return called_fn(f=1, owner=2)

        compiled_fn = api.to_graph(test_fn)

        self.assertEqual(compiled_fn(), 3)

    def test_to_graph_with_kwargs_clashing_unconverted_call(self):
        @api.do_not_convert
        def called_fn(**kwargs):
            return kwargs["f"] + kwargs["owner"]

        def test_fn():
            # These arg names intentionally match _call_unconverted's
            return called_fn(f=1, owner=2)

        compiled_fn = api.to_graph(test_fn)

        self.assertEqual(compiled_fn(), 3)

    def test_to_graph_caching(self):
        def test_fn(x):
            if x > 0:
                return x
            else:
                return -x

        converted_functions = tuple(api.to_graph(test_fn) for _ in (-1, 0, 1))

        # All outputs are from the same module. We can't use __module__ because
        # that's reset when we instantiate the function (see conversion.py).
        # TODO(mdan): Can and should we overwrite __module__ instead?
        module_names = frozenset(f.ag_module for f in converted_functions)
        self.assertEqual(len(module_names), 1)
        self.assertNotIn("__main__", module_names)

        self.assertEqual(len(frozenset(id(f) for f in converted_functions)), 3)

    def test_to_graph_caching_different_options(self):
        def called_fn():
            pass

        def test_fn():
            return called_fn()

        converted_recursive = api.to_graph(test_fn, recursive=True)
        converted_non_recursive = api.to_graph(test_fn, recursive=False)

        self.assertNotEqual(converted_recursive.ag_module, converted_non_recursive.ag_module)
        self.assertRegex(
            inspect.getsource(converted_recursive), "FunctionScope(.*recursive=True.*)"
        )
        self.assertRegex(
            inspect.getsource(converted_non_recursive), "FunctionScope(.*recursive=False.*)"
        )

    def test_to_graph_preserves_bindings(self):
        y = 3

        def test_fn():
            return y

        converted = api.to_graph(test_fn)

        self.assertEqual(converted(), 3)

        y = 7

        self.assertEqual(converted(), 7)

    def test_to_graph_source_map(self):
        def test_fn(y):
            return y**2

        self.assertTrue(hasattr(api.to_graph(test_fn), "ag_source_map"))

    def test_to_code_basic(self):
        def test_fn(x, s):
            while reduce(add, x) > s:
                x /= 2
            return x

        # Just check that the output is parsable Python code.
        self.assertIsNotNone(parser.parse(api.to_code(test_fn)))

    def test_tf_convert_overrides_current_context(self):
        def f(expect_converted):
            self.assertEqual(converter_testing.is_inside_generated_code(), expect_converted)

        @api.do_not_convert
        def test_fn(ctx, expect_converted):
            return api.tf_convert(f, ctx)(expect_converted)

        test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.ENABLED), True)
        test_fn(ag_ctx.ControlStatusCtx(status=ag_ctx.Status.DISABLED), False)

    def test_super_with_one_arg(self):
        test_case_self = self

        class TestBase(object):
            def plus_three(self, x):
                return x + 3

        class TestSubclass(TestBase):
            def plus_three(self, x):
                test_case_self.fail("This should never be called.")

            def one_arg(self, x):
                test_base_unbound = super(TestSubclass)
                test_base = test_base_unbound.__get__(self, TestSubclass)
                return test_base.plus_three(x)

        tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)

        self.assertEqual(5, tc.one_arg(2))

    def test_super_with_two_args(self):
        test_case_self = self

        class TestBase(object):
            def plus_three(self, x):
                return x + 3

        class TestSubclass(TestBase):
            def plus_three(self, x):
                test_case_self.fail("This should never be called.")

            def two_args(self, x):
                return super(TestSubclass, self).plus_three(x)

        tc = api.converted_call(TestSubclass, (), None, options=DEFAULT_RECURSIVE)

        self.assertEqual(5, tc.two_args(2))

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import functools
import importlib.util
import os
import pathlib
import tempfile
from collections.abc import Callable

import numpy as np
from ndd_utils import _is_compiled
from nose_utils import assert_raises
from test_utils import get_dali_extra_path

import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali.types
from nvidia.dali.types import DALIDataType

dali_extra_path = get_dali_extra_path()
images_root = os.path.join(dali_extra_path, "db", "single", "jpeg")


def assert_compiled_matches_eager(
    transform: Callable[[ndd.Batch], ndd.Batch],
    expect_captured: bool,
) -> None:
    """Apply `transform` to an eager and compiled epoch. Assert that the results are identical."""
    reader_dyn = ndd.readers.File(file_root=images_root)
    reader_comp = ndd.readers.File(file_root=images_root)

    dyn = []
    for jpegs, _ in reader_dyn.next_epoch(batch_size=2):
        images = ndd.decoders.image(jpegs, device="gpu")
        dyn.append(ndd.as_tensor(transform(images), pad=True).cpu())

    comp = []
    for jpegs, _ in reader_comp.next_epoch(batch_size=2, compile=True):
        images = ndd.decoders.image(jpegs, device="gpu")
        assert _is_compiled(images)
        out = transform(images)
        assert _is_compiled(out) is expect_captured
        comp.append(ndd.as_tensor(out, pad=True).cpu())

    for d, c in zip(dyn, comp, strict=True):
        np.testing.assert_array_equal(d, c)


def compiled_test(*, expect_captured: bool):
    def decorator(transform: Callable[[ndd.Batch], ndd.Batch]):
        @functools.wraps(transform)
        def test():
            assert_compiled_matches_eager(transform, expect_captured=expect_captured)

        return test

    return decorator


_MODULE_DTYPE = ndd.float32
_GLOBAL_ANGLE = 60
_INVARIANT_ANGLE = ndd.compile.invariant(0.0)
_INVARIANT_SIZE = ndd.compile.invariant(4)


class _Cfg:
    DTYPE = ndd.float32


class _InvariantCfg:
    angle = 60


_INVARIANT_CFG = ndd.compile.invariant(_InvariantCfg())


class _DaliHolder:
    pkg = nvidia.dali


# Tests for captured cases


@compiled_test(expect_captured=True)
def test_literal_scalar(images):
    return ndd.rotate(images, angle=60)


@compiled_test(expect_captured=True)
def test_literal_list(images):
    return ndd.resize(images, size=[224, 224])


@compiled_test(expect_captured=True)
def test_local_scalar(images):
    c = 60
    return ndd.rotate(images, angle=c)


@compiled_test(expect_captured=True)
def test_invariant_names(images):
    c = 224
    return ndd.resize(images, size=[c, c])


@compiled_test(expect_captured=True)
def test_computed_local(images):
    c = 60 + 16
    return ndd.rotate(images, angle=c)


@compiled_test(expect_captured=True)
def test_chained_locals(images):
    a = 60
    c = a + 16
    return ndd.rotate(images, angle=c)


@compiled_test(expect_captured=True)
def test_annotated_local(images):
    a: int = 60
    return ndd.rotate(images, angle=a)


@compiled_test(expect_captured=True)
def test_chained_plain(images):
    _ = a = 60
    return ndd.rotate(images, angle=a)


@compiled_test(expect_captured=True)
def test_tuple_unpack(images):
    a, _ = 60, 90
    return ndd.rotate(images, angle=a)


@compiled_test(expect_captured=True)
def test_starred_sibling(images):
    a, *_ = 60, 1, 2
    return ndd.rotate(images, angle=a)


@compiled_test(expect_captured=True)
def test_starred_immutable_name(images):
    x = (64, 64)
    return ndd.resize(images, size=(*x,))


@compiled_test(expect_captured=True)
def test_chained_unpack(images):
    _ = (a, _) = (60, 90)
    return ndd.rotate(images, angle=a)


@compiled_test(expect_captured=True)
def test_nested_unpack_inner(images):
    _, (x, _), _ = 1, (60, 2), 90
    return ndd.rotate(images, angle=x)


@compiled_test(expect_captured=True)
def test_nested_unpack_outer(images):
    _, (_, _), c = 1, (60, 2), 90
    return ndd.rotate(images, angle=c)


@compiled_test(expect_captured=True)
def test_walrus_direct(images):
    return ndd.rotate(images, angle=(c := 60))  # noqa: F841


@compiled_test(expect_captured=True)
def test_walrus_named(images):
    if (c := 60) > 0:
        return ndd.rotate(images, angle=c)


@compiled_test(expect_captured=True)
def test_local_in_block(images):
    if images is not None:
        c = 60
        return ndd.rotate(images, angle=c)
    assert False


@compiled_test(expect_captured=True)
def test_dali_attribute(images):
    return ndd.cast(images, dtype=ndd.int32)


@compiled_test(expect_captured=True)
def test_dali_attribute_local(images):
    dtype = ndd.int32
    return ndd.cast(images, dtype=dtype)


@compiled_test(expect_captured=True)
def test_dali_fully_qualified(images):
    return ndd.cast(images, dtype=nvidia.dali.types.DALIDataType.FLOAT)


@compiled_test(expect_captured=True)
def test_dali_imported_class(images):
    return ndd.cast(images, dtype=DALIDataType.FLOAT)


@compiled_test(expect_captured=True)
def test_list_comprehension(images):
    items = [images]
    return [ndd.resize(x, size=[64, 64]) for x in items][0]


@compiled_test(expect_captured=True)
def test_non_ascii_line(images):
    carré = (64, 64)  # noqa: E501
    return ndd.resize(images, size=carré)


def test_not_decorated_callsite():
    def transform(images):
        c = 64
        return ndd.resize(images, size=[c, c])

    compiled_test(expect_captured=True)(transform)()


@compiled_test(expect_captured=True)
def test_param_literal(images):
    def rotate(imgs, angle):
        return ndd.rotate(imgs, angle=angle)

    return rotate(images, 60)


@compiled_test(expect_captured=True)
def test_param_expression(images):
    def rotate(imgs, angle, /):
        return ndd.rotate(imgs, angle=angle)

    return rotate(images, 40 + 20)


@compiled_test(expect_captured=True)
def test_param_dali_attribute(images):
    def cast(imgs, dtype):
        return ndd.cast(imgs, dtype=dtype)

    return cast(images, ndd.int32)


@compiled_test(expect_captured=True)
def test_param_chained(images):
    def inner(imgs, angle):
        return ndd.rotate(imgs, angle=angle)

    def outer(imgs, angle):
        return inner(imgs, angle)

    return outer(images, 60)


@compiled_test(expect_captured=True)
def test_param_recursive(images):
    def _rotate_recursive(images, angle, depth):
        if depth == 0:
            return ndd.rotate(images, angle=angle)
        return _rotate_recursive(images, angle, depth - 1)

    return _rotate_recursive(images, 60, 4)


@compiled_test(expect_captured=True)
def test_param_cross_file(images):
    source = """
import nvidia.dali.experimental.dynamic as ndd

def resize(*, images, size):
    return ndd.resize(images, size=size)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        module_path = pathlib.Path(tmpdir) / "module.py"
        module_path.write_text(source)

        # Import the module
        spec = importlib.util.spec_from_file_location("module", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.resize(images=images, size=(224, 224))


@compiled_test(expect_captured=True)
def test_param_method(images):
    class Aug:
        def rotate(self, imgs, angle):
            return ndd.rotate(imgs, angle=angle)

    aug = Aug()
    return aug.rotate(images, 60)


@compiled_test(expect_captured=True)
def test_param_classmethod(images):
    class Aug:
        @classmethod
        def rotate(cls, imgs, angle):
            return ndd.rotate(imgs, angle=angle)

    return Aug.rotate(images, 60)


@compiled_test(expect_captured=True)
def test_param_staticmethod(images):
    class Aug:
        @staticmethod
        def rotate(imgs, angle):
            return ndd.rotate(imgs, angle=angle)

    return Aug.rotate(images, 60)


@compiled_test(expect_captured=True)
def test_param_partial_keyword(images):
    def resize(imgs, width, height):
        return ndd.resize(imgs, size=[width, height])

    resize_partial = functools.partial(resize, width=64)
    return resize_partial(images, height=128)


@compiled_test(expect_captured=True)
def test_closure_param(images):
    def make_rotate(angle):
        def rotate():
            return ndd.rotate(images, angle=angle)

        return rotate

    return make_rotate(60)()


@compiled_test(expect_captured=True)
def test_closure_local(images):
    def make_rotate():
        def rotate():
            return ndd.rotate(images, angle=angle)

        angle = 60
        return rotate

    return make_rotate()()


@compiled_test(expect_captured=True)
def test_closure_live_parent(images):
    def rotate(angle):
        def transform():
            return ndd.rotate(images, angle=angle)

        return transform()

    return rotate(60)


@compiled_test(expect_captured=True)
def test_param_default_literal(images):
    def rotate(imgs, angle=60):
        return ndd.rotate(imgs, angle=angle)

    return rotate(images)


@compiled_test(expect_captured=True)
def test_param_default_dali_attribute(images):
    def cast(imgs, dtype=ndd.int32):
        return ndd.cast(imgs, dtype=dtype)

    return cast(images)


@compiled_test(expect_captured=True)
def test_param_default_name(images):
    angle = 60

    def rotate(imgs, angle=angle):
        return ndd.rotate(imgs, angle=angle)

    return rotate(images)


@compiled_test(expect_captured=True)
def test_invariant_marker(images):
    angle = ndd.compile.invariant(10) + _INVARIANT_CFG.angle
    return ndd.rotate(images, angle=angle, fill_value=ndd.compile.invariant(None))


@compiled_test(expect_captured=True)
def test_invariant_marker_parameter(images):
    def rotate(imgs, angle):
        return ndd.rotate(imgs, angle=angle)

    return rotate(images, _INVARIANT_ANGLE)


@compiled_test(expect_captured=True)
def test_invariant_marker_list(images):
    images = ndd.resize(images, size=[_INVARIANT_SIZE, _INVARIANT_SIZE])
    size = ndd.compile.invariant([4, 4])
    return ndd.resize(images, size=size)


@compiled_test(expect_captured=True)
def test_invariant_marker_without_source(images):
    source = """
def transform(images, angle):
    return ndd.rotate(images, angle=angle)
"""
    namespace = {"ndd": ndd}
    exec(compile(source, "<source-unavailable>", "exec"), namespace)
    return namespace["transform"](images, _INVARIANT_ANGLE)


# Tests for rejected cases


@compiled_test(expect_captured=False)
def test_branch_rebind(images):
    if images is not None:
        c = 60
    else:
        c = 90
    return ndd.rotate(images, angle=c)


@compiled_test(expect_captured=False)
def test_nonlocal_rebind(images):
    def rebind():
        nonlocal c
        c = 60

    c = 42
    rebind()
    return ndd.rotate(images, angle=c)


@compiled_test(expect_captured=False)
def test_global_name(images):
    return ndd.rotate(images, angle=_GLOBAL_ANGLE)


@compiled_test(expect_captured=False)
def test_mutable_name(images):
    c = [224, 224]  # named mutables are not captured
    return ndd.resize(images, size=c)


@compiled_test(expect_captured=False)
def test_rhs_unpack(images):
    x = [60]
    a, _ = *x, 1  # RHS itself unpacks, so the bound length is not statically known
    return ndd.rotate(images, angle=a)


@compiled_test(expect_captured=False)
def test_starred_target(images):
    _, *b = 0, 64, 64  # b is the *star target, a runtime list
    return ndd.resize(images, size=b)


@compiled_test(expect_captured=False)
def test_for_target(images):
    for c in (0, 60, 90):
        return ndd.rotate(images, angle=c)
    assert False


@compiled_test(expect_captured=False)
def test_augassign(images):
    c = 60
    image = ndd.rotate(images, angle=c)
    # In this specific case, c is not accessed again so it could theoretically be captured,
    # this needs to be proven statically. This requires data-flow analysis.
    c += 5
    return image


@compiled_test(expect_captured=False)
def test_call_result(images):
    c = abs(-60)  # We don't resolve calls yet
    return ndd.rotate(images, angle=c)


@compiled_test(expect_captured=False)
def test_global_dali_attribute(images):
    return ndd.cast(images, dtype=_MODULE_DTYPE)


@compiled_test(expect_captured=False)
def test_non_dali_class_chain(images):
    return ndd.cast(images, dtype=_Cfg.DTYPE)


@compiled_test(expect_captured=False)
def test_user_rooted_dali_chain(images):
    return ndd.cast(images, dtype=_DaliHolder.pkg.types.DALIDataType.FLOAT)


@compiled_test(expect_captured=False)
def test_import_name(images):
    from math import pi

    return ndd.rotate(images, angle=pi)


@compiled_test(expect_captured=False)
def test_param_mutable(images):
    def resize(imgs, size):
        size[1] = 42
        return ndd.resize(imgs, size=size)

    return resize(images, [224, 224])


@compiled_test(expect_captured=False)
def test_param_varargs(images):
    def resize(imgs, *size):
        return ndd.resize(imgs, size=size)

    return resize(images, 224, 224)


@compiled_test(expect_captured=False)
def test_default_mutable(images):
    def resize(imgs, size=[224, 224]):  # noqa: B006
        return ndd.resize(imgs, size=size)

    return resize(images)


@compiled_test(expect_captured=False)
def test_closure_mutable_cell(images):
    def make_resize(size):
        def resize():
            return ndd.resize(images, size=size)

        return resize

    return make_resize([224, 224])()


@compiled_test(expect_captured=False)
def test_closure_nonlocal_rebind(images):
    def make_rotate():
        angle = 60

        def rebind():
            nonlocal angle
            angle = 90

        def rotate():
            return ndd.rotate(images, angle=angle)

        rebind()
        return rotate

    return make_rotate()()


@compiled_test(expect_captured=False)
def test_param_through_decorator(images):
    def rotate(imgs, angle):
        return ndd.rotate(imgs, angle=angle)

    @functools.wraps(rotate)
    def wrapped(*args, **kwargs):
        return rotate(*args, **kwargs)

    return wrapped(images, 60)


@compiled_test(expect_captured=False)
def test_param_inline_call_target(images):
    def make_rotate():
        def rotate(imgs, angle):
            return ndd.rotate(imgs, angle=angle)

        return rotate

    return make_rotate()(images, 60)


@compiled_test(expect_captured=False)
def test_param_callable_instance_attr(images):
    class Aug:
        def __call__(self, imgs, angle):
            return ndd.rotate(imgs, angle=angle)

    class Holder:
        aug = Aug()

    return Holder.aug(images, 60)


@compiled_test(expect_captured=False)
def test_param_global_arg(images):
    def augment(imgs, crop):
        return ndd.rotate(imgs, angle=crop)

    return augment(images, _GLOBAL_ANGLE)


@compiled_test(expect_captured=False)
def test_invariant_marker_with_global(images):
    angle = ndd.compile.invariant(0.0) + _GLOBAL_ANGLE
    return ndd.rotate(images, angle=angle)


def test_invariant_marker_removed():
    es = ndd.ExternalSource(lambda: ndd.zeros(batch_size=2, shape=(4, 4, 3), layout="HWC"))
    angles = iter((ndd.compile.invariant(0.0), 0.0))

    with assert_raises(RuntimeError, glob="marked with ndd.compile.invariant*remain marked"):
        for images in es.compiled(batch_size=2):
            ndd.rotate(images, angle=next(angles))

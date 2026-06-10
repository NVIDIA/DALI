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
import os
from collections.abc import Callable

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
import nvidia.dali.types
from ndd_utils import _is_compiled
from nvidia.dali.types import DALIDataType
from test_utils import get_dali_extra_path

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


# Module-level fixtures for the rejection tests
#
_MODULE_DTYPE = ndd.float32
_GLOBAL_ANGLE = 60


class _Cfg:
    DTYPE = ndd.float32


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

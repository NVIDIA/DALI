# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

from nose_utils import raises

from nvidia.dali import pipeline_def, fn
from nvidia.dali.pipeline import do_not_convert
from nvidia.dali._autograph import is_autograph_artifact


def assert_autograph_artifact(artifact_name, expected):
    if is_autograph_artifact(artifact_name) != expected:
        raise AssertionError(f"Expected {artifact_name} to be {expected}")


def es_with_local_source(parallel=False):
    """Runs a pipeline that has a locally defined source function for external source"""

    @pipeline_def(
        batch_size=4,
        num_threads=1,
        device_id=None,
        enable_conditionals=True,
        py_start_method="spawn",
    )
    def pipe_with_local():
        def source_local(si):
            assert_autograph_artifact(source_local, True)
            return np.full((2,), 1)

        return fn.external_source(source=source_local, parallel=parallel, batch=False)

    p = pipe_with_local()
    (out,) = p.run()
    assert np.array_equal(np.array(out.as_tensor()), np.full((4, 2), 1))


def es_with_nonlocal_converted_source(parallel=False):
    """Runs a pipeline that has a source function created by a factory function, defined
    out of scope of the pipeline, it is not marked with @do_not_convert - so due to the
    converted_call being made, the source_in_factory is also converted by AutoGraph.
    """

    def source_factory(size):
        def source_in_factory(si):
            assert_autograph_artifact(source_in_factory, True)
            return np.full(size, 10)

        return source_in_factory

    @pipeline_def(
        batch_size=4,
        num_threads=1,
        device_id=None,
        enable_conditionals=True,
        py_start_method="spawn",
    )
    def pipe_with_converted_factory():
        source = source_factory((3,))
        assert_autograph_artifact(source, True)
        return fn.external_source(source=source, parallel=parallel, batch=False)

    p = pipe_with_converted_factory()
    (out,) = p.run()
    assert np.array_equal(np.array(out.as_tensor()), np.full((4, 3), 10))


def es_with_nonlocal_not_converted_source(parallel=False):
    """Runs a pipeline that has a source function created by a factory function, defined
    out of scope of the pipeline, marked with @do_not_convert - so the source_in_factory is run
    without AutoGraph.
    """

    @do_not_convert
    def source_factory(size):
        def source_in_factory(si):
            assert_autograph_artifact(source_in_factory, False)
            return np.full(size, 10)

        return source_in_factory

    @pipeline_def(
        batch_size=4,
        num_threads=1,
        device_id=None,
        enable_conditionals=True,
        py_start_method="spawn",
    )
    def pipe_with_converted_factory():
        source = source_factory((3,))
        assert_autograph_artifact(source, False)
        return fn.external_source(source=source, parallel=parallel, batch=False)

    p = pipe_with_converted_factory()
    (out,) = p.run()
    assert np.array_equal(np.array(out.as_tensor()), np.full((4, 3), 10))


# Sanity test: check if indeed those source callbacks are converted to AutoGraph artifacts
def test_es_with_converted_sources():
    es_with_local_source()
    es_with_nonlocal_converted_source()


@raises(
    ValueError,
    "To allow the `source` to be correctly used*, it must remain unconverted",
)
def test_parallel_es_with_local_source():
    es_with_local_source(parallel=True)


@raises(
    ValueError,
    "To allow the `source` to be correctly used*, it must remain unconverted",
)
def test_parallel_es_with_nonlocal_source():
    es_with_nonlocal_converted_source(parallel=True)


def test_es_with_not_converted_source():
    es_with_nonlocal_not_converted_source()


def test_parallel_es_with_not_converted_callback():
    es_with_nonlocal_not_converted_source(True)

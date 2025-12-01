# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copyreg
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.pickling as dali_pickle
import os
from nvidia.dali import pipeline_def
from pickle import PicklingError

from nose_utils import raises
from test_utils import get_dali_extra_path, restrict_python_version

tests_dali_pickling = []

tests_dill_pickling = []

tests_cloudpickle_pickling = []


def register_case(suite):
    def decorator(test_case):
        suite.append(test_case)
        return test_case

    return decorator


def _simple_callback(sample_info):
    return np.full((5, 6), sample_info.idx_in_epoch, dtype=np.int32)


@dali_pickle.pickle_by_value
def _simple_callback_by_value(sample_info):
    return np.full((5, 6), sample_info.idx_in_epoch, dtype=np.int32)


def callback_const_42(sample_info):
    return np.full((10, 20), 42, dtype=np.uint8)


def callback_const_84(sample_info):
    return np.full((10, 20), 84, dtype=np.uint8)


def standard_global_callback(sample_info):
    return np.full((10, 20), sample_info.idx_in_batch, dtype=np.uint8)


def callback_idx(i):
    return np.full((10, 20), i, dtype=np.uint8)


@dali_pickle.pickle_by_value
def callback_idx_by_value(i):
    return np.full((10, 20), i, dtype=np.uint8)


def dumps(obj, **kwargs):
    if kwargs.get("special_dumps_param") != 42:
        raise ValueError("Expected special_dumps_param among kwargs, got {}".format(kwargs))
    return dali_pickle._DaliPickle.dumps(obj)


def loads(data, **kwargs):
    obj = dali_pickle._DaliPickle.loads(data)
    if kwargs.get("special_loads_param") == 84:
        return obj if obj.__name__ != "callback_const_84" else callback_const_42
    return obj


# Register dummy reducer for custom type to check if DALI pickler did not interfere with
# ability to register custom reducers for user defined types
class DummyCb:
    def __call__(self, sample_info):
        return np.int32([1])


class DummyCb42:
    def __call__(self, sample_info):
        return np.int32([42])


def crazy_reducer(obj):
    return DummyCb42().__reduce__()


copyreg.pickle(DummyCb, crazy_reducer)

# use process pid to assert that arrays were passed by value
# and are equal for externalsource run in the same and separate process
global_numpy_arrays = [np.full((10, 10), os.getpid() + i) for i in range(30)]


def create_closure_callback_numpy(shape, data_set_size):
    # use process pid to assert that arrays were passed by value
    # and are equal for externalsource run in the same and separate process
    data = [np.full(shape, os.getpid()) for _ in range(data_set_size)]

    def callback(sample_info):
        if sample_info.idx_in_epoch >= data_set_size:
            raise StopIteration
        return data[sample_info.idx_in_epoch]

    return callback


def create_closure_callback_img_reader(data_set_size):
    data_root = get_dali_extra_path()
    images_dir = os.path.join(data_root, "db", "single", "jpeg")

    with open(os.path.join(images_dir, "image_list.txt"), "r") as f:
        file_label = [line.rstrip().split(" ") for line in f if line != ""]
        files, labels = zip(*file_label)

    def py_file_reader(sample_info):
        if sample_info.idx_in_epoch >= data_set_size:
            raise StopIteration
        sample_idx = sample_info.idx_in_epoch % len(files)
        jpeg_filename = files[sample_idx]
        label = np.int32([labels[sample_idx]])
        with open(os.path.join(images_dir, jpeg_filename), "rb") as f:
            encoded_img = np.frombuffer(f.read(), dtype=np.uint8)
        return encoded_img, label

    return py_file_reader


def create_closure_generator_img_reader(batch_size, data_set_size):
    data_root = get_dali_extra_path()
    images_dir = os.path.join(data_root, "db", "single", "jpeg")

    with open(os.path.join(images_dir, "image_list.txt"), "r") as f:
        file_label = [line.rstrip().split(" ") for line in f if line != ""]
        files, labels = zip(*file_label)

    def py_file_gen_reader():
        i = 0
        while i + batch_size <= data_set_size:
            batch_imgs, batch_labels = [], []
            for _ in range(batch_size):
                jpeg_filename = files[i]
                with open(os.path.join(images_dir, jpeg_filename), "rb") as f:
                    batch_imgs.append(np.frombuffer(f.read(), dtype=np.uint8))
                batch_labels.append(np.int32([labels[i]]))
                i += 1
            yield batch_imgs, batch_labels

    return py_file_gen_reader


jpeg_file = os.path.join(
    get_dali_extra_path(), "db", "single", "jpeg", "510", "ship-1083562_640.jpg"
)
sequence_lenght = 4


def create_callback_with_syntactically_nested_code_referencing_global_var():
    def cb_from_closure():
        return 0

    def get_data(sample_info):
        def nested_in_cb():
            def super_nested():
                return np.fromfile(jpeg_file, dtype=np.uint8) + cb_from_closure()

            return super_nested

        return nested_in_cb()()

    return get_data


def create_callback_with_list_comprehension_referencing_global_var():
    def get_data(sample_info):
        return [
            x
            for xs in [
                [np.fromfile(jpeg_file, dtype=np.uint8) for _ in range(sequence_lenght)]
                for i in range(2)
            ]
            for x in xs
        ]

    return get_data


def create_simple_pipeline(
    callback,
    py_callback_pickler,
    batch_size,
    parallel=True,
    py_num_workers=None,
    py_start_method="spawn",
):
    extra = {}
    if parallel:
        extra["py_num_workers"] = py_num_workers
        extra["py_start_method"] = py_start_method
        if py_callback_pickler is not None:
            extra["py_callback_pickler"] = py_callback_pickler

    @pipeline_def(batch_size=batch_size, num_threads=2, device_id=0, **extra)
    def create_pipline():
        outputs = fn.external_source(source=callback, batch=False, parallel=parallel)
        return outputs

    return create_pipline()


def create_stacking_pipeline(
    callback,
    py_callback_pickler,
    batch_size,
    parallel=True,
    py_num_workers=None,
    py_start_method="spawn",
):
    extra = {}
    if parallel:
        extra["py_num_workers"] = py_num_workers
        extra["py_start_method"] = py_start_method
        if py_callback_pickler is not None:
            extra["py_callback_pickler"] = py_callback_pickler

    @pipeline_def(batch_size=batch_size, num_threads=2, device_id=0, **extra)
    def create_pipline():
        jpegs = fn.external_source(
            source=callback, num_outputs=sequence_lenght * 2, parallel=parallel, batch=False
        )
        images = fn.decoders.image(jpegs, device="cpu")
        sequence = fn.stack(*images)
        sequence = fn.reshape(sequence, layout="DHWC")
        return sequence

    return create_pipline()


def create_decoding_pipeline(
    callback,
    py_callback_pickler,
    batch_size,
    parallel=True,
    py_num_workers=None,
    py_start_method="spawn",
    batch=False,
):
    extra = {}
    if parallel:
        extra["py_num_workers"] = py_num_workers
        extra["py_start_method"] = py_start_method
        if py_callback_pickler is not None:
            extra["py_callback_pickler"] = py_callback_pickler

    @pipeline_def(batch_size=batch_size, num_threads=2, device_id=0, **extra)
    def create_pipline():
        jpegs, labels = fn.external_source(
            source=callback, num_outputs=2, batch=batch, parallel=parallel
        )
        images = fn.decoders.image(jpegs, device="cpu")
        return images, labels

    return create_pipline()


def _run_and_compare_outputs(batch_size, parallel_pipeline, serial_pipeline):
    parallel_batch = parallel_pipeline.run()
    serial_batch = serial_pipeline.run()
    for parallel_output, serial_output in zip(parallel_batch, serial_batch):
        assert len(parallel_output) == batch_size
        assert len(serial_output) == batch_size
        for i in range(batch_size):
            assert np.array_equal(parallel_output[i], serial_output[i])


def _build_and_compare_pipelines_epochs(epochs_num, batch_size, parallel_pipeline, serial_pipeline):
    parallel_pipeline.build()
    serial_pipeline.build()
    assert parallel_pipeline._py_pool is not None
    assert serial_pipeline._py_pool is None
    for _ in range(epochs_num):
        try:
            while True:
                _run_and_compare_outputs(batch_size, parallel_pipeline, serial_pipeline)
        except StopIteration:
            parallel_pipeline.reset()
            serial_pipeline.reset()


def _create_and_compare_simple_pipelines(
    cb, py_callback_pickler, batch_size, py_num_workers=2, py_start_method="spawn"
):
    parallel_pipeline = create_simple_pipeline(
        cb,
        py_callback_pickler,
        batch_size=batch_size,
        py_num_workers=py_num_workers,
        py_start_method=py_start_method,
        parallel=True,
    )
    serial_pipeline = create_simple_pipeline(cb, None, batch_size=batch_size, parallel=False)
    parallel_pipeline.build()
    serial_pipeline.build()
    for _ in range(3):
        _run_and_compare_outputs(batch_size, parallel_pipeline, serial_pipeline)


# It uses fork method to start so need to be run as the first test
def test_no_pickling_in_forking_mode():
    # modify callback name so that an attempt to pickle it in spawn mode would fail
    _simple_callback.__name__ = _simple_callback.__qualname__ = "simple_callback"
    _create_and_compare_simple_pipelines(
        _simple_callback, None, batch_size=8, py_num_workers=2, py_start_method="fork"
    )


# Run this one as sanity check that standard serialization is not broken by the change
def test_standard_global_function_serialization():
    _create_and_compare_simple_pipelines(
        standard_global_callback, None, batch_size=4, py_num_workers=2
    )


def test_if_custom_type_reducers_are_respected_by_dali_reducer():
    batch_size = 8
    parallel_pipeline = create_simple_pipeline(
        DummyCb(), None, batch_size=batch_size, py_num_workers=2, parallel=True
    )
    parallel_pipeline.build()
    (batch,) = parallel_pipeline.run()
    assert len(batch) == batch_size
    for i in range(batch_size):
        assert np.array_equal(batch[i], np.int32([42]))


@register_case(tests_dali_pickling)
@raises(PicklingError, "Can't pickle * attribute lookup simple_callback on * failed")
def _test_global_function_pickled_by_reference(name, py_callback_pickler):
    # modify callback name so that an attempt to pickle by reference,
    # which is default Python behavior, fails
    _simple_callback.__name__ = _simple_callback.__qualname__ = "simple_callback"
    _create_and_compare_simple_pipelines(
        _simple_callback, py_callback_pickler, batch_size=4, py_num_workers=2
    )


@register_case(tests_dali_pickling)
def _test_pickle_by_value_decorator_on_global_function(name, py_callback_pickler):
    # modify callback name so that an attempt to pickle by reference,
    # which is default Python behavior, would fail
    _simple_callback_by_value.__name__ = _simple_callback_by_value.__qualname__ = (
        "simple_callback_by_value"
    )
    _create_and_compare_simple_pipelines(
        _simple_callback_by_value, py_callback_pickler, batch_size=4, py_num_workers=2
    )


@register_case(tests_dali_pickling)
@raises(ValueError, "Expected special_dumps_param among kwargs, got *")
def _test_pickle_does_not_pass_extra_params_function(name, py_callback_pickler):
    this_module = __import__(__name__)
    _create_and_compare_simple_pipelines(
        callback_const_42, this_module, batch_size=4, py_num_workers=2
    )


@register_case(tests_dali_pickling)
def _test_pickle_passes_extra_dumps_params_function(name, py_callback_pickler):
    this_module = __import__(__name__)
    _create_and_compare_simple_pipelines(
        callback_const_42,
        (this_module, {"special_dumps_param": 42}),
        batch_size=4,
        py_num_workers=2,
    )


@register_case(tests_dali_pickling)
def _test_pickle_passes_extra_dumps_loads_params_function(name, py_callback_pickler):
    this_module = __import__(__name__)
    batch_size = 4
    # this_module.loads replaces callback_const_84 to callback_const_42
    # iff it receives special_loads_param
    parallel_pipeline = create_simple_pipeline(
        callback_const_84,
        (this_module, {"special_dumps_param": 42}, {"special_loads_param": 84}),
        batch_size=batch_size,
        py_num_workers=2,
        parallel=True,
    )
    serial_pipeline = create_simple_pipeline(
        callback_const_42, None, batch_size=batch_size, parallel=False
    )
    parallel_pipeline.build()
    serial_pipeline.build()
    for _ in range(3):
        _run_and_compare_outputs(batch_size, parallel_pipeline, serial_pipeline)


@register_case(tests_dali_pickling)
def _test_global_function_wrapped_in_lambda_by_value(name, py_callback_pickler):
    # modify callback name so that an attempt to pickle by reference,
    # which is default Python behavior, would fail
    callback_idx_by_value.__name__ = callback_idx_by_value.__qualname__ = "_scrambled_name"
    _create_and_compare_simple_pipelines(
        lambda x: callback_idx_by_value(x.idx_in_epoch),
        py_callback_pickler,
        batch_size=8,
        py_num_workers=2,
    )


@register_case(tests_dali_pickling)
@register_case(tests_dill_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_lambda_np_full(name, py_callback_pickler):
    _create_and_compare_simple_pipelines(
        lambda x: np.full((100, 100), x.idx_in_epoch),
        py_callback_pickler,
        batch_size=8,
        py_num_workers=2,
    )


@register_case(tests_dali_pickling)
@register_case(tests_dill_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_lambda_np_readfromfile(name, py_callback_pickler):
    data_root = get_dali_extra_path()
    images_dir = os.path.join(data_root, "db", "single", "jpeg")

    with open(os.path.join(images_dir, "image_list.txt"), "r") as f:
        file_label = [line.rstrip().split(" ") for line in f if line != ""]
        files, _ = zip(*file_label)

    _create_and_compare_simple_pipelines(
        lambda x: (
            np.fromfile(
                os.path.join(images_dir, files[x.idx_in_epoch % len(files)]), dtype=np.uint8
            )
        ),
        py_callback_pickler,
        batch_size=8,
        py_num_workers=2,
    )


@register_case(tests_dali_pickling)
@register_case(tests_dill_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_serialization_of_globals_from_code_nested_in_cb(name, py_callback_pickler):
    _create_and_compare_simple_pipelines(
        create_callback_with_syntactically_nested_code_referencing_global_var(),
        py_callback_pickler,
        batch_size=8,
        py_num_workers=2,
    )


@register_case(tests_dali_pickling)
@register_case(tests_dill_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_serialization_of_globals_referenced_in_list_comprehension_in_cb(
    name, py_callback_pickler
):
    batch_size = 10

    parallel_pipeline = create_stacking_pipeline(
        create_callback_with_list_comprehension_referencing_global_var(),
        py_callback_pickler,
        batch_size=batch_size,
        py_num_workers=2,
        parallel=True,
    )
    serial_pipeline = create_stacking_pipeline(
        create_callback_with_list_comprehension_referencing_global_var(),
        None,
        batch_size=batch_size,
        parallel=False,
    )
    parallel_pipeline.build()
    serial_pipeline.build()
    for _ in range(3):
        _run_and_compare_outputs(batch_size, parallel_pipeline, serial_pipeline)


@register_case(tests_dali_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_mutually_recursive_functions(name, py_callback_pickler):
    def div_by_2(n, acc=0):
        return acc if n <= 0 else add_one(n // 2, acc)

    add_one = lambda n, acc: div_by_2(n, acc + 1)  # noqa: E731
    _create_and_compare_simple_pipelines(
        lambda x: np.int32([div_by_2(x.idx_in_epoch)]),
        py_callback_pickler,
        batch_size=15,
        py_num_workers=2,
    )


@register_case(tests_dali_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_builtin_functions_usage_in_cb(name, py_callback_pickler):
    def div_by_2(n, acc=0):
        return acc if n <= 0 else add_one(n // 2, acc)

    add_one = lambda n, acc: div_by_2(n, acc + 1)  # noqa: E731
    _create_and_compare_simple_pipelines(
        lambda x: np.int32([div_by_2(x.idx_in_epoch)]) + len(dir(np)),
        py_callback_pickler,
        batch_size=15,
        py_num_workers=2,
    )


@register_case(tests_dali_pickling)
@register_case(tests_dill_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_module_dependency(name, py_callback_pickler):
    import import_module_test_helper

    _create_and_compare_simple_pipelines(
        lambda x: import_module_test_helper.cb(x),
        py_callback_pickler,
        batch_size=15,
        py_num_workers=2,
    )


@register_case(tests_dali_pickling)
@register_case(tests_dill_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_module_dependency_unqualified(name, py_callback_pickler):
    from import_module_test_helper import cb

    _create_and_compare_simple_pipelines(
        lambda x: cb(x), py_callback_pickler, batch_size=15, py_num_workers=2
    )


@register_case(tests_dali_pickling)
@register_case(tests_dill_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_module_dependency_by_reference(name, py_callback_pickler):
    from import_module_test_helper import cb

    _create_and_compare_simple_pipelines(cb, py_callback_pickler, batch_size=15, py_num_workers=2)


@register_case(tests_dali_pickling)
@register_case(tests_dill_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_accessing_global_np_list(name, py_callback_pickler):
    _create_and_compare_simple_pipelines(
        lambda x: global_numpy_arrays[x.idx_in_epoch % len(global_numpy_arrays)],
        py_callback_pickler,
        batch_size=9,
        py_num_workers=2,
    )


def __test_numpy_closure(shape, py_callback_pickler):
    batch_size = 8
    epochs_num = 3
    callback = create_closure_callback_numpy(shape, data_set_size=epochs_num * batch_size)
    parallel_pipeline = create_simple_pipeline(
        callback, py_callback_pickler, batch_size=batch_size, py_num_workers=2, parallel=True
    )
    serial_pipeline = create_simple_pipeline(callback, None, batch_size=batch_size, parallel=False)
    _build_and_compare_pipelines_epochs(epochs_num, batch_size, parallel_pipeline, serial_pipeline)


@register_case(tests_dali_pickling)
@register_case(tests_dill_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_numpy_closure(name, py_callback_pickler):
    for shape in [
        tuple(),
        (
            5,
            5,
            5,
        ),
    ]:
        yield __test_numpy_closure, shape, py_callback_pickler


@register_case(tests_dali_pickling)
@register_case(tests_dill_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_reader_closure(name, py_callback_pickler):
    batch_size = 7
    batches_in_epoch = 3
    epochs_num = 3
    callback = create_closure_callback_img_reader(data_set_size=batches_in_epoch * batch_size)
    parallel_pipeline = create_decoding_pipeline(
        callback, py_callback_pickler, batch_size=batch_size, py_num_workers=2, parallel=True
    )
    serial_pipeline = create_decoding_pipeline(
        callback, None, batch_size=batch_size, parallel=False
    )
    _build_and_compare_pipelines_epochs(epochs_num, batch_size, parallel_pipeline, serial_pipeline)


@register_case(tests_dali_pickling)
@register_case(tests_dill_pickling)
@register_case(tests_cloudpickle_pickling)
def _test_generator_closure(name, py_callback_pickler):
    batch_size = 7
    batches_in_epoch = 3
    epochs_num = 3
    callback = create_closure_generator_img_reader(
        batch_size=batch_size, data_set_size=batches_in_epoch * batch_size
    )
    parallel_pipeline = create_decoding_pipeline(
        callback,
        py_callback_pickler,
        batch_size=batch_size,
        py_num_workers=1,
        parallel=True,
        batch=True,
    )
    serial_pipeline = create_decoding_pipeline(
        callback, None, batch_size=batch_size, parallel=False, batch=True
    )
    _build_and_compare_pipelines_epochs(epochs_num, batch_size, parallel_pipeline, serial_pipeline)


@restrict_python_version(3, 8)
def test_dali_pickling():
    for i, test in enumerate(tests_dali_pickling, start=1):
        yield test, "{}. {}".format(i, test.__name__.strip("_")), None


def test_cloudpickle_pickling():
    import cloudpickle

    for i, test in enumerate(tests_cloudpickle_pickling, start=1):
        yield test, "{}. {}".format(i, test.__name__.strip("_")), cloudpickle


def test_dill_pickling():
    import dill

    for i, test in enumerate(tests_dill_pickling, start=1):
        yield test, "{}. {}".format(i, test.__name__.strip("_")), (dill, {"recurse": True})

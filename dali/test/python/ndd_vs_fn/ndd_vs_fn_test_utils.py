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

from dataclasses import dataclass, field
import functools
import random
import numpy as np
from nvidia.dali.experimental.dynamic._batch import Batch
from nvidia.dali.backend_impl import TensorListCPU, TensorListGPU
import nvidia.dali.fn as fn
import nvidia.dali.experimental.dynamic as ndd
import inspect
from nvidia.dali.pipeline import pipeline_def
import test_utils

MAX_BATCH_SIZE = 31
N_ITERATIONS = 13
RNG_SEED = 5318008


@dataclass
class OperatorTestConfig:
    """Configuration for a single operator test."""

    name: str
    args: dict = field(default_factory=dict)
    devices: list[str] = field(default_factory=lambda: ["cpu", "gpu"])

    def generate_test_tuples(self):
        """Generate (device, fn_operator, ndd_operator, args) tuples for this config."""
        return [
            (device, get_fn_operator(self.name), get_ndd_operator(self.name), self.args)
            for device in self.devices
        ]


def use_fn_api(func):
    """Decorator that injects 'fn' as the api parameter.

    This decorator allows writing operator functions that are API-agnostic by automatically
    injecting the nvidia.dali.fn module as the first parameter. This is useful for testing
    operators with both the fn API and ndd API using the same function definition.

    Args:
        func: A function that expects an API module (fn or ndd) as its first parameter,
              followed by input tensors and operator arguments.

    Returns:
        A function that can be decorated with @pipeline_def.

    Usage:
        Define an operation that uses the 'api' parameter instead of directly calling fn:

        >>> def operation(api, *inp, **kwargs):
        ...     out = api.expand_dims(*inp, axes=[0, 2])
        ...     out = api.squeeze(out, axis_names="Z")
        ...     return out

        Use the decorator to create an fn-based version:

        >>> fn_operation = use_fn_api(operation)
        >>> # Now fn_operation will automatically inject fn as the api parameter
        >>> # Can be used in pipeline definitions:
        >>> pipe = pipeline_es_feed_input_wrapper(fn_operation, device="gpu")

        For ndd testing, use the companion decorator use_ndd_api:

        >>> ndd_operation = use_ndd_api(operation)
        >>> ndd_out = ndd_operation(ndd.as_batch(inp, device="gpu"))

    See also:
        use_ndd_api: Companion decorator that injects ndd instead of fn
    """

    @functools.wraps(func)
    def wrapper(*inp, **operator_args):
        return func(fn, *inp, **operator_args)

    return wrapper


def use_ndd_api(func):
    """Decorator that injects 'ndd' as the api parameter.

    This decorator allows writing operator functions that are API-agnostic by automatically
    injecting the nvidia.dali.experimental.dynamic (ndd) module as the first parameter.
    This is the companion decorator to use_fn_api and is used for testing ndd operations.

    Args:
        func: A function that expects an API module (fn or ndd) as its first parameter,
              followed by input tensors and operator arguments.

    Returns:
        A function that can be run as a standalone function with ndd API.

    Usage:
        >>> def operation(api, *inp, **kwargs):
        ...     mean = api.reductions.mean(*inp)
        ...     reduced = api.reductions.std_dev(*inp, mean)
        ...     return reduced

        Use the decorator to create an ndd-based version:

        >>> ndd_operation = use_ndd_api(operation)
        >>> ndd_out = ndd_operation(ndd.as_batch(inp, device="gpu"))

    See also:
        use_fn_api: Companion decorator that injects fn instead of ndd
    """

    @functools.wraps(func)
    def wrapper(*inp, **operator_args):
        return func(ndd, *inp, **operator_args)

    return wrapper


def image_like_shape_generator():
    return random.randint(160, 161), random.randint(80, 81), 3


def array_1d_shape_generator():
    return (random.randint(300, 400),)  # The comma is important


def custom_shape_generator(*args):
    """
    Fully configurable shape generator.
    Returns a callable which serves as a non-uniform & random shape generator to generate_epoch

    Usage:
    custom_shape_generator(dim1_lo, dim1_hi, dim2_lo, dim2_hi, etc...)
    """
    assert len(args) % 2 == 0, "Incorrect number of arguments"
    ndims = len(args) // 2
    gen_conf = [[args[2 * i], args[2 * i + 1]] for i in range(ndims)]
    return lambda: tuple([random.randint(lohi[0], lohi[1]) for lohi in gen_conf])


def generate_data(
    sample_shape,
    max_batch_size=MAX_BATCH_SIZE,
    n_iter=N_ITERATIONS,
    lo=0.0,
    hi=1.0,
    dtype=np.float32,
    batch_sizes=None,
):
    """
    Generates an epoch of data, that will be used for variable batch size verification.

    :param max_batch_size: Actual sizes of every batch in the epoch will be less or equal
                           to max_batch_size
    :param n_iter: Number of iterations in the epoch
    :param sample_shape: If sample_shape is callable, shape of every sample will be determined by
                         calling sample_shape. In this case, every call to sample_shape has to
                         return a tuple of integers. If sample_shape is a tuple, this will be a
                         shape of every sample.
    :param lo: Begin of the random range
    :param hi: End of the random range
    :param dtype: Numpy data type
    :return: An epoch of data
    """
    if batch_sizes is None:
        batch_sizes = np.array([max_batch_size // 2, max_batch_size // 4, max_batch_size])
    elif isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes] * 3

    if isinstance(sample_shape, tuple):

        def sample_shape_wrapper():
            return sample_shape

        size_fn = sample_shape_wrapper
    elif inspect.isfunction(sample_shape):
        size_fn = sample_shape
    else:
        raise RuntimeError(
            "`sample_shape` shall be either a tuple or a callable. "
            "Provide `(val,)` tuple for 1D shape"
        )

    if np.issubdtype(dtype, np.integer):
        return [
            np.random.randint(lo, hi, size=(bs,) + size_fn(), dtype=dtype) for bs in batch_sizes
        ]
    elif np.issubdtype(dtype, np.float32):
        ret = (np.random.random_sample(size=(bs,) + size_fn()) for bs in batch_sizes)
        ret = map(lambda batch: (hi - lo) * batch + lo, ret)
        ret = map(lambda batch: batch.astype(dtype), ret)
        return list(ret)
    elif np.issubdtype(dtype, bool):
        assert isinstance(lo, bool)
        assert isinstance(hi, bool)
        return [np.random.choice(a=[lo, hi], size=(bs,) + size_fn()) for bs in batch_sizes]
    else:
        raise RuntimeError(f"Invalid type argument: {dtype}")
    

def generate_decoders_data(data_dir, data_extension, exclude_subdirs=[]):
    # File reader won't work, so I need to load audio files into external_source manually
    fnames = test_utils.filter_files(data_dir, data_extension, exclude_subdirs=exclude_subdirs)

    nfiles = len(fnames)
    # TODO(janton): Workaround for audio data (not enough samples)
    #               To be removed when more audio samples are added
    for i in range(len(fnames), 10):  # At least 10 elements
        fnames.append(fnames[-1])
    nfiles = len(fnames)
    _input_epoch = [
        list(map(lambda fname: test_utils.read_file_bin(fname), fnames[: nfiles // 3])),
        list(
            map(
                lambda fname: test_utils.read_file_bin(fname),
                fnames[nfiles // 3 : nfiles // 2],
            )
        ),
        list(map(lambda fname: test_utils.read_file_bin(fname), fnames[nfiles // 2 :])),
    ]

    # Since we pack buffers into ndarray, we need to pad samples with 0.
    input_epoch = []
    for inp in _input_epoch:
        max_len = max(sample.shape[0] for sample in inp)
        inp = map(lambda sample: np.pad(sample, (0, max_len - sample.shape[0])), inp)
        input_epoch.append(np.stack(list(inp)))
    input_epoch = list(map(lambda batch: np.reshape(batch, batch.shape), input_epoch))

    return input_epoch


def get_batch_size(batch):
    """
    Returns the batch size in samples

    :param batch: List of input batches, if there is one input a batch can be either
                  a numpy array or a list, for multiple inputs it can be tuple of lists or
                  numpy arrays.
    """
    if isinstance(batch, tuple):
        return get_batch_size(batch[0])
    else:
        if isinstance(batch, list):
            return len(batch)
        else:
            return batch.shape[0]


def external_source_of_random_states(rng, batch_size, n_states, **external_source_args):
    """
    Create an external source operator that generates random states.

    Example:
    @pipeline_def(num_threads=4, device_id=0, batch_size=batch_size)
    def rn50_pipeline():
        state_1, state_2, state_3 = external_source_of_random_states(
            rng=rng, batch_size=batch_size, n_states=3, num_outputs=3
        )
        xy = fn.random.uniform(range=[0, 1], shape=2, _random_state=state_1)
        do_mirror = fn.random.coin_flip(probability=0.5, _random_state=state_2)
        size = fn.random.uniform(range=[256, 480], _random_state=state_3)
    """
    source_fun = _random_state_source_factory(rng, batch_size, n_states)
    return fn.external_source(source=source_fun, **external_source_args)


def _random_state_source_factory(rng, batch_size, n_states):
    STATE_SIZE = 7
    STATE_TYPE = np.uint32

    def source_fun():
        states = [
            np.array([rng() for _ in range(STATE_SIZE)], dtype=STATE_TYPE) for _ in range(n_states)
        ]
        out = tuple([state] * batch_size for state in states)
        return tuple(out)

    return source_fun


def create_rngs():
    ndd_rng = ndd.random.RNG(seed=RNG_SEED)
    fn_rng = ndd_rng.clone()
    return fn_rng, ndd_rng


def feed_input(dali_pipeline, input_data):
    """
    Feeds input data to DALI pipeline. Supports multiple inputs.

    :param dali_pipeline: DALI pipeline object
    :param input_data: Input data to be fed. For single input it can be either a numpy array
                       or a list of numpy arrays. For multiple inputs it has to be a tuple
                       of lists or numpy arrays.
    """
    if isinstance(input_data, tuple):
        for i in range(len(input_data)):
            dali_pipeline.feed_input(f"INPUT{i}", input_data[i])
    else:
        dali_pipeline.feed_input("INPUT0", input_data)


def tuple_to_batch_multi_input(multi_input_tuple: tuple, device=None, layout=None) -> tuple[Batch]:
    """
    Converts a tuple of inputs to a tuple of ndd Batch objects.

    :param multi_input_tuple: Tuple of inputs, each input can be either a numpy array
                              or a list of numpy arrays.
    :return: Tuple of ndd Batch objects.
    """
    return tuple(
        ndd.as_batch(input_data, device=device, layout=layout) for input_data in multi_input_tuple
    )


def to_numpy(ndd_batch: Batch) -> np.ndarray:
    return np.array(ndd.as_tensor(ndd_batch.cpu()))


def _cmp(pipe_out: TensorListCPU | TensorListGPU, ndd_out: Batch) -> bool:
    assert (
        len(pipe_out.shape()) == ndd_out.batch_size
    ), f"Batch size mismatch: {len(pipe_out.shape())=} != {ndd_out.batch_size=}"
    for pipe_sample, ndd_sample in zip(pipe_out, ndd_out):
        ndd_numpy = to_numpy(ndd_sample)
        pipe_numpy = np.array(pipe_sample)
        if not np.array_equal(pipe_numpy, ndd_numpy):
            return False
    return True


def compare(
    pipe_out: tuple[TensorListCPU] | tuple[TensorListGPU], ndd_out: tuple[Batch] | Batch
) -> bool:
    if isinstance(ndd_out, tuple):
        assert len(pipe_out) == len(
            ndd_out
        ), f"Number of outputs mismatch: {len(pipe_out)=} != {len(ndd_out)=}"
        for pout, nout in zip(pipe_out, ndd_out):
            if not _cmp(pout.as_cpu(), nout):
                return False
        return True
    else:
        assert (
            len(pipe_out) == 1
        ), f"Expected single output from DALI pipeline, got {len(pipe_out)=}"
        return _cmp(pipe_out[0].as_cpu(), ndd_out)


def compare_no_input(
    pipe_out: tuple[TensorListCPU] | tuple[TensorListGPU], ndd_out: tuple[Batch] | Batch
) -> bool:
    """Comparison function for no-input operators."""
    assert len(pipe_out) == 2, f"Expected two outputs from DALI pipeline, got {len(pipe_out)=}"
    return _cmp(pipe_out[0].as_cpu(), ndd_out)


def pipeline_es_feed_input_wrapper(
    operator_under_test,
    device,
    max_batch_size=MAX_BATCH_SIZE,
    input_layout=None,
    needs_input=True,
    num_inputs=1,
    rng=None,
    **operator_args,
):
    """Creates a DALI pipeline with external source as an input."""
    rs = (
        external_source_of_random_states(
            rng=rng, batch_size=max_batch_size, n_states=1, num_outputs=1
        )[
            0
        ]  # [0], because it's tuple
        if rng is not None
        else None
    )

    @pipeline_def(
        batch_size=max_batch_size,
        device_id=0,
        num_threads=ndd.get_num_threads(),
        prefetch_queue_depth=1,
    )
    def pipe():
        inp = [
            fn.external_source(name=f"INPUT{i}", device=device, layout=input_layout)
            for i in range(num_inputs)
        ]
        if needs_input:
            output = operator_under_test(*inp, device=device, _random_state=rs, **operator_args)
        else:
            output = operator_under_test(device=device, _random_state=rs, **operator_args)
        if needs_input:
            if isinstance(output, list):  # DALI uses list to collect operator multi-output
                return tuple(output[i] for i in range(len(output)))
            else:
                return output
        else:
            # set input as an output to make sure it is not pruned from the graph
            return output, *inp

    p = pipe()
    p.build()
    return p


def generate_image_like_data():
    return generate_data(image_like_shape_generator, lo=0, hi=255, dtype=np.uint8)


def run_operator_test(
    input_epoch,
    fn_operator,
    ndd_operator,
    device,
    operator_args={},
    num_inputs=1,
    input_layout=None,
    compare_fn=compare,
    random=False,
):
    """Generic test runner for operator comparison tests."""

    if random:
        fn_rng, ndd_rng = create_rngs()
    else:
        fn_rng = ndd_rng = None

    # Create a DALI pipeline
    pipe = pipeline_es_feed_input_wrapper(
        fn_operator,
        device,
        num_inputs=num_inputs,
        input_layout=input_layout,
        rng=fn_rng,
        **operator_args,
    )

    # Iterate over input epoch
    for inp in input_epoch:
        # Run the DALI pipeline, collect the output.
        feed_input(pipe, inp)
        pipe_out = pipe.run()

        # Run the dynamic operator, collect the output.
        if random:
            if num_inputs > 1:
                ndd_inp = tuple_to_batch_multi_input(inp, device=device, layout=input_layout)
                ndd_out = ndd_operator(*ndd_inp, rng=ndd_rng, **operator_args)
            else:
                ndd_out = ndd_operator(
                    ndd.as_batch(inp, layout=input_layout, device=device),
                    rng=ndd_rng,
                    **operator_args,
                )
        else:
            if num_inputs > 1:
                ndd_inp = tuple_to_batch_multi_input(inp, device=device, layout=input_layout)
                ndd_out = ndd_operator(*ndd_inp, **operator_args)
            else:
                ndd_out = ndd_operator(
                    ndd.as_batch(inp, layout=input_layout, device=device), **operator_args
                )

        # Compare the outputs.
        assert compare_fn(pipe_out, ndd_out)


def get_nested_attr(obj, attr_path):
    """
    Get nested attribute by splitting attr_path on '.' and calling getattr for each part.

    Example:
        get_nested_attr(fn, "experimental.median_blur")
        returns fn.experimental.median_blur
    """
    parts = attr_path.split(".")
    result = obj
    for part in parts:
        result = getattr(result, part)
    return result


def get_fn_operator(operator_name: str):
    """Get operator from fn module by name, supporting nested attributes."""
    try:
        fn_operator = get_nested_attr(fn, operator_name)
    except AttributeError:
        raise AttributeError(
            f"""Couldn't find operator {operator_name} in fn module.
            Please check the string with operator name specification."""
        )
    return fn_operator


def get_ndd_operator(operator_name: str):
    """Get operator from ndd module by name, supporting nested attributes."""
    try:
        ndd_operator = get_nested_attr(ndd, operator_name)
    except AttributeError:
        raise AttributeError(
            f"""Couldn't find operator {operator_name} in ndd module.
            Please check the string with operator name specification."""
        )
    return ndd_operator


def generate_op_tuple(op_name: str, device: str, operator_args: dict):
    """Generate test configuration tuple for an operator."""
    return (device, get_fn_operator(op_name), get_ndd_operator(op_name), operator_args)


def flatten_operator_configs(configs: list[OperatorTestConfig]) -> list[tuple]:
    """Flatten OperatorTestConfig list into test tuples."""
    result = []
    for config in configs:
        result.extend(config.generate_test_tuples())
    return result

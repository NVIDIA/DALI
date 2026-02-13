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
from nvidia.dali.pipeline import pipeline_def
import test_utils

MAX_BATCH_SIZE = 31
N_ITERATIONS = 3
RNG_SEED = 5318008
random.seed(RNG_SEED)

# This needs to be here because if it lands in `test_ndd_vs_fn_coverage`, which is both a test
# entry point and an import, the sign_off registry was created twice.
sign_off = test_utils.create_sign_off_registry()


@dataclass
class OperatorTestConfig:
    """Configuration for a single operator test."""

    name: str
    args: dict = field(default_factory=dict)

    def generate_test_tuples(self):
        """Generate (device, fn_operator, ndd_operator, args) tuples for this config."""
        fn_op = get_fn_operator(self.name)
        ndd_op = get_ndd_operator(self.name)
        return [
            (device, self.name, fn_op, ndd_op, self.args)
            for device in ndd_op._op_class._supported_backends
        ]


def use_fn_api(func):
    """
    We expect the tested operation to be generalized over `fn` an `ndd` APIs
    by taking the appropriate module as the first argument.
    This and the accompanying function `use_ndd_api` bind the selected API
    before invoking the operation with the same set of arguments during testing.
        Example:
        >>> def operation(api, *inp, **kwargs):
        ...     mean = api.reductions.mean(*inp)
        ...     reduced = api.reductions.std_dev(*inp, mean)
        ...     return reduced
    """

    @functools.wraps(func)
    def wrapper(*inp, **operator_args):
        return func(fn, *inp, **operator_args)

    return wrapper


def use_ndd_api(func):
    """See :func:use_fn_api"""

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


def _default_batch_sizes(max_batch_size):
    return [max_batch_size // 2, max_batch_size // 4, max_batch_size]


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
    :param lo: Begin of the random range for the data values.
    :param hi: End of the random range for the data values.
    :param dtype: Numpy data type
    :param batch_sizes: List of batch sizes to generate data for. If None,
                        the set of default batch sizes will be used. If int, it will be
                        used for each test iteration.
    :return: An epoch of data
    """
    if batch_sizes is None:
        batch_sizes = np.array(_default_batch_sizes(max_batch_size))
    elif isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes] * n_iter

    if isinstance(sample_shape, tuple):

        def size_fn():
            return sample_shape

    elif callable(sample_shape):
        size_fn = sample_shape
    else:
        raise RuntimeError(
            "`sample_shape` shall be either a tuple or a callable. "
            "Provide `(val,)` tuple for 1D shape"
        )

    rng = np.random.default_rng(RNG_SEED)
    if np.issubdtype(dtype, np.integer):
        return [rng.integers(lo, hi, size=(bs,) + size_fn(), dtype=dtype) for bs in batch_sizes]
    elif np.issubdtype(dtype, np.floating):
        ret = (rng.random(size=(bs,) + size_fn()) for bs in batch_sizes)
        ret = map(lambda batch: (hi - lo) * batch + lo, ret)
        ret = map(lambda batch: batch.astype(dtype), ret)
        return list(ret)
    elif np.issubdtype(dtype, bool):
        assert isinstance(lo, bool)
        assert isinstance(hi, bool)
        return [rng.choice(a=[lo, hi], size=(bs,) + size_fn()) for bs in batch_sizes]
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
    input_epoch = [
        list(map(test_utils.read_file_bin, fnames[: nfiles // 3])),
        list(map(test_utils.read_file_bin, fnames[nfiles // 3 : nfiles // 2])),
        list(map(test_utils.read_file_bin, fnames[nfiles // 2 :])),
    ]

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


def external_source_of_random_states(
    rng, max_batch_size, n_states, batch_sizes=None, **external_source_args
):
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
    source_fun = _random_state_source_factory(rng, max_batch_size, batch_sizes, n_states)
    return fn.external_source(source=source_fun, **external_source_args, cycle=True)


def _random_state_source_factory(rng, max_batch_size, batch_sizes, n_states):
    STATE_SIZE = 7
    STATE_TYPE = np.uint32

    if batch_sizes is None:
        batch_sizes = _default_batch_sizes(max_batch_size)

    def source_fun():
        for batch_size in batch_sizes:
            states = [
                np.array([rng() for _ in range(STATE_SIZE)], dtype=STATE_TYPE)
                for _ in range(n_states)
            ]
            out = tuple([state] * batch_size for state in states)
            yield tuple(out)

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


def _cmp_metadata(
    pipe_out: TensorListCPU | TensorListGPU, ndd_out: Batch, output_idx: int = 0
) -> None:
    """
    Compare metadata (shapes, dtypes, layouts, device) between DALI pipeline output and ndd Batch.

    :param pipe_out: DALI pipeline output (TensorListCPU or TensorListGPU)
    :param ndd_out: ndd Batch output
    :param output_idx: Index of the output being compared (for error reporting)
    :raises AssertionError: If metadata doesn't match
    """
    # Validate device
    pipe_is_gpu = isinstance(pipe_out, TensorListGPU)
    ndd_is_gpu = ndd_out.device.device_type == "gpu"
    assert (
        pipe_is_gpu == ndd_is_gpu
    ), f"""[Output {output_idx}] Device mismatch:
  Pipeline: {'gpu' if pipe_is_gpu else 'cpu'}
  NDD:      {ndd_out.device.device_type}"""

    # Validate batch size
    pipe_batch_size = len(pipe_out.shape())
    ndd_batch_size = ndd_out.batch_size
    assert (
        pipe_batch_size == ndd_batch_size
    ), f"""[Output {output_idx}] Batch size mismatch:
  Pipeline: {pipe_batch_size}
  NDD:      {ndd_batch_size}"""

    # Validate data type
    pipe_dtype = pipe_out.dtype
    ndd_dtype = ndd_out.dtype.type_id
    assert (
        pipe_dtype == ndd_dtype
    ), f"""[Output {output_idx}] Data type mismatch:
  Pipeline: {pipe_dtype}
  NDD:      {ndd_dtype}"""

    # Validate layout if available
    pipe_layout = pipe_out.layout() if hasattr(pipe_out, "layout") and pipe_out.layout() else None
    ndd_layout = ndd_out.layout if hasattr(ndd_out, "layout") else None
    assert (
        pipe_layout == ndd_layout
    ), f"""[Output {output_idx}] Layout mismatch:
  Pipeline: {pipe_layout}
  NDD:      {ndd_layout}"""

    # Validate individual sample shapes
    for i, (psh, nddsh) in enumerate(zip(pipe_out.shape(), ndd_out.shape, strict=True)):
        assert (
            psh == nddsh
        ), f"""[Output {output_idx}, Sample {i}] Shape mismatch:
  Pipeline: {psh}
  NDD:      {nddsh}"""


def _cmp_values(
    pipe_out: TensorListCPU | TensorListGPU, ndd_out: Batch, output_idx: int = 0
) -> None:
    for sample_idx, (pipe_sample, ndd_sample) in enumerate(zip(pipe_out, ndd_out)):
        ndd_numpy = to_numpy(ndd_sample)
        pipe_numpy = np.array(pipe_sample)
        assert np.array_equal(
            pipe_numpy, ndd_numpy
        ), f"""[Output {output_idx}, Sample {sample_idx}] Values mismatch between
pipeline and NDD outputs"""


def _cmp(pipe_out: TensorListCPU | TensorListGPU, ndd_out: Batch, output_idx: int = 0) -> None:
    _cmp_metadata(pipe_out, ndd_out, output_idx=output_idx)
    _cmp_values(pipe_out.as_cpu(), ndd_out, output_idx=output_idx)


def compare(
    pipe_out: tuple[TensorListCPU] | tuple[TensorListGPU], ndd_out: tuple[Batch] | Batch
) -> None:
    """
    Compare DALI pipeline outputs with ndd outputs.

    Performs comprehensive comparison including:
    - Metadata validation (batch size, shapes, dtypes, layouts, devices)
    - Bit-exact value comparison
    - Detailed error reporting on mismatch

    Supports both single and multiple outputs.

    :param pipe_out: Tuple of DALI pipeline outputs
    :param ndd_out: Either a single ndd Batch or tuple of ndd Batches
    :raises AssertionError: If outputs don't match
    """
    if isinstance(ndd_out, tuple):
        pipe_out_len = len(pipe_out)
        ndd_out_len = len(ndd_out)
        assert (
            pipe_out_len == ndd_out_len
        ), f"""Number of outputs mismatch:
  Pipeline: {pipe_out_len}
  NDD:      {ndd_out_len}"""

        for idx, (pout, nout) in enumerate(zip(pipe_out, ndd_out, strict=True)):
            _cmp(pout, nout, output_idx=idx)
    else:
        assert (
            len(pipe_out) == 1
        ), f"""Expected single output from DALI pipeline, got {len(pipe_out)}"""

        _cmp(pipe_out[0], ndd_out, output_idx=0)


def compare_no_input(
    pipe_outs: tuple[TensorListCPU] | tuple[TensorListGPU], ndd_outs: tuple[Batch] | Batch
) -> None:
    """Comparison function for no-input operators."""
    if not isinstance(ndd_outs, tuple):
        ndd_outs = (ndd_outs,)
    if not isinstance(pipe_outs, tuple):
        pipe_outs = (pipe_outs,)
    for idx, (pipe_out, ndd_out) in enumerate(zip(pipe_outs, ndd_outs, strict=True)):
        _cmp(pipe_out, ndd_out, output_idx=idx)


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
            rng=rng, max_batch_size=max_batch_size, n_states=1, num_outputs=1
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
        if needs_input:
            inp = [
                fn.external_source(name=f"INPUT{i}", device=device, layout=input_layout)
                for i in range(num_inputs)
            ]
            output = operator_under_test(*inp, device=device, _random_state=rs, **operator_args)
        else:
            output = operator_under_test(device=device, _random_state=rs, **operator_args)
        if needs_input:
            if isinstance(output, list):  # DALI uses list to collect operator multi-output
                return tuple(output[i] for i in range(len(output)))
            else:
                return output
        else:
            return output

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
    operator_args=None,
    num_inputs=1,
    input_layout=None,
    compare_fn=compare,
    random=False,
    batch_size=MAX_BATCH_SIZE,
):
    """Run fn vs ndd operator comparison over an epoch of batches and assert outputs match.

    For each batch in input_epoch: builds/uses a pipeline that runs the fn operator,
    runs the ndd operator on the same data, then calls compare_fn(pipe_out, ndd_out).
    Use this to test that an operator has equivalent behavior in fn and ndd APIs.

    How to invoke
    -------------
    1. Get operators by name (e.g. from OperatorTestConfig):
       fn_op = get_fn_operator("resize")   # or nested: "reductions.mean"
       ndd_op = get_ndd_operator("resize")
    2. Prepare input_epoch: list of batches (one per "iteration"). Each batch is
       a numpy array (single input) or a tuple of arrays (num_inputs > 1).
       Example: generate_image_like_data() or generate_data(shape_gen, ...).
    3. Call:
       run_operator_test(
           input_epoch=data,
           fn_operator=fn_op,
           ndd_operator=ndd_op,
           device="cpu" or "gpu",
           operator_args={"resize_x": 50, "resize_y": 50},  # passed to both
           num_inputs=1,
           input_layout="HWC",   # optional, for image-like
           compare_fn=compare,   # or compare_no_input for no-input ops
           random=False,         # True if op uses rng (fn/ndd RNGs synced)
           batch_size=MAX_BATCH_SIZE,
       )

    Parameters
    ----------
    input_epoch : list
        List of batches. Each element is a numpy array (num_inputs=1) or tuple of
        arrays (num_inputs>1). Batch shape: (batch_dim,) + sample_shape.
    fn_operator : callable
        Operator from fn API (e.g. fn.resize). Must accept (data, device=..., **operator_args).
    ndd_operator : callable
        Operator from ndd API (e.g. ndd.resize). Must accept (batch, **operator_args);
        if random=True, also accepts rng=.
    device : str
        "cpu" or "gpu".
    operator_args : dict, optional
        Keyword arguments passed to both fn and ndd operator (e.g. resize_x=50).
    num_inputs : int, optional
        Number of inputs. If > 1, input_epoch batches must be tuples of arrays.
    input_layout : str, optional
        Layout for external source (e.g. "HWC" for image-like).
    compare_fn : callable, optional
        (pipe_out, ndd_out) -> None. Default compare; use compare_no_input for
        no-input operators (e.g. random generators).
    random : bool, optional
        If True, create and pass matching RNGs to fn/ndd for reproducible comparison.
    batch_size : int, optional
        Max batch size used when building the pipeline.
    """
    if operator_args is None:
        operator_args = {}

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
        max_batch_size=batch_size,
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
        compare_fn(pipe_out, ndd_out)


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


def flatten_operator_configs(configs: list[OperatorTestConfig]) -> list[tuple]:
    """Flatten OperatorTestConfig list into test tuples."""
    result = []
    for config in configs:
        result.extend(config.generate_test_tuples())
        sign_off.register_test(config.name)
    return result

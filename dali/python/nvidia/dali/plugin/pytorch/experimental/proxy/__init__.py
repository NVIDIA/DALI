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

__all__ = ["DALIServer", "DataLoader"]

from torch.cuda import nvtx as _nvtx
import torch.multiprocessing as _mp
from torch.utils import data as _torchdata
from torch.utils.data._utils.collate import default_collate_fn_map as _default_collate_fn_map
from nvidia.dali import Pipeline as _Pipeline
from nvidia.dali.external_source import ExternalSource as _ExternalSource
import threading
import queue
from queue import Empty
from nvidia.dali.plugin.pytorch.torch_utils import to_torch_tensor
from inspect import Parameter, Signature

# DALI proxy is a PyTorch specific layer that connects a multi-process PyTorch DataLoader with
# a single DALI pipeline. This allows to run CUDA processing in a single process, avoiding the
# problem of having multiple CUDA contexts which hurts the performance.
#
# The diagram below shows how the different processes and thread interact with each other
# via shared queues. req_n_k represents the k-th processing request from data worker n,
# consisting of a batch identifier (n, k) and a set of inputs. data_n_k represents the
# outputs of a DALI pipeline corresponding to the same batch identifier, consisting of the
# batch identifier and a set of outputs.
#
# +-------+   +---------------+   +-------------+ +---------------+   +-----------+ +-----------+
# | main  |   | dali_output_q |   | data_thread | | dali_input_q  |   | worker_0  | | worker_1  |
# +-------+   +---------------+   +-------------+ +---------------+   +-----------+ +-----------+
#     |~~~get()~~~~~~>|                  |                |                 |             |
#     |               |                  |~~~get()~~~~~~~>|                 |             |
#     |               |                  |                |                 |             |
#     |               |                  |                |           -------------       |
#     |               |                  |                |           | collate   |       |
#     |               |                  |                |           -------------       |
#     |               |                  |                |                 |             |
#     |               |                  |                |<~~put(req_0_0)~~|             |
#     |               |                  |                |                 |             |
#     |               |                  |                |---------------->|             |
#     |               |                  |                |                 |             |
#     |               |                  |<--req_0_0------|                 |             |
#     |               |                  |                |                 |             |
#     |               |           ---------------         |                 |             |
#     |               |           | run         |         |                 |             |
#     |               |           ---------------         |                 |             |
#     |               |                  |                |                 |       -------------
#     |               |                  |                |                 |       | collate   |
#     |               |                  |                |                 |       -------------
#     |               |                  |                |                 |             |
#     |               |                  |                |<~~put(req_1_0)~~~~~~~~~~~~~~~~|
#     |               |                  |                |                 |             |
#     |               |                  |                |------------------------------>|
#     |               |                  |                |                 |             |
#     |               |<~~put(data_0_0)~~|                |                 |             |
#     |               |                  |                |                 |             |
#     |               |----------------->|                |                 |             |
#     |               |                  |                |                 |             |
#     |               |                  |                |                 |       -------------
#     |               |                  |                |                 |       | collate   |
#     |               |                  |                |                 |       -------------
#     |               |                  |                |                 |             |
#     |               |                  |                |<~~put(req_1_1)~~~~~~~~~~~~~~~~|
#     |               |                  |                |                 |             |
#     |               |                  |                |------------------------------>|
#     |               |                  |                |                 |             |
#     |<--data_0_0----|                  |                |                 |             |
#     |               |                  |                |                 |             |
#     |~~~get()~~~~~~>|                  |                |                 |             |
#     |               |                  |~~~get()~~~~~~~>|                 |             |
#     |               |                  |                |                 |             |
#     |               |                  |<--req_1_0------|                 |             |
#     |               |                  |                |                 |             |
#     |               |           ---------------         |          -------------        |
#     |               |           | run         |         |          | collate   |        |
#     |               |           ---------------         |          -------------        |
#     |               |                  |                |                 |             |
#     |               |                  |                |<~~put(req_0_1)~~|             |
#     |               |                  |                |                 |             |
#     |               |                  |                |-----------------|             |
#     |               |                  |                |                 |             |
#     |               |<~~put(data_1_0)~~|                |                 |             |
#     |               |                  |                |                 |             |
#     |               |----------------->|                |                 |             |
#     |               |                  |                |                 |             |
#     |<--data_1_0----|                  |                |                 |             |
#     |               |                  |                |                 |             |
# +-------+   +---------------+   +-------------+ +---------------+   +-----------+ +-----------+
# | main  |   | dali_output_q |   | data_thread | | dali_input_q  |   | worker_0  | | worker_1  |
# +-------+   +---------------+   +-------------+ +---------------+   +-----------+ +-----------+


def _modify_signature(new_sig):
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                bound_args = new_sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                return func(*bound_args.args, **bound_args.kwargs)
            except Exception as err:
                args_str = ", ".join([f"{type(arg)}" for arg in args])
                kwargs_str = ", ".join([f"{key}={type(arg)}" for key, arg in kwargs.items()])
                raise ValueError(
                    f"Expected signature is: {new_sig}. "
                    f"Got: args=({args_str}), kwargs={kwargs_str}, error: {err}"
                )

        wrapper.__signature__ = new_sig
        return wrapper

    return decorator


def _external_source_node_names(pipeline):
    """
    extract the names of all the ExternalSource nodes in the pipeline
    """
    # TODO(janton): Add a native function to query those names, so that we can do it
    # also on deserialized pipelines
    if pipeline._deserialized:
        raise RuntimeError(
            "Not able to find the external source "
            "operator names, since the pipeline was deserialized"
        )
    if not pipeline._py_graph_built:
        pipeline._build_graph()
    input_node_names = []
    for op in pipeline._ops:
        if isinstance(op._op, _ExternalSource):
            input_node_names.append(op.name)
    return input_node_names


class DALIOutputSampleRef:
    """
    Reference for a single sample output bound to a pipeline run.
    """

    def __init__(self, proxy, pipe_run_ref, output_idx, sample_idx):
        """
        Args:
            proxy (_DALIProxy): The proxy object used for communication or data handling.
            pipe_run_ref (DALIPipelineRunRef): A reference to the pipeline run.
            output_idx (int): The index of the output in the pipeline.
            sample_idx (int): The index of the sample within the batch.
        """
        self.proxy = proxy
        self.pipe_run_ref = pipe_run_ref
        self.output_idx = output_idx
        self.sample_idx = sample_idx

    def __repr__(self):
        return (
            f"DALIOutputSampleRef({self.pipe_run_ref}, "
            + f"output_idx={self.output_idx}, sample_idx={self.sample_idx})"
        )


class DALIPipelineRunRef:
    """
    Reference for a DALI pipeline run iteration.
    """

    def __init__(self, batch_id):
        """
        Args:
            batch_id (tuple(int, int)): A tuple that uniquely identifies the batch. The first
                element represent the worker, and the second the batch index for that worker
        """
        self.batch_id = batch_id
        self.call_args = {}
        self.is_scheduled = False
        self.is_complete = False

    def __repr__(self):
        return (
            f"DALIPipelineRunRef(batch_id={self.batch_id}, call_args={self.call_args} "
            f"is_scheduled={self.is_scheduled}, is_complete={self.is_complete})"
        )


class DALIOutputBatchRef:
    """
    Reference for a batched output bound to a pipeline run.
    """

    def __init__(self, pipe_run_ref, output_idx):
        """
        Args:
            pipe_run_ref (DALIPipelineRunRef): A reference to the pipeline run.
            output_idx (int): The index of the output in the pipeline.
        """
        self.pipe_run_ref = pipe_run_ref
        self.output_idx = output_idx

    def __repr__(self):
        return f"DALIOutputBatchRef(pipe_run_ref={self.pipe_run_ref}, output_idx={self.output_idx})"


def _collate_dali_output_sample_ref_fn(samples, *, collate_fn_map=None):
    """
    Special collate function that schedules a DALI iteration for execution
    """
    assert len(samples) > 0
    pipe_run_ref = samples[0].pipe_run_ref
    output_idx = samples[0].output_idx
    proxy = samples[0].proxy
    for i, sample in enumerate(samples):
        if (
            sample.proxy != proxy
            or sample.pipe_run_ref != pipe_run_ref
            or sample.output_idx != output_idx
        ):
            raise RuntimeError("All samples should belong to the same batch")

        if sample.sample_idx != i:
            raise RuntimeError("Unexpected sample order")

    pipe_run_ref.is_complete = True
    if not proxy._deterministic and not pipe_run_ref.is_scheduled:
        pipe_run_ref = proxy._schedule_batch(pipe_run_ref)
    return DALIOutputBatchRef(pipe_run_ref, output_idx)


# In-place modify `default_collate_fn_map` to handle DALIOutputSampleRef
_default_collate_fn_map.update({DALIOutputSampleRef: _collate_dali_output_sample_ref_fn})


class _DALIProxy:
    class _WorkerData:
        def __init__(self, worker_id):
            self.worker_id = worker_id
            self.pipe_run_ref = None
            self.next_worker_batch_idx = 0
            self.batch_sample_idx = 0

    def __init__(self, signature, dali_input_q, deterministic):
        # If True, the request is not sent to DALI upon creation, so that it can be scheduled
        # always in the same order by the main process. This comes at a cost of performance
        self._deterministic = deterministic
        # Shared queue with the server
        self._dali_input_q = dali_input_q
        # Per worker
        self._worker_data = None
        # Override callable signature
        self._signature = signature
        # Current batch
        self._curr_batch_params = {}
        # get num outputs
        self._num_outputs = self._get_num_outputs()

    def _get_num_outputs(self):
        return_annotation = self._signature.return_annotation
        if isinstance(return_annotation, tuple):
            return len(return_annotation)
        elif return_annotation is Signature.empty:
            return 0
        else:
            return 1

    def _get_worker_id(self):
        """
        Getter for 'worker_id'. In case of torch data worker it is the worker info, and
        in case of a thread the thread identifier
        """
        worker_info = _torchdata.get_worker_info()
        return worker_info.id if worker_info else threading.get_ident()

    def _get_worker_data(self):
        if self._worker_data is None:
            self._worker_data = _DALIProxy._WorkerData(self._get_worker_id())
        return self._worker_data

    def _add_sample(self, inputs):
        """
        Adds a sample to the current batch. In the collate function, we mark the batch as
        complete. When a completed batch is encountered, a new batch should be started.
        """
        worker_data = self._get_worker_data()
        if worker_data.pipe_run_ref is None or worker_data.pipe_run_ref.is_complete:
            worker_data.pipe_run_ref = DALIPipelineRunRef(
                batch_id=(worker_data.worker_id, worker_data.next_worker_batch_idx)
            )
            worker_data.next_worker_batch_idx += 1
            worker_data.batch_sample_idx = 0

        for name, value in inputs.items():
            # we want to transfer only the arguments to the caller side, not the the self reference
            if name == "self":
                continue
            if name not in worker_data.pipe_run_ref.call_args:
                worker_data.pipe_run_ref.call_args[name] = []
            worker_data.pipe_run_ref.call_args[name].append(value)

        ret = tuple(
            DALIOutputSampleRef(
                self,
                pipe_run_ref=worker_data.pipe_run_ref,
                output_idx=i,
                sample_idx=worker_data.batch_sample_idx,
            )
            for i in range(self._num_outputs)
        )
        # unpack single element tuple
        if len(ret) == 1:
            ret = ret[0]
        worker_data.batch_sample_idx += 1
        return ret

    def _schedule_batch(self, pipe_run_ref):
        """
        Schedules a batch for execution by appending it to the DALI input queue.
        """
        if not pipe_run_ref.call_args:
            raise RuntimeError("No inputs for the pipeline to run (was it already scheduled?)")
        if not pipe_run_ref.is_complete:
            raise RuntimeError("Batch is not marked as complete")
        _nvtx.range_push(f"dali_input_q put {pipe_run_ref.batch_id}")

        dali_input_q_item = DALIPipelineRunRef(pipe_run_ref.batch_id)
        dali_input_q_item.call_args = pipe_run_ref.call_args
        dali_input_q_item.is_complete = True
        dali_input_q_item.is_scheduled = True
        self._dali_input_q.put(dali_input_q_item)

        pipe_run_ref.call_args = {}
        pipe_run_ref.is_scheduled = True
        _nvtx.range_pop()
        return pipe_run_ref


class DALIServer:
    def __init__(self, pipeline, deterministic=False):
        """
        Initializes a new DALI server instance.

        Args:
            pipeline (Pipeline): DALI pipeline to run.
            deterministic (bool): If True, it ensures that the order of execution is always
                                the same, which is important when the pipeline has a state
                                and we are interested in obtaining reproducible results.
                                Also, if enabled, the execution will be less performant, as
                                the DALI processing can be scheduled only after the data
                                loader has returned the batch information, and not as soon
                                as data worker collates the batch.

        Example 1 - Full integration with PyTorch via DALI proxy DataLoader:

            .. code-block:: python

                @pipeline_def
                def rn50_train_pipe():
                    rng = fn.random.coin_flip(probability=0.5)
                    filepaths = fn.external_source(name="images", no_copy=True)
                    jpegs = fn.io.file.read(filepaths)
                    images = fn.decoders.image_random_crop(
                        jpegs,
                        device="mixed",
                        output_type=types.RGB,
                        random_aspect_ratio=[0.75, 4.0 / 3.0],
                        random_area=[0.08, 1.0],
                    )
                    images = fn.resize(
                        images,
                        size=[224, 224],
                        interp_type=types.INTERP_LINEAR,
                        antialias=False,
                    )
                    output = fn.crop_mirror_normalize(
                        images,
                        dtype=types.FLOAT,
                        output_layout="CHW",
                        crop=(224, 224),
                        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                        mirror=rng,
                    )
                    return output

                def read_filepath(path):
                    return np.frombuffer(path.encode(), dtype=np.int8)

                nworkers = 8
                pipe = rn50_train_pipe(
                    batch_size=16, num_threads=3, device_id=0,
                    prefetch_queue_depth=2*nworkers)

                # The scope makes sure the server starts and stops at enter/exit
                with dali_proxy.DALIServer(pipe) as dali_server:
                    # DALI proxy instance can be used as a transform callable
                    dataset = torchvision.datasets.ImageFolder(
                        jpeg, transform=dali_server.proxy, loader=read_filepath)

                    # Same interface as torch DataLoader, but takes a dali_server as first argument
                    loader = nvidia.dali.plugin.pytorch.experimental.proxy.DataLoader(
                        dali_server,
                        dataset,
                        batch_size=batch_size,
                        num_workers=nworkers,
                        drop_last=True,
                    )

                    for data, target in loader:
                        # consume it

        Example 2 - Manual execution using DALI proxy / DALI server and PyTorch's default_collate:

            .. code-block:: python

                @pipeline_def
                def my_pipe():
                    a = fn.external_source(name="a", no_copy=True)
                    b = fn.external_source(name="b", no_copy=True)
                    return a + b, a - b

                with dali_proxy.DALIServer(
                    my_pipe(device='cpu', batch_size=batch_size,
                            num_threads=3, device_id=None)) as dali_server:

                    outs = []
                    for _ in range(batch_size):
                        a = np.array(np.random.rand(3, 3), dtype=np.float32)
                        b = np.array(np.random.rand(3, 3), dtype=np.float32)
                        out0, out1 = dali_server.proxy(a=a, b=b)
                        outs.append((a, b, out0, out1))

                    outs = torch.utils.data.dataloader.default_collate(outs)

                    a, b, a_plus_b, a_minus_b = dali_server.produce_data(outs)

        Example 3 - Full integration with PyTorch but using the original PyTorch DataLoader

            .. code-block:: python

                pipe = rn50_train_pipe(...)
                with dali_proxy.DALIServer(pipe) as dali_server:
                    dataset = torchvision.datasets.ImageFolder(
                        jpeg, transform=dali_server.proxy, loader=read_filepath)

                    # Using PyTorch DataLoader directly
                    loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=nworkers,
                        drop_last=True,
                    )

                    for data, target in loader:
                        # replaces the output reference with actual data
                        data = dali_server.produce_data(data)
                        ...
        """
        if not isinstance(pipeline, _Pipeline):
            raise TypeError(f"Expected an NVIDIA DALI pipeline, got: {pipeline}")
        else:
            self._pipe = pipeline

        # get the dali pipeline input names
        self._dali_input_names = _external_source_node_names(self._pipe)
        num_inputs = len(self._dali_input_names)
        if num_inputs == 0:
            raise RuntimeError("The provided pipeline doesn't have any inputs")

        parameters = [Parameter("self", Parameter.POSITIONAL_OR_KEYWORD)]
        parameter_kind = (
            Parameter.POSITIONAL_OR_KEYWORD if num_inputs == 1 else Parameter.KEYWORD_ONLY
        )
        for input_name in self._dali_input_names:
            parameters.append(Parameter(input_name, parameter_kind))
        return_annotation = tuple(DALIOutputSampleRef for _ in range(self._pipe.num_outputs))
        self._signature = Signature(parameters, return_annotation=return_annotation)

        # Multi-process queue used to transfer data from the pytorch workers to the main process
        self._dali_input_q = _mp.Queue()
        # Multi-process queue used by the main process to consume outputs from the DALI pipeline
        self._dali_output_q = queue.Queue()
        # Thread
        self._thread = None
        self._thread_stop_event = None
        # Cache
        self._cache_outputs = dict()
        # Whether we want the order of DALI execution to be reproducible
        self._deterministic = deterministic
        # Proxy
        self._proxy = None

    def __del__(self):
        self.stop_thread()

    @property
    def proxy(self):
        if not self._proxy:

            class _DALIProxyCallable(_DALIProxy):
                def __init__(self, signature, dali_input_q, deterministic):
                    super().__init__(signature, dali_input_q, deterministic)

                @_modify_signature(self._signature)
                def __call__(self, *args, **kwargs):
                    bound_args = self._signature.bind(self, *args, **kwargs)
                    return self._add_sample(bound_args.arguments)

            self._proxy = _DALIProxyCallable(
                self._signature, self._dali_input_q, self._deterministic
            )
        return self._proxy

    def _get_outputs(self, pipe_run_ref: DALIPipelineRunRef):
        """
        Gets the pipeline outputs for a specific batch id. It will keep reading data until the
        right batch is found, caching the results that were not consumed until a future call
        """
        req_batch_id = pipe_run_ref.batch_id

        # In case we haven't scheduled the iteration yet (i.e. deterministic config), do it now
        if not pipe_run_ref.is_scheduled:
            _nvtx.range_push(f"dali_input_q put {pipe_run_ref.batch_id}")
            self._dali_input_q.put(pipe_run_ref)
            pipe_run_ref.is_scheduled = True
            _nvtx.range_pop()

        # Wait for the requested output to be ready
        req_outputs = None
        # If the data was already read, just return it (and clear the cache entry)
        if req_batch_id in self._cache_outputs:
            req_outputs = self._cache_outputs[req_batch_id]
            del self._cache_outputs[req_batch_id]

        else:
            curr_batch_id = None
            # If not the data we are looking for, store it and keep processing until we find it
            while req_batch_id != curr_batch_id:
                _nvtx.range_push("dali_output_q.get")
                curr_batch_id, curr_processed_outputs, err = self._dali_output_q.get()
                _nvtx.range_pop()

                if err is not None:
                    raise err

                if curr_batch_id == req_batch_id:
                    req_outputs = curr_processed_outputs
                else:
                    self._cache_outputs[curr_batch_id] = curr_processed_outputs
        return req_outputs

    @staticmethod
    def _need_conversion(obj, need_conversion_cache):
        """Return True if the object or any of its members need conversion."""
        obj_id = id(obj)
        if obj_id in need_conversion_cache:
            return need_conversion_cache[obj_id]

        if isinstance(obj, DALIOutputBatchRef):
            need_conversion_cache[obj_id] = True
            return True

        need_conversion_cache[obj_id] = False  # Prevent infinite recursion
        for item in (
            obj
            if isinstance(obj, (list, tuple))
            else obj.values() if isinstance(obj, dict) else getattr(obj, "__dict__", {}).values()
        ):
            if DALIServer._need_conversion(item, need_conversion_cache):
                need_conversion_cache[obj_id] = True
                return True

        return False

    def _produce_data_impl(self, obj, cache, need_conversion_cache):
        """
        Recursive single-pass implementation of produce_data with in-place modifications.
        """

        obj_id = id(obj)
        if obj_id in cache:  # Return cached result to prevent infinite recursion
            return cache[obj_id]

        # If it doesn't need conversion, return immediately
        if not DALIServer._need_conversion(obj, need_conversion_cache):
            cache[obj_id] = obj
            return obj

        # Handle DALIOutputBatchRef
        if isinstance(obj, DALIOutputBatchRef):
            pipe_run_ref_id = id(obj.pipe_run_ref)
            if pipe_run_ref_id not in cache:
                cache[pipe_run_ref_id] = self._get_outputs(obj.pipe_run_ref)
            outputs = cache[pipe_run_ref_id]
            result = outputs[obj.output_idx]
            cache[obj_id] = result
            return result

        # Handle lists (modify in place)
        if isinstance(obj, list):
            cache[obj_id] = obj  # Cache before recursion to prevent infinite loops
            for i in range(len(obj)):
                obj[i] = self._produce_data_impl(obj[i], cache, need_conversion_cache)
            return obj

        # Handle tuples (regular, named, or custom)
        if isinstance(obj, tuple):
            # Named tuple: Reconstruct using `_replace`
            if hasattr(obj, "_replace") and hasattr(obj, "_fields"):
                result = obj._replace(
                    **{
                        field: self._produce_data_impl(
                            getattr(obj, field), cache, need_conversion_cache
                        )
                        for field in obj._fields
                    }
                )
            # Regular or custom tuple: Reconstruct using type(obj)
            else:
                result = type(obj)(
                    self._produce_data_impl(item, cache, need_conversion_cache) for item in obj
                )
            cache[obj_id] = result
            return result

        # Handle dictionaries (modify in place)
        if isinstance(obj, dict):
            cache[obj_id] = obj  # Cache before recursion to prevent infinite loops
            for key in list(obj.keys()):  # Ensure we handle dynamic changes in the dictionary
                obj[key] = self._produce_data_impl(obj[key], cache, need_conversion_cache)
            return obj

        # Handle custom objects with attributes (modify in place)
        if hasattr(obj, "__dict__"):
            cache[obj_id] = obj  # Cache before recursion to prevent infinite loops
            for attr_name, attr_value in obj.__dict__.items():
                setattr(
                    obj,
                    attr_name,
                    self._produce_data_impl(attr_value, cache, need_conversion_cache),
                )
            return obj

        # Default case (shouldn't be reached since we exited early in case of no-op)
        cache[obj_id] = obj
        return obj

    def produce_data(self, obj):
        """
        A generic function to recursively visits all elements in a nested structure and replace
        instances of DALIOutputBatchRef with the actual data provided by the DALI server
        See :class:`nvidia.dali.plugin.pytorch.experimental.proxy.DALIServer` for a full example.

        Args:
            obj: The object to map (can be an instance of any class).

        Returns:
            A new object where any instance of DALIOutputBatchRef has been replaced with actual
            data.

        """
        cache = {}
        need_conversion_cache = {}
        ret = self._produce_data_impl(obj, cache, need_conversion_cache)
        return ret

    def _get_input_batches(self, max_num_batches, timeout=None):
        _nvtx.range_push("dali_input_q.get")
        count = 0
        batches = []
        if timeout is not None:
            try:
                batches.append(self._dali_input_q.get(timeout=timeout))
                count = count + 1
            except Empty:
                return None
            except _mp.TimeoutError:
                return None

        while count < max_num_batches:
            try:
                batches.append(self._dali_input_q.get_nowait())
                count = count + 1
            except Empty:
                break
        _nvtx.range_pop()
        return batches

    def _thread_fn(self):
        """
        Asynchronous DALI thread that gets iteration data from the queue and schedules it
        for execution
        """
        fed_batches = []
        while not self._thread_stop_event.is_set():
            _nvtx.range_push("get_input_batches")
            timeout = 5 if len(fed_batches) == 0 else None
            # We try to feed as many batches as the prefetch queue (if available)
            batches = self._get_input_batches(
                self._pipe.prefetch_queue_depth - len(fed_batches), timeout=timeout
            )
            _nvtx.range_pop()
            if batches is not None and len(batches) > 0:
                _nvtx.range_push("feed_pipeline")
                for pipe_run_ref in batches:
                    for name, data in pipe_run_ref.call_args.items():
                        self._pipe.feed_input(name, data)
                    self._pipe._run_once()
                    fed_batches.append(pipe_run_ref.batch_id)
                _nvtx.range_pop()

            # If no batches to consume, continue
            if len(fed_batches) == 0:
                continue

            _nvtx.range_push("outputs")
            batch_id = fed_batches.pop(0)  # we are sure there's at least one
            err = None
            torch_outputs = None
            try:
                pipe_outputs = self._pipe.outputs()
                torch_outputs = tuple(
                    [
                        to_torch_tensor(out.as_tensor(), not self._pipe.exec_dynamic)
                        for out in pipe_outputs
                    ]
                )
            except Exception as exception:
                err = exception

            self._dali_output_q.put((batch_id, torch_outputs, err))
            _nvtx.range_pop()

    def start_thread(self):
        """
        Starts the DALI pipeline thread. Note: Using scope's __enter__/__exit__ is preferred
        """
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=DALIServer._thread_fn, args=(self,))
        self._thread_stop_event = threading.Event()
        self._thread.start()

    def stop_thread(self):
        """
        Stops the DALI pipeline thread. Note: Using scope's __enter__/__exit__ is preferred
        """
        if self._thread_stop_event is None:
            return
        self._thread_stop_event.set()
        self._thread.join()
        self._thread = None
        self._thread_stop_event = None

    def _is_thread_running(self):
        return self._thread is not None

    def __enter__(self):
        """
        Starts the DALI pipeline thread
        """
        self.start_thread()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Stops the DALI pipeline thread
        """
        self.stop_thread()
        return False  # Return False to propagate exceptions


class DataLoader(_torchdata.dataloader.DataLoader):
    """
    DALI data loader to be used in the main loop, which replaces the pipeline run references
    with actual data produced by the DALI server.
    See :class:`nvidia.dali.plugin.pytorch.experimental.proxy.DALIServer` for a full example.
    """

    class _Iter(_torchdata.dataloader._MultiProcessingDataLoaderIter):
        """
        Data loader iterator used by the DALI proxy data loader
        """

        def __init__(self, loader):
            super().__init__(loader)
            self.loader = loader

        def _next_data(self):
            self.loader.ensure_server_listening()
            data = super()._next_data()
            return self.loader.dali_server.produce_data(data)

    def __init__(self, dali_server, *args, **kwargs):
        """
        Same interface as PyTorch's DataLoader except for the extra DALIServer argument
        """
        super().__init__(*args, **kwargs)
        self.dali_server = dali_server
        self.server_started_by_loader = False

    def __del__(self):
        if self.server_started_by_loader and self.dali_server._is_thread_running():
            print("Stop")
            self.dali_server.stop_thread()

    def ensure_server_listening(self):
        if not self.dali_server._is_thread_running():
            self.dali_server.start_thread()
            self.server_started_by_loader = True

    def _get_iterator(self):
        return DataLoader._Iter(self)

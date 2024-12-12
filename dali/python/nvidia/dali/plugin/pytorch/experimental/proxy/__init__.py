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

import torch
import torch.multiprocessing as mp
from torch.utils import data as torchdata
from torch.utils.data._utils.collate import default_collate_fn_map
from nvidia.dali import Pipeline
from nvidia.dali.external_source import ExternalSource
import threading
import queue
from queue import Empty
from nvidia.dali.plugin.pytorch.torch_utils import to_torch_tensor
import tree
import warnings

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
#     |               |           ---------------         |                 |             |
#     |               |           | run         |         |                 |             |
#     |               |           ---------------         |                 |             |
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
        if isinstance(op._op, ExternalSource):
            input_node_names.append(op.name)
    return input_node_names


class DALIPipelineRunRef:
    """
    Reference for a DALI pipeline run iteration.
    """

    def __init__(self, batch_id, inputs):
        """
        batch_id: Identifier of the batch
        is_scheduled: Whether the iteration has been scheduled for execution already
        inputs: Inputs to be used when scheduling the iteration (makes sense only if
        is_scheduled is False)
        """
        self.batch_id = batch_id
        self.inputs = inputs
        self.is_scheduled = False


class DALIProcessedSampleRef:
    """
    Placeholder for a pipeline run reference, which is returned by the data worker instead of
    the actual data. The PyTorch worker returns this trivial object, only containing information
    about this proxy instance and the input data to the pipeline. Later in the collate function,
    we send the data for execution to DALI.
    """

    def __init__(self, proxy, inputs):
        self.proxy = proxy
        self.inputs = inputs
        if len(inputs) != len(self.proxy.dali_input_names):
            raise RuntimeError(
                f"Unexpected number of inputs. Expected: {self.dali_input_names}, got: {inputs}"
            )


def _collate_dali_processed_sample_ref_fn(samples, *, collate_fn_map=None):
    """
    Special collate function that schedules a DALI iteration for execution
    """
    assert len(samples) > 0
    sample = samples[0]
    inputs = [[] for _ in range(len(sample.inputs))]
    proxy = sample.proxy
    for sample in samples:
        assert proxy == sample.proxy
        for idx, input_ref in enumerate(sample.inputs):
            inputs[idx].append(input_ref)
    pipe_run_ref = proxy._create_pipe_run_ref(inputs)
    if not proxy.deterministic:
        proxy._schedule_batch(pipe_run_ref)
        # If we scheduled the iteration, we don't need to transfer the inputs back to the consumer
        return DALIPipelineRunRef(pipe_run_ref.batch_id, inputs=None)
    else:
        return pipe_run_ref


# In-place modify `default_collate_fn_map` to handle DALIProcessedSampleRef
default_collate_fn_map.update({DALIProcessedSampleRef: _collate_dali_processed_sample_ref_fn})


class _DALIProxy:
    def __init__(self, dali_input_names, dali_input_q, deterministic):
        # External source instance names
        self.dali_input_names = dali_input_names
        # If True, the request is not sent to DALI upon creation, so that it can be scheduled
        # always in the same order by the main process. This comes at a cost of performance
        self.deterministic = deterministic
        # Shared queue with the server
        self.dali_input_q = dali_input_q
        # Torch worker id, to be filled on first call to worker_id()
        self._worker_id = None
        # Iteration index for the current worker
        self.data_idx = 0

    @property
    def worker_id(self):
        """
        Getter for 'worker_id'. In case of torch data worker it is the worker info, and
        in case of a thread the thread identifier
        """
        if self._worker_id is None:
            worker_info = torchdata.get_worker_info()
            self._worker_id = worker_info.id if worker_info else threading.get_ident()
        return self._worker_id

    def _create_pipe_run_ref(self, inputs):
        # Identifier of this request
        batch_id = (self.worker_id, self.data_idx)
        self.data_idx = self.data_idx + 1
        return DALIPipelineRunRef(batch_id, inputs=inputs)

    def _schedule_batch(self, pipe_run_ref):
        torch.cuda.nvtx.range_push(f"dali_proxy.dali_input_q.put {pipe_run_ref.batch_id}")
        self.dali_input_q.put((pipe_run_ref.batch_id, pipe_run_ref.inputs))
        torch.cuda.nvtx.range_pop()

    def __call__(self, *inputs):
        """
        Returns a reference to the pipeline run
        """
        if len(inputs) != len(self.dali_input_names):
            raise RuntimeError(
                f"Unexpected number of inputs. Expected: {self.dali_input_names}, got: {inputs}"
            )
        return DALIProcessedSampleRef(self, inputs)


class DALIServer:
    def __init__(self, pipeline, input_names=None, deterministic=False):
        """
        Initializes a new DALI server instance.

        Args:
            input_names (list): list of strings representing the inputs to the pipeline. Those
                                should match the names of the ``external_source`` nodes in the
                                DALI pipeline. If the pipeline has a single input, there is no
                                need to provide this argument.
            deterministic (bool): If True, it ensures that the order of execution is always
                                the same, which is important when the pipeline has a state
                                and we are interested in obtaining reproducible results.
                                Also, if enabled, the execution will be less performant, as
                                the DALI processing can be scheduled only after the data
                                loader has returned the batch information, and not as soon
                                as data worker collates the batch.

        Example:

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
                        pass

        """
        assert isinstance(pipeline, Pipeline), f"Expected an NVIDIA DALI pipeline, got: {pipeline}"
        self.pipe = pipeline
        self.pipe_input_names = _external_source_node_names(self.pipe)
        if len(self.pipe_input_names) == 0:
            raise RuntimeError("The provided pipeline doesn't have any inputs")
        elif len(self.pipe_input_names) == 1:
            assert input_names is None or input_names[0] == self.pipe_input_names[0]
            self.dali_input_names = self.pipe_input_names
        elif input_names is None or len(input_names) != len(self.pipe_input_names):
            raise RuntimeError(
                "The provided pipeline has more than one output. In such case, the argument "
                "`input_names` should containi the same exact number of strings, one for "
                "each pipeline input to be mapped by the proxy callable object"
            )
        self.num_inputs = len(self.dali_input_names)

        # Multi-process queue used to transfer data from the pytorch workers to the main process
        self.dali_input_q = mp.Queue()
        # Multi-process queue used by the main process to consume outputs from the DALI pipeline
        self.dali_output_q = queue.Queue()
        # Thread
        self.thread = None
        # Cache
        self.cache_outputs = dict()
        # Whether we want the order of DALI execution to be reproducible
        self.deterministic = deterministic

    @property
    def proxy(self):
        return _DALIProxy(self.dali_input_names, self.dali_input_q, self.deterministic)


    def _schedule_batch(self, pipe_run_ref):
        torch.cuda.nvtx.range_push(f"dali_proxy.dali_input_q.put {pipe_run_ref.batch_id}")
        self.dali_input_q.put((pipe_run_ref.batch_id, pipe_run_ref.inputs))
        torch.cuda.nvtx.range_pop()

    def _get_outputs(self, pipe_run_ref: DALIPipelineRunRef):
        """
        Gets the pipeline outputs for a specific batch id. It will keep reading data until the
        right batch is found, caching the results that were not consumed until a future call
        """
        req_batch_id = pipe_run_ref.batch_id

        # In case we haven't scheduled the iteration yet (i.e. deterministic config), do it now
        if not pipe_run_ref.is_scheduled:
            self._schedule_batch(pipe_run_ref)
            pipe_run_ref.is_scheduled = True

        # Wait for the requested output to be ready
        req_outputs = None
        # If the data was already read, just return it (and clear the cache entry)
        if req_batch_id in self.cache_outputs:
            req_outputs = self.cache_outputs[req_batch_id]
            del self.cache_outputs[req_batch_id]

        else:
            curr_batch_id = None
            # If not the data we are looking for, store it and keep processing until we find it
            while req_batch_id != curr_batch_id:
                torch.cuda.nvtx.range_push("dali_output_q.get")
                curr_batch_id, curr_processed_outputs = self.dali_output_q.get()
                torch.cuda.nvtx.range_pop()
                if curr_batch_id == req_batch_id:
                    req_outputs = curr_processed_outputs
                else:
                    self.cache_outputs[curr_batch_id] = curr_processed_outputs
        # Unpack single element tuples
        if isinstance(req_outputs, tuple) and len(req_outputs) == 1:
            req_outputs = req_outputs[0]
        return req_outputs

    def produce_data(self, obj):
        """
        A generic function to recursively visits all elements in a nested structure and replace
        instances of DALIPipelineRunRef with the actual data provided by the DALI server

        Args:
            obj: The object to map (can be an instance of any class).

        Returns:
            A new object where any instance of DALIPipelineRunRef has been replaced with actual
            data.
        """

        # If it is an instance of DALIPipelineRunRef, replace it with data
        if isinstance(obj, DALIPipelineRunRef):
            return self._get_outputs(obj)
        # If it is a custom class, recursively call produce data on its members
        elif hasattr(obj, '__dict__'):
            new_obj = obj.__class__.__new__(obj.__class__)
            for attr_name, attr_value in obj.__dict__.items():
                setattr(new_obj, attr_name, self.produce_data(attr_value))
            return new_obj

        # If it's a list, recursively apply the function to each element
        elif isinstance(obj, list):
            return [self.produce_data(item) for item in obj]

        # If it's a tuple, recursively apply the function to each element (and preserve tuple type)
        elif isinstance(obj, tuple):
            return tuple(self.produce_data(item) for item in obj)

        # If it's a dictionary, apply the function to both keys and values
        elif isinstance(obj, dict):
            return {key: self.produce_data(value) for key, value in obj.items()}

        else:  # return directly anything else
            return obj

    def _thread_fn(self):
        """
        Asynchronous DALI thread that gets iteration data from the queue and schedules it
        for execution
        """
        self.pipe.build()  # just in case

        while not self.thread_stop_event.is_set():
            try:
                torch.cuda.nvtx.range_push("dali_input_q.get")
                batch_id, inputs = self.dali_input_q.get(timeout=5)
                torch.cuda.nvtx.range_pop()
            except mp.TimeoutError:
                continue
            except Empty:
                continue

            torch.cuda.nvtx.range_push(f"schedule iteration {batch_id}")
            for idx, input_name in enumerate(self.dali_input_names):
                self.pipe.feed_input(input_name, inputs[idx])
            self.pipe._run_once()
            pipe_outputs = self.pipe.outputs()
            torch_outputs = tuple(
                [
                    to_torch_tensor(out.as_tensor(), not self.pipe.exec_dynamic)
                    for out in pipe_outputs
                ]
            )
            self.dali_output_q.put((batch_id, torch_outputs))
            torch.cuda.nvtx.range_pop()

    def start_thread(self):
        """
        Starts the DALI pipeline thread
        """
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=DALIServer._thread_fn, args=(self,))
        self.thread_stop_event = threading.Event()
        self.thread.start()

    def stop_thread(self):
        """
        Stops the DALI pipeline thread
        """
        if self.thread_stop_event is None:
            return
        self.thread_stop_event.set()
        self.thread.join()
        self.thread = None
        self.thread_stop_event = None

    def __enter__(self):
        self.start_thread()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.stop_thread()
        if exc_type is not None:
            warnings.warn(f"An exception occurred: {exc_value}", category=UserWarning)
        return False  # Return False to propagate exceptions


class DataLoader(torchdata.dataloader.DataLoader):
    """
    DALI data loader to be used in the main loop, which runs the DALI pipeline doing the
    processing asynchronously with regards to the training.
    """

    class _Iter(torchdata.dataloader._MultiProcessingDataLoaderIter):
        """
        Data loader iterator used by the DALI proxy data loader
        """

        def __init__(self, loader):
            super().__init__(loader)
            self.loader = loader

        def _next_data(self):
            data = super()._next_data()
            if self.loader.dali_server.thread is None:
                raise RuntimeError("DALI server is not running")
            data = self.dali_server.produce_data(data)
            return data

    def __init__(self, dali_server, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dali_server = dali_server

    def _get_iterator(self):
        return DataLoader._Iter(self)

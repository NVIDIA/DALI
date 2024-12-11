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
from nvidia.dali.backend import TensorGPU
from nvidia.dali import Pipeline
from nvidia.dali.external_source import ExternalSource
import threading
import queue
from queue import Empty
from nvidia.dali.plugin.pytorch.torch_utils import to_torch_type, feed_ndarray
import tree
import warnings


def _external_source_node_names(pipeline):
    """
    extract the names of all the ExternalSource nodes in the pipeline
    """
    if not pipeline._py_graph_built:
        pipeline._build_graph()
    input_node_names = []
    for op in pipeline._ops:
        if isinstance(op._op, ExternalSource):
            input_node_names.append(op.name)
    return input_node_names


class DALIPipelineOutputRef:
    """
    Placeholder for a pipeline output reference, after the iteration has been scheduled to DALI.
    """

    def __init__(self, batch_id):
        self.batch_id = batch_id


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


class _DALIProxy:
    def __init__(self, dali_input_names, dali_input_q):
        self.dali_input_names = dali_input_names
        # Shared queue with the server
        self.dali_input_q = dali_input_q
        # Torch worker id, to be filled on first call to worker_id()
        self._worker_id = None
        # Iteration index for the current worker
        self.data_idx = 0

    @property
    def worker_id(self):
        """
        Getter for 'worker_id'. In case of torch data worker it is the worker info,
        and in case of a thread the thread identifier
        """
        if self._worker_id is None:
            worker_info = torchdata.get_worker_info()
            self._worker_id = worker_info.id if worker_info else threading.get_ident()
        return self._worker_id

    def _schedule_batch(self, inputs):
        # Identifier of this request
        batch_id = (self.worker_id, self.data_idx)
        torch.cuda.nvtx.range_push(f"dali_proxy.dali_input_q.put {batch_id}")
        self.dali_input_q.put((batch_id, inputs))
        torch.cuda.nvtx.range_pop()
        self.data_idx = self.data_idx + 1
        # Returns a placeholder, which is replaced with the actual data once the iteration completes
        return DALIPipelineOutputRef(batch_id)

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

    def __init__(self, pipeline, input_names=None):
        """
        Initializes a new DALI server instance.

        Args:
            input_names (list): list of strings representing the inputs to the pipeline. Those
            should match the names of the ``external_source`` nodes in the DALI pipeline.
            If the pipeline has a single input, there is no need to provide this argument.
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

    @property
    def proxy(self):
        return _DALIProxy(self.dali_input_names, self.dali_input_q)

    def get_outputs(self, pipe_out_ref: DALIPipelineOutputRef):
        """
        Gets the pipeline outputs for a specific batch id. It will keep reading data until the
        right batch is found, caching the results that were not consumed until a future call
        """
        req_batch_id = pipe_out_ref.batch_id
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

    def thread_fn(self):
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

            # If exec_dynamic, we can avoid copying the data
            if not self.pipe.exec_dynamic:
                torch_outputs = []
                for pipe_output in pipe_outputs:
                    tensor = pipe_output.as_tensor()
                    torch_dtype = to_torch_type[tensor.dtype]
                    if isinstance(tensor, TensorGPU):
                        torch_device = torch.device("cuda", self.pipe.device_id)
                    else:
                        torch_device = torch.device("cpu")
                    torch_output = torch.empty(
                        tensor.shape(),
                        dtype=torch_dtype,
                        device=torch_device,
                    )
                    cuda_stream = (
                        torch.cuda.current_stream(device=torch_device)
                        if isinstance(tensor, TensorGPU)
                        else None
                    )
                    feed_ndarray(tensor, torch_output, cuda_stream=cuda_stream)
                    torch_outputs.append(torch_output)
                torch_outputs = tuple(torch_outputs)
            else:
                torch_outputs = tuple(
                    [torch.from_dlpack(pipe_output.as_tensor()) for pipe_output in pipe_outputs]
                )

            self.dali_output_q.put((batch_id, torch_outputs))
            torch.cuda.nvtx.range_pop()

    def start_thread(self):
        """
        Starts the DALI pipeline thread
        """
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=DALIServer.thread_fn, args=(self,))
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


def _collate_dali_processed_sample_ref_fn(pipe_out, *, collate_fn_map=None):
    """
    Special collate function that schedules a DALI iteration for execution
    """
    assert len(pipe_out) > 0
    first_elem = pipe_out[0]
    inputs = [[] for _ in range(len(first_elem.inputs))]
    proxy = first_elem.proxy
    for elem in pipe_out:
        assert proxy == elem.proxy
        for idx, input_ref in enumerate(elem.inputs):
            inputs[idx].append(input_ref)
    ret = proxy._schedule_batch(inputs)
    return ret


# In-place modify `default_collate_fn_map` to handle DALIProcessedSampleRef
default_collate_fn_map.update({DALIProcessedSampleRef: _collate_dali_processed_sample_ref_fn})


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
            if not hasattr(data, "__iter__"):
                warnings.warn(
                    "Warning: Non iterable returned from dataloader. Please "
                    " review the code, since it usually indicates a bug in the pipeline.",
                    category=UserWarning,
                )
                data = [data]
            if self.loader.dali_server.thread is None:
                raise RuntimeError("DALI server is not running")
            data = tree.map_structure(
                lambda entry: (
                    self.loader.dali_server.get_outputs(entry)
                    if isinstance(entry, DALIPipelineOutputRef)
                    else entry
                ),
                data,
            )
            return data

    def __init__(self, dali_server, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dali_server = dali_server

    def _get_iterator(self):
        return DataLoader._Iter(self)

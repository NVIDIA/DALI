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
from torch.utils.data._utils.collate import collate
from nvidia.dali.backend import TensorGPU, TensorListCPU, TensorListGPU
from nvidia.dali import types, Pipeline
from nvidia.dali.external_source import ExternalSource
import ctypes
import threading
from queue import Empty
from .. import to_torch_type


def _external_source_node_names(pipeline):
    if not pipeline._py_graph_built:
        pipeline._build_graph()
    input_node_names = []
    for op in pipeline._ops:
        if isinstance(op._op, ExternalSource):
            input_node_names.append(op.name)
    return input_node_names


def _to_torch_tensor(tensor_or_tl, device_id=0):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `tensor_or_tl` : TensorGPU or TensorListGPU
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    if isinstance(tensor_or_tl, (TensorListGPU, TensorListCPU)):
        dali_tensor = tensor_or_tl.as_tensor()
    else:
        dali_tensor = tensor_or_tl

    if isinstance(dali_tensor, (TensorGPU)):
        torch_device = torch.device("cuda", device_id)
    else:
        torch_device = torch.device("cpu")

    out_torch = torch.empty(
        dali_tensor.shape(),
        dtype=to_torch_type[dali_tensor.dtype],
        device=torch_device,
    )

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(out_torch.data_ptr())
    if isinstance(dali_tensor, (TensorGPU)):
        non_blocking = True
        cuda_stream = torch.cuda.current_stream(device=torch_device)
        cuda_stream = types._raw_cuda_stream(cuda_stream)
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        tensor_or_tl.copy_to_external(c_type_pointer, stream, non_blocking)
    else:
        tensor_or_tl.copy_to_external(c_type_pointer)

    return out_torch


class DALIPipelineOutputRef:
    """
    Placeholder for a pipeline output reference, after the iteration has been scheduled to DALI.
    """

    def __init__(self, info):
        self.info = info


class DALIProxy:
    def __init__(self, input_names, send_q):
        self.input_names = input_names
        # Shared queue with the server
        self.send_q = send_q
        # Torch worker id, to be filled on first call to worker_id()
        self._worker_id = None
        # Iteration index for the current worker
        self.data_idx = 0

    @property
    def worker_id(self):
        """Getter for 'worker_id'"""
        if self._worker_id is None:
            self._worker_id = torchdata.get_worker_info().id
        return self._worker_id

    class _PipelineRunRef:
        """
        Placeholder for a pipeline run reference, which is returned by the data worker instead of
        the actual data. The PyTorch worker returns this trivial object, only containing information
        about this proxy instance and the input data to the pipeline. Later in the collate function,
        we send the data for execution to DALI.
        """

        def __init__(self, proxy, inputs):
            self.proxy = proxy
            self.inputs = inputs
            if len(inputs) != len(self.proxy.input_names):
                raise RuntimeError(
                    f"Unexpected number of inputs. Expected: {self.input_names}, got: {inputs}"
                )

    def schedule_batch(self, inputs):
        # Identifier of this request
        info = (self.worker_id(), self.data_idx)
        torch.cuda.nvtx.range_push(f"dali_proxy.send_q.put {info}")
        self.send_q.put((info, inputs))
        torch.cuda.nvtx.range_pop()
        self.data_idx = self.data_idx + 1
        # Returns a placeholder, which is replaced with the actual data once the iteration completes
        return DALIPipelineOutputRef(self.info)

    def __call__(self, *inputs):
        """
        Returns a reference to the pipeline run
        """
        if len(inputs) != len(self.input_names):
            raise RuntimeError(
                f"Unexpected number of inputs. Expected: {self.input_names}, got: {inputs}"
            )
        return self._PipelineRunRef(self, inputs)


class DALIServer:
    def __init__(self, pipeline, input_names=None):
        """
        Initializes a new DALI proxy instance.

        Args:
            input_names (list): list of strings representing the inputs to the pipeline. Those
            should match the names of the ``external_source`` nodes in the DALI pipeline.
        """
        assert isinstance(pipeline, Pipeline), f"Expected an NVIDIA DALI pipeline, got: {pipeline}"
        self.pipe = pipeline
        self.pipe_input_names = _external_source_node_names(self.pipe)
        if len(self.pipe_input_names) == 0:
            raise RuntimeError("The provided pipeline doesn't have any inputs")
        if len(self.pipe_input_names) == 1:
            assert input_names is None or input_names[0] == self.pipe_input_names[0]
            self.input_names = self.pipe_input_names
        elif input_names is None or len(input_names) != len(self.pipe_input_names):
            raise RuntimeError(
                "The provided pipeline has more than one output. In such case, the argument "
                "`input_names` should containi the same exact number of strings, one for "
                "each pipeline input to be mapped by the proxy callable object"
            )
        self.input_names = input_names
        self.num_inputs = len(input_names)

        # Multi-process queue used to transfer data from the pytorch workers to the main process
        self.send_q = mp.Queue()
        # Multi-process queue used by the main process to remember the actual order
        # of execution of the requests
        self.order_q = mp.Queue()

    @property
    def proxy(self):
        return DALIProxy(self.input_names, self.send_q)

    def next_outputs(self):
        # Get the information about the order of execution, so that we know which one is
        # the next iteration
        torch.cuda.nvtx.range_push("order_q.get")
        info = self.dali_proxy.order_q.get()
        torch.cuda.nvtx.range_pop()

        # Get the outputs from the current iteration
        torch.cuda.nvtx.range_push(f"pipe.outputs {info}")
        outputs = self.pipe.outputs()
        torch.cuda.nvtx.range_pop()

        # Return information about the iteration, together with the data
        processed_outputs = tuple(
            [_to_torch_tensor(output, device_id=self.pipe.device_id) for output in outputs]
        )
        return (info, processed_outputs)

    def get_outputs(self, req_info):
        req_outputs = None
        # If the data was already read, just return it (and clear the cache entry)
        if req_info in self.cache_outputs:
            req_outputs = self.cache_outputs[req_info]
            del self.cache_outputs[req_info]
            del self.cache_inputs[req_info]
        else:
            info = None
            # If not the data we are looking for, store it and keep processing until we find it
            while req_info != info:
                info, processed_outputs = self.next_outputs()
                if info == req_info:
                    req_outputs = processed_outputs
                    del self.cache_inputs[req_info]
                else:
                    self.cache_outputs[info] = processed_outputs
        # Unpack single element tuples
        if isinstance(req_outputs, tuple) and len(req_outputs) == 1:
            req_outputs = req_outputs[0]
        return req_outputs

    def thread_fn(self):
        """
        Asynchronous DALI thread that gets iteration data from the queue and schedules it
        for execution
        """
        while not self.thread_stop_event.is_set():
            try:
                torch.cuda.nvtx.range_push("send_q.get")
                info, inputs = self.send_q.get(timeout=5)
                torch.cuda.nvtx.range_pop()
                self.cache_inputs[info] = inputs
            except mp.TimeoutError:
                continue
            except Empty:
                continue
            torch.cuda.nvtx.range_push(f"order_q.put {info}")
            self.order_q.put(info)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"feed_input {info}")
            for idx, input_name in enumerate(self.input_names):
                self.pipe.feed_input(input_name, inputs[idx])
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"schedule_run {info}")
            self.pipe.schedule_run()
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

    def __exit__(self, exc_type, exc_value, tb):
        self.stop_thread()


def _collate_pipeline_run_ref_fn(pipe_out, *, collate_fn_map=None):
    """
    Special collate function that schedules a batch for execution
    """
    assert len(pipe_out) > 0
    first_elem = pipe_out[0]
    inputs = [[] for idx in range(len(first_elem.inputs))]
    proxy = first_elem.proxy
    for elem in pipe_out:
        assert proxy == elem.proxy
        for idx, input_ref in enumerate(elem.inputs):
            inputs[idx].append(input_ref)
    return proxy.schedule_batch(inputs)


def _custom_collate(batch):
    """
    Subscribe a special collate function for PipelineRunRef, that handles the scheduling
    of the iteration on the fly
    """
    collate_fn_map = torchdata._utils.collate.default_collate_fn_map
    collate_fn_map.update({DALIServer.PipelineRunRef: _collate_pipeline_run_ref_fn})
    return collate(batch, collate_fn_map=collate_fn_map)


class DataLoader(torchdata.dataloader.DataLoader):
    """
    DALI data loader to be used in the main loop, which runs the DALI pipeline doing the
    processing asynchronously with regards to the training.
    """

    class Iterator(torchdata.dataloader._MultiProcessingDataLoaderIter):
        """
        Data loader iterator used by the DALI proxy data loader
        """

        def __init__(self, loader):
            super().__init__(loader)
            self.loader = loader

        def _next_data(self):
            data = super()._next_data()
            if not hasattr(data, "__iter__"):
                print(
                    "Warning: Non iterable returned from dataloader. Please "
                    " review the code, since it usually indicates a bug in the pipeline."
                )
                data = [data]
            for data_idx, data_elem in enumerate(data):
                # If loader returns a dictionary the iterator iterates over its keys.
                # We need to access a value. Probably need to address more casess.
                if isinstance(data, dict):
                    if isinstance(data[data_elem], DALIPipelineOutputRef):
                        data[data_elem] = self.loader.dali_server.get_outputs(data[data_elem].info)
                if isinstance(data_elem, DALIPipelineOutputRef):
                    data[data_idx] = self.loader.dali_server.get_outputs(data_elem.info)
            return data

    def __init__(self, dali_server, *args, **kwargs):
        self.dali_server = dali_server
        super().__init__()
        if "collate_fn" in kwargs and kwargs["collate_fn"] is not None:
            print(
                "Warning: Make sure to handle DALIServer.PipelineRunRef when providing"
                " a custom collate_fn (see collate_pipeline_run_ref_fn)"
            )
        else:
            kwargs["collate_fn"] = _custom_collate

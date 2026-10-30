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

from abc import ABC, abstractmethod
from typing import List, Sequence, Callable, Literal, Union

import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.backend import TensorListCPU, TensorListGPU

from .operator import _ValidateTensorOrImage
from .totensor import ToPureTensor, PILToTensor, ToPILImage
from .functional.totensor import to_pil_image

import numpy as np
import multiprocessing
from PIL import Image
import torch

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_THREADS = 1 if multiprocessing.cpu_count() == 1 else multiprocessing.cpu_count() // 2


def _to_torch_tensor(tensor_or_tl: TensorListGPU | TensorListCPU) -> torch.Tensor:
    if isinstance(tensor_or_tl, (TensorListGPU, TensorListCPU)):
        dali_tensor = tensor_or_tl.as_tensor()
    else:
        dali_tensor = tensor_or_tl

    return torch.from_dlpack(dali_tensor)


def to_torch_tensor(
    x: Union[tuple, "TensorListGPU", "TensorListCPU"],
) -> Union[torch.Tensor, tuple]:
    """
    Converts a DALI tensor or tensor list to a PyTorch tensor.

    Parameters
    ----------
        tensor_or_tl : tuple, TensorListGPU, TensorListCPU
            DALI tensor or tensor list.
    """
    if isinstance(x, (TensorListGPU, TensorListCPU)):
        return to_torch_tensor(x.as_tensor())
    elif isinstance(x, tuple):
        if len(x) == 1:
            return _to_torch_tensor(x[0])
        return tuple(to_torch_tensor(elem) for elem in x)
    else:
        return torch.from_dlpack(x)


@pipeline_def(enable_conditionals=True, exec_dynamic=True, prefetch_queue_depth=1)
def _pipeline_function(op_list: Sequence, layout: str = "HWC", input_device: str = "gpu"):
    """
    Builds a DALI pipeline from a list of operators.

    Parameters
    ----------
        op_list : list
            List of DALI operators.
        layout : str
            Layout of the data.
    """
    input_node = fn.external_source(name="input_data", layout=layout, device=input_device)
    for op in op_list:
        input_node = op._invoke(input_node)
    return input_node


class PipelineWithLayout(ABC):
    """Base class for pipeline layouts.

    This class is a base class for DALI pipelines with a specific layout. It is used to handle
    the layout of the data.
    Single DALI Pipeline can only use one layout at a time.

    Parameters
    ----------
    op_list : list
        List of DALI operators.
    layout : str
        Layout of the data.
    batch_size : int, optional, default = DEFAULT_BATCH_SIZE
        Batch size.
    num_threads : int, optional, default = DEFAULT_NUM_THREADS
        Number of threads.
    **dali_pipeline_kwargs
        Additional keyword arguments for the DALI pipeline.
    """

    def _cuda_run(self, data_input):
        if isinstance(data_input, torch.Tensor) and data_input.is_cuda:
            device_id = data_input.device.index
        else:
            device_id = torch.cuda.current_device()

        stream = torch.cuda.current_stream(device=device_id)

        with torch.cuda.stream(stream):
            output = self.pipe.run(stream, input_data=data_input)

        return output

    def _cpu_run(self, data_input):
        return self.pipe.run(input_data=data_input)

    @staticmethod
    def _get_output_format(op_list: List[Callable[..., Sequence[_DataNode] | _DataNode]]) -> str:
        output_type = "default"
        for op in op_list:
            if isinstance(op, (ToPureTensor, PILToTensor)):
                output_type = "tensor"
            elif isinstance(op, ToPILImage):
                output_type = "pil"

        return output_type

    def __init__(
        self,
        op_list: List[Callable[..., Sequence[_DataNode] | _DataNode]],
        layout: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int = DEFAULT_NUM_THREADS,
        **dali_pipeline_kwargs,
    ):
        self.batch_size = batch_size
        self.output_type = PipelineWithLayout._get_output_format(op_list)
        self.device = op_list[0].device if len(op_list) > 0 else "cpu"
        self.torch_device_type = "cuda" if self.device == "gpu" else "cpu"

        self.pipe = _pipeline_function(
            op_list,
            layout=layout,
            input_device=self.device,
            batch_size=batch_size,
            num_threads=num_threads,
            **dali_pipeline_kwargs,
        )
        self._internal_run = self._cuda_run if torch.cuda.is_available() else self._cpu_run

    def run(self, data_input):

        output = self._internal_run(data_input)

        if output is None:
            return output

        return to_torch_tensor(output)

    @abstractmethod
    def output_to_tensor(self, output: torch.Tensor) -> torch.Tensor:
        """Return the pipeline output as a CHW ``torch.Tensor``."""
        ...

    @abstractmethod
    def output_to_pil(self, output: torch.Tensor) -> Image.Image:
        """Return the pipeline output tensor as a ``PIL.Image``."""
        ...

    @abstractmethod
    def get_layout(self) -> str: ...

    @abstractmethod
    def get_channel_reverse_idx(self) -> int: ...

    @abstractmethod
    def verify_layout(self, data) -> None: ...

    def is_conversion_to_tensor(self) -> bool:
        return self.output_type == "tensor"


class PipelineHWC(PipelineWithLayout):
    """Handles ``PIL.Image`` in HWC format.

    This class prepares data to be passed to a DALI pipeline, runs the pipeline and converts
    the output to a ``PIL.Image``.

    Parameters
    ----------
    op_list : list
        List of DALI operators.
    batch_size : int, optional, default = DEFAULT_BATCH_SIZE
        Batch size.
    num_threads : int, optional, default = DEFAULT_NUM_THREADS
        Number of threads.
    **dali_pipeline_kwargs
        Additional keyword arguments for the DALI pipeline.
    """

    def __init__(
        self,
        op_list: List[Callable[..., Sequence[_DataNode] | _DataNode]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int = DEFAULT_NUM_THREADS,
        **dali_pipeline_kwargs,
    ):
        super().__init__(
            op_list,
            layout="HWC",
            batch_size=batch_size,
            num_threads=num_threads,
            **dali_pipeline_kwargs,
        )

    def run(self, data_input):
        if isinstance(data_input, Image.Image):
            _input = torch.as_tensor(np.array(data_input, copy=True)).unsqueeze(0)
            if data_input.mode == "L":
                _input = _input.unsqueeze(-1)
        else:
            raise ValueError("HWC layout is currently supported for PIL Images only.\
                Please check if samples have the same format.")

        output = super().run(_input)  # (H, W, C)

        if self.output_type == "tensor":
            return self.output_to_tensor(output)

        # default: PIL output
        if output.shape[0] > 1:
            return [self.output_to_pil(output[i]) for i in range(output.shape[0])]
        return self.output_to_pil(output[0])

    def output_to_tensor(self, output: torch.Tensor) -> torch.Tensor:
        """Convert a single-sample HWC tensor to a CHW tensor"""
        if output.shape[0] == 1:
            return output.squeeze(0).permute(2, 0, 1).contiguous()  # → (C, H, W)
        return output.permute(0, 3, 1, 2).contiguous()  # → (N, C, H, W)

    def output_to_pil(self, output: torch.Tensor) -> Image.Image:
        """Convert a single-sample HWC tensor to a ``PIL.Image``."""
        # Pipeline stores data as HWC; to_pil_image expects CHW, so permute first.
        return to_pil_image(output, input_layout="HWC")

    def get_layout(self) -> str:
        return "HWC"

    def get_channel_reverse_idx(self) -> int:
        return -1

    def verify_layout(self, data_input) -> None:
        if not isinstance(data_input, Image.Image):
            raise TypeError(f"The pipeline expects PIL.Images as input got {type(data_input)}")


class PipelineCHW(PipelineWithLayout):
    """Handles ``torch.Tensors`` in CHW (unbatched) or NCHW (batched) format.

    Both layouts describe samples in ``CHW`` layout and differ only in whether the input has
    a leading batch dimension. Random parameters are sampled once per ``CHW`` sample.
    ``CHW`` expects exactly 3 dimensions. For ``NCHW`` any leading dimensions are flattened into
    the batch dimension, and restored in the output.

    This class prepares data to be passed to a DALI pipeline and runs the pipeline, converting
    the output to a ``torch.Tensor``.

    Parameters
    ----------
    op_list : list
        List of DALI operators.
    batch_size : int, optional, default = DEFAULT_BATCH_SIZE
        Batch size.
    num_threads : int, optional, default = DEFAULT_NUM_THREADS
        Number of threads.
    batched : bool, optional, default = False
        Whether the input has a leading batch dimension (``NCHW``) or not (``CHW``).
    **dali_pipeline_kwargs
        Additional keyword arguments for the DALI pipeline.
    """

    sample_layout = "CHW"

    def __init__(
        self,
        op_list: List[Callable[..., Sequence[_DataNode] | _DataNode]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_threads: int = DEFAULT_NUM_THREADS,
        batched: bool = False,
        **dali_pipeline_kwargs,
    ):
        self.batched = batched
        super().__init__(
            op_list,
            layout=self.sample_layout,
            batch_size=batch_size,
            num_threads=num_threads,
            **dali_pipeline_kwargs,
        )

    def run(self, data_input):
        if not isinstance(data_input, torch.Tensor):
            raise ValueError(
                f"{self.get_layout()} layout is currently supported for torch.Tensor only.\
                Please check if samples have the same format."
            )

        shape = data_input.shape
        if self.batched:
            # Any leading dimensions are flattened into the batch dimension
            tail = len(self.sample_layout)
            supported = data_input.ndim > tail
            expected = f"at least {tail + 1} dimensions"
        else:
            tail = 3
            if self.sample_layout == "FCHW":
                # Any leading dimensions are flattened into the frames dimension, as in torchvision
                supported = data_input.ndim > tail
                expected = "at least 4 dimensions"
            else:
                supported = data_input.ndim == tail
                expected = "exactly 3 dimensions"
        if not supported:
            raise ValueError(
                f"Unsupported input: a tensor with {data_input.ndim} dimensions for layout "
                f"{self.get_layout()}, expected {expected}."
            )
        if data_input.numel() == 0:
            raise ValueError(f"Empty input of shape {tuple(shape)} is not supported.")
        leading = shape[:-tail]

        _input = data_input.reshape(-1, *shape[-tail:])
        if self.batched and _input.shape[0] > self.batch_size:
            raise ValueError(
                f"Input of shape {tuple(shape)} flattens to {_input.shape[0]} samples "
                f"for layout {self.get_layout()}, but batch_size is {self.batch_size}. "
                "Increase batch_size."
            )
        if self.sample_layout == "FCHW" and not self.batched:
            # DALI requires batch size to be present
            _input = _input.unsqueeze(0)
        output = super().run(_input)

        if self.batched:
            output = output.reshape(*leading, *output.shape[1:])
        else:
            # Remove the batch dimension we added above
            output = output.squeeze(0)
            output = output.reshape(*leading, *output.shape[-3:])

        if self.output_type == "pil":
            return self.output_to_pil(output)

        return output

    def output_to_tensor(self, output: torch.Tensor) -> torch.Tensor:
        """Return the pipeline output as a ``torch.Tensor``."""
        return output

    def output_to_pil(self, output: torch.Tensor):
        """Convert the output to ``PIL.Image``. Extra leading dimensions yield (nested) lists."""
        if output.ndim == len(self.sample_layout):
            if self.sample_layout == "CHW":
                return to_pil_image(output)
            return [to_pil_image(frame) for frame in output]
        if output.shape[0] > 1:
            return [self.output_to_pil(output[i]) for i in range(output.shape[0])]
        return self.output_to_pil(output[0])

    def get_layout(self) -> str:
        return ("N" if self.batched else "") + self.sample_layout

    def get_channel_reverse_idx(self) -> int:
        return -3

    def verify_layout(self, data_input) -> None:
        if not isinstance(data_input, torch.Tensor):
            raise TypeError(f"The pipeline expects torch.Tensor as input got {type(data_input)}")


class PipelineFCHW(PipelineCHW):
    """Handles ``torch.Tensors`` in FCHW (unbatched) or NFCHW (batched) format.

    Both layouts describe samples in ``FCHW`` layout (a sequence of frames) and differ only
    in whether the input has a leading batch dimension. Random parameters are sampled once per
    ``FCHW`` sample, so all frames of a sample are transformed identically.
    """

    sample_layout = "FCHW"


# layout -> (pipeline class, input has a batch dimension)
PipelineLayouts = {
    "HWC": (PipelineHWC, False),
    "CHW": (PipelineCHW, False),
    "NCHW": (PipelineCHW, True),
    "FCHW": (PipelineFCHW, False),
    "NFCHW": (PipelineFCHW, True),
}
_UNSUPPORTED_LAYOUTS = ("NHWC", "FHWC", "NFHWC")
_ALLOWED_LAYOUTS = (None, "HWC", "CHW", "NCHW", "FCHW", "NFCHW")


class Compose:
    """
    Composes transforms together in a single pipeline

    This class chains multiple DALI operations in a sequential manner, similar to
    ``torchvision.transforms.Compose``. The ``Compose`` class implements a callable which runs
    the pipeline.

    The DALI pipeline is built on the first call, and its input layout and type are fixed from then
    on. Use a separate ``Compose`` instance for inputs with a different layout (e.g. ``CHW`` and
    ``FCHW`` tensors) or type (``PIL.Image`` and ``torch.Tensor``).

    Parameters
    ----------
    op_list : list
        List of DALI operators.
    batch_size : int, optional, default = None
        Maximum number of samples in a batch, must be at least 1. If not set, ``Compose`` follows
        torchvision: every call is a single sample with one set of random parameters. Setting it
        enables the batched mode, which is an extension to torchvision: each sample in the batch
        gets its own random parameters. ``batch_size > 1`` contradicts a ``layout`` without ``N``
        and raises ``ValueError``.
    num_threads : int, optional, default = DEFAULT_NUM_THREADS
        Number of threads.
    layout : str, optional, default = None
        Layout of the input data. One of ``None``, ``"HWC"``, ``"CHW"``, ``"NCHW"``, ``"FCHW"``,
        ``"NFCHW"`` (``"NHWC"``, ``"FHWC"`` and ``"NFHWC"`` are not supported yet).
        ``N`` is a batch of samples, ``F`` is a sequence of frames. Random parameters are sampled
        once per sample, i.e. per ``CHW`` image for ``NCHW`` and per ``FCHW`` sequence for
        ``NFCHW``. ``NCHW`` and ``NFCHW`` are extensions to torchvision and require
        ``batch_size``. Leading dimensions beyond the layout are flattened into ``N`` (or into
        ``F`` for ``FCHW``).
        If ``None``, the layout is inferred from the input: ``PIL.Image`` is ``HWC``.
        For ``torch.Tensor`` without ``batch_size`` (torchvision behavior): 3 dimensions is
        ``CHW``, 4 or more is ``FCHW``. With ``batch_size`` set: 3 dimensions is ``CHW``,
        4 is ``NCHW``, 5 or more is ``NFCHW``.
    **dali_pipeline_kwargs
        Additional keyword arguments for the DALI pipeline.
    """

    def __init__(
        self,
        op_list: List[Callable[..., Sequence[_DataNode] | _DataNode]],
        batch_size: int | None = None,
        num_threads: int = DEFAULT_NUM_THREADS,
        layout: Literal[
            None, "HWC", "CHW", "NHWC", "NCHW", "FCHW", "FHWC", "NFCHW", "NFHWC"
        ] = None,
        **dali_pipeline_kwargs,
    ):
        if layout not in _ALLOWED_LAYOUTS:
            raise ValueError(f"Unknown layout {layout!r}. Allowed values: {_ALLOWED_LAYOUTS}.")
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer or None, got {batch_size}.")
        if layout is not None:
            if layout in _UNSUPPORTED_LAYOUTS:
                raise NotImplementedError(f"Layout {layout!r} is currently not supported.")
            if layout.startswith("N") and batch_size is None:
                raise ValueError(f"Layout {layout!r} requires batch_size.")
            if not layout.startswith("N") and batch_size is not None and batch_size > 1:
                raise ValueError(f"Layout {layout!r} has no batch dimension, got {batch_size=}.")
        self.op_list = op_list
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.layout = layout
        self.active_pipeline = None
        self.dali_pipeline_kwargs = dali_pipeline_kwargs

    def _infer_layout(self, data_input) -> str:
        """Infer the layout from the input type, its dimensions and the batch size."""
        if isinstance(data_input, Image.Image):
            return "HWC"
        if not isinstance(data_input, torch.Tensor):
            raise ValueError("Currently only PILImages and torch.Tensors are supported")

        ndim = data_input.ndim
        if ndim < 3:
            raise ValueError(
                f"Unsupported input: a tensor with {ndim} dimensions, expected at least 3 "
                "dimensions (..., C, H, W)."
            )
        if ndim == 3:
            return "CHW"
        if not self.batch_size:
            return "FCHW"
        return "NCHW" if ndim == 4 else "NFCHW"

    def _verify_pil_support(self):
        """Raise if an operator that does not accept PIL images would receive one."""
        for op in self.op_list:
            if isinstance(op, (ToPureTensor, PILToTensor)):
                return  # the data is a tensor from here on
            # Wrappers that are not Operator subclasses (e.g. Pad) have no such attribute
            if not getattr(op, "supports_pil", True):
                raise TypeError(
                    f"{type(op).__name__} does not support PIL images. "
                    "Convert the image to a tensor first, e.g. with PILToTensor."
                )

    def _build_pipeline(self, data_input):
        if isinstance(data_input, Image.Image):
            self._verify_pil_support()
        layout = self.layout or self._infer_layout(data_input)
        pipeline_cls, batched = PipelineLayouts[layout]
        kwargs = dict(self.dali_pipeline_kwargs)
        if pipeline_cls is not PipelineHWC:
            kwargs["batched"] = batched
        pipeline_batch_size = self.batch_size if batched else 1
        self.active_pipeline = pipeline_cls(
            self.op_list, pipeline_batch_size, self.num_threads, **kwargs
        )

    def __call__(self, data_input):
        """
        Runs the pipeline

        The ``Pipeline`` class builds a graph based on the operations list passed in
        the constructor. Next, whenever the ``Compose`` object is called it starts the pipeline
        and returns results.

        Parameters
        ----------
            data_input: Tensor or PIL Image
                In case of PIL image it will be converted to tensor before sending to pipeline
        """

        _ValidateTensorOrImage.verify(data_input)

        if self.active_pipeline is None:
            self._build_pipeline(data_input)

        self.active_pipeline.verify_layout(data_input)

        return self.active_pipeline.run(data_input=data_input)

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import collections
from typing import Sequence, Union, Literal
from types import NoneType


import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn

from .operator import Operator


def get_inputHW(data_input):
    """
    Gets the height and width of the input data.

    Parameters
    ----------
    data_input : Tensor
        Input data to get the height and width of.

    Returns
    -------
    input_height : int
        Height of the input data.
    input_width : int
        Width of the input data.
    """
    layout = data_input.property("layout")[0]

    # CWH
    if layout == np.frombuffer(bytes("C", "utf-8"), dtype=np.uint8)[0]:
        input_height = data_input.shape()[1]
        input_width = data_input.shape()[2]
    # HWC
    else:
        input_height = data_input.shape()[0]
        input_width = data_input.shape()[1]

    return input_height, input_width, data_input


class PadBase(Operator):
    """
    Base class for padding operators.

    Set ``border_type`` in child class to the desired border type.

    Parameters
    ----------
    padding : int or sequence
        Padding on each border. If a single int is provided this is used to pad all borders.
        If sequence of length 2 is provided this is the padding on left/right and top/bottom
        respectively. If a sequence of length 4 is provided this is the padding for the left,
        top, right and bottom borders respectively.
    fill : number or tuple or dict, optional, default = 0
        Pixel fill value used when the padding_mode is constant.
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the padding. Can be ``"cpu"`` or ``"gpu"``.
    """

    preprocess_data = get_inputHW
    border_type = ""

    @classmethod
    def get_padding(cls, padding):
        if isinstance(padding, int):
            return (
                padding,
                padding,
                padding,
                padding,
            )
        elif isinstance(padding, Sequence):
            if len(padding) == 2:
                return (padding[0], padding[1], padding[0], padding[1])
            elif len(padding) == 4:
                return padding

        raise TypeError(
            f"Padding must be an int or a sequence of length 2 or 4, got {type(padding)}"
        )

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        fill: Union[
            int,
            float,
            Sequence[int],
            Sequence[float],
            None,
            dict[
                Union[type, str],
                Union[
                    int,
                    float,
                    collections.abc.Sequence[int],
                    collections.abc.Sequence[float],
                    NoneType,
                ],
            ],
        ] = 0,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        super().__init__(device=device)
        self.pad_left, self.pad_top, self.pad_right, self.pad_bottom = PadBase.get_padding(padding)
        self.fill = fill
        self.border_type = type(self).border_type

    def _kernel(self, data_input):
        """ "
        Applies the padding to the input data using the ``fn.slice`` operator.

        Parameters
        ----------
        data_input : Tensor
            Input data to apply the padding to.

        Returns
        -------
        Tensor
            Padded data.
        """
        input_height, input_width, data_input = data_input

        data_input = fn.slice(
            data_input,
            fn.stack(
                fn.cast(-self.pad_left, dtype=dali.types.INT64),
                fn.cast(-self.pad_top, dtype=dali.types.INT64),
            ),  # __anchor
            fn.stack(
                input_width + self.pad_right + self.pad_left,
                input_height + self.pad_bottom + self.pad_top,
            ),  # __shape
            out_of_bounds_policy=self.border_type,
            fill_values=self.fill,
            device=self.device,
        )

        return data_input


class PadConstant(PadBase):
    """
    Implementation of padding with a constant value.
    """

    border_type = "pad"


class PadEdge(PadBase):
    """
    Implementation of padding with edges,e.g.:
    input: [1, 2, 3, 4, 5, 6]
    padded: [1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6]
    """

    border_type = "clamp"


class PadSymmetric(PadBase):
    """
    Implementation of symmetric padding:
    input: [1, 2, 3, 4, 5, 6]
    padded: [3, 2, 1, 1, 2, 3, 4, 5, 6, 6, 5, 4]
    notice, that the first value in padding is repeated
    """

    border_type = "reflect_1001"


class PadReflect(PadBase):
    """
    Implementation of reflection padding:
    input: [1, 2, 3, 4, 5, 6]
    padded: [3, 2, 1,  2, 3, 4, 5, 6, 5, 4]
    notice, that the first value in padding is not repeated
    """

    border_type = "reflect_101"


PADDING_CLASS = {
    "constant": PadConstant,
    "edge": PadEdge,
    "reflect": PadReflect,
    "symmetric": PadSymmetric,
}


class Pad:
    """
    Pad the input on all sides with the given ``padding_mode`` type and ``fill`` value.

    Parameters
    ----------
    padding : int or sequence
        Padding on each border. If a single int is provided this is used to pad all borders.
        If sequence of length 2 is provided this is the padding on left/right and top/bottom
        respectively. If a sequence of length 4 is provided this is the padding for the left,
        top, right and bottom borders respectively.
    fill : number or tuple or dict, optional, default = 0
        Pixel fill value used when the padding_mode is constant.
    padding_mode : Literal["constant", "edge", "reflect", "symmetric"], optional,
        Type of padding. Should be: constant, edge, reflect or symmetric. Default is “constant”.
            constant: pads with a constant value, this value is specified with ``fill``
            edge: pads with the last value at the edge of the image.
            reflect: pads with reflection of image without repeating the last value on the edge.
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image repeating the last value on the edge.
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    device : Literal["cpu", "gpu"], optional, default = "cpu"
    """

    def __init__(
        self,
        padding: Union[int, Sequence[int]],
        fill: Union[
            int,
            float,
            Sequence[int],
            Sequence[float],
            None,
            dict[
                Union[type, str],
                Union[
                    int,
                    float,
                    collections.abc.Sequence[int],
                    collections.abc.Sequence[float],
                    NoneType,
                ],
            ],
        ] = 0,
        padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        self.device = device
        self.pad = PADDING_CLASS[padding_mode](padding, fill, device)

    def __call__(self, data_input):
        return self.pad(data_input)

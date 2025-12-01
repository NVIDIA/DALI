# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Callable, Type, Tuple, Optional, Union

from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.auto_aug.core._augmentation import Augmentation

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "Could not import numpy. DALI's automatic augmentation examples depend on numpy. "
        "Please install numpy to use the examples."
    )

from numpy import typing as npt


def augmentation(
    function: Optional[Callable[..., _DataNode]] = None,
    *,
    mag_range: Optional[Union[Tuple[float, float], np.ndarray]] = None,
    randomly_negate: Optional[bool] = None,
    mag_to_param: Optional[Callable[[float], npt.ArrayLike]] = None,
    param_device: Optional[str] = None,
    name: Optional[str] = None,
    augmentation_cls: Optional[Type[Augmentation]] = None,
):
    """
    A decorator turning transformations implemented with DALI into augmentations that
    can be used by automatic augmentations (e.g. AutoAugment, RandAugment, TrivialAugment).

    The `function` must accept at least two args: a sample and a parameter.
    The decorator handles computation of the parameter. Instead of the parameter, the
    decorated augmentation accepts magnitude bin and the total number of bins.
    Then, the bin is used to compute the parameter as if by calling
    `mag_to_param(magnitudes[magnitude_bin] * ((-1) ** random_sign))`, where
    `magnitudes=linspace(mag_range[0], mag_range[1], num_magnitude_bins)`.

    Args
    ----
    function : callable
        A function that accepts at least two positional args: a batch
        (represented as DataNode) to be processed, and a parameter of the transformation.
        The function must use DALI operators to process the batch and return a single output
        of such processing.
    mag_range : (number, number) or np.ndarray
        Specifies the range of applicable magnitudes for the operation.
        If the tuple is provided, the magnitudes will be computed as
        `linspace(mag_range[0], mag_range[1], num_magnitude_bins)`.
        If the np.ndarray is provided, it will be used directly instead of the linspace.
        If no `mag_range` is specified, the parameter passed to the `function` will be `None`.
    randomly_negate: bool
        If `True`, the magnitude from the mag_range will be randomly negated for every sample.
    mag_to_param: callable
        A callback that transforms the magnitude into a parameter. The parameter will be passed to
        the decorated operation instead of the plain magnitude. This way, the parameters for the
        range of magnitudes can be computed once in advance and stored as a Constant node.
        Note, the callback must return numpy arrays or data directly convertible to numpy arrays
        (in particular, no pipelined DALI operators can be used in the callback).
        The output type and dimensionality must be consistent and not depend on the magnitude.
    param_device: str
        A "cpu", "gpu", or "auto"; defaults to "cpu". Describes where to store the precomputed
        parameters (i.e. the `mag_to_param` outputs). If "auto" is specified, the CPU or GPU
        backend will be selected to match the `sample`'s backend.
    name: str
        Name of the augmentation. By default, the name of the decorated function is used.

    Returns
    -------
    Augmentation
        The operation wrapped with the Augmentation class so that it can be used with the `auto_aug`
        transforms.
    """

    def decorator(function):
        cls = augmentation_cls or Augmentation
        return cls(
            function,
            mag_range=mag_range,
            mag_to_param=mag_to_param,
            randomly_negate=randomly_negate,
            param_device=param_device,
            name=name,
        )

    if function is None:
        return decorator
    else:
        if not callable(function):
            raise Exception(
                f"The `@augmentation` decorator was used to decorate the object that "
                f"is not callable: {function}."
            )
        elif isinstance(function, Augmentation):
            # it's not clear if we should go with "update the setup" or
            # "discard and create" semantics here
            raise Exception(
                f"The `@augmentation` was applied to already decorated Augmentation. "
                f"Please call `{function.name}.augmentation` method to modify the augmentation "
                f"setup or apply the decorator to the underlying `{function.name}.op` directly.\n"
                f"Error in augmentation: {function}."
            )
        return decorator(function)

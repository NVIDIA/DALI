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

import warnings

from typing import Callable, Tuple, Optional, Union

from nvidia.dali import fn, types
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.auto_aug.core._args import (
    filter_extra_accepted_kwargs,
    get_missing_kwargs,
    get_num_positional_args,
    MissingArgException,
)

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "Could not import numpy. DALI's automatic augmentation examples depend on numpy. "
        "Please install numpy to use the examples."
    )

from numpy import typing as npt


class _UndefinedParam:
    """Use _UndefinedParam as a kwarg default when it matters to distinguish between `kwarg=None`
    and not specifying the kwarg"""


class _SignedMagnitudeBin:
    def __init__(
        self,
        magnitude_bin: Union[int, _DataNode],
        random_sign: _DataNode,
        signed_magnitude_idx: _DataNode,
    ):
        self._magnitude_bin = magnitude_bin
        self._random_sign = random_sign
        self._signed_magnitude_idx = signed_magnitude_idx

    def __getitem__(self, idx: int):
        """
        Indexing simplifies creation of "signed magnitude bins" in cases when a single sample
        may be processed by a sequence of random augmentations - we can sample random signs once
        for the full sequence and then use indexing to access each single signed magnitude bin.
        """
        if isinstance(self._magnitude_bin, int):
            magnitude_bin = self._magnitude_bin
        else:
            magnitude_bin = self._magnitude_bin[idx]
        cls = self.__class__
        return cls(magnitude_bin, self._random_sign[idx], self._signed_magnitude_idx[idx])

    @classmethod
    def create_from_bin(
        cls,
        magnitude_bin: Union[int, _DataNode],
        random_sign: Optional[_DataNode] = None,
        seed: Optional[int] = None,
        shape: Optional[Tuple] = None,
    ):
        if not isinstance(magnitude_bin, (int, _DataNode)):
            raise Exception(
                f"The `magnitude_bin` must be an int or _DataNode (output of DALI op "
                f"or `types.Constant`) representing batch of ints from "
                f"`[0..num_magnitude_bins-1]` range. Got {magnitude_bin} instead."
            )
        if random_sign is not None and any(arg is not None for arg in (seed, shape)):
            raise Exception(
                "The `random_sign` cannot be specified together with neither `seed` nor `shape`."
            )
        if random_sign is None:
            random_sign = fn.random.uniform(
                values=[0, 1], dtype=types.INT32, seed=seed, shape=shape
            )
        # it is important to compute it as soon as possible - we may be created at the top level
        # in the pipeline, while it may be read in conditional split
        signed_magnitude_idx = 2 * magnitude_bin + random_sign
        return cls(magnitude_bin, random_sign, signed_magnitude_idx)

    @staticmethod
    def _remap_to_signed_magnitudes(magnitudes):
        def remap_bin_idx(bin_idx):
            magnitude = magnitudes[bin_idx // 2]
            if bin_idx % 2:
                magnitude = -magnitude
            return magnitude

        return np.array([remap_bin_idx(bin_idx) for bin_idx in range(2 * len(magnitudes))])

    @property
    def bin(self):
        return self._magnitude_bin

    @property
    def random_sign(self):
        return self._random_sign

    @property
    def signed_magnitude_idx(self):
        return self._signed_magnitude_idx


def signed_bin(
    magnitude_bin: Union[int, _DataNode],
    random_sign: Optional[_DataNode] = None,
    seed: Optional[int] = None,
    shape: Optional[Tuple] = None,
) -> _SignedMagnitudeBin:
    """
    Combines the `magnitude_bin` with information about the sign of the magnitude.
    The Augmentation wrapper can generate and handle the random sign on its own. Yet,
    if the augmentation is called inside conditional split, it is better to combine
    magnitude_bins and sign in advance, before the split, so that the sign handling is done
    once for the whole batch rather than multiple times for each op operating on the split batch.

    Args
    ----
    magnitude_bin: int or DataNode
        The magnitude bin from range `[0, num_magnitude_bins - 1]`. Can be plain int or
        a batch (_DataNode) of ints.
    random_sign : DataNode, optional
        A batch of {0, 1} integers. For augmentations declared with `randomly_negate=True`,
        it determines if the magnitude is negated (for 1) or not (for 0).
    """
    return _SignedMagnitudeBin.create_from_bin(magnitude_bin, random_sign, seed, shape)


class Augmentation:
    """
    Wrapper for transformations implemented with DALI that are meant to be used with
    automatic augmentations. You should not need to instantiate this class directly,
    use `@augmentation` decorator instead.
    """

    def __init__(
        self,
        op: Callable[..., _DataNode],
        mag_range: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        randomly_negate: Optional[bool] = None,
        mag_to_param: Optional[Callable[[float], npt.ArrayLike]] = None,
        param_device: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self._op = op
        self._mag_range = mag_range
        self._randomly_negate = randomly_negate
        self._mag_to_param = mag_to_param
        self._param_device = param_device
        self._name = name
        self._validate_op_sig()

    def __repr__(self):
        params = [
            f"{name}={repr(val)}" for name, val in self._get_config().items() if val is not None
        ]
        return f"Augmentation({', '.join([repr(self.op)] + params)})"

    def __call__(
        self,
        data: _DataNode,
        *,
        magnitude_bin: Optional[Union[int, _DataNode, _SignedMagnitudeBin]] = None,
        num_magnitude_bins: Optional[int] = None,
        **kwargs,
    ) -> _DataNode:
        """
        Applies the decorated transformation to the `data` as if by calling
        `self.op(data, param, **kwargs)` where
        `param = mag_to_param(magnitudes[magnitude_bin] * ((-1) ** random_sign))`.

        Args
        ----
        data : DataNode
            A batch of samples to be transformed.
        magnitude_bin: int, DataNode, or _SignedMagnitudeBin
            The magnitude bin from range `[0, num_magnitude_bins - 1]`. The bin is used to get
            parameter for the transformation. The parameter is computed as if by calling
            `mag_to_param(magnitudes[magnitude_bin] * ((-1) ** random_sign))`, where
            `magnitudes=linspace(mag_range[0], mag_range[1], num_magnitude_bins)`.
            If the `mag_range` is custom `np.ndarray`, it will be used as `magnitudes` directly.
        num_magnitude_bins: int
            The total number of magnitude bins (limits the accepted range of
            `magnitude_bin` to `[0, num_magnitude_bins - 1]`).
        kwargs
            Dictionary with extra arguments to pass to the `self.op`. The op's signature
            is checked for any additional arguments (apart from the ``data`` and ``parameter``) and
            the arguments with matching names are passed to the call.

        Returns
        -------
        DataNode
            A batch of transformed samples.
        """
        num_mandatory_positional_args = 2
        param_device = self._infer_param_device(data)
        params = self._get_param(magnitude_bin, num_magnitude_bins, param_device)
        op_kwargs = filter_extra_accepted_kwargs(self.op, kwargs, num_mandatory_positional_args)
        missing_args = get_missing_kwargs(self.op, kwargs, num_mandatory_positional_args)
        if missing_args:
            raise MissingArgException(
                f"The augmentation `{self.name}` requires following named argument(s) "
                f"which were not provided to the call: {', '.join(missing_args)}. "
                f"Please make sure to pass the required arguments when calling the "
                f"augmentation.",
                augmentation=self,
                missing_args=missing_args,
            )
        return self.op(data, params, **op_kwargs)

    @property
    def op(self):
        return self._op

    @property
    def mag_range(self):
        return self._mag_range

    @property
    def mag_to_param(self):
        return self._mag_to_param or _np_wrap

    @property
    def randomly_negate(self):
        return self._randomly_negate or False

    @property
    def param_device(self):
        return self._param_device or "cpu"

    @property
    def name(self):
        return self._name or self.op.__name__

    def augmentation(
        self,
        mag_range=_UndefinedParam,
        randomly_negate=_UndefinedParam,
        mag_to_param=_UndefinedParam,
        param_device=_UndefinedParam,
        name=_UndefinedParam,
        augmentation_cls=None,
    ):
        """
        The method to update augmentation parameters specified with `@augmentation` decorator.
        Returns a new augmentation with the original operation decorated but updated parameters.
        Parameters that are not specified are kept as in the initial augmentation.
        """
        cls = augmentation_cls or self.__class__
        config = self._get_config()
        for key, value in dict(
            mag_range=mag_range,
            randomly_negate=randomly_negate,
            mag_to_param=mag_to_param,
            param_device=param_device,
            name=name,
        ).items():
            assert key in config
            if value is not _UndefinedParam:
                config[key] = value
        return cls(self.op, **config)

    def _get_config(self):
        return {
            "mag_range": self._mag_range,
            "mag_to_param": self._mag_to_param,
            "randomly_negate": self._randomly_negate,
            "param_device": self._param_device,
            "name": self._name,
        }

    def _infer_param_device(self, sample: _DataNode):
        if self.param_device != "auto":
            return self.param_device
        return sample.device or "cpu"

    def _has_custom_magnitudes(self):
        return isinstance(self.mag_range, np.ndarray)

    def _map_mag_to_param(self, magnitude):
        param = self.mag_to_param(magnitude)
        if _contains_data_node(param):
            raise Exception(
                f"The `mag_to_param` callback of `{self.name}` augmentation returned `DataNode`, "
                f"i.e. an output of DALI pipelined operator, which is not supported there. "
                f"Instead, the `mag_to_param` callback must return parameter that is `np.ndarray` "
                f"or is directly convertible to `np.ndarray`, so that the all parameters can "
                f"be precomputed and reused across iterations.\n\n"
                f"You can move DALI operators from `mag_to_param` callback to the "
                f"decorated augmentation code or replace the DALI operators in `mag_to_param` "
                f"callback with their `dali.experimental.eger` counterparts.\n\n"
                f"Error in augmentation: {self}."
            )
        return np.array(param)

    def _map_mags_to_params(self, magnitudes):
        params = [self._map_mag_to_param(magnitude) for magnitude in magnitudes]
        if len(params) >= 2:
            ref_shape = params[0].shape
            ref_dtype = params[0].dtype
            for param, mag in zip(params, magnitudes):
                if param.shape != ref_shape or param.dtype != ref_dtype:
                    raise Exception(
                        f"The `mag_to_param` callback of `{self.name}` augmentation must return "
                        f"the arrays of the same type and shape for different magnitudes. "
                        f"Got param of shape {ref_shape} and {ref_dtype} type for magnitude "
                        f"{magnitudes[0]}, but for magnitude {mag} the returned array "
                        f"has shape {param.shape} and type {param.dtype}.\n\n"
                        f"Error in augmentation: {self}."
                    )
        return np.array(params)

    def _get_magnitudes(self, num_magnitude_bins):
        mag_range = self.mag_range
        if mag_range is None:
            return None
        if self._has_custom_magnitudes():
            if num_magnitude_bins is not None and len(mag_range) != num_magnitude_bins:
                raise Exception(
                    f"The augmentation `{self.name}` has nd.array of length {len(mag_range)} "
                    f"specified as the `mag_range`. However, the `num_magnitude_bins` "
                    f"passed to the call is {num_magnitude_bins}."
                )
            return mag_range
        if num_magnitude_bins is None:
            raise Exception(
                f"The `num_magnitude_bins` argument is missing in the call of "
                f"the `{self.name}` augmentation. Please specify the `num_magnitude_bins` "
                f"along with the samples and magnitude_bin."
                f"\nError in augmentation: {self}."
            )
        if not hasattr(mag_range, "__len__") or len(mag_range) != 2:
            raise Exception(
                f"The `mag_range` must be a tuple of (low, high) ends of magnitude range or "
                f"nd.array of explicitly defined magnitudes. Got `{self.mag_range}` for "
                f"augmentation `{self.name}`."
            )
        lo, hi = mag_range
        return np.linspace(lo, hi, num_magnitude_bins, dtype=np.float32)

    def _get_param(self, magnitude_bin, num_magnitude_bins, param_device):
        magnitudes = self._get_magnitudes(num_magnitude_bins)
        if magnitudes is None:
            return None
        if magnitude_bin is None:
            raise Exception(
                f"The augmentation `{self.name}` has `mag_range` specified, "  # nosec B608
                f"so when called, it requires `magnitude_bin` parameter to select "
                f"the magnitude from the `mag_range`.\nError in augmentation: {self}."
            )
        if self.randomly_negate and not isinstance(magnitude_bin, _SignedMagnitudeBin):
            magnitude_bin = signed_bin(magnitude_bin)
            warnings.warn(
                f"The augmentation `{self.name}` was declared with `random_negate=True`, "
                f"but unsigned `magnitude_bin` was passed to the augmentation call. "
                f"The augmentation will randomly negate the magnitudes manually. "
                f"However, for better performance, if you conditionally split batch "
                f"between multiple augmentations, it is better to call "
                f"`signed_magnitude_bin = signed_bin(magnitude_bin)` before the split "
                f"and pass the signed bins instead.",
                Warning,
            )
        if self.randomly_negate:
            assert isinstance(magnitude_bin, _SignedMagnitudeBin)  # by the two checks above
            if isinstance(magnitude_bin.bin, int):
                magnitudes = [magnitudes[magnitude_bin.bin]]
                param_idx = magnitude_bin.random_sign
            else:
                param_idx = magnitude_bin.signed_magnitude_idx
            magnitudes = _SignedMagnitudeBin._remap_to_signed_magnitudes(magnitudes)
            params = self._map_mags_to_params(magnitudes)
            params = types.Constant(params, device=param_device)
            return params[param_idx]
        else:
            # other augmentations in the suite may need sign and we got it along the magnitude bin,
            # just unpack the plain magnitude bin
            bin_idx = (
                magnitude_bin.bin
                if isinstance(magnitude_bin, _SignedMagnitudeBin)
                else magnitude_bin
            )
            if isinstance(bin_idx, int):
                magnitude = magnitudes[bin_idx]
                param = self._map_mag_to_param(magnitude)
                return types.Constant(param, device=param_device)
            else:
                params = self._map_mags_to_params(magnitudes)
                params = types.Constant(params, device=param_device)
                return params[bin_idx]

    def _validate_op_sig(self):
        num_positional = get_num_positional_args(self.op)
        if num_positional <= 1:
            raise Exception(
                f"The {self.op} accepts {num_positional} positional argument(s), "
                f"but the functions decorated with `@augmentation` must accept at least two "
                f"positional arguments: the samples and parameters.\nError in: {self}."
            )


def _np_wrap(mag):
    return np.array(mag)


def _contains_data_node(obj):
    if isinstance(obj, (tuple, list)):
        return any(_contains_data_node(el) for el in obj)
    return isinstance(obj, _DataNode)

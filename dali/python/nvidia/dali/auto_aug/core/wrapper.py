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

import inspect

from nvidia.dali import types
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.auto_aug.core.utils import remap_bins_to_signed_magnitudes

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        "Could not import numpy. DALI's automatic augmentation examples depend on numpy. "
        "Please install numpy to use the examples.")


def _np_wrap(mag):
    return np.array(mag)


class _DummyParam:
    """Use DummyParam as a kwarg default when it matters to distinguish between `kwarg=None`
    and not specifying the kwarg"""


class Augmentation:

    def __init__(self, op, mag_range=None, randomly_negate=None, as_param=None, param_device=None):
        self._op = op
        self._mag_range = mag_range
        self._randomly_negate = randomly_negate
        self._as_param = as_param
        self._param_device = param_device

    @property
    def mag_range(self):
        return self._mag_range or (0, 0)

    @property
    def as_param(self):
        return self._as_param or _np_wrap

    @property
    def randomly_negate(self):
        return self._randomly_negate or False

    @property
    def param_device(self):
        return self._param_device or "cpu"

    def augmentation(self, mag_range=_DummyParam, randomly_negate=_DummyParam, as_param=_DummyParam,
                     param_device=_DummyParam, augmentation_cls=None):
        """
        The method to override augmentation parameters specified with `@augmentation` decorator.
        Returns a new augmentation with the original operation decorated but updated parameters.
        Parameters that are not specified are inherited from the initial augmentation.
        """
        cls = augmentation_cls or self.__class__
        config = {name: value for name, value in self._get_config()}
        for name, value in (
            ('mag_range', mag_range),
            ('as_param', as_param),
            ('randomly_negate', randomly_negate),
            ('param_device', param_device),
        ):
            if value is not _DummyParam:
                config[name] = value
        return cls(self._op, **config)

    def _get_config(self):
        return [(name, value) for name, value in (
            ('mag_range', self._mag_range),
            ('as_param', self._as_param),
            ('randomly_negate', self._randomly_negate),
            ('param_device', self._param_device),
        )]

    def __repr__(self):
        aug_params_repr = [repr(self._op)]
        config = [(name, val) for name, val in self._get_config() if val is not None]
        config_reprs = [f"{name}={repr(param)}" for name, param in config]
        aug_params_repr.extend(config_reprs)
        return f"Augmentation({', '.join(aug_params_repr)})"

    def __call__(self, samples, magnitude_bin_idx, num_magnitude_bins=31, random_sign=None,
                 **kwargs):
        params = self.get_param(magnitude_bin_idx, num_magnitude_bins, random_sign)
        fun_args = inspect.getfullargspec(self._op).args[2:]
        op_kwargs = {name: param for name, param in kwargs.items() if name in fun_args}
        return self._op(samples, params, **op_kwargs)

    def get_mag_range(self, num_bins):
        mag_range = self.mag_range
        if isinstance(mag_range, tuple) and len(mag_range) == 2:
            lo, hi = mag_range
            return np.linspace(lo, hi, num_bins, dtype=np.float32)
        if len(mag_range) != num_bins:
            raise Exception(f"Got `mag_range` of length {len(mag_range)} while the "
                            f"`num_bins` specified is {num_bins}.")
        return mag_range

    def get_param(self, magnitude_bin_idx, num_magnitude_bins, random_sign=None):
        assert random_sign is None or isinstance(random_sign, _DataNode)
        magnitudes = self.get_mag_range(num_magnitude_bins)
        if isinstance(magnitude_bin_idx, _DataNode):
            if random_sign is not None:
                magnitudes = remap_bins_to_signed_magnitudes(magnitudes, self.randomly_negate)
                magnitude_bin_idx = 2 * magnitude_bin_idx + random_sign
            params = np.array([self.as_param(magnitude) for magnitude in magnitudes])
            params = types.Constant(params, device=self.param_device)
            return params[magnitude_bin_idx]
        else:
            if random_sign is None:
                magnitude = magnitudes[magnitude_bin_idx]
                param = np.array(self.as_param(magnitude))
                return types.Constant(param, device=self.param_device)
            else:
                magnitudes = [magnitudes[magnitude_bin_idx]]
                magnitudes = remap_bins_to_signed_magnitudes(magnitudes, self.randomly_negate)
                params = np.array([self.as_param(magnitude) for magnitude in magnitudes])
                params = types.Constant(params, device=self.param_device)
                return params[random_sign]


def augmentation(function=None, *, mag_range=None, randomly_negate=None, as_param=None,
                 param_device=None, augmentation_cls=None):
    """
    A decorator turning DALI operation into an augmentation that can be used with the
    `auto_aug` transformations such as RandAugment.

    Parameter
    ---------
    mag_range : (int, int)
        Specifies the range of applicable magnitudes for the operation.
    randomly_negate: bool
        If true, the magnitude from the mag_range will be randomly negated for every sample.
    as_param: callable
        A callback that transforms the magnitude into a parameter. The parameter will be passed to
        the decorated operation instead of the plain magnitude. This way, the parameters for the
        range of magnitudes can be computed once in advance and stored as a Constant node.
    param_device: str
        A "cpu" or "gpu", describes where to store the precomputed parameters.

    Returns
    -------
    Augmentation
        The operation wrapped with the Augmentation class so that it can be used with the `auto_aug`
        transforms.
    """

    def decorator(function):
        cls = augmentation_cls or Augmentation
        return cls(function, mag_range=mag_range, as_param=as_param,
                   randomly_negate=randomly_negate, param_device=param_device)

    if function is None:
        return decorator
    else:
        if not callable(function):
            raise Exception(f"The `@augmentation` decorator was used to decorate the object that "
                            f"is not callable: {function}.")
        return decorator(function)

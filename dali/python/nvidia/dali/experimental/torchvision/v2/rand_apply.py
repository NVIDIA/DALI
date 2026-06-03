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

from typing import Callable, Literal, Sequence

from .operator import Operator, _ValidateIfZeroOneRange
import nvidia.dali.fn as fn
import nvidia.dali as dali


class RandomApply(Operator):
    """
    Apply randomly a list of transformations with a given probability.

    Parameters
    ----------
    op_list : Sequence[Callable]
        List of transformations to apply.
    p : float, optional, default = 0.5
        Probability of applying the transformations.
    device : Literal["cpu", "gpu"], optional, default = "cpu"
        Device to use for the operator. Can be ``"cpu"`` or ``"gpu"``.
    """

    arg_rules = [_ValidateIfZeroOneRange]

    def __init__(
        self,
        op_list: Sequence[Callable],
        p: float = 0.5,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        super().__init__(device=device, p=p)
        self.p = p
        self._op_list = op_list

    def _kernel(self, data_input):
        """
        Randomly applies each operator in op_list sequentially.
        """
        output = data_input
        convert = fn.random.coin_flip(dtype=dali.types.DALIDataType.BOOL, probability=self.p)
        if convert:
            for op in self._op_list:
                output = op._invoke(output)

        return output

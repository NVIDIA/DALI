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


class ToTensor:
    """
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor
    of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes
    (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8

    [DEPRECATED but used]
    """

    def __init__(self): ...

    def __call__(self, data_input):
        """
        Performs to tensor conversion it only converts to float, the remaining part is being done
        in Compose.__call__
        """
        # TODO: if data_input.dtype==types.DALIDataType.UINT8:
        data_input = data_input / 255.0
        return data_input

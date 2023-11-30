# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def autoserialize(dali_pipeline):
    """
    Decorator, that marks a DALI pipeline (represented by :meth:`nvidia.dali.pipeline_def`) for
    autoserialization in [DALI Backend's]
    (https://github.com/triton-inference-server/dali_backend#dali-triton-backend) model repository.

    For details about the autoserialization feature, please refer to the
    [DALI Backend documentation]
    (https://github.com/triton-inference-server/dali_backend#autoserialization).

    Only a ``pipeline_def`` can be decorated with ``autoserialize``.

    Only one ``pipeline_def`` may be decorated with ``autoserialize`` in a given program.

    To perform autoserialization, please refer to :meth:`nvidia.dali._utils.invoke_autoserialize`.

    For more information about Triton, please refer to
    [Triton documentation]
    (https://github.com/triton-inference-server/server#triton-inference-server).

    :param dali_pipeline: DALI Python model definition (``pipeline_def``).
    """
    if not getattr(dali_pipeline, "_is_pipeline_def", False):
        raise TypeError("Only `@pipeline_def` can be decorated with `@triton.autoserialize`.")
    dali_pipeline._is_autoserialize = True
    return dali_pipeline

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys

pipeline_autoserialized = False


def autoserialize(dali_pipeline):
    """
    Decorator, that marks a DALI pipeline (represented by :meth:`nvidia.dali.pipeline_def`) for
    autoserialization in [DALI Backend's](https://github.com/triton-inference-server/dali_backend#dali-triton-backend) model repository.

    For details about the autoserialization feature, please refer to the
    [DALI Backend documentation](https://github.com/triton-inference-server/dali_backend#autoserialization).

    To properly autoserialize the DALI pipeline, the caller needs to run the Python script, that
    contains the pipeline marked with ``autoserialize``, pass the target file name
    (i.e. the path to the file, where the serialized model will be saved) and the magical command
    ``autoserialize.me``. For example:

        # identity.py
        import nvidia.dali as dali

        @dali.triton.autoserialize
        @dali.pipeline_def(batch_size=3, num_threads=1, device_id=0)
        def pipe():
            data = dali.fn.external_source(device="cpu", name="DALI_INPUT_0")
            return data


        $ python identity.py /tmp/model.dali autoserialize.me


    The above command will serialize a DALI Pipeline, no matter where in the module it exists.
    On the other hand, invoking a python script in any other way than presented above, will be
    a no-op regarding pipeline serialization. For example, let's consider the following file:

        # imported.py
        import identity  # the identity.py above


        $ python imported.py /tmp/model.dali autoserialize.me  # serializes DALI pipeline into the
                                                               # /tmp/model.dali file
        $ python imported.py  # nothing happens

    Only a ``pipeline_def`` can be decorated with ``autoserialize``.

    Only one ``pipeline_Def`` may be decorated with ``autoserialize`` in given program.

    :param dali_pipeline: DALI Python model definition (``pipeline_def``)
    """
    if len(sys.argv) != 3 or sys.argv[2] != "autoserialize.me":
        return
    global pipeline_autoserialized
    assert not pipeline_autoserialized, f"There can be only one autoserialized pipeline in a file. Offending pipeline name: {dali_pipeline.__qualname__}."
    assert getattr(dali_pipeline, "is_pipeline_def", False), "Only `@pipeline_def` can be decorated with `@triton.autoserialize`."
    filepath = sys.argv[1]
    dali_pipeline().serialize(filename=filepath)
    pipeline_autoserialized = True

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

from pathlib import Path

from nvidia.dali import fn
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali import types

from test_utils import get_dali_extra_path

_test_root = Path(get_dali_extra_path())


@pipeline_def(batch_size=10, device_id=0, num_threads=4, enable_conditionals=True)
def cond_pipe():
    enc, label = fn.readers.file(
        files=[str(_test_root / "db/single/jpeg/113/snail-4291306_1280.jpg")], name="FileReader")
    imgs = fn.decoders.image(enc, device="mixed")
    resized = fn.resize(imgs, size=[224, 224], interp_type=types.DALIInterpType.INTERP_LINEAR)
    if fn.random.uniform(range=[0, 1]) < 0.25:
        out = fn.rotate(resized, angle=fn.random.uniform(range=[30, 60]))
    else:
        out = resized
    return out, label.gpu()


pipe = cond_pipe()
pipe.build()
pipe.run()

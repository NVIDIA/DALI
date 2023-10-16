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


# TODO(klecki): It would be nice, if we could actually have those without defaults
@pipeline_def(batch_size=10, device_id=0, num_threads=4)
def rn50_pipe():
    enc, label = fn.readers.file(
        files=[str(_test_root / "db/single/jpeg/113/snail-4291306_1280.jpg")], name="FileReader")
    imgs = fn.decoders.image(enc, device="mixed")
    rng = fn.random.coin_flip(probability=0.5)
    resized = fn.random_resized_crop(imgs, size=[224, 224])
    normalized = fn.crop_mirror_normalize(resized, mirror=rng,
                                          dtype=types.DALIDataType.DALI_FLOAT16,
                                          output_layout="HWC", crop=(224, 224),
                                          mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                          std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return normalized, label.gpu()


pipe = rn50_pipe()
pipe.build()
pipe.run()

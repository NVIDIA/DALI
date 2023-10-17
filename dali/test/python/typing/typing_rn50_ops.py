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

from nvidia.dali import ops
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali import types

from test_utils import get_dali_extra_path

_test_root = Path(get_dali_extra_path())


@pipeline_def(batch_size=10, device_id=0, num_threads=4)
def rn50_pipe():
    Reader = ops.readers.File(files=[str(_test_root / "db/single/jpeg/113/snail-4291306_1280.jpg")],
                              name="FileReader")
    Decoder = ops.decoders.Image(device="mixed")
    Rng = ops.random.CoinFlip(probability=0.5)
    Rrc = ops.RandomResizedCrop(size=[224, 224])
    Cmn = ops.CropMirrorNormalize(mirror=Rng(), dtype=types.DALIDataType.FLOAT16,
                                  output_layout="HWC", crop=(224, 244),
                                  mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                  std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    enc, label = Reader()
    imgs = Decoder(enc)
    resized = Rrc(imgs)
    normalized = Cmn(resized)
    return normalized, label.gpu()


pipe = rn50_pipe()
pipe.build()
pipe.run()

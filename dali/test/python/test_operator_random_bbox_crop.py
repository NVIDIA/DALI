# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
import numpy as np
import os

from test_utils import ConstantDataIterator

class RandomBBoxCropSynthDataPipeline(Pipeline):
    def __init__(self, device, batch_size, boxes,
                 input_shape=None, crop_shape=None,
                 num_threads=1, device_id=0, num_gpus=1):
        super(RandomBBoxCropSynthDataPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=1234)
        self.device = device
        self.boxes = boxes
        self.inputs = ops.ExternalSource()
        self.bbox_crop = ops.RandomBBoxCrop(
            device=self.device,
            aspect_ratio=[0.5, 2.0] if crop_shape is None else None,
            scaling=[0.3, 1.0] if crop_shape is None else None,
            thresholds=[0, 0.01, 0.05, 0.1, 0.15],
            ltrb=True,
            num_attempts=100,
            allow_no_crop=False,
            input_shape=input_shape,
            crop_shape=crop_shape)

    def define_graph(self):
        self.data = self.inputs()
        input_data = self.data
        data = input_data.gpu() if self.device == 'gpu' else input_data
        out1, out2, out3 = self.bbox_crop(data)
        return out1, out2, out3

    def iter_setup(self):
        self.feed_input(self.data, self.boxes)

bbox_2d_ltrb_1 = [0.0, 0.0, 0.9, 0.9]
bbox_2d_ltrb_2 = [0.1, 0.1, 0.99, 0.99]
bbox_2d_ltrb_3 = [0.3, 0.3, 0.5, 0.5]
bbox_3d_ltrb_1 = [0.5, 0.5, 0.5, 0.7, 0.7, 0.7]
bbox_3d_ltrb_2 = [0.1, 0.1, 0.1, 0.6, 0.6, 0.6]
bbox_3d_ltrb_3 = [0.4, 0.4, 0.4, 0.9, 0.9, 0.9]

def test_random_bbox_crop_2d():
    device = 'cpu'
    batch_size = 1
    boxes = [np.array([bbox_3d_ltrb_1, bbox_3d_ltrb_2, bbox_3d_ltrb_3], dtype=np.float32)]
    pipe = RandomBBoxCropSynthDataPipeline(device='cpu', batch_size=batch_size, boxes=boxes,
                                           input_shape=[1200, 800, 64], crop_shape=[1200, 800, 64])
    pipe.build()
    for i in range(100):
        outputs = pipe.run()
        print("Out 0:\n{}".format(outputs[0].at(0)))
        print("Out 1:\n{}".format(outputs[1].at(0)))
        print("Out 2:\n{}".format(outputs[2].at(0)))

def main():
  test_random_bbox_crop_2d()

if __name__ == '__main__':
  main()

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
# limitations under the License

import json
import os
from PIL import Image
import torch
import urllib

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as dali_pytorch


batch_size = 4


class ReadMetaExternalInput:
    def __init__(self, batch_size, root_dir):
        with open(
            os.path.join(root_dir, "evaluation/example_gt.json"), "r"
        ) as file:
            json_file = json.load(file)
            self.images = json_file["images"]
            self.annotations = json_file["annotations"]

        self.batch_size = batch_size
        self.full_iterations = len(self.images) // self.batch_size
        self.root_dir = root_dir

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        batch = []
        bbox = []
        for _ in range(self.batch_size):
            if self.i == self.full_iterations * self.batch_size:
                self.__iter__()
                raise StopIteration()

            img_url = self.images[self.i]["coco_url"]
            img_id = self.images[self.i]["id"]

            annotations = [
                annotation
                for annotation in self.annotations
                if annotation["image_id"] == img_id
            ]
            bbox.append(torch.tensor(annotations[0]["bbox"], dtype=torch.float))

            with urllib.request.urlopen(img_url) as f:
                batch.append(
                    torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
                )

            self.i += 1

        return batch, bbox


@pipeline_def(batch_size=4, num_threads=4, device_id=0, exec_dynamic=True)
def read_meta_pipeline(root_dir):

    image, bbox = fn.external_source(
        source=ReadMetaExternalInput(batch_size, root_dir),
        num_outputs=2,
        dtype=[types.UINT8, types.FLOAT],
    )

    decoded = fn.decoders.image(
        image,
        device="mixed",
        output_type=types.RGB,
        use_fast_idct=False,
        jpeg_fancy_upsampling=True,
    )

    start = fn.cast(bbox[0:2], dtype=types.UINT32)
    end = fn.cast(bbox[0:2] + bbox[2:4], dtype=types.UINT32)

    # Crop image
    img_crop = decoded[start[1] : end[1], start[0] : end[0]]
    img_crop = fn.resize(img_crop, size=(640, 480))

    return img_crop


crop = []

# The root dir of the COCO-WholeBody repository
# The repository can be found in: https://github.com/jin-s13/COCO-WholeBody/
root_dir = "./COCO-WholeBody/"

pipeline = read_meta_pipeline(root_dir)
pipeline.build()

dali_iter = dali_pytorch.DALIGenericIterator(pipeline, ["img_crop"])
# Iterate over the data
for i, data in enumerate(dali_iter):
    crop += data[0]["img_crop"]

for i, crp in enumerate(crop):
    img = Image.fromarray(crp.cpu().numpy())
    img.save(f"{i}.jpg")

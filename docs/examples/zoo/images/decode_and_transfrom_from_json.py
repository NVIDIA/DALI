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

import argparse
import json
import numpy as np
import os
from PIL import Image
from pathlib import Path
import torch

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as dali_pytorch


batch_size = 4


class ReadMetaExternalInput:
    def __init__(self, batch_size, root_dir):
        with open(os.path.join(root_dir, "instances.json"), "r") as file:
            self.json_data = json.load(file)

        self.images = self.json_data["images"]
        self.annotations = self.json_data["annotations"]

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

            img_name = self.images[self.i]["file_name"]
            img_id = self.images[self.i]["id"]

            batch.append(
                np.fromfile(
                    Path(self.root_dir) / "images" / img_name, dtype=np.uint8
                )
            )

            annotations = [
                annotation
                for annotation in self.annotations
                if annotation["image_id"] == img_id
            ]
            bbox.append(torch.tensor(annotations[0]["bbox"], dtype=torch.float))

            self.i += 1

        return batch, bbox


@pipeline_def(
    batch_size=4,
    num_threads=4,
    device_id=0,
    exec_dynamic=True,
    enable_conditionals=True,
)
def read_meta_pipeline(coco_dir):

    image, bbox = fn.external_source(
        source=ReadMetaExternalInput(batch_size, coco_dir),
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
    left, top = start[0], start[1]
    end = fn.cast(bbox[0:2] + bbox[2:4], dtype=types.UINT32)
    right, bottom = end[0], end[1]

    # Crop image
    img_crop = decoded[top:bottom, left:right]
    img_crop = fn.resize(img_crop, size=(640, 480))
    hflip = fn.random.coin_flip()
    if hflip:
        img_crop = fn.flip(img_crop)

    return img_crop


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Example of DALI image processing with json input data"
    )
    parser.add_argument(
        "--coco_dir",
        type=str,
        default="../DALI_extra/db/coco/",
        help="COCO DALI_extra root directory",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=True,
        help="If set to true saves images in the current directory, displays them otherwise",
    )

    args = parser.parse_args()

    crop = []

    pipeline = read_meta_pipeline(args.coco_dir)
    pipeline.build()

    dali_iter = dali_pytorch.DALIGenericIterator(pipeline, ["img_crop"])
    # Iterate over the data
    for i, data in enumerate(dali_iter):
        crop += data[0]["img_crop"]

    for i, crp in enumerate(crop):
        img = Image.fromarray(crp.cpu().numpy())
        if args.save:
            img.save(f"{i}.jpg")
        else:
            img.show()

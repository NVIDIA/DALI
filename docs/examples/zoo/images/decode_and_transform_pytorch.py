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
import numpy as np
import os
import torch
from torch.utils.data import Dataset

import nvidia.dali.plugin.pytorch.experimental.proxy as dali_proxy
from nvidia.dali import pipeline_def, fn, types
import glob
from pathlib import Path

class ImageDataset(Dataset):
    def __init__(self, image_dir, json_path, transform=None):
        """
        Initialize the dataset.

        :param image_dir: Directory containing the images.
        :param json_dir: Directory containing the JSON labels.
        :param transform: Optional transform to apply on images.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = glob.glob("*.jpg", root_dir=image_dir)

        # Load the JSON label
        with open(json_path, "r") as f:
            self.labels = json.load(f)

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Load an image and its corresponding label.

        :param idx: Index of the image to load.
        :return: Tuple containing the image tensor and its label.
        """
        image_id = Path(image_dir) / self.image_ids[idx]
        # Find the corresponding label in labels based on image_id
        for label_entry in self.labels:
            label = None
            if label_entry["image_id"] == image_id.name:
                label = label_entry["label"]
                break

        if label is None:
            raise ValueError(f"No label found for image {image_id}")

        # Load the image
        encoded_img = np.expand_dims(
            np.fromfile(image_id, dtype=np.uint8), axis=0
        )
        label_tensor = torch.tensor(label)
        # Apply transform if provided and if not return the encoded image
        if self.transform:
            return self.transform(encoded_img), label_tensor
        else:
            return encoded_img, label_tensor


@pipeline_def
def image_pipe(img_hw=(320, 200)):
    encoded_images = fn.external_source(name="images", no_copy=True)

    decoded = fn.decoders.image(
        encoded_images,
        device="mixed",
        output_type=types.RGB,
        use_fast_idct=False,
        jpeg_fancy_upsampling=True,
    )

    images = fn.resize(
        decoded,
        size=img_hw,
        interp_type=types.INTERP_LINEAR,
        antialias=False,
    )

    return images


if __name__ == "__main__":
    image_dir = "img/"
    json_file = "img/labels.json"
    batch_size = 8
    nworkers = 8
    dali_server = dali_proxy.DALIServer(
        image_pipe(batch_size=batch_size, num_threads=4, device_id=0)
    )

    dataset = ImageDataset(image_dir, json_file, transform=dali_server.proxy)

    # Create a DataLoader

    dataloader = dali_proxy.DataLoader(
        dali_server,
        dataset,
        batch_size=batch_size,
        num_workers=nworkers,
        drop_last=True,
    )
    # Iterate over the dataset
    for images, labels in dataloader:
        print(f"Batch size: {images.size(0)}")
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")

    del dataloader

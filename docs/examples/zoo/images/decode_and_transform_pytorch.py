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
import glob
from pathlib import Path
import os
import sys

import numpy as np
from torch.utils.data import Dataset

import nvidia.dali.plugin.pytorch.experimental.proxy as dali_proxy
from nvidia.dali import pipeline_def, fn, types


class ImageDataset(Dataset):
    def __init__(self, landmarks_dir, transform=None):
        """
        Initialize the dataset.

        :param landmarks_dir: Directory containing the images and landmarks.
        :param transform: Optional transform to apply on images.
        """
        self.landmarks_dir = landmarks_dir
        self.transform = transform
        self.image_ids = glob.glob("*.jpeg", root_dir=landmarks_dir)

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
        image_id = Path(self.landmarks_dir) / self.image_ids[idx]
        landmark_id = (
            Path(self.landmarks_dir) / f"{self.image_ids[idx][:-4]}npy"
        )

        # Load the image
        encoded_img = np.expand_dims(
            np.fromfile(image_id, dtype=np.uint8), axis=0
        )

        # Load landmark
        landmarks = np.fromfile(landmark_id)

        # Apply transform if provided and if not return the encoded image
        if self.transform is not None:
            return self.transform(encoded_img), landmarks
        else:
            return encoded_img, landmarks


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
    parser = argparse.ArgumentParser(
        description="Example of DALI and Pytorch image processing integration"
    )
    parser.add_argument(
        "--landmarks_dir",
        type=str,
        required=True,
        help="Images and face landmark directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for image processing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of CPU workers for Pytorch dataloader",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of CPU threads used by the pipeline",
    )

    args = parser.parse_args()

    if not os.path.exists(args.landmarks_dir):
        sys.exit(f"Invalid DALI_extra landmarks path: {args.landmarks_dir}")

    # DALI Server is the key member of DALI Proxy
    # For further information, please refer to:
    # https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/pytorch_dali_proxy.html
    with dali_proxy.DALIServer(
        image_pipe(
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            device_id=0,
        )
    ) as dali_server:

        dataset = ImageDataset(args.landmarks_dir, transform=dali_server.proxy)

        # Create a DataLoader
        dataloader = dali_proxy.DataLoader(
            dali_server,
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
        )
        # Iterate over the dataset
        for images, landmarks in dataloader:
            print(f"Batch size: {images.size(0)}")
            print(f"Image shape: {images.shape}")
            print(f"Landmark shape: {landmarks.shape}")

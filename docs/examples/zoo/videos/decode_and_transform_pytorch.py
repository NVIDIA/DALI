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
import numpy as np
from torch.utils.data import Dataset

import nvidia.dali.plugin.pytorch.experimental.proxy as dali_proxy
from nvidia.dali import pipeline_def, fn, types
import pathlib


class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        """
        Initialize the dataset.

        :param video_dir: Directory containing the videos.
        :param transform: Optional transform to apply on images.
        """
        self.video_dir = video_dir
        self.transform = transform
        self.video_ids = list(pathlib.Path(video_dir).glob("*.mp4"))

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.video_ids)

    def __getitem__(self, idx):
        """
        Load an image and its corresponding label.

        :param idx: Index of the image to load.
        :return: Tuple containing the image tensor and its label.
        """
        video_id = self.video_ids[idx]
        # Load the image
        encoded_video = np.fromfile(video_id, dtype=np.uint8)
        # Apply transform if provided and return encoded video if not
        if self.transform:
            return self.transform(encoded_video)
        else:
            return encoded_video


@pipeline_def
def video_pipe(video_hw=(720, 1280)):
    encoded_videos = fn.external_source(name="videos", no_copy=True)

    decoded = fn.experimental.decoders.video(
        encoded_videos,
        device="mixed",
        sequence_length=32,
    )

    # Resize the video to the desired size preserving aspect ratio
    # Since the video resoultions will differ with, due to that method of  resize,
    # the batch size is set to 1, to avoid creating batches of tensors with various shapes.
    out = fn.resize(
        decoded,
        size=video_hw,
        mode="not_larger",
        interp_type=types.INTERP_CUBIC,
    )

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example of DALI video decoding"
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="../DALI_extra/db/video/sintel/video_files/",
        help="Videos directory",
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

    args = parser.parse_args()
    # Iterate through all files in the directory
    dali_server = dali_proxy.DALIServer(
        video_pipe(batch_size=args.batch_size, num_threads=4, device_id=0)
    )

    dataset = VideoDataset(args.videos_dir, transform=dali_server.proxy)

    # Create a DataLoader

    dataloader = dali_proxy.DataLoader(
        dali_server,
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )
    # Iterate over the dataset
    for videos in dataloader:
        print(f"Batch size: {videos.size(0)}")
        print(f"Video shape: {videos.shape}")

    del dataloader

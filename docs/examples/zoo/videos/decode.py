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
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch.torch_utils import to_torch_tensor


@pipeline_def(batch_size=1, num_threads=4, device_id=0, exec_dynamic=True)
def decode_pipeline(source_name):

    # Read the input video
    encoded_video = fn.external_source(
        device="cpu",
        name=source_name,
        no_copy=False,
        blocking=True,
        dtype=types.UINT8,
    )

    # Decode the video
    decoded = fn.experimental.decoders.video(
        encoded_video, device="mixed", start_frame=0, sequence_length=30
    )

    # Resize the video to 1280x720 and flip vertically
    decoded = fn.resize_crop_mirror(decoded, size=(720, 1280), mirror=2)

    return decoded


if __name__ == "__main__":
    # Create and build the decoding pipeline
    pipe = decode_pipeline("encoded_video", prefetch_queue_depth=1)
    pipe.build()

    parser = argparse.ArgumentParser(
        description="Example of DALI video decoding"
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="Videos directory",
    )

    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Saves first 2 frames from each of processed videos",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./",
        help="Path where frames will be saved",
    )

    args = parser.parse_args()

    if not os.path.exists(args.videos_dir):
        sys.exit(f"Invalid videos path: {args.videos_dir}")

    # Iterate through all files in the directory
    for i, file_name in enumerate(os.listdir(args.videos_dir)):
        file_path = os.path.join(args.videos_dir, file_name)
        try:
            # Read the file into a numpy array of shape (1, video_size)
            # Send the tensor to the pipeline, run the pipeline and retrieve the output
            decoded = pipe.run(
                encoded_video=np.expand_dims(
                    np.fromfile(file_path, dtype=np.uint8), axis=0
                )
            )

            if args.save:
                video = to_torch_tensor(decoded[0][0], True).cpu()
                Image.fromarray(video[0].numpy()).save(
                    Path(args.save_path) / f"{file_name}_0.jpg"
                )
                Image.fromarray(video[1].numpy()).save(
                    Path(args.save_path) / f"{file_name}_1.jpg"
                )

        except Exception as e:
            print(f"Error loading {file_name}: {e}")

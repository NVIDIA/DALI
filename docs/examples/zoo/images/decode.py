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


@pipeline_def(batch_size=4, num_threads=4, device_id=0, exec_dynamic=True)
def decode_pipeline(source_name):

    # Read the input image
    inputs = fn.external_source(
        device="cpu",
        name=source_name,
        no_copy=False,
        blocking=True,
        dtype=types.UINT8,
    )

    # Decode the encoded image on GPU
    decoded = fn.decoders.image(
        inputs,
        device="mixed",
        output_type=types.RGB,
        jpeg_fancy_upsampling=True,
    )

    return decoded


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Example of DALI image processing"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Folder with images",
    )
    parser.add_argument(
        "--show",
        type=bool,
        default=False,
        help="If set to true displays images",
    )

    args = parser.parse_args()

    if not os.path.exists(args.images_dir):
        sys.exit(f"Invalid images path: {args.images_dir}")

    # Create and build the decoding pipeline
    pipe = decode_pipeline("encoded_img", prefetch_queue_depth=1)
    pipe.build()

    # The directory with images to decode
    # Iterate through all files in the directory
    for i, file_name in enumerate(Path(args.images_dir).iterdir()):
        # Read the file into a numpy array of shape (1, img_size)
        # Send the tensor to the pipeline, run the pipeline and retrieve the output
        decoded = pipe.run(
            encoded_img=np.expand_dims(
                np.fromfile(file_name, dtype=np.uint8), axis=0
            )
        )
        img_on_gpu = to_torch_tensor(decoded[0][0], copy=False)

        if args.show:
            # Display decoded image. Note: The image is in GPU memory
            # and needs to be retrieved to CPU first.
            img_to_show = img_on_gpu.cpu().numpy()
            Image.fromarray(img_to_show[0]).show()

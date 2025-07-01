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

# pip install ffmpeg-python # not the ffmpeg!
import argparse
import ffmpeg
import json
import numpy as np
from pathlib import Path
import os
from PIL import Image
import sys
import torch

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as dali_pytorch
from nvidia.dali.plugin.base_iterator import LastBatchPolicy as LastBatchPolicy


batch_size = 4


class ReadVideosAndAnnotationsExternalInput:
    """Reads videos and annotations from the DomainSpecificHighlight dataset.

    This class reads videos and annotations from the DomainSpecificHighlight repository
    (https://github.com/aliensunmin/DomainSpecificHighlight). The root_dir parameter
    should point to one of the activity folders (e.g. surfing) from the repository.

    The videos need to be downloaded beforehand using the following script:
    `python code/downloadVideo.py ./`

    For more details, see: https://github.com/aliensunmin/DomainSpecificHighlight

    Args:
        batch_size (int): Number of videos to load in each batch
        root_dir (str): Path to activity folder containing video subfolders
    """

    def __init__(self, batch_size, root_dir):
        # Get all subfolders in the root directory
        self.video_paths = []
        for path in Path(root_dir).rglob("*.mp4"):
            self.video_paths.append(path)

        self.batch_size = batch_size
        self.full_iterations = len(self.video_paths) // self.batch_size
        self.root_dir = root_dir

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        batch = []
        clips = []
        for _ in range(self.batch_size):
            if self.i == self.full_iterations * self.batch_size:
                self.__iter__()
                raise StopIteration()

            video_path = self.video_paths[self.i]
            batch.append(np.fromfile(video_path, dtype=np.uint8))
            meta_data = ffmpeg.probe(video_path)
            video_stream = next(
                (
                    stream
                    for stream in meta_data["streams"]
                    if stream["codec_type"] == "video"
                ),
                None,
            )
            fps = int(video_stream["r_frame_rate"].split("/")[0]) / int(
                video_stream["r_frame_rate"].split("/")[1]
            )
            fpms = fps / 1000
            # "clip.json" is a json file that contains the start and end frames of clips
            with open(
                os.path.join(os.path.dirname(video_path), "clip.json"),
                "r",
            ) as file:
                clip_json = json.load(file)
                clip_frames = torch.tensor(clip_json)
                for cf in range(clip_frames.shape[0]):
                    clip_frames[cf][0] = int(
                        clip_frames[cf][0] * fpms
                    )  # Convert ms to frames
                    clip_frames[cf][1] = int(
                        clip_frames[cf][1] * fpms
                    )  # Convert ms to frames
            clips.append(clip_frames.int())
            self.i += 1
        return batch, clips


@pipeline_def(
    batch_size=4,
    num_threads=4,
    device_id=0,
    exec_dynamic=True,
    enable_conditionals=True,
)
def read_clips_pipeline(root_dir, sequence_length):

    video_file, clips = fn.external_source(
        source=ReadVideosAndAnnotationsExternalInput(batch_size, root_dir),
        num_outputs=2,
        dtype=[types.UINT8, types.INT32],
    )
    start_frame = clips[0][0]
    # If not enough frames extend to the desired sequnce length by padding with the last frame
    # Decode the video - start and end frames are available for DALI >= 1.48.0
    decoded = fn.experimental.decoders.video(
        video_file,
        device="mixed",
        start_frame=start_frame,
        sequence_length=sequence_length,
        pad_mode="edge",
    )
    # Resize the video to 640x480
    decoded = fn.resize(decoded, size=(640, 480))

    return decoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example of DALI video decoding"
    )
    # The root dir of the DomainSpecificHighlight repository
    # The repository can be found in: https://github.com/aliensunmin/DomainSpecificHighlight/
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
        help="If set to true saves sample frames",
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

    pipeline = read_clips_pipeline(args.videos_dir, sequence_length=7)
    pipeline.build()

    dali_iter = dali_pytorch.DALIRaggedIterator(
        pipeline, ["decoded"], last_batch_policy=LastBatchPolicy.FILL
    )
    # Iterate over the data
    for i, data in enumerate(dali_iter):
        for element in range(batch_size):
            if args.save:
                for j in range(2):
                    img = Image.fromarray(
                        data[0]["decoded"][element][j].cpu().numpy()
                    )
                    img.save(Path(args.save_path) / f"{i}_{element}_{j}.jpg")

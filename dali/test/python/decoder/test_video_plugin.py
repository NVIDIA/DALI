# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from nvidia.dali import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import cv2

import numpy as np

import nvidia.dali.fn.plugin.video as video


from test_utils import get_dali_extra_path, is_mulit_gpu, skip_if_m60

data_path = get_dali_extra_path() + "/db/video/cfr/test_2.mp4"


def test_vfr_mp4_decoding():
    
    @pipeline_def(device_id=0, num_threads=4, batch_size=1)
    def video_decoder_pipeline():
        data = fn.external_source(source=lambda: [np.fromfile(data_path, np.uint8),], dtype=types.UINT8)
        return video.decoder(data, device="mixed")
    
    pipeline = video_decoder_pipeline()
    pipeline.build()
    
    (output,) = pipeline.run()
    frames = output.as_cpu().as_array()
    print(len(frames[0]), "frames")
    
    # Read frame file names from the text file
    with open(f"{get_dali_extra_path()}/db/video/cfr/frames_2/frames_list.txt") as f:
        frame_list = f.read().splitlines()
    
    for i, frame in enumerate(frames[0]):
        ground_truth = cv2.imread(f"{get_dali_extra_path()}/db/video/cfr/frames_2/{frame_list[i]}", cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not np.average(frame - ground_truth) < 10:
            print(f"Frame {i} is not equal")
            cv2.imwrite(f"{get_dali_extra_path()}/db/video/cfr/frames_2/gt_{frame_list[i]}", ground_truth)
            cv2.imwrite(f"{get_dali_extra_path()}/db/video/cfr/frames_2/error_{frame_list[i]}", frame)
            assert False
        
    
    
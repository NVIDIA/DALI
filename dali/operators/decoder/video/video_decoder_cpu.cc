// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/operators/decoder/video/video_decoder_base.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"

namespace dali {

DALI_SCHEMA(experimental__decoders__Video)
    .DocStr(
        R"code(Decodes video files from memory buffers into sequences of frames.

The operator accepts video files in common container formats (e.g. MP4, AVI). For CPU backend,
FFmpeg is used for decoding. For Mixed backend, NVIDIA's Video Codec SDK (NVDEC) is used.
Each output sample is a sequence of frames with shape ``(F, H, W, C)`` where:

* ``F`` is the number of frames in the sequence (can vary between samples)
* ``H`` is the frame height in pixels
* ``W`` is the frame width in pixels
* ``C`` is the number of color channels

Example 1: Extract a sequence of arbitrary frames:

.. code-block:: python

    video_decoder = dali.experimental.decoders.video(
        encoded=encoded_video,
        frames=[0, 10, 20, 30, 40, 50, 40, 30, 20, 10, 0]
        ...,
    )

Example 2: Extract a sequence of evenly spaced frames, starting from frame 0,
with a stride of 2, until 10 frames are reached:

.. code-block:: python

    video_decoder = dali.experimental.decoders.Video(
        encoded=encoded_video,
        start_frame=0, sequence_length=10, stride=2
        ...,
    )

Example 3: Pad the sequence with the last frame in the video, until 100 frames are reached:

.. code-block:: python

    video_decoder = dali.experimental.decoders.Video(
        encoded=encoded_video,
        start_frame=0, sequence_length=100, stride=2, pad_mode="edge"
        ...,
    )

Example 4: Pad the sequence with a constant value of 128, until 100 frames are reached:

.. code-block:: python

    video_decoder = dali.experimental.decoders.Video(
        encoded=encoded_video,
        start_frame=0, sequence_length=100, stride=2, pad_mode="constant", fill_value=128
        ...,

Example 5: Pad the sequence with a constant RGB value of (118, 185, 0), until 100 frames are reached:

.. code-block:: python

    video_decoder = dali.experimental.decoders.Video(
        encoded=encoded_video,
        start_frame=0, sequence_length=100, stride=2, pad_mode="constant", fill_value=[118, 185, 0]
        ...,
)code")
    .NumInput(1)
    .NumOutput(1)
    .InputDox(0, "encoded", "TensorList", "Encoded video stream")
    .AddOptionalArg("affine",
                    R"code(Whether to pin threads to CPU cores (mixed backend only).

If True, each thread in the internal thread pool will be pinned to a specific CPU core.
If False, threads can migrate between cores based on OS scheduling.)code",
                    true)
    .AddOptionalArg<std::vector<int>>(
        "frames",
        R"code(Specifies which frames to extract from each video by their indices.

The indices can be provided in any order and can include duplicates. For example, ``[0,10,5,10]`` would extract:

* Frame 0 (first frame)
* Frame 10 
* Frame 5
* Frame 10 (again)

This argument cannot be used together with ``start_frame``, ``sequence_length``, ``stride``.)code",
        nullptr, true)
    .AddOptionalArg<int>(
        "start_frame",
        R"code(Index of the first frame to extract from each video. Cannot be used together with ``frames`` argument.)code",
        nullptr, true)
    .AddOptionalArg<int>(
        "stride",
        R"code(Number of frames to skip between each extracted frame. Cannot be used together with ``frames`` argument.)code",
        nullptr, true)
    .AddOptionalArg<int>(
        "sequence_length",
        R"code(Number of frames to extract from each video. If not provided, the whole video is decoded. Cannot be used together with ``frames`` argument.)code",
        nullptr, true)
    .AddOptionalArg<std::string>(
        "pad_mode",
        R"code(How to handle videos with insufficient frames when using start_frame/sequence_length/stride:

* ``'none'``: Return shorter sequences if not enough frames: ABC -> ABC
* ``'constant'``: Pad with a fixed value (specified by ``pad_value``): ABC -> ABCPPP  
* ``'edge'`` or ``'repeat'``: Repeat the last valid frame: ABC -> ABCCCC
* ``'reflect_1001'`` or ``'symmetric'``: Reflect padding, including the last element: ABC -> ABCCBA
* ``'reflect_101'`` or ``'reflect'``: Reflect padding, not including the last element: ABC -> ABCBA

Not relevant when using ``frames`` argument.)code",
        "constant", true)
    .AddOptionalArg(
        "fill_value",
        R"code(Value(s) used to pad missing frames when ``pad_mode='constant'``'.

Each value must be in range [0, 255].
If a single value is provided, it will be used for all channels. 
Otherwise, the number of values must match the number of channels in the video.)code",
        std::vector<int>{0, })
    .AddOptionalArg("build_index",
                    R"code(Controls whether to build a frame index during initialization.

Building an index allows faster seeking to specific frames, but requires additional CPU memory
to store frame metadata and longer initialization time to scan the entire video file. The index
stores metadata, such as whether it is a key frame and the presentation timestamp (PTS).

Building an index is particularly useful when decoding a small number of frames spaced far
apart or starting playback from a frame deep into the video)code",
                    true);

class VideoDecoderCpu : public VideoDecoderBase<CPUBackend, FramesDecoder> {
 public:
  explicit VideoDecoderCpu(const OpSpec &spec) :
    VideoDecoderBase<CPUBackend, FramesDecoder>(spec) {}
};

DALI_REGISTER_OPERATOR(experimental__decoders__Video, VideoDecoderCpu, CPU);

}  // namespace dali

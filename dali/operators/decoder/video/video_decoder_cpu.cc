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
#include "dali/operators/reader/loader/video/frames_decoder_cpu.h"

namespace dali {

DALI_SCHEMA(experimental__decoders__Video)
    .DocStr(
        R"code(Decodes videos from in-memory streams.

The operator supports most common video container formats using libavformat (FFmpeg).
The operator utilizes either libavcodec (FFmpeg) or NVIDIA Video Codec SDK (NVDEC) for decoding the frames.

The following video codecs are supported by both CPU and Mixed backends:

* H.264/AVC
* H.265/HEVC
* VP8
* VP9
* MJPEG

The following codecs are supported by the Mixed backend only:

* AV1
* MPEG-4

Each output sample is a sequence of frames with shape ``(F, H, W, C)`` where:

* ``F`` is the number of frames in the sequence (can vary between samples)
* ``H`` is the frame height in pixels
* ``W`` is the frame width in pixels
* ``C`` is the number of color channels

The operator provides several ways to select which frames to extract from the video:

* Using no frame selection arguments:

  * When no frame selection arguments are provided, all frames in the video are decoded
  * Frames are extracted sequentially from start to end with stride=1
  * For example, a 10-frame video would extract frames [0,1,2,3,4,5,6,7,8,9]

* Using the ``frames`` argument:

  * Accepts a list of frame indices to extract from the video
  * Frame indices can be specified in any order and can repeat frames 
  * Each index must be non-negative and may exceed the bounds of the video, if the ``pad_mode`` is not ``none``

* Using ``start_frame``, ``end_frame`` and ``stride``:

  * ``start_frame``: First frame to extract (default: 0)
  * ``end_frame``: Last frame to extract (exclusive)
  * ``stride``: Number of frames to skip between each extracted frame (default: 1)
  * Extracts frames in the range [start_frame, end_frame) advancing by stride
  * For example, with start_frame=0, end_frame=10, stride=2 extracts frames [0,2,4,6,8]

* Using ``start_frame``, ``sequence_length`` and ``stride``:

  * ``start_frame``: First frame to extract (default: 0)
  * ``sequence_length``: Number of frames to extract
  * ``stride``: Number of frames to skip between each extracted frame (default: 1) 
  * Extracts sequence_length frames starting at start_frame, advancing by stride
  * For example, with start_frame=0, sequence_length=5, stride=2 extracts frames [0,2,4,6,8]

If the requested frames exceed the bounds of the video, the behavior depends on
``pad_mode``. If pad_mode is ``none``, it causes an error. Otherwise, the sequence is padded according to the
``pad_mode`` argument (see ``pad_mode`` for details).

Example 1: Extract a sequence of arbitrary frames:

.. code-block:: python

    video_decoder = dali.experimental.decoders.video(
        encoded=encoded_video,
        frames=[0, 10, 20, 30, 40, 50, 40, 30, 20, 10, 0]
        ...,
    )

Example 2: Extract a sequence of evenly spaced frames, starting from frame 0,
with a stride of 2, until frame 20 (exclusive):

.. code-block:: python

    video_decoder = dali.experimental.decoders.Video(
        encoded=encoded_video,
        start_frame=0, end_frame=20, stride=2
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
        R"code(Number of frames to extract from each video. Cannot be used together with ``frames`` or ``end_frame`` arguments.)code",
        nullptr, true)
    .AddOptionalArg<int>(
        "end_frame",
        R"code(Last frame to extract from each video (exclusive). Cannot be used with ``frames`` or ``sequence_length``.)code",
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
    .AddOptionalArg("fill_value",
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
apart or starting playback from a frame deep into the video.)code",
                    true);

class VideoDecoderCpu : public VideoDecoderBase<CPUBackend, FramesDecoderCpu> {
 public:
  explicit VideoDecoderCpu(const OpSpec &spec) :
    VideoDecoderBase<CPUBackend, FramesDecoderCpu>(spec) {}
};

DALI_REGISTER_OPERATOR(experimental__decoders__Video, VideoDecoderCpu, CPU);

}  // namespace dali

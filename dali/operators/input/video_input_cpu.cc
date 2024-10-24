// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/input/video_input.h"
#include <memory>

namespace dali {


template<>
void VideoInput<CPUBackend, dali::FramesDecoder>::CreateDecoder(const Workspace &ws) {
  auto sample = encoded_video_[0];
  auto data = reinterpret_cast<const char *>(sample.data<uint8_t>());
  size_t size = sample.shape().num_elements();
  this->frames_decoders_[0] = std::make_unique<dali::FramesDecoder>(data, size, false);
  DALI_ENFORCE(this->frames_decoders_[0]->IsValid(),
               "Failed to create video decoder for provided video data");
}


DALI_SCHEMA(experimental__inputs__Video)
                .DocStr(
                        R"code(
Streams and decodes a video from a memory buffer. To be used with long and high resolution videos.

Returns a batch of sequences of frames, with the layout: ``(F, H, W, C)``, where:

* ``F`` - number of frames in a sequence,
* ``H`` - height of the frame,
* ``W`` - width of the frame,
* ``C`` - number of channels in the frame.

When using ``fn.inputs.video`` operator inside the DALI Pipeline, the user needs to provide the data
using :meth:`Pipeline.feed_input`. When the Operator is fed with data, the Pipeline can be run
multiple times and the ``fn.inputs.video`` operator will return consecutive sequences, as long as
there is enough data to decode. When the source of the frames (the video file) depletes, user needs
to call another ``feed_input`` again to provide the next video file to the operator. This Operator
has an inner-queue for the data, so the ``feed_input`` may be called multiple times and when given
video file ends, the Operator will fetch the next one automatically from the top of the queue.
Running the pipeline while there is no data for the ``fn.inputs.video`` to run results in an error.

This operator takes only one video as and input (i.e. ``input_batch_size=1``) and will return
batches of sequences. Every output batch will have the ``max_batch_size`` samples, set during
the Pipeline creation. When the number of frames in the video file does not allow to split
the frames uniformly across batches, the last batch returned by this operator for a given video
will be partial and the last sequence in this batch will be determined using
``last_sequence_policy`` parameter. For example::


    This is a video that consists of 67 frames (every '-' is a frame):
    -------------------------------------------------------------------


    User decided that there shall be 5 frames per sequence and
    the last_sequence_policy='partial':
    -------------------------------------------------------------------
    [   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][]
    -------------------------------------------------------------------
    Since there are not enough frames, the last sequence comprises 2 frames.


    The Pipeline has max_batch_size=3, therefore the operator will return
    5 batches of sequences.
    First 4 batches comprise 3 sequences and the last batch is partial and
    comprises 2 sequences.
    ---------------   ---------------   ---------------   ---------------   -------
    [   ][   ][   ]   [   ][   ][   ]   [   ][   ][   ]   [   ][   ][   ]   [   ][]
    ---------------   ---------------   ---------------   ---------------   -------


    With the last_sequence_policy='pad', the last sequence of the last batch
    will be padded with 0:
    ---------------   ---------------   ---------------   ---------------   -------000
    [   ][   ][   ]   [   ][   ][   ]   [   ][   ][   ]   [   ][   ][   ]   [   ][   ]
    ---------------   ---------------   ---------------   ---------------   -------000


The difference between ``fn.inputs.video`` and ``fn.readers.video`` is that the former
reads an encoded video from memory and the latter reads the encoded video from disk.

The difference between ``fn.inputs.video`` and ``fn.decoders.video`` is that the former
does not decode the whole video file in one go. This behaviour is needed for longer videos. E.g.
5-min, 4k, 30fps decoded video takes about 1.7 TB of memory.

This operator accepts most of the video containers and file formats. FFmpeg is used to parse
the video container. In the situations, that the container does not contain required metadata
(e.g. frames sizes, number of frames, etc...), the operator needs to find it out itself,
which may result in a slowdown.
)code")
                .NumInput(0)
                .NumOutput(1)
                .AddArg("sequence_length", R"code(
Number of frames in each sequence.
)code", DALI_INT32)
                .AddOptionalArg("last_sequence_policy", R"code(
Specifies, how to handle the last sequence in the video file.

For a given number of frames in the video file and ``frames_per_sequence`` parameter,
it might happen that the video can't be split uniformly across sequences. If the
``last_sequence_policy='partial'``, the last sequence might have fewer frames than
``frames_per_sequence`` value specified. If the ``last_sequence_policy='partial'``,
the last sequence will always have ``frames_per_sequence`` frames and will
be padded with empty frames.

Allowed values are ``'partial'`` and ``'pad'``.
)code", "partial")
                .AddOptionalArg("affine", R"code(
Applies only to the mixed backend type.
If set to True, each thread in the internal thread pool will be tied to a specific CPU core.
 Otherwise, the threads can be reassigned to any CPU core by the operating system.
)code", true)
                .AddParent("InputOperatorBase");


class VideoInputCpu : public VideoInput<CPUBackend> {
  /*
   * This awkward class originates from an API inconsistency between
   * Operator<CPUBackend> and Operator<MixedBackend>. Operator<CPUBackend> has a `RunImpl` function
   * to be overriden, while Operator<MixedBackend> has `Run` function to be overriden.
   * Can't sort it out using SFINAE, since these are virtual functions.
   */
 public:
  explicit VideoInputCpu(const OpSpec &spec) : VideoInput<CPUBackend>(spec) {}
  void RunImpl(Workspace &ws) override { VideoInputRunImpl(ws); }
};


DALI_REGISTER_OPERATOR(experimental__inputs__Video, VideoInputCpu, CPU);

}  // namespace dali

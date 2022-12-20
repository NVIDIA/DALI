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
#include <vector>

namespace dali {

/// By definition, the input batch size of this Operator is always 1.
constexpr int input_batch_size = 1;


template<>
bool VideoInput<CPUBackend>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                       const Workspace &ws) {
  if (!valid_) {
    InputOperator<CPUBackend>::HandleDataAvailability();
    TensorList<CPUBackend> encoded_videos;
    encoded_videos.set_pinned(device_id_ != CPU_ONLY_DEVICE_ID);
    frames_decoders_.resize(input_batch_size);
    auto &thread_pool = ws.GetThreadPool();
    this->ForwardCurrentData(encoded_videos, thread_pool);

    // Creating FramesDecoders
    auto sample = encoded_videos[0];
    auto data = reinterpret_cast<const char *>(sample.data<uint8_t>());
    size_t size = sample.shape().num_elements();
    frames_decoders_[0] = std::make_unique<FramesDecoder>(data, size, false);

    assert(output_descs_.empty());
    DetermineOutputDescs(static_cast<int>(frames_decoders_[0]->NumFrames()));

    // This has to be done for every video file, since we need to know the shape of the frames.
    if (last_sequence_policy_ == "pad") {
      InitializePadValue(0);
    }

    valid_ = true;
  }
  output_desc.resize(1);
  output_desc[0] = output_descs_.front();
  output_descs_.pop_front();
  return true;
}


template<>
void VideoInput<CPUBackend>::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  output.SetLayout("FHWC");
  bool full_sequence;
  for (int64_t s = 0; s < output.num_samples(); s++) {
    auto pad_value =
            last_sequence_policy_ == "pad" ? std::optional<SampleView<CPUBackend>>(GetPadFrame())
                                           : std::nullopt;
    full_sequence = DecodeFrames(output[s], 0, frames_per_sequence_, pad_value);
    if (!full_sequence) {
      break;
    }
  }
  if (!full_sequence || frames_decoders_[0]->NextFrameIdx() == -1) {
    Invalidate();
  }
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


    User decided that there shall be 5 frames per sequence and the last_sequence_policy='partial':
    -------------------------------------------------------------------
    [   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][   ][]
    -------------------------------------------------------------------
                      Since there are not enough frames, the last sequence comprises 2 frames.


    The Pipeline has max_batch_size=3, therefore the operator will return 5 batches of sequences.
    First 4 batches comprise 3 sequences and the last batch is partial and comprises 2 sequences.
    ---------------   ---------------   ---------------   ---------------   -------
    [   ][   ][   ]   [   ][   ][   ]   [   ][   ][   ]   [   ][   ][   ]   [   ][]
    ---------------   ---------------   ---------------   ---------------   -------


    With the last_sequence_policy='pad', the last sequence of the last batch will be padded with 0:
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
                .AddArg("frames_per_sequence", R"code(
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
                .AddParent("InputOperatorBase");


DALI_REGISTER_OPERATOR(experimental__inputs__Video, VideoInput<CPUBackend>, CPU);

}  // namespace dali

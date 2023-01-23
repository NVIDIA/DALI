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
#include <utility>
#include <vector>

namespace dali {

template<typename Backend, typename FramesDecoder>
void VideoInput<Backend, FramesDecoder>::LoadDataFromInputOperator(ThreadPool &thread_pool) {
  // By definition, the input batch size of this Operator is always 1.
  static constexpr int input_batch_size = 1;
  assert(needs_data_load_);  // Data shall not be loaded if it's not needed.
  assert(this->HasDataInQueue());  // Data shall not be loaded if there's no data in queue.
  encoded_video_.Reset();
  encoded_video_.set_pinned(device_id_ != CPU_ONLY_DEVICE_ID);
  this->frames_decoders_.resize(input_batch_size);
  this->ForwardCurrentData(encoded_video_, data_id_, thread_pool);
  needs_data_load_ = false;
}


template<typename Backend, typename FramesDecoder>
void
VideoInput<Backend, FramesDecoder>::SetNextDataIdTrace(Workspace &ws, std::string next_data_id) {
  ws.SetOperatorTrace(next_output_data_id_trace_name_, std::move(next_data_id));
}


template<typename Backend, typename FramesDecoder>
void VideoInput<Backend, FramesDecoder>::EraseNextDataIdTrace(Workspace &ws) {
  ws.EraseOperatorTrace(next_output_data_id_trace_name_);
}


template<typename Backend, typename FramesDecoder>
void VideoInput<Backend, FramesDecoder>::CreateDecoder(const Workspace &ws) {
  auto sample = encoded_video_[0];
  auto data = reinterpret_cast<const char *>(sample.data<uint8_t>());
  size_t size = sample.shape().num_elements();
  if constexpr (std::is_same_v<FramesDecoder, dali::FramesDecoder>) {
    frames_decoders_[0] = std::make_unique<FramesDecoder>(data, size, false);
  } else if constexpr (std::is_same_v<FramesDecoder, FramesDecoderGpu>) {
    frames_decoders_[0] = std::make_unique<FramesDecoder>(data, size, ws.stream(), false);
  }
}


template<typename Backend, typename FramesDecoder>
bool VideoInput<Backend, FramesDecoder>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                       const Workspace &ws) {
  if (!initialized_) {
    if (needs_data_load_) {
      InputOperator<InBackend>::HandleDataAvailability();
      auto &tp = GetThreadPool(ws);
      LoadDataFromInputOperator(tp);
    }

    CreateDecoder(ws);

    assert(output_descs_.empty());
    DetermineOutputDescs(static_cast<int>(frames_decoders_[0]->NumFrames()));

    // This has to be done for every video file, since we need to know the shape of the frames.
    if (last_sequence_policy_ == "pad") {
      InitializePadValue(0);
    }

    initialized_ = true;
  }
  output_desc.resize(1);
  output_desc[0] = output_descs_.front();
  output_descs_.pop_front();
  return true;
}


template<typename Backend, typename FramesDecoder>
void VideoInput<Backend, FramesDecoder>::RunImpl(Workspace &ws) {
  auto &output = ws.Output<OutBackend>(0);
  output.SetLayout("FHWC");

  auto stream = !detail::is_cpu<Backend>() ? std::make_optional(ws.stream()) : std::nullopt;

  bool full_sequence;
  for (int64_t s = 0; s < output.num_samples(); s++) {
    auto pad_value =
            last_sequence_policy_ == "pad" ? std::optional<SampleView<OutBackend>>(GetPadFrame())
                                           : std::nullopt;
    full_sequence = DecodeFrames(output[s], 0, sequence_length_, pad_value, stream);
    if (!full_sequence) {
      break;
    }
  }

  // There won't be any more output using the current input.
  bool input_sample_depleted = !full_sequence || frames_decoders_[0]->NextFrameIdx() == -1;

  if (input_sample_depleted) {
    Invalidate();
    if (HasDataInQueue()) {
      /*
       * Loading the next input (if available).
       * Instead of doing this in Setup, it's done in Run so that operator can assign proper
       * "next_output_data_id" trace.
       */
      LoadDataFromInputOperator(GetThreadPool(ws));
    }
  }

  if (data_id_) {
    SetNextDataIdTrace(ws, *data_id_);
  } else {
    EraseNextDataIdTrace(ws);
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


DALI_REGISTER_OPERATOR(experimental__inputs__Video, VideoInput<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(experimental__inputs__Video, VideoInput<MixedBackend>, Mixed);

}  // namespace dali

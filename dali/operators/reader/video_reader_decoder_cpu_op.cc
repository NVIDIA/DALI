// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/reader/video_reader_decoder_cpu_op.h"

#include <string>
#include <vector>

namespace dali {

VideoReaderDecoderCpu::VideoReaderDecoderCpu(const OpSpec &spec)
    : DataReader<CPUBackend, VideoSampleCpu, VideoSampleCpu, true>(spec),
      has_labels_(spec.HasArgument("labels")),
      has_frame_idx_(spec.GetArgument<bool>("enable_frame_num")) {
      loader_ = InitLoader<VideoLoaderDecoderCpu>(spec);
      this->SetInitialSnapshot();
}

void VideoReaderDecoderCpu::RunImpl(SampleWorkspace &ws) {
  const auto &sample = GetSample(ws.data_idx());
  auto &video_output = ws.Output<CPUBackend>(0);

  video_output.Copy(sample.data_);
  video_output.SetSourceInfo(sample.data_.GetSourceInfo());

  int out_index = 1;
  if (has_labels_) {
    auto &label_output = ws.Output<CPUBackend>(out_index);
    label_output.Resize({}, DALIDataType::DALI_INT32);
    label_output.mutable_data<int>()[0] = sample.label_;
    out_index++;
  }
  if (has_frame_idx_) {
    auto &frame_idx_output = ws.Output<CPUBackend>(out_index);
    frame_idx_output.Resize({}, DALIDataType::DALI_INT32);
    frame_idx_output.mutable_data<int>()[0] = sample.first_frame_;
    out_index++;
  }
}

namespace detail {
inline int VideoReaderDecoderOutputFn(const OpSpec &spec) {
  bool has_labels = spec.HasArgument("labels");
  bool has_frame_num_output  = spec.GetArgument<bool>("enable_frame_num");
  return 1 + has_labels + has_frame_num_output;
}
}  // namespace detail

DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoderCpu, CPU);

DALI_SCHEMA(experimental__readers__Video)
  .DocStr(R"code(Loads and decodes video files from disk.

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
  
.. note::
  Containers which do not support indexing, like MPEG, require DALI to build the index.
DALI will go through the video and mark keyframes to be able to seek effectively,
even in the constant frame rate scenario.
)code")
  .NumInput(0)
  .OutputFn(detail::VideoReaderDecoderOutputFn)
  .AddOptionalArg("filenames",
      R"code(Absolute paths to the video files to load.)code",
      std::vector<std::string>{})
    .AddOptionalArg<vector<int>>("labels", R"(Labels associated with the files listed in
`filenames` argument. If not provided, no labels will be yielded.)", nullptr)
  .AddArg("sequence_length",
      R"code(Frames to load per sequence.)code",
      DALI_INT32)
  .AddOptionalArg("enable_frame_num",
      R"code(If set, returns the index of the first frame in the decoded sequence
as an additional output.)code",
      false)
  .AddOptionalArg("step",
      R"code(Frame interval between each sequence.

When the value is less than 0, `step` is set to `sequence_length`.)code",
      -1)
  .AddOptionalArg("stride",
      R"code(Distance between consecutive frames in the sequence.)code", 1u, false)
  .AddParent("LoaderBase");

}  // namespace dali

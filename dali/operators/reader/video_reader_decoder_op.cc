// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/reader/video_reader_decoder_op.h"

#include <string>
#include <vector>

namespace dali {

VideoReaderDecoder::VideoReaderDecoder(const OpSpec &spec)
    : DataReader<CPUBackend, VideoSample>(spec),
      has_labels_(spec.HasArgument("labels")) {
      loader_ = InitLoader<VideoLoaderDecoder>(spec);
}

void VideoReaderDecoder::RunImpl(SampleWorkspace &ws) {
  const auto &sample = GetSample(ws.data_idx());
  auto &video_output = ws.Output<CPUBackend>(0);

  video_output.Copy(sample.data_, 0);

  if (has_labels_) {
    auto &label_output = ws.Output<CPUBackend>(1);
    label_output.Resize({});
    label_output.set_type<float>();
    label_output.mutable_data<int>()[0] = sample.label_;
  }
}

namespace detail {
inline int VideoReaderDecoderOutputFn(const OpSpec &spec) {
  return spec.HasArgument("labels") ? 2 : 1;
}
}  // namespace detail

DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder, CPU);

DALI_SCHEMA(experimental__readers__Video)
  .DocStr(R"code(Loads and decodes video files using FFmpeg.

The video streams can be in most of the container file formats. FFmpeg is used to parse video
containers and returns a batch of sequences of ``sequence_length`` frames with shape
``(N, F, H, W, C)``, where ``N`` is the batch size, and ``F`` is the number of frames).

.. note::
  Containers which doesn't support indexing, like mpeg, requires DALI to build the index.)code")
  .NumInput(0)
  .OutputFn(detail::VideoReaderDecoderOutputFn)
  .AddOptionalArg("filenames",
      R"code(File names of the video files to load.)code",
      std::vector<std::string>{})
    .AddOptionalArg<vector<int>>("labels", R"(Labels associated with the files listed in
``filenames`` argument. If not provided, no labels will be yielded.)", nullptr)
  .AddArg("sequence_length",
      R"code(Frames to load per sequence.)code",
      DALI_INT32)
  .AddOptionalArg("step",
      R"code(Frame interval between each sequence.

When the value is less than 0, ``step`` is set to ``sequence_length``.)code",
      -1)
  .AddOptionalArg("stride",
      R"code(Distance between consecutive frames in the sequence.)code", 1u, false)
  .AddParent("LoaderBase");

}  // namespace dali

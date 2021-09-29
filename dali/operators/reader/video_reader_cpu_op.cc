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
#include "dali/operators/reader/video_reader_cpu_op.h"

namespace dali {

VideoReaderCPU::VideoReaderCPU(const OpSpec &spec)
    : DataReader<CPUBackend, VideoSample>(spec),
      has_labels_(spec.HasArgument("labels")) {
      loader_ = InitLoader<VideoLoaderCPU>(spec);
}

void VideoReaderCPU::RunImpl(SampleWorkspace &ws) {
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

}  // namespace dali

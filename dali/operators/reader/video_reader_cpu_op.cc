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
#include "dali/operators/reader/loader/video_loader_cpu.h"

namespace dali {

VideoReaderCPU::VideoReaderCPU(const OpSpec &spec)
    : DataReader<CPUBackend, Tensor<CPUBackend>>(spec) {
      loader_ = InitLoader<VideoLoaderCPU>(spec);
}

void VideoReaderCPU::RunImpl(SampleWorkspace &ws) {
  const auto &video_sample = GetSample(ws.data_idx());
  auto &video_output = ws.Output<CPUBackend>(0);

  video_output.Copy(video_sample, 0);
}

}  // namespace dali

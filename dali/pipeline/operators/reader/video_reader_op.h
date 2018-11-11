// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_READER_VIDEO_READER_OP_H_
#define DALI_PIPELINE_OPERATORS_READER_VIDEO_READER_OP_H_

#include "dali/pipeline/operators/reader/reader_op.h"
#include "dali/pipeline/operators/reader/loader/video_loader.h"
//#include "dali/pipeline/operators/reader/parser/video_parser.h"

namespace dali {

class VideoReader : public DataReader<GPUBackend, SequenceWrapper> {
 public:
  explicit VideoReader(const OpSpec &spec)
  : DataReader<GPUBackend, SequenceWrapper>(spec),
    filenames_(spec.GetRepeatedArgument<std::string>("filenames")),
    count_(spec.GetArgument<int>("count")) {
    	loader_.reset(new VideoLoader(spec, filenames_));
  }

  virtual inline ~VideoReader() = default;

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx) override {
    const int data_idx = ws->data_idx();
    auto* sequence = prefetched_batch_[data_idx];
    auto* sequence_ouput = ws->Output<GPUBackend>(0);
  }
  void SetupSharedSampleParams(SampleWorkspace *ws) override;

 private:
  std::vector<std::string> filenames_;
  int count_;

  USE_READER_OPERATOR_MEMBERS(GPUBackend, SequenceWrapper);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_VIDEO_READER_OP_H_
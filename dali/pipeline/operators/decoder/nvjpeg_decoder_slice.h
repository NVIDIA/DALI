// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_SLICE_H_
#define DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_SLICE_H_

#include <vector>
#include "dali/pipeline/operators/decoder/nvjpeg_decoder.h"

namespace dali {

class nvJPEGDecoderSlice : public nvJPEGDecoder {
 public:
  explicit nvJPEGDecoderSlice(const OpSpec& spec);
  ~nvJPEGDecoderSlice() noexcept(false) override = default;

  DISABLE_COPY_MOVE_ASSIGN(nvJPEGDecoderSlice);

 protected:
  using OperatorBase::Run;
  void Run(MixedWorkspace *ws) override {
    DataDependentSetup(ws);
    nvJPEGDecoder::Run(ws);
  }

  inline CropWindowGenerator GetCropWindowGenerator(int data_idx) const override {
    return per_sample_crop_window_generators_[data_idx];
  }

 private:
  void DataDependentSetup(MixedWorkspace *ws);

  std::vector<CropWindowGenerator> per_sample_crop_window_generators_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_NVJPEG_DECODER_SLICE_H_

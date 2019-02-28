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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_SLICE_H_
#define DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_SLICE_H_

#include <vector>
#include "dali/common.h"
#include "dali/pipeline/operators/decoder/host_decoder.h"
#include "dali/pipeline/operators/crop/slice_attr.h"

namespace dali {

class HostDecoderSlice : public HostDecoder, public SliceAttr {
 public:
  explicit HostDecoderSlice(const OpSpec &spec);

  inline ~HostDecoderSlice() override = default;
  DISABLE_COPY_MOVE_ASSIGN(HostDecoderSlice);

 protected:
  inline void RunImpl(SampleWorkspace *ws, const int idx) override {
    SliceAttr::ProcessArguments(ws);
    HostDecoder::RunImpl(ws, idx);
  }

  inline CropWindowGenerator GetCropWindowGenerator(int data_idx) const override {
    return SliceAttr::GetCropWindowGenerator(data_idx);
  }

 private:
  std::vector<CropWindowGenerator> per_sample_crop_window_generators_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_SLICE_H_

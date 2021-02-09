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

#ifndef DALI_OPERATORS_DECODER_HOST_FUSED_HOST_DECODER_SLICE_H_
#define DALI_OPERATORS_DECODER_HOST_FUSED_HOST_DECODER_SLICE_H_

#include <vector>
#include "dali/core/common.h"
#include "dali/operators/generic/slice/slice_attr.h"
#include "dali/operators/decoder/host/host_decoder.h"

namespace dali {

class HostDecoderSlice : public HostDecoder {
 public:
  explicit HostDecoderSlice(const OpSpec &spec);

  inline ~HostDecoderSlice() override = default;
  DISABLE_COPY_MOVE_ASSIGN(HostDecoderSlice);

 protected:
  inline void RunImpl(HostWorkspace &ws) override {
    slice_attr_.ProcessArguments<CPUBackend>(ws);
    Operator<CPUBackend>::RunImpl(ws);
  }

  inline void RunImpl(SampleWorkspace &ws) override {
    HostDecoder::RunImpl(ws);
  }

  inline CropWindowGenerator GetCropWindowGenerator(int data_idx) const override {
    return slice_attr_.GetCropWindowGenerator(data_idx);
  }

 private:
  SliceAttr slice_attr_;
  std::vector<CropWindowGenerator> per_sample_crop_window_generators_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_HOST_FUSED_HOST_DECODER_SLICE_H_

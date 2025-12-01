// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/decoder/host/host_decoder.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <tuple>
#include "dali/core/nvtx.h"
#include "dali/operators/decoder/image/image_factory.h"

namespace dali {

void HostDecoder::RunImpl(SampleWorkspace &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);
  auto file_name = input.GetSourceInfo();

  // Verify input
  DALI_ENFORCE(input.ndim() == 1, "Input must be 1D encoded jpeg string.");
  DALI_ENFORCE(IsType<uint8_t>(input.type()), "Input must be stored as uint8 data.");

  std::unique_ptr<Image> img;
  try {
    DomainTimeRange tr(make_string("Decode #", ws.data_idx(), " fast_idct=", use_fast_idct_),
                       DomainTimeRange::kBlue1);
    img = ImageFactory::CreateImage(input.data<uint8_t>(), input.size(), output_type_);
    img->SetCropWindowGenerator(GetCropWindowGenerator(ws.data_idx()));
    img->SetUseFastIdct(use_fast_idct_);
    img->Decode();
  } catch (std::exception &e) {
    DALI_FAIL(e.what(), ". File: ", file_name);
  }
  const auto decoded = img->GetImage();
  const auto shape = img->GetShape();
  output.Resize(shape, DALI_UINT8);
  output.SetLayout("HWC");
  auto *out_data = output.mutable_data<uint8_t>();
  DomainTimeRange tr(make_string("memcpy #", ws.data_idx()), DomainTimeRange::kBlue1);
  std::memcpy(out_data, decoded.get(), volume(shape));
}

DALI_REGISTER_OPERATOR(decoders__Image, HostDecoder, CPU);
DALI_REGISTER_OPERATOR(ImageDecoder, HostDecoder, CPU);

}  // namespace dali

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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_H_
#define DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_H_

#include <opencv2/opencv.hpp>
#include <tuple>
#include <memory>
#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/image/image_factory.h"


namespace dali {

class HostDecoder : public Operator<CPUBackend> {
 public:
  explicit inline HostDecoder(const OpSpec &spec) :
          Operator<CPUBackend>(spec),
          output_type_(spec.GetArgument<DALIImageType>("output_type")),
          c_(IsColor(output_type_) ? 3 : 1) {}


  inline ~HostDecoder() override = default;
  DISABLE_COPY_MOVE_ASSIGN(HostDecoder);

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx) override {
    auto &input = ws->Input<CPUBackend>(idx);
    auto output = ws->Output<CPUBackend>(idx);
    auto file_name = input.GetSourceInfo();

    // Verify input
    DALI_ENFORCE(input.ndim() == 1,
                 "Input must be 1D encoded jpeg string.");
    DALI_ENFORCE(IsType<uint8>(input.type()),
                 "Input must be stored as uint8 data.");

    std::unique_ptr<Image> img;
    try {
      img = ImageFactory::CreateImage(input.data<uint8>(), input.size(), output_type_);
      img->Decode();
    } catch (std::runtime_error &e) {
      DALI_FAIL(e.what() + "File: " + file_name);
    }
    const auto decoded = img->GetImage();
    const auto hwc = img->GetImageDims();
    const auto h = std::get<0>(hwc);
    const auto w = std::get<1>(hwc);
    const auto c = std::get<2>(hwc);

    output->Resize({static_cast<int>(h), static_cast<int>(w), static_cast<int>(c)});
    unsigned char *out_data = output->mutable_data<unsigned char>();
    std::memcpy(out_data, decoded.get(), h * w * c);
  }


  DALIImageType output_type_;
  int c_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_H_

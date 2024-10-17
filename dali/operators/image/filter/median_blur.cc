// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <optional>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <cvcuda/OpMedianBlur.hpp>
#include "dali/core/dev_buffer.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/common/utils.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"

#include "dali/operators/nvcvop/nvcvop.h"

namespace dali {


DALI_SCHEMA(experimental__MedianBlur)
  .DocStr(R"doc(
Median blur performs smoothing of an image or sequence of images by replacing each pixel
with the median color of a surrounding rectangular region.
  )doc")
  .NumInput(1)
  .InputDox(0, "input", "TensorList",
            "Input data. Must be images in HWC or CHW layout, or a sequence of those.")
  .NumOutput(1)
  .InputLayout({"HW", "HWC", "FHWC", "CHW", "FCHW"})
  .AddOptionalArg("window_size",
    "The size of the window over which the smoothing is performed",
    std::vector<int>({3, 3}),
    true);


class MedianBlur : public nvcvop::NVCVSequenceOperator<StatelessOperator> {
 public:
  explicit MedianBlur(const OpSpec &spec) :
    nvcvop::NVCVSequenceOperator<StatelessOperator>(spec) {}

  bool ShouldExpandChannels(int input_idx) const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    const auto &input = ws.Input<GPUBackend>(0);
    auto sh = input.shape();
    output_desc.resize(1);
    output_desc[0] = {sh, input.type()};
    return true;
  }

  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<GPUBackend>(0);
    auto &output = ws.Output<GPUBackend>(0);
    output.SetLayout(input.GetLayout());

    kernels::DynamicScratchpad scratchpad({}, AccessOrder(ws.stream()));
    auto ksize = AcquireTensorArgument<int32_t>(ws, scratchpad, ksize_arg_,
                                                TensorShape<1>(2),
                                                nvcvop::GetDataType<int32_t>(), "W");

    auto input_images = GetInputBatch(ws, 0);
    auto output_images = GetOutputBatch(ws, 0);
    if (!median_blur_ || input.num_samples() > op_batch_size_) {
      op_batch_size_ = std::max(op_batch_size_ * 2, input.num_samples());
      median_blur_.emplace(op_batch_size_);
    }
    (*median_blur_)(ws.stream(), input_images, output_images, ksize);
  }

 private:
  USE_OPERATOR_MEMBERS();
  ArgValue<int, 1> ksize_arg_{"window_size", spec_};
  int op_batch_size_ = 0;
  std::optional<cvcuda::MedianBlur> median_blur_{};
};

DALI_REGISTER_OPERATOR(experimental__MedianBlur, MedianBlur, GPU);

}  // namespace dali

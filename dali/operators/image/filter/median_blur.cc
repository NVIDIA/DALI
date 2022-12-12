// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include <nvcv/ImageBatch.hpp>
#include <operators/OpMedianBlur.hpp>

namespace dali {

namespace nvcv = nv::cv;
namespace nvcvop = nv::cvop;

DALI_SCHEMA(MedianBlur)
  .NumInput(1)
  .NumOutput(1)
  .InputLayout({"HWC", "FHWC", "CHW", "FCHW"})
  .AddOptionalArg("window_size",
    "The size of the window over which the smoothing is performed", 3, true);


class MedianBlur : public Operator<GPUBackend> {
 public:
  MedianBlur(const OpSpec &spec) : Operator<GPUBackend>(spec) {
    if (ksize_.HasExplicitConstant()) {
      int ksize = *ksize_[0].data;
      DALI_ENFORCE(ksize > 1 && (ksize&1),
                   "The window size must be an odd integer greater than one.");
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    return true;
  }

  void RunImpl(Workspace &ws) {
    auto &input = ws.Input<CPUBackend>(0);
    int effective_batch_size = CalculateEffectiveBatchSize(input);
    if (effective_batch_size > effective_batch_size_) {
      effective_batch_size_ = std::max(2*effective_batch_size_, effective_batch_size);
      impl_.reset();
      impl_ = std::make_unique<nvcvop::MedianBlur>(effective_batch_size_);
      batch_ = nvcv::IImageBatchVarShape
    }
  }

  bool CanInferOutputs() const override {
    return true;
  }

  int effective_batch_size_ = 0;
  std::unique_ptr<nvcvop::MedianBlur> impl_;
  ArgValue<int> ksize_;
  nvcv::ImageBatchVarShape batch_;
};

}  // namespace dali

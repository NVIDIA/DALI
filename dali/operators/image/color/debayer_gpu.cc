// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include "dali/core/span.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/color_manipulation/debayer/debayer.h"
#include "dali/kernels/imgproc/color_manipulation/debayer/debayer_npp.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/image/color/debayer.h"

namespace dali {

namespace debayer {

template <typename InOutT>
class DebayerImplGPU : public DebayerImplBase<GPUBackend> {
 public:
  using Kernel = kernels::debayer::NppDebayerKernel<InOutT>;
  explicit DebayerImplGPU(const OpSpec &spec, const std::vector<debayer::DALIBayerPattern> &pattern)
      : pattern_{pattern} {
    kmgr_.Resize<Kernel>(1, spec.template GetArgument<int>("device_id"));
  }

  void RunImpl(Workspace &ws) override {
    auto &output = ws.Output<GPUBackend>(0);
    output.SetLayout("HWC");
    ctx_.gpu.stream = ws.stream();

    auto out_view = view<InOutT, 3>(output);
    auto in_view = GetInView(ws);
    kmgr_.Run<Kernel>(0, ctx_, out_view, in_view, make_span(pattern_));
  }

 protected:
  TensorListView<StorageGPU, const InOutT, 2> GetInView(Workspace &ws) {
    const auto &input = ws.Input<GPUBackend>(0);
    const auto &in_shape = input.shape();
    assert(in_shape.sample_dim() == 2 ||
           in_shape.sample_dim() == 3);  // by Debayer op's base class setupimpl
    if (in_shape.sample_dim() == 2) {
      return view<const InOutT, 2>(input);
    }
    assert(([&]() {
      for (int sample_idx = 0; sample_idx < in_shape.num_samples(); sample_idx++) {
        if (in_shape[sample_idx][2] != 1) {
          return false;
        }
      }
      return true;
    })());  // by Debayer op's base class setupimpl
    auto collapsed_shape = collapse_dims<2>(in_shape, {{1, 2}});
    auto in_view = view<const InOutT, 3>(input);
    return reshape<2>(in_view, collapsed_shape);
  }

  const std::vector<debayer::DALIBayerPattern> &pattern_;
  kernels::KernelManager kmgr_;
  kernels::KernelContext ctx_;
};

}  // namespace debayer

class DebayerGPU : public Debayer<GPUBackend> {
 public:
  explicit DebayerGPU(const OpSpec &spec) : Debayer<GPUBackend>(spec) {}

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;
  void RunImpl(Workspace &ws) override;
  std::unique_ptr<debayer::DebayerImplBase<GPUBackend>> impl_ = nullptr;
};

bool DebayerGPU::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  bool has_inferred = Debayer<GPUBackend>::SetupImpl(output_desc, ws);
  // If the algorithm is set to default, use default npp
  if (alg_ == debayer::DALIDebayerAlgorithm::DALI_DEBAYER_DEFAULT) {
    alg_ = debayer::DALIDebayerAlgorithm::DALI_DEBAYER_DEFAULT_NPP;
  }
  assert(has_inferred);
  if (impl_ == nullptr) {
    DALI_ENFORCE(alg_ == debayer::DALIDebayerAlgorithm::DALI_DEBAYER_DEFAULT_NPP,
                 "Only default and default_npp algorithm is supported on GPU.");
    const auto type = ws.GetInputDataType(0);
    TYPE_SWITCH(type, type2id, InT, DEBAYER_SUPPORTED_TYPES_GPU, (
      impl_ = std::make_unique<debayer::DebayerImplGPU<InT>>(spec_, pattern_);
    ), DALI_FAIL(make_string("Unsupported input type for debayer operator: ", type,  // NOLINT
                             ". Only tensors of uint8_t and uint16_t type are supported."));
    );  // NOLINT
  }
  return true;
}

void DebayerGPU::RunImpl(Workspace &ws) {
  assert(impl_ != nullptr);
  impl_->RunImpl(ws);
}

// Kept for backwards compatibility
DALI_REGISTER_OPERATOR(experimental__Debayer, DebayerGPU, GPU);

DALI_REGISTER_OPERATOR(Debayer, DebayerGPU, GPU);

}  // namespace dali

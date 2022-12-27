// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include <vector>
#include "dali/operators/audio/resample.h"
#include "dali/operators/audio/resampling_params.h"
#include "dali/kernels/signal/resampling_gpu.h"
#include "dali/kernels/kernel_params.h"
#include "dali/kernels/kernel_manager.h"

namespace dali {
namespace audio {

using kernels::InListGPU;
using kernels::OutListGPU;

class ResampleGPU : public ResampleBase<GPUBackend> {
 public:
  using Base = ResampleBase<GPUBackend>;
  explicit ResampleGPU(const OpSpec &spec) : Base(spec) {}

  void RunImpl(Workspace &ws) override {
    auto &out = ws.Output<GPUBackend>(0);
    const auto &in = ws.Input<GPUBackend>(0);
    out.SetLayout(in.GetLayout());

    int N = in.num_samples();
    assert(N == static_cast<int>(args_.size()));
    assert(out.type() == dtype_);

    TYPE_SWITCH(dtype_, type2id, Out, (AUDIO_RESAMPLE_TYPES), (
      TYPE_SWITCH(in.type(), type2id, In, (AUDIO_RESAMPLE_TYPES), (
        ResampleTyped<Out, In>(view<Out>(out), view<const In>(in), ws.stream());
      ), (  // NOLINT
        DALI_FAIL(
          make_string("Unsupported input type: ", in.type(), "\nSupported types are : ",
                      ListTypeNames<AUDIO_RESAMPLE_TYPES>()));
      ));  // NOLINT
    ), (assert(!"Unreachable code.")));  // NOLINT
  }

  template <typename Out, typename In>
  void ResampleTyped(const OutListGPU<Out> &out, const InListGPU<const In> &in,
                     cudaStream_t stream) {
    using Kernel = kernels::signal::resampling::ResamplerGPU<Out, In>;
    if (kmgr_.NumInstances() == 0) {
      kmgr_.Resize<Kernel>(1);
      auto params = ResamplingParams::FromQuality(quality_);
      kmgr_.Get<Kernel>(0).Initialize(params.lobes, params.lookup_size);
    }
    auto args = make_cspan(args_);
    kernels::KernelContext ctx;
    ctx.gpu.stream = stream;
    kmgr_.Setup<Kernel>(0, ctx, in, args);
    kmgr_.Run<Kernel>(0, ctx, out, in, args);
  }

 private:
  kernels::KernelManager kmgr_;
};


}  // namespace audio


// Kept for backwards compatibility
DALI_REGISTER_OPERATOR(experimental__AudioResample, audio::ResampleGPU, GPU);

DALI_REGISTER_OPERATOR(AudioResample, audio::ResampleGPU, GPU);

}  // namespace dali

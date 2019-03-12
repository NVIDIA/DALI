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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_CPU_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_CPU_H_

#include "dali/kernels/imgproc/resample/params.h"
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"
#include "dali/kernels/imgproc/resample/resampling_impl_cpu.h"
#include "dali/kernels/imgproc/resample/resampling_setup.h"

namespace dali {
namespace kernels {

struct ResamplingSetupCPU : SeparableResamplingSetup {
  ResamplingSetupCPU() {
    InitializeCPU();
  }

  static bool IsPureNN(const SampleDesc &desc) {
    return
      desc.filter_type[0] == ResamplingFilterType::Nearest &&
      desc.filter_type[1] == ResamplingFilterType::Nearest;
  }

  struct MemoryReq {
    size_t tmp_size = 0;
    size_t coeffs_size = 0;
    size_t indices_size = 0;

    MemoryReq &extend(const MemoryReq &other) {
      if (other.tmp_size > tmp_size)
        tmp_size = other.tmp_size;
      if (other.coeffs_size > coeffs_size)
        coeffs_size = other.coeffs_size;
      if (other.indices_size > indices_size)
        indices_size = other.indices_size;
      return *this;
    }
  };

  MemoryReq GetMemoryRequirements(const SampleDesc &desc) {
    if (IsPureNN(desc)) {
      return {};  // no memory required for Nearest Neighbour on CPU
    } else {
      MemoryReq req;
      req.tmp_size = volume(desc.tmp_shape()) * desc.channels;
      int axis_order[2];
      axis_order[0] = desc.order == HorzVert ? 1 : 0;  // axis 1 is horizontal, HWC layout
      axis_order[1] = 1 - axis_order[0];

      req.indices_size = std::max(desc.tmp_shape()[axis_order[0]],
                                  desc.out_shape()[axis_order[1]]);

      int support[2] = { desc.filter[0].support(), desc.filter[1].support() };
      req.coeffs_size = std::max(desc.tmp_shape()[axis_order[0]] * support[axis_order[0]],
                                 desc.out_shape()[axis_order[1]] * support[axis_order[1]]);
      return req;
    }
  }
};

struct ResamplingSetupSingleImage : ResamplingSetupCPU {
  void Setup(const TensorShape<3> &in_shape, const ResamplingParams2D &params) {
    SetupSample(desc, in_shape, params);
    memory = GetMemoryRequirements(desc);
  }

  SampleDesc desc;
  MemoryReq memory;
};

template <typename OutputElement, typename InputElement>
struct SeparableResampleCPU  {
  using Input =  InTensorCPU<InputElement, 3>;
  using Output = OutTensorCPU<OutputElement, 3>;

  KernelRequirements Setup(KernelContext &context,
                           const Input &input,
                           const ResamplingParams2D &params) {
    setup.Setup(input.shape, params);

    TensorShape<3> out_shape =
        { setup.desc.out_shape()[0], setup.desc.out_shape()[1], setup.desc.channels };

    ScratchpadEstimator se;
    se.add<float>(AllocType::Host, setup.memory.tmp_size);
    se.add<float>(AllocType::Host, setup.memory.coeffs_size);
    se.add<int32_t>(AllocType::Host, setup.memory.indices_size);

    TensorListShape<> out_tls({ out_shape });

    KernelRequirements req;
    req.output_shapes = { out_tls };
    req.scratch_sizes = se.sizes;

    return req;
  }

  void Run(KernelContext &context,
           const Output &output,
           const Input &input,
           const ResamplingParams2D &params) {
    auto &desc = setup.desc;

    auto in_ROI = as_surface_HWC(input);
    in_ROI.width  = desc.in_shape()[1];
    in_ROI.height = desc.in_shape()[0];
    in_ROI.data  += desc.in_offset();

    auto out_ROI = as_surface_HWC(output);
    out_ROI.width  = desc.out_shape()[1];
    out_ROI.height = desc.out_shape()[0];
    out_ROI.data  += desc.out_offset();

    if (setup.IsPureNN(desc)) {
      ResampleNN(out_ROI, in_ROI,
                 desc.origin[1], desc.origin[0], desc.scale[1], desc.scale[0]);
    } else {
      TensorShape<3> tmp_shape = { desc.tmp_shape()[0], desc.tmp_shape()[1], desc.channels };
      auto tmp = context.scratchpad->AllocTensor<AllocType::Host, float, 3>(tmp_shape);

      auto tmp_surf = as_surface_HWC(tmp);

      void *filter_mem = context.scratchpad->Allocate<int32_t>(AllocType::Host,
        setup.memory.coeffs_size + setup.memory.indices_size);

      if (desc.order == setup.VertHorz) {
        ResamplePass<0, float, InputElement>(tmp_surf, in_ROI, filter_mem);
        ResamplePass<1, OutputElement, float>(out_ROI, tmp_surf, filter_mem);
      } else {
        ResamplePass<1, float, InputElement>(tmp_surf, in_ROI, filter_mem);
        ResamplePass<0, OutputElement, float>(out_ROI, tmp_surf, filter_mem);
      }
    }
  }

  template <int axis, typename PassOutput, typename PassInput>
  void ResamplePass(const Surface2D<PassOutput> &out,
                    const Surface2D<const PassInput> &in,
                    void *mem) {
    auto &desc = setup.desc;

    if (desc.filter_type[axis] == ResamplingFilterType::Nearest) {
      // use specialized NN resampling pass - should be faster
      ResampleNN(out, in,
        desc.origin[1], desc.origin[0],
        (axis == 1 ? desc.scale[1] : 1.0f), (axis == 0 ? desc.scale[0] : 1.0f));
    } else {
      int32_t *indices = static_cast<int32_t*>(mem);
      float *coeffs = static_cast<float*>(static_cast<void*>(indices + desc.out_shape()[axis]));
      int support = desc.filter[axis].support();

      InitializeResamplingFilter(indices, coeffs, desc.out_shape()[axis],
                                 desc.origin[axis], desc.scale[axis], desc.filter[axis]);

      ResampleAxis(out, in, indices, coeffs, support, axis);
    }
  }

  ResamplingSetupSingleImage setup;
};

}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_CPU_H_

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
#include "dali/kernels/imgproc/resample/resampling_windows.h"
#include "dali/kernels/imgproc/resample/resampling_impl_cpu.h"

namespace dali {
namespace kernels {

struct ResamplingSetupCPU {
  struct SampleDesc {
    float origin[2];
    float scale[2];
    FilterWindow filter[2];
    ResamplingFilterType filter_type[2];
    TensorShape<2> tmp_shape, out_shape;
    int channels;
    bool vert_horz;

    bool IsPureNN() const {
      return
        filter_type[0] == ResamplingFilterType::Nearest &&
        filter_type[1] == ResamplingFilterType::Nearest;
    }
  };

  static FilterWindow GetFilter(const FilterDesc &desc) {
    switch (desc.type) {
    case ResamplingFilterType::Nearest:
      return NNFilter();
    case ResamplingFilterType::Linear:
      return LinearFilter();
    case ResamplingFilterType::Triangular:
      return TriangularFilter(desc.radius);
    case ResamplingFilterType::Gaussian:
      return GaussianFilter(desc.radius);
    case ResamplingFilterType::Cubic:
      return CubicFilter();
    case ResamplingFilterType::Lanczos3:
      return Lanczos3Filter();
    default:
      assert(!"Unsupported filter type");
      return LinearFilter();
    }
  }

  SampleDesc GetSampleDesc(const TensorShape<3> &in_shape, const ResamplingParams2D &params) {
    SampleDesc desc;
    desc.channels = in_shape[2];

    for (int i = 0; i < 2; i++) {
      desc.out_shape[i] = params[i].output_size == KeepOriginalSize
        ? in_shape[i] : params[i].output_size;

      FilterDesc filter = desc.out_shape[i] < in_shape[i]
        ? params[i].min_filter : params[i].mag_filter;

      if (!filter.radius)
        filter.radius = DefaultFilterRadius(filter.type, in_shape[i], desc.out_shape[i]);

      desc.filter_type[i] = filter.type;
      desc.filter[i] = GetFilter(filter);
    }

    int in_H = in_shape[0];
    int in_W = in_shape[1];
    int out_H = desc.out_shape[0];
    int out_W = desc.out_shape[1];
    int flt_vert = desc.filter[0].support();
    int flt_horz = desc.filter[1].support();

    desc.scale[0] = static_cast<float>(in_H)/out_H;
    desc.scale[1] = static_cast<float>(in_W)/out_W;
    desc.origin[0] = 0;
    desc.origin[1] = 0;

    // vertical resampling is cheaper
    float horz_cost = 3;
    float vert_cost = 1;
    // calculate the cost of resampling in different orders
    float cost_hv = in_H * out_W * flt_horz * horz_cost + out_H * out_W * flt_vert * vert_cost;
    float cost_vh = out_H * in_W * flt_vert * vert_cost + out_H * out_W * flt_horz * horz_cost;
    desc.vert_horz = (cost_vh < cost_hv);
    if (desc.vert_horz) {
      desc.tmp_shape = { out_H, in_W };
    } else {
      desc.tmp_shape = { in_H, out_W };
    }

    return desc;
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
    if (desc.IsPureNN()) {
      return {};
    } else {
      MemoryReq req;
      req.tmp_size = volume(desc.tmp_shape) * desc.channels;
      req.indices_size = std::max(desc.out_shape[0], desc.out_shape[1]);
      int vsupport = desc.filter[0].support();
      int hsupport = desc.filter[1].support();
      req.coeffs_size = std::max(desc.out_shape[0] * vsupport, desc.out_shape[1] * hsupport);
      return req;
    }
  }
};

struct ResamplingSetupSingleImage : ResamplingSetupCPU {
  void Setup(const TensorShape<3> &in_shape, const ResamplingParams2D &params) {
    desc = GetSampleDesc(in_shape, params);
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
        { setup.desc.out_shape[0], setup.desc.out_shape[1], setup.desc.channels };

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
    if (desc.IsPureNN()) {
      ResampleNN(as_surface_HWC(output), as_surface_HWC(input),
                 desc.origin[1], desc.origin[0], desc.scale[1], desc.scale[0]);
    } else {
      TensorShape<3> tmp_shape = shape_cat(desc.tmp_shape, desc.channels);
      auto tmp = context.scratchpad->AllocTensor<AllocType::Host, float, 3>(tmp_shape);

      void *filter_mem = context.scratchpad->Allocate<int32_t>(AllocType::Host,
        setup.memory.coeffs_size + setup.memory.indices_size);

      if (desc.vert_horz) {
        ResamplePass<0, float, InputElement>(tmp, input, filter_mem);
        ResamplePass<1, OutputElement, float>(output, tmp, filter_mem);
      } else {
        ResamplePass<1, float, InputElement>(tmp, input, filter_mem);
        ResamplePass<0, OutputElement, float>(output, tmp, filter_mem);
      }
    }
  }

  template <int axis, typename PassOutput, typename PassInput>
  void ResamplePass(const OutTensorCPU<PassOutput, 3> &out,
                    const InTensorCPU<PassInput, 3> &in,
                    void *mem) {
    auto &desc = setup.desc;

    if (desc.filter_type[axis] == ResamplingFilterType::Nearest) {
      // use specialized NN resampling pass - should be faster
      ResampleNN(as_surface_HWC(out), as_surface_HWC(in),
        desc.origin[1], desc.origin[0],
        (axis == 1 ? desc.scale[1] : 1.0f), (axis == 0 ? desc.scale[0] : 1.0f));
    } else {
      int32_t *indices = static_cast<int32_t*>(mem);
      float *coeffs = static_cast<float*>(static_cast<void*>(indices + desc.out_shape[axis]));
      int support = desc.filter[axis].support();

      InitializeResamplingFilter(indices, coeffs, desc.out_shape[axis],
                                 desc.origin[axis], desc.scale[axis], desc.filter[axis]);

      ResampleAxis(as_surface_HWC(out), as_surface_HWC(in), indices, coeffs, support, axis);
    }
  }

  ResamplingSetupSingleImage setup;
};

}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_CPU_H_

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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_KERNEL_CPU_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_KERNEL_CPU_H_

#include "dali/kernels/imgproc/resample/params.h"
#include "dali/kernels/imgproc/resample/resampling_windows.h"
#include "dali/kernels/imgproc/resample/resampling_impl_cpu.h"

namespace dali {
namespace kernels {

struct ResamplingSetupCPU {
  struct SampleDesc {
    float x0, y0;
    float scale_x, scale_y;
    bool VertHorz;
    FilterWindow filter[2];
    ResamplingFilterType filter_type[2];
    std::array<int, 2> tmp_shape, out_shape;
    int channels;

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

      if (!filter.radius == 0)
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

    desc.scale_x = static_cast<float>(in_W)/out_W;
    desc.scale_y = static_cast<float>(in_H)/out_H;
    desc.x0 = 0;
    desc.y0 = 0;

    // vertical resampling is cheaper
    float horz_cost = 3;
    float vert_cost = 1;
    // calculate the cost of resampling in different orders
    float cost_hv = in_H * out_W * flt_horz * horz_cost + out_H * out_W * flt_vert * vert_cost;
    float cost_vh = out_H * in_W * flt_vert * vert_cost + out_H * out_W * flt_horz * horz_cost;
    desc.VertHorz = (cost_vh < cost_hv);
    if (desc.VertHorz) {
      desc.tmp_shape = { out_H, in_W };
    } else {
      desc.tmp_shape = { in_H, out_W };
    }

    return desc;
  };
};

struct ResamplingSetupSingleImage : ResamplingSetupCPU {

  void Setup(const TensorShape<3> &in_shape, const ResamplingParams2D &params) {
    desc = GetSampleDesc(in_shape, params);
    if (desc.IsPureNN()) {
      tmp_size = 0;
    } else {
      tmp_size = volume(desc.tmp_shape) * desc.channels;
    }
  }

  SampleDesc desc;
  size_t tmp_size;
  size_t coeffs_size;
  size_t indices_size;
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
    se.add<float>(AllocType::Host, setup.tmp_size);

    KernelRequirements req;
    req.output_shapes = { out_shape };
    req.scratch_sizes = se;

    return req;
  }

  void Run(KernelContext &context,
           const Output &output,
           const Input &input,
           const ResamplingParams2D &params) {

    auto &desc = setup.desc;
    if (desc.IsPureNN()) {
      ResampleNN(as_surface_HWC(output), as_surface_HWC(input), desc.x0, desc.y0, desc.scale_x, desc.scale_y);
    }

  }

  ResamplingSetupSingleImage setup;
};

}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_KERNEL_CPU_H_

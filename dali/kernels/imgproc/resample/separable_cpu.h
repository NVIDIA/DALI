// Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
namespace resampling {

template <int _spatial_ndim>
struct ResamplingSetupCPU : SeparableResamplingSetup<_spatial_ndim> {
  using Base = SeparableResamplingSetup<_spatial_ndim>;
  using Base::spatial_ndim;
  using typename Base::SampleDesc;

  ResamplingSetupCPU() {
    this->InitializeCPU();
  }

  static bool IsPureNN(const SampleDesc &desc) {
    for (auto flt_type : desc.filter_type)
      if (flt_type != ResamplingFilterType::Nearest)
        return false;
    return true;
  }

  struct MemoryReq {
    // size, in elements, of intermediate buffers
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
    if (this->IsPureNN(desc)) {
      return {};  // no memory required for Nearest Neighbour on CPU
    } else {
      MemoryReq req;
      // Temporary buffers are swapped every second stage; they are allocated
      // at the beginning and at the end of the temporary buffer area.
      size_t prev_tmp = 0;
      size_t max_tmp = 0;
      for (int i = 0; i < desc.num_tmp_buffers; i++) {
        size_t v = volume(desc.tmp_shape(i)) * desc.channels;
        if (v + prev_tmp > max_tmp)
          max_tmp = v + prev_tmp;
        prev_tmp = v;
      }
      req.tmp_size = max_tmp;

      const int num_stages = spatial_ndim;

      // returns extent of the dimensions resized at given stage - e.g. if processing
      // (Y, Z, X)), then extent for stage 0 is height, stage 1 - depth, stage 2 - width
      auto resized_dim_extent = [&](int stage) {
        return stage == num_stages - 1
          ? desc.out_shape()[desc.order[num_stages - 1]]
          : desc.tmp_shape(stage)[desc.order[stage]];
      };

      // maximum support of the filter used at given stage
      auto filter_support = [&](int stage) {
        return desc.filter[desc.order[stage]].support();
      };

      req.indices_size = 0;
      req.coeffs_size = 0;

      for (int stage = 0; stage < num_stages; stage++) {
        size_t extent = resized_dim_extent(stage);
        size_t support = filter_support(stage);
        size_t num_coeffs = extent * support;

        if (extent > req.indices_size)
          req.indices_size = extent;
        if (num_coeffs > req.coeffs_size)
          req.coeffs_size = num_coeffs;
      }

      return req;
    }
  }
};

template <int spatial_ndim>
struct ResamplingSetupSingleImage : ResamplingSetupCPU<spatial_ndim> {
  using Base = ResamplingSetupCPU<spatial_ndim>;
  using typename Base::SampleDesc;
  using typename Base::MemoryReq;
  using Base::tensor_ndim;

  void Setup(const TensorShape<tensor_ndim> &in_shape,
             const ResamplingParamsND<spatial_ndim> &params) {
    this->SetupSample(desc, in_shape, params);
    memory = this->GetMemoryRequirements(desc);
  }

  SampleDesc desc;
  MemoryReq memory;
};

template <typename OutputElement, typename InputElement, int _spatial_ndim>
struct SeparableResampleCPU  {
  static constexpr int spatial_ndim = _spatial_ndim;
  static constexpr int tensor_ndim = spatial_ndim + 1;
  using Input =  InTensorCPU<InputElement, tensor_ndim>;
  using Output = OutTensorCPU<OutputElement, tensor_ndim>;

  KernelRequirements Setup(KernelContext &context,
                           const Input &input,
                           const ResamplingParamsND<spatial_ndim> &params) {
    setup.Setup(input.shape, params);

    TensorShape<tensor_ndim> out_shape =
      shape_cat(vec2shape(setup.desc.out_shape()), setup.desc.channels);

    ScratchpadEstimator se;
    if (out_shape.num_elements() > 0) {
      se.add<mm::memory_kind::host, float>(setup.memory.tmp_size);
      se.add<mm::memory_kind::host, float>(setup.memory.coeffs_size);
      se.add<mm::memory_kind::host, int32_t>(setup.memory.indices_size);
    }

    TensorListShape<> out_tls({ out_shape });

    KernelRequirements req;
    req.output_shapes = { out_tls };
    req.scratch_sizes = se.sizes;

    return req;
  }

  void Run(KernelContext &context,
           const Output &output,
           const Input &input,
           const ResamplingParamsND<spatial_ndim> &params) {
    if (output.shape.num_elements() == 0)
      return;

    auto &desc = setup.desc;

    desc.set_base_pointers(input.data, nullptr, output.data);

    auto in_ROI = as_surface_channel_last(input);
    in_ROI.size = desc.in_shape();
    in_ROI.data = desc.template in_ptr<InputElement>();

    auto out_ROI = as_surface_channel_last(output);
    out_ROI.size = desc.out_shape();
    out_ROI.data = desc.template out_ptr<OutputElement>();

    if (setup.IsPureNN(desc)) {
      ResampleNN(out_ROI, in_ROI, desc.origin, desc.scale);
    } else {
      TensorShape<tensor_ndim> tmp_shapes[num_tmp_buffers];
      for (int i = 0; i < num_tmp_buffers; i++) {
        tmp_shapes[i] = shape_cat(vec2shape(desc.tmp_shape(i)), desc.channels);
      }

      float *tmp_buf = context.scratchpad->AllocateHost<float>(setup.memory.tmp_size);
      void *filter_mem = context.scratchpad->AllocateHost<int32_t>(
          setup.memory.coeffs_size + setup.memory.indices_size);

      Surface<spatial_ndim, float> tmp_surf = {}, tmp_prev = {};

      for (int stage = 0; stage < spatial_ndim; stage++) {
        if (stage < spatial_ndim - 1) {
          ptrdiff_t tmp_size = volume(tmp_shapes[stage]);
          ptrdiff_t tmp_ofs = stage & 1 ? setup.memory.tmp_size - tmp_size : 0;
          assert(tmp_ofs >= 0 && tmp_ofs+tmp_size <= static_cast<ptrdiff_t>(setup.memory.tmp_size));
          tmp_surf.data = tmp_buf + tmp_ofs;
          tmp_surf.size = setup.desc.tmp_shape(stage);
          tmp_surf.channels = setup.desc.channels;

          tmp_surf.channel_stride = 1;
          tmp_surf.strides.x = tmp_surf.channels;
          for (int i = 1; i < spatial_ndim; i++) {
            tmp_surf.strides[i] = tmp_surf.strides[i-1] * tmp_surf.size[i-1];
          }
          assert(&tmp_surf(tmp_surf.size - 1) < tmp_buf + setup.memory.tmp_size);
        }

        if (stage == 0)  // in -> tmp(0)
          ResamplePass<float, InputElement>(tmp_surf, in_ROI, filter_mem, desc.order[stage]);
        else if (stage < spatial_ndim - 1)  // tmp(i) -> tmp(i+1)
          ResamplePass<float, float>(tmp_surf, tmp_prev, filter_mem, desc.order[stage]);
        else  // tmp(spatial_ndim-1) -> out
          ResamplePass<OutputElement, float>(out_ROI, tmp_prev, filter_mem, desc.order[stage]);

        tmp_prev = tmp_surf;
      }
    }
  }

  template <typename PassOutput, typename PassInput>
  void ResamplePass(const Surface<spatial_ndim, PassOutput> &out,
                    const Surface<spatial_ndim, const PassInput> &in,
                    void *mem,
                    int axis) {
    auto &desc = setup.desc;

    if (desc.filter_type[axis] == ResamplingFilterType::Nearest) {
      // use specialized NN resampling pass - should be faster
      auto scale = desc.scale;
      for (int i = 0; i < spatial_ndim; i++) {
        if (i != axis)
          scale[i] = 1;
      }
      ResampleNN(out, in, desc.origin, scale);
    } else {
      int32_t *indices = static_cast<int32_t*>(mem);
      int out_size = desc.out_shape()[axis];
      float *coeffs = static_cast<float*>(static_cast<void*>(indices + out_size));
      int support = desc.filter[axis].support();

      InitializeResamplingFilter(indices, coeffs, out_size,
                                 desc.origin[axis], desc.scale[axis],
                                 desc.filter[axis]);

      ResampleAxis(out, in, indices, coeffs, support, axis);
    }
  }

  using ResamplingSetup = ResamplingSetupSingleImage<spatial_ndim>;
  ResamplingSetup setup;
  static constexpr int num_tmp_buffers = ResamplingSetup::num_tmp_buffers;
};

}  // namespace resampling
}  // namespace kernels
}  // namespace dali


#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_CPU_H_

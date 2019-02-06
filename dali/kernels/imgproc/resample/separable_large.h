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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_LARGE_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_LARGE_H_

#include "dali/kernels/imgproc/resample/separable.h"

namespace dali {
namespace kernels {

template <typename OutputElement, typename InputElement,
          typename IntermediateElement = float,
          typename Base_ = SeparableResamplingFilter<OutputElement, InputElement>>
struct LargeSeparableResamplingGPU : Base_ {
  using Base = Base_;
  using typename Base::Params;
  using typename Base::Input;
  using typename Base::Output;
  using Intermediate = OutListGPU<IntermediateElement, 3>;

  enum ProcessingOrder : int8_t {
    HorzVert,
    VertHorz
  };

  struct ProcessingSetup {
    std::vector<ProcessingOrder> order;
    TensorListShape<3> intermediate_shape, output_shape;
    size_t intermediate_size;
  } setup;

  struct SampleDesc {
    ptrdiff_t      in_offset, tmp_offset, out_offset;
    int            in_stride, tmp_stride, out_stride;
    TensorShape<2> in_shape,  tmp_shape,  out_shape;

    ProcessingOrder order;
    uint8_t channels;
    ResamplingParamsBase<2> params;
  };

  void MakeSampleDescs(std::vector<SampleDesc> &descs, const Output &out, const Intermediate &tmp, const Input &in, const Params &params) {
    int N = setup.order.size();
    for (int i = 0; i < N; i++) {
      SampleDesc &desc = descs[i];
      desc.in_offset = in.offsets[i];
      desc.tmp_offset = tmp.offsets[i];
      desc.out_offset = out.offsets[i];
      desc.in_shapes = in.shapes.tensor_shape(i).template first<2>();
      desc.tmp_shapes = tmp.shapes.tensor_shape(i).template first<2>();
      desc.out_shapes = out.shapes.tensor_shape(i).template first<2>();
      desc.in_stride = desc.in_shape[1];
      desc.tmp_stride = desc.tmp.shape[1];
      desc.out_stride = desc.out_shape[1];
      desc.channels = in.tensor_shape_span(i)[2];
      desc.params = static_cast<const ResamplingParamsBase<2>&>(params[i]);
    }
  }

  void SetupComputation(const Input &in, const Params &params) {
    int N = in.num_samples();
    assert(params.size() == N);

    setup.descs.resize(N);
    setup.order.resize(N);
    setup.intermediate_shape.resize(N);
    setup.output_shape.resize(N);
    setup.intermediate_size = 0;

    ScratchpadEstimator se;
    se.add<SampleDesc>(AllocType::GPU, N);

    for (int i = 0; i < N; i++) {
      auto ts_in = in.shape.tensor_shape_span(i);
      int H = ts_in[0];
      int W = ts_in[1];
      int C = ts_in[2];
      int out_H = params[i].output_size[0];
      int out_W = params[i].output_size[1];

      if (out_H == KeepOriginalSize) out_H = H;
      if (out_W == KeepOriginalSize) out_W = W;

      int size_vert = W * out_H;
      int size_horz = H * out_W;
      int compute_vert = size_vert * (2 * params[i].radii[0] + 1);
      int compute_horz = size_horz * (2 * params[i].radii[1] + 1);

      // TODO: fine tune the size/compute weights?
      const float size_weight = 3;
      float cost_vert = size_weight*size_vert + compute_vert;
      float cost_horz = size_weight*size_horz + compute_horz;

      auto ts_tmp = setup.intermediate_shape.tensor_shape_span[i];
      if (cost_vert < cost_horz) {
        setup.order[i] = VertHorz;
        ts_tmp[0] = out_H;
        ts_tmp[1] = W;
      } else {
        setup.order[i] = HorzVert;
        ts_tmp[0] = H;
        ts_tmp[1] = out_W;
      }
      ts_tmp[2] = C;
      setup.intermediate_size += volume(ts_tmp) * sizeof(IntermediateElement);

      auto ts_out = setup.output_shape.tensor_shape_span[i];
      ts_out[0] = out_H;
      ts_out[1] = out_W;
      ts_out[2] = C;
    }

  }


  virtual KernelRequirements Setup(KernelContext &context, const Input &in, const Params &params) {
    KernelRequirements req;
    SetupComputation();
    ScratchpadEstimator se;
    se.add<SampleDesc>(AllocType::GPU, in.num_samples());
    se.add<IntermediateElement>(AllocType::GPU, setup.intermediate_size);
    req.scratch_sizes = se.sizes;
    req.output_shapes = setup.output_shape;
    return req;
  }

  virtual void Run(KernelContext &context, const Output &out, const Input &in, const Params &params) {
    //context.scratchpad->AllocTensorList<
  }
};

}  // namespace kerenls
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_LARGE_H_

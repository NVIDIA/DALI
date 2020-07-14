// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/imgproc/resample.h"
#include "dali/kernels/imgproc/resample_cpu.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/pipeline/data/views.h"

namespace dali {

using namespace kernels;  // NOLINT

inline ResamplingFilterType interp2resample(DALIInterpType interp) {
#define DALI_MAP_INTERP_TO_RESAMPLE(interp, resample) case DALI_INTERP_##interp:\
  return ResamplingFilterType::resample;

  switch (interp) {
    DALI_MAP_INTERP_TO_RESAMPLE(NN, Nearest);
    DALI_MAP_INTERP_TO_RESAMPLE(LINEAR, Linear);
    DALI_MAP_INTERP_TO_RESAMPLE(CUBIC, Cubic);
    DALI_MAP_INTERP_TO_RESAMPLE(LANCZOS3, Lanczos3);
    DALI_MAP_INTERP_TO_RESAMPLE(GAUSSIAN, Gaussian);
    DALI_MAP_INTERP_TO_RESAMPLE(TRIANGULAR, Triangular);
  default:
    DALI_FAIL("Unknown interpolation type");
  }
#undef DALI_MAP_INTERP_TO_RESAMPLE
}

DALI_SCHEMA(ResamplingFilterAttr)
  .DocStr(R"code(Resampling filter attribute placeholder)code")
  .AddOptionalArg("interp_type",
      R"code(Type of interpolation used. Use `min_filter` and `mag_filter` to specify
different filtering for downscaling and upscaling.)code",
      DALI_INTERP_LINEAR)
  .AddOptionalArg("mag_filter", "Filter used when scaling up",
      DALI_INTERP_LINEAR)
  .AddOptionalArg("min_filter", "Filter used when scaling down",
      DALI_INTERP_LINEAR)
  .AddOptionalArg("temp_buffer_hint",
      "Initial size, in bytes, of a temporary buffer for resampling.\n"
      "Ingored for CPU variant.\n",
      0)
  .AddOptionalArg("minibatch_size", "Maximum number of images processed in a single kernel call",
      32);

ResamplingFilterAttr::ResamplingFilterAttr(const OpSpec &spec) {
  DALIInterpType interp_min = DALIInterpType::DALI_INTERP_LINEAR;
  DALIInterpType interp_mag = DALIInterpType::DALI_INTERP_LINEAR;

  if (spec.HasArgument("min_filter"))
    interp_min = spec.GetArgument<DALIInterpType>("min_filter");
  else if (spec.HasArgument("interp_type"))
    interp_min = spec.GetArgument<DALIInterpType>("interp_type");

  if (spec.HasArgument("mag_filter"))
    interp_mag = spec.GetArgument<DALIInterpType>("mag_filter");
  else if (spec.HasArgument("interp_type"))
    interp_mag = spec.GetArgument<DALIInterpType>("interp_type");

  min_filter_ = { interp2resample(interp_min), 0 };
  mag_filter_ = { interp2resample(interp_mag), 0 };

  temp_buffer_hint_ = spec.GetArgument<int64_t>("temp_buffer_hint");
}

template <typename Backend>
ResizeBase<Backend>::ResizeBase(const OpSpec &spec) : ResamplingFilterAttr(spec) {}

template <typename Backend>
ResizeBase<Backend>::~ResizeBase() = default;

template <>
struct ResizeBase<GPUBackend>::Impl {
  virtual void RunResize(TensorList<GPUBackend> &output,
                         const TensorList<GPUBackend> &input,
                         cudaStream_t stream) = 0;

  virtual void Setup(const TensorListShape<> &in_shape,
                     const TensorLayout &layout,
                     span<const kernels::ResamplingParams> params) = 0;

  virtual ~Impl() = default;
};

template <int spatial_ndim, int out_ndim, int in_ndim>
void GetFrameShapesAndParams(TensorListShape<out_ndim> &frame_shape,
                             std::vector<ResamplingParamsND<out_ndim>> &frame_params,
                             const TensorListShape<in_ndim> &in_shape,
                             const span<ResamplingParams> &in_params,
                             int first_spatial_dim) {
  static_assert(out_ndim == spatial_ndim + 1);

  int N = in_shape.num_samples();
  int total_frames = 0;

  for (int i = 0; i < N; i++) {
    auto in_sample_shape = in_shape.tensor_shape_span(i);
    total_frames += volume(&in_sample_shape[0], &in_sample_shape[first_spatial_dim]);
  }

  frame_params.resize(total_frames);

  assert(in_shape.dim() == out_ndim + 1 + (channel_dim_idx >= 0));

  for (int i = 0, flat_frame_idx = 0 = 0; i < N; i++) {
    auto in_sample_shape = in_shape.tensor_shape_span(i);

    for (int d = 0; d < in_sample_shape.size(); d++) {

    }
    /*int seq_len = 1;  // just an image
    if (frame_dim_idx >= 0) {
      in_shape.tensor_shape_span(i)[frame_dim_idx];
      if (seq_len == 0)
        continue;  // skip empty sequences
    }

    ResamplingParamsND<out_ndim> &frame_par = frame_params[i];
    TensorShape<out_ndim> frame_shape;
    frame_shape.resize(frame_ndim);

    for (int d = 0, od = 0; d < in_shape.dim(); d++) {
      if (d != frame_dim_idx)
        frame_shape[od++] = in_shape.tensor_shape_span(i)[d];
    }
    if (channel_dim_idx < 0)
      frame_shape[od++] = 1;  // add degenerate channel dim

    for (int f = 0; f < seq_len; f++) {
      for (int d = 0, od = 0; d < in_shape.dim(); d++) {
        if (d != frame_dim_idx && d != channel_dim_idx)
          frame_par[od++] = in_params[flat_frame_idx * od + od];
      }
    }*/
  }
}



template <typename Out, typename In, int spatial_ndim>
struct ResizeOpImplGPU : ResizeBase<GPUBackend>::Impl {
  ResizeOpImplGPU(int minibatch_size) : minibatch_size_(minibatch_size) {}

  static_assert(spatial_ndim == 2, "NOT IMPLEMENTED. Only 2D resize is supported");

  using Kernel = SeparableResample<Out, In, spatial_ndim>;

  /// Dimensnionality of each separate frame. If input contains no channel dimension, one is added
  constexpr int frame_ndim = spatial_ndim + 1;

  void Setup(const TensorListShape<> &in_shape,
             int first_spatial_dim,
             span<const kernels::ResamplingParams> params) override {
    GetFrameShapesAndParams<spatial_ndim>(in_shape_, params_, in_shape, params,
                                          first_spatial_dim);

    SetNumFrames(in_shape_.num_samples());
  }

  void RunResize(TensorList<GPUBackend> &output,
                 const TensorList<GPUBackend> &input,
                 cudaStream_t stream) override {

  }

  void SetNumFrames(int n) {
    int num_minibatches = CalculateMinibatchPartition(n, minibatch_size_);
    if (kmgr_.GetNumInstances() < num_minibatches)
      kmgr_.Resize<Kernel>(num_minibatches);
  }

  int CalculateMinibatchPartition(int total_frames, int minibatch_size) {
    int num_minibatches = div_ceil(total_frames, minibatch_size);

    minibatches_.resize(num_minibatches);
    int start = 0;
    for (int i = 0; i < num_minibatches; i++) {
      int end = (i + 1) * total_frames / num_minibatches
      minibatches_[i].start = start;
      minibatches_[i].count = end - start;
    }
    return num_minibatches;
  }

  TensorListShape<ndim> in_shape_;
  std::vector<ResamplingParamsND<ndim>> params_;

  kernels::KernelManager kmgr_;

  struct MiniBatch {
    int start, count;
    TensorListShape<> out_shape;
    kernels::InListGPU<In, ndim> input;
    kernels::OutListGPU<Out, ndim> output;
  };

  std::vector<MiniBatch> minibatches_;

  void SubdivideInput(const kernels::InListGPU<In, 3> &in) {
    mb.input = sample_range(mb.input, in, mb.start, mb.start + mb.count);
  }

  void SubdivideOutput(const kernels::OutListGPU<Out, 3> &out) {
    mb.output = sample_range(mb.output, out, mb.start, mb.start + mb.count);
  }

  int minibatch_size_;
};


template <typename Backend>
void ResizeBase::SetupResize(const Workspace &ws,
                             TensorListShape<> &out_shape,
                             const input_type &input,
                             span<const kernels::ResamplingParams> params,
                             DALIDataType out_type,
                             int spatial_ndim,
                             int first_spatial_dim) {
  DALIDataType in_type = input.type().id();
  if (out_type == in_type) {
    TYPE_SWITCH(out_type, typ2id, OutputType, (uint8_t, int16_t, uint16_t, float),
      SetupResizeTyped<OutputType, OutputType>(ws, out_shape, input, params, out_type,
                                               spatial_ndim, first_spatial_dim);
  }
}


template <>
void ResizeBase<GPUBackend>::RunResize(DeviceWorkspace &ws) {
  using Kernel = kernels::ResampleGPU<uint8_t, uint8_t>;

  auto in_view = view<const uint8_t, 3>(input);
  SubdivideInput(in_view);

  out_shape_ = TensorListShape<>();
  out_shape_.resize(in_view.num_samples(), in_view.sample_dim());
  int sample_idx = 0;

  kernels::KernelContext context;
  context.gpu.stream = stream;
  for (size_t b = 0; b < minibatches_.size(); b++) {
    MiniBatch &mb = minibatches_[b];

    auto &req = kmgr_.Setup<Kernel>(b, context,
        mb.input, make_span(resample_params_.data() + mb.start, mb.count));

    mb.out_shape = req.output_shapes[0];
    // minbatches have uniform dim
    for (int i = 0; i < mb.out_shape.size(); i++) {
      out_shape_.set_tensor_shape(i + sample_idx, mb.out_shape[i]);
    }
    sample_idx += mb.out_shape.size();
  }

  output.Resize(out_shape_);

  auto out_view = view<uint8_t, 3>(output);
  SubdivideOutput(out_view);

  for (size_t b = 0; b < minibatches_.size(); b++) {
    MiniBatch &mb = minibatches_[b];

    kmgr_.Run<Kernel>(0, b, context,
        mb.output, mb.input, make_span(resample_params_.data() + mb.start, mb.count));
  }
}

template <>
void ResizeBase<CPUBackend>::InitializeCPU(int num_threads) {
  if (num_threads != num_threads_) {
    impl_->reset();
    num_threads_ = num_threads;
  }
}

template <>
void ResizeBase<GPUBackend>::InitializeGPU(int minibatch_size) {
  if (minibatch_size != minibatch_size_) {
    impl_->reset();
    minibatch_size_ = minibatch_size;
  }

  kmgr_.SetMemoryHint(kernels::AllocType::GPU, temp_buffer_hint_);
  kmgr_.ReserveMaxScratchpad(0);
}

template <>
void ResizeBase<CPUBackend>::RunResize(HostWorkspace &ws) {
  /*using Kernel = kernels::ResampleCPU<uint8_t, uint8_t>;
  auto in_view = view<const uint8_t, 3>(input);
  kernels::KernelContext context;
  auto &req = kmgr_.Setup<Kernel>(
      thread_idx, context,
      in_view, resample_params_[thread_idx]);

  const auto &input_shape = input.shape();

  auto out_shape = req.output_shapes[0][0];
  out_shape_.set_tensor_shape(thread_idx, out_shape.shape);

  // Resize the output & run
  output.Resize(out_shape_[thread_idx]);
  auto out_view = view<uint8_t, 3>(output);
  kmgr_.Run<Kernel>(
      thread_idx, thread_idx, context,
      out_view, in_view, resample_params_[thread_idx]);*/
}


template class DLL_PUBLIC ResizeBase<CPUBackend>;
template class DLL_PUBLIC ResizeBase<GPUBackend>;

}  // namespace dali

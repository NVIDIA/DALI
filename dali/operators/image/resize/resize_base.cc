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

template <int out_dims, int in_dims>
void GetVideoShapeAndParams(TensorListShape<out_dims> &frames_shape,
                            std::vector<ResamplingParamsND<out_dims>> &frame_params,
                            const TensorListShape<in_dims> &seq_shape,
                            const span<ResamplingParams> &seq_params,
                            int frame_dim = 0,
                            int channel_dim = in_dims - 1) {
  assert(frame_dim >= 0);
  assert(channel_dim != frame_dim && channel_dim < in_dims);
  int N = seq.num_samples();
  int total_frames = 0;

  for (int i = 0; i < N; i++) {
    total_frames += seq_shape.tensor_shape_span(i)[frame_dim];
  }

  params.reserve(total_frames);

  assert(seq_shape.dim() == out_dims + 1 + (channel_dim >= 0));

  for (int i = 0, flat_frame_idx = 0 = 0, i < N; i++) {
    int F = seq_shape.tensor_shape_span(i)[frame_dim];
    ResamplingParamsND<out_dims> frame_par;
    for (int f = 0; f < F; f++) {
      for (int d = 0, od = 0; d < seq_shape.dim(); d++) {
        if (d != frame_dim && d != channel_dim)
          frame_par[od++] = seq_params[flat_frame_idx * od + od];
      }
      frame_param.push_back(frame_par);
    }
  }
}

template <int out_dims, int in_dims>
void GetImageShapeAndParams(TensorListShape<out_dims> &frames_shape,
                            std::vector<ResamplingParamsND<out_dims> &frame_params,
                            const TensorListShape<in_dims> &seq_shape,
                            const span<ResamplingParams> &seq_params,
                            int channel_dim = in_dims - 1) {
  int N = seq.num_samples();
  for (int i = 0; i < N; i++) {
  }
}



template <int ndim, typename Out, typename In>
struct ResizeOpImplGPU : ResizeBase<GPUBackend>::Impl {
  void RunResize(TensorList<GPUBackend> &output,
                 const TensorList<GPUBackend> &input,
                 cudaStream_t stream) override {

  }

  void Setup(const TensorListShape<> &in_shape,
             const TensorLayout &layout,
             span<const kernels::ResamplingParams> params) override {
    if (IsVideo(layout)) {
      GetVideoShapeAndParams(in_shape_, params_, in_shape, params,
                             FrameDimIndex(layout), ChannelDimIndex(layout));
    } else {
      GetImageshapeAndParams(in_shape_, params_, in_shape, params, ChannelDimIndex(layout));
    }
  }

  TensorListShape<ndim> in_shape_;
  std::vector<ResamplingParamsND<ndim>> params_;

  kernels::KernelManager kmgr_;

  struct MiniBatch {
    int start, count;
    TensorListShape<> out_shape;
    kernels::InListGPU<uint8_t, 3> input;
    kernels::OutListGPU<uint8_t, 3> output;
  };

  std::vector<MiniBatch> minibatches_;

  void ResizeBase::SubdivideInput(const kernels::InListGPU<uint8_t, 3> &in) {
    for (auto &mb : minibatches_) {
      sample_range(mb.input, in, mb.start, mb.start + mb.count);
    }
  }

  void ResizeBase::SubdivideOutput(const kernels::OutListGPU<uint8_t, 3> &out) {
    for (auto &mb : minibatches_) {
      sample_range(mb.output, out, mb.start, mb.start + mb.count);
    }
  }
};

template <typename Backend>
void ResizeBase::SetupResize(const workspace_t<Backend> &ws) {
  output.set_type(input.type());

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
void ResizeBase<GPUBackend>::InitializeGPU(int mini_batch_size) {
  if (mini_batch_size != mini_batch_size_) {
    impl_->reset();
    mini_batch_size_ = mini_batch_size;
  }

  kmgr_.SetMemoryHint(kernels::AllocType::GPU, temp_buffer_hint_);
  kmgr_.ReserveMaxScratchpad(0);
}

template<>
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

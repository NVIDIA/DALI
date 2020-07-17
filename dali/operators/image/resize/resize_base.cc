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

  virtual void Setup(TensorListShape<> &out_shape,
                     const TensorListShape<> &in_shape,
                     const TensorLayout &layout,
                     span<const kernels::ResamplingParams> params) = 0;

  virtual ~Impl() = default;
};

template <int spatial_ndim, int out_ndim, int in_ndim>
void GetFrameShapesAndParams(
      TensorListShape<out_ndim> &frame_shapes,
      std::vector<ResamplingParamsND<out_ndim>> &frame_params,
      const TensorListShape<in_ndim> &in_shape,
      const span<ResamplingParams> &in_params,
      int first_spatial_dim) {
  assert(first_spatial_dim + spatial_ndim <= in_ndim.sample_dim());
  const int frame_ndim = spatial_ndim + 1;
  static_assert(out_ndim == frame_ndim || out_ndim < 0, "Invalid frame tensor rank.");

  int N = in_shape.num_samples();
  int total_frames = 0;

  for (int i = 0; i < N; i++) {
    auto in_sample_shape = in_shape.tensor_shape_span(i);
    total_frames += volume(&in_sample_shape[0], &in_sample_shape[first_spatial_dim]);
  }

  frame_params.resize(total_frames);
  frame_shapes.resize(total_frames, frame_ndim);

  assert(in_shape.dim() == out_ndim + 1 + (channel_dim_idx >= 0));

  int ndim = in_shape.sample_dim();
  for (int i = 0, flat_frame_idx = 0 = 0; i < N; i++) {
    auto in_sample_shape = in_shape.tensor_shape_span(i);
    // Collapse leading dimensions, if any, as frame dim. This handles channel-first.
    int seq_len = volume(&in_sample_shape[0], &in_sample_shape[first_spatial_dim]);
    if (seq_len == 0)
      continue;  // skip empty sequences

    ResamplingParamsND<out_ndim> &frame_par = frame_params[i];
    TensorShape<out_ndim> frame_shape;
    frame_shape.resize(frame_ndim);

    for (int d = first_spatial_dim, od = 0; od < spatial_ndim; d++, od++) {
      frame_shape[od] = in_sample_shape[d];
    }
    // Collapse trailing dimensions, if any, as channel dim.
    int num_channels = volume(&in_sample_shape[first_spatial_dim + spatial_ndim],
                              &in_sample_shape[ndim]);
    frame_shape[frame_ndim] = num_channels;

    // Replicate parameters and frame shape.
    for (int f = 0; f < seq_len; f++, flat_frame_idx++) {
      for (int d = 0; d < spatial_ndim; d++)
        frame_par[flat_frame_idx * spatial_ndim + d] = in_params[i * spatial_ndim + d];

      frame_shapes.set_tensor_shape(flat_frame_idx, frame_shape);
    }
  }
}

template <int out_ndim, int in_ndim>
void GetResizedShape(
      TensorListShape<out_ndim> &out_shape, const TensorListShape<in_ndim> &in_shape,
      span<const ResamplingParams> params, int spatial_ndim, int first_spatial_dim) {
  assert(params.size() == spatial_ndim * in_shape.num_samples());
  assert(first_spatial_dim >= 0 && first_spatial_dim + spatial_ndim <= in_shape.sample_dim());

  out_shape = in_shape;
  for (int i = 0; i < N; i++) {
    auto out_sample_shape = out_shape.tensor_shape_span(i);
    for (int d = 0; d < spatial_ndim; d++) {
      auto out_extent = params[i * spatial_ndim + d].output_size;
      if (out_extent != dali::kernels::KeepOriginalSize)
        out_sample_shape[d + first_spatial_dim] = out_extent;
    }
  }
}

template <int out_ndim, int in_ndim, int spatial_ndim>
void GetResizedShape(
      TensorListShape<out_ndim> &out_shape, const TensorListShape<in_ndim> &in_shape,
      span<const ResamplingParamsND<spatial_ndim>> params, int first_spatial_dim) {
  GetResizedShape(out_shape, flatten(params));
}

template <typename Out, typename In, int spatial_ndim>
struct ResizeOpImplGPU : ResizeBase<GPUBackend>::Impl {
  ResizeOpImplGPU(int minibatch_size) : minibatch_size_(minibatch_size) {}

  static_assert(spatial_ndim == 2, "NOT IMPLEMENTED. Only 2D resize is supported");

  using Kernel = ResampleGPU<Out, In>;

  /// Dimensnionality of each separate frame. If input contains no channel dimension, one is added
  constexpr int frame_ndim = spatial_ndim + 1;

  void Setup(TensorListShape<> &out_shape,
             const TensorListShape<> &in_shape,
             int first_spatial_dim,
             span<const kernels::ResamplingParams> params) override {
    GetResizedShape(out_shape, in_shape, spatial_ndim, first_spatial_dim, params);

    // Create "frames" from outer dimensions and "channels" from inner dimensions.
    GetFrameShapesAndParams<spatial_ndim>(in_shape_, params_, in_shape, params,
                                          first_spatial_dim);

    // Now that we have per-frame parameters, we can calculate the output frame shape.
    GetResizedShape(out_shape_, in_shape_, params_);

    // Now that we know how many logical frames there are, calculate batch subdivision.
    SetNumFrames(in_shape_.num_samples());

    SetupKernel();
  }

  void SetupKernel() {
    const int dim = in_shape_.sample_dim();
    KernelContext ctx;
    for (int mb_idx = 0, num_mb = minbatches_.size(); mb_idx < num_mb; mb_idx++) {
      auto &mb = minibatches_[mb_idx];
      auto &in_slice = mb.input;
      in_slice.shape.resize(mb.end - mb.begin, dim);
      for (int i = mb.begin, j = 0; i < mb.end; i++, j++) {
        for (int d = 0; d < dim; d++)
          in_slice.tensor_shape_span(j)[d] = in_shape_.tensor_shape_span(i)[d];
      }

      kmgr_.Setup<Kernel>(mb_idx, ctx, mb.input, make_span(&params[mb.begin], &prams[mb.end]));
    }
  }

  void RunResize(TensorList<GPUBackend> &output,
                 const TensorList<GPUBackend> &input,
                 cudaStream_t stream) override {

    auto in_view = view<const InputType>(input);
    auto in_frames_view = reshape(in_view, in_shape_, true);
    SubdivideInput(in_frames_view);

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
      auto &mb = minibatches_[i];
      mb.start = start;
      mb.count = end - start;
    }
    return num_minibatches;
  }

  TensorListShape<ndim> in_shape_, out_shape_;
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
      (SetupResizeTyped<OutputType, OutputType>(ws, out_shape, input.shape, params,
        spatial_ndim, first_spatial_dim)),
      (DALI_FAIL(make_string("Unsupported type: ", out_type,
        ". Supported types are: uint8, int16, uint16 and float"))));
  } else {
    DALI_ENFORCE(out_type == DALI_FLOAT,
      make_string("Resize must output original type or float. Got: ", out_type));
    TYPE_SWITCH(in_type, typ2id, InputType, (uint8_t, int16_t, uint16_t),
      (SetupResizeTyped<OutputType, InputType>(ws, out_shape, input.shape, params,
        spatial_ndim, first_spatial_dim)),
      (DALI_FAIL(make_string("Unsupported type: ", in_type,
        ". Supported types are: uint8, int16, uint16 and float"))));
  }
}


template <typename Backend>
template <typename OutputType, typename InputType>
void ResizeBase<Backend>::SetupResizeTyped<OutputType, InputType>(
      const Workspace &ws,
      TensorListShape<> &out_shape,
      const TensorListShape<> &in_shape,
      span<const kernels::ResamplingParams> params,
      int spatial_ndim,
      int first_spatial_dim) {
  VALUE_SWITCH(spatial_ndim, static_spatial_ndim, (2),
  (SetupResizeStatic<static_spatial_ndim, OutputType, OutputType>(
      ws, out_shape, input.shape(), params, static_spatial_ndim, first_spatial_dim)),
  (DALI_FAIL(make_string("Unsupported number of spatial dimensions: ", spatial_ndim))));
}

template <typename Backend>
template <typename OutputType, typename InputType, int spatial_ndim>
void ResizeBase<Backend>::SetupResizeTyped<OutputType, InputType, spatial_ndim>(
      const Workspace &ws,
      TensorListShape<> &out_shape,
      const TensorListShape<> &in_shape,
      span<const kernels::ResamplingParams> params,
      int first_spatial_dim) {
  using ImplType = ResizeOpImplGPU<OutputType, InputType, spatial_ndim>;
  auto *impl = dynamic_cast<ImplType*>(impl_.get());
  if (!impl) {
    impl_.reset();
    impl_ = make_unique<ImplType>();
    impl = static_cast<ImplType*>(impl_.get());
  }
  impl->Setup
}


template <>
void ResizeBase<GPUBackend>::RunResize(DeviceWorkspace &ws) {
  impl_->RunResize(ws);
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

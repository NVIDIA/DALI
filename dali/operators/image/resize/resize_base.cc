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
#include "dali/core/static_switch.h"

namespace dali {

using namespace kernels;  // NOLINT

template <typename Backend>
struct ResizeBase<Backend>::Impl {
  using input_type =  typename Workspace::template input_t<Backend>::element_type;
  using output_type = typename Workspace::template output_t<Backend>::element_type;

  virtual void RunResize(output_type &output,
                         const input_type &input,
                         cudaStream_t stream) = 0;

  virtual void Setup(TensorListShape<> &out_shape,
                     const TensorListShape<> &in_shape,
                     int first_spatial_dim,
                     span<const kernels::ResamplingParams> paramss) = 0;

  virtual ~Impl() = default;
};


template <typename Backend>
ResizeBase<Backend>::ResizeBase(const OpSpec &spec) : ResamplingFilterAttr(spec) {}

template <typename Backend>
ResizeBase<Backend>::~ResizeBase() = default;

template <int spatial_ndim, int out_ndim, int in_ndim>
void GetFrameShapesAndParams(
      TensorListShape<out_ndim> &frame_shapes,
      std::vector<ResamplingParamsND<spatial_ndim>> &frame_params,
      const TensorListShape<in_ndim> &in_shape,
      const span<const ResamplingParams> &in_params,
      int first_spatial_dim) {
  assert(first_spatial_dim + spatial_ndim <= in_shape.sample_dim());
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

  int ndim = in_shape.sample_dim();
  for (int i = 0, flat_frame_idx = 0; i < N; i++) {
    auto in_sample_shape = in_shape.tensor_shape_span(i);
    // Collapse leading dimensions, if any, as frame dim. This handles channel-first.
    int seq_len = volume(&in_sample_shape[0], &in_sample_shape[first_spatial_dim]);
    if (seq_len == 0)
      continue;  // skip empty sequences

    ResamplingParamsND<spatial_ndim> &frame_par = frame_params[i];
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
  int N = out_shape.num_samples();
  for (int i = 0; i < N; i++) {
    auto out_sample_shape = out_shape.tensor_shape_span(i);
    for (int d = 0; d < spatial_ndim; d++) {
      auto out_extent = params[i * spatial_ndim + d].output_size;
      if (out_extent != dali::kernels::KeepOriginalSize)
        out_sample_shape[d + first_spatial_dim] = out_extent;
    }
  }
}

template <size_t spatial_ndim, int out_ndim, int in_ndim>
void GetResizedShape(
      TensorListShape<out_ndim> &out_shape, const TensorListShape<in_ndim> &in_shape,
      span<const ResamplingParamsND<spatial_ndim>> params, int first_spatial_dim) {
  GetResizedShape(out_shape, in_shape, flatten(params), spatial_ndim, first_spatial_dim);
}

template <typename Out, typename In, int spatial_ndim>
class ResizeOpImplGPU : public ResizeBase<GPUBackend>::Impl {
 public:
  ResizeOpImplGPU(int minibatch_size) : minibatch_size_(minibatch_size) {}

  static_assert(spatial_ndim == 2, "NOT IMPLEMENTED. Only 2D resize is supported");

  using Kernel = ResampleGPU<Out, In>;

  /// Dimensnionality of each separate frame. If input contains no channel dimension, one is added
  static constexpr int frame_ndim = spatial_ndim + 1;

  void Setup(TensorListShape<> &out_shape,
             const TensorListShape<> &in_shape,
             int first_spatial_dim,
             span<const kernels::ResamplingParams> params) override {
    GetResizedShape(out_shape, in_shape, params, spatial_ndim, first_spatial_dim);

    // Create "frames" from outer dimensions and "channels" from inner dimensions.
    GetFrameShapesAndParams<spatial_ndim>(in_shape_, params_, in_shape, params,
                                          first_spatial_dim);

    // Now that we have per-frame parameters, we can calculate the output frame shape.
    GetResizedShape(out_shape_, in_shape_, make_cspan(params_), first_spatial_dim);

    // Now that we know how many logical frames there are, calculate batch subdivision.
    SetNumFrames(in_shape_.num_samples());

    SetupKernel();
  }

  void SetupKernel() {
    const int dim = in_shape_.sample_dim();
    KernelContext ctx;
    for (int mb_idx = 0, num_mb = minibatches_.size(); mb_idx < num_mb; mb_idx++) {
      auto &mb = minibatches_[mb_idx];
      auto &in_slice = mb.input;
      in_slice.shape.resize(mb.count, dim);
      int end = mb.start + mb.count;
      for (int i = mb.start, j = 0; i < end; i++, j++) {
        for (int d = 0; d < dim; d++)
          in_slice.tensor_shape_span(j)[d] = in_shape_.tensor_shape_span(i)[d];
      }

      kmgr_.Setup<Kernel>(mb_idx, ctx, mb.input, make_span(&params_[mb.start], mb.count));
    }
  }

  void RunResize(TensorList<GPUBackend> &output,
                 const TensorList<GPUBackend> &input,
                 cudaStream_t stream) override {
    auto in_view = view<const In>(input);
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
          mb.input, make_span(params_.data() + mb.start, mb.count));

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
          mb.output, mb.input, make_span(params_.data() + mb.start, mb.count));
    }
  }

  void SetNumFrames(int n) {
    int num_minibatches = CalculateMinibatchPartition(n, minibatch_size_);
    if (static_cast<int>(kmgr_.NumInstances()) < num_minibatches)
      kmgr_.Resize<Kernel>(1, num_minibatches);
  }

  int CalculateMinibatchPartition(int total_frames, int minibatch_size) {
    int num_minibatches = div_ceil(total_frames, minibatch_size);

    minibatches_.resize(num_minibatches);
    int start = 0;
    for (int i = 0; i < num_minibatches; i++) {
      int end = (i + 1) * total_frames / num_minibatches;
      auto &mb = minibatches_[i];
      mb.start = start;
      mb.count = end - start;
    }
    return num_minibatches;
  }

  TensorListShape<frame_ndim> in_shape_, out_shape_;
  std::vector<ResamplingParamsND<spatial_ndim>> params_;

  kernels::KernelManager kmgr_;

  struct MiniBatch {
    int start, count;
    TensorListShape<> out_shape;
    kernels::InListGPU<In, frame_ndim> input;
    kernels::OutListGPU<Out, frame_ndim> output;
  };

  std::vector<MiniBatch> minibatches_;

  void SubdivideInput(const kernels::InListGPU<In, frame_ndim> &in) {
    for (auto &mb : minibatches_)
      mb.input = sample_range(mb.input, in, mb.start, mb.start + mb.count);
  }

  void SubdivideOutput(const kernels::OutListGPU<Out, frame_ndim> &out) {
    for (auto &mb : minibatches_)
      mb.output = sample_range(mb.output, out, mb.start, mb.start + mb.count);
  }

  int minibatch_size_;
};

template <typename Backend>
void ResizeBase<Backend>::SetupResize(TensorListShape<> &out_shape,
                                      DALIDataType out_type,
                                      const TensorListShape<> &in_shape,
                                      DALIDataType in_type,
                                      span<const kernels::ResamplingParams> params,
                                      int spatial_ndim,
                                      int first_spatial_dim) {
  if (out_type == in_type) {
    TYPE_SWITCH(out_type, type2id, OutputType, (uint8_t, int16_t, uint16_t, float),
      (this->template SetupResizeTyped<OutputType, OutputType>(out_shape, in_shape, params,
        spatial_ndim, first_spatial_dim)),
      (DALI_FAIL(make_string("Unsupported type: ", out_type,
        ". Supported types are: uint8, int16, uint16 and float"))));
  } else {
    DALI_ENFORCE(out_type == DALI_FLOAT,
      make_string("Resize must output original type or float. Got: ", out_type));
    TYPE_SWITCH(in_type, type2id, InputType, (uint8_t, int16_t, uint16_t),
      (this->template SetupResizeTyped<float, InputType>(out_shape, in_shape, params,
        spatial_ndim, first_spatial_dim)),
      (DALI_FAIL(make_string("Unsupported type: ", in_type,
        ". Supported types are: uint8, int16, uint16 and float"))));
  }
}


template <typename Backend>
template <typename OutputType, typename InputType>
void ResizeBase<Backend>::SetupResizeTyped(
      TensorListShape<> &out_shape,
      const TensorListShape<> &in_shape,
      span<const kernels::ResamplingParams> params,
      int spatial_ndim,
      int first_spatial_dim) {
  VALUE_SWITCH(spatial_ndim, static_spatial_ndim, (2),
  (SetupResizeStatic<OutputType, OutputType, static_spatial_ndim>(
      out_shape, in_shape, params, first_spatial_dim)),
  (DALI_FAIL(make_string("Unsupported number of spatial dimensions: ", spatial_ndim))));
}

template <>
template <typename OutputType, typename InputType, int spatial_ndim>
void ResizeBase<GPUBackend>::SetupResizeStatic(
      TensorListShape<> &out_shape,
      const TensorListShape<> &in_shape,
      span<const kernels::ResamplingParams> params,
      int first_spatial_dim) {
  using ImplType = ResizeOpImplGPU<OutputType, InputType, spatial_ndim>;
  auto *impl = dynamic_cast<ImplType*>(impl_.get());
  if (!impl) {
    impl_.reset();
    auto unq_impl = std::make_unique<ImplType>(minibatch_size_);
    impl = unq_impl.get();
    impl_ = std::move(unq_impl);
  }
  impl->Setup(out_shape, in_shape, first_spatial_dim, params);
}


template <>
void ResizeBase<CPUBackend>::InitializeCPU(int num_threads) {
  if (num_threads != num_threads_) {
    impl_.reset();
    num_threads_ = num_threads;
  }
}

template <>
void ResizeBase<GPUBackend>::InitializeGPU(int minibatch_size) {
  if (minibatch_size != minibatch_size_) {
    impl_.reset();
    minibatch_size_ = minibatch_size;
  }
}

template <typename Backend>
void ResizeBase<Backend>::RunResize(Workspace &ws, output_type &output, const input_type &input) {
  impl_->RunResize(output, input, ws.stream());
}


template class ResizeBase<CPUBackend>;
template class ResizeBase<GPUBackend>;

}  // namespace dali

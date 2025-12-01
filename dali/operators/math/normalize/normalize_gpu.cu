// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/math/normalize/normalize.h"
#include "dali/core/math_util.h"
#include "dali/core/tensor_layout.h"
#include "dali/kernels/normalize/normalize_gpu.h"
#include "dali/kernels/reduce/reduce_gpu.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/dynamic_scratchpad.h"

namespace dali {

using namespace kernels; // NOLINT

template <>
class Normalize<GPUBackend> : public NormalizeBase<GPUBackend> {
 public:
  explicit Normalize(const OpSpec &spec) : NormalizeBase<GPUBackend>(spec) {}

 private:
  friend class NormalizeBase<GPUBackend>;

  template <typename OutputType, typename InputType>
  void SetupTyped(const Workspace &ws);

  template <typename OutputType, typename InputType>
  void RunTyped(Workspace &ws);

  void AllocTempStorage();
  void FoldMeans();
  void FoldStdDev();

  template <typename ParamType, typename InputType>
  MeanGPU<ParamType, InputType> &GetMeanKernel() {
    return mean_kernel_.create_or_get<MeanGPU<ParamType, InputType>>();
  }

  template <typename ParamType, typename InputType>
  InvStdDevGPU<ParamType, InputType> &GetInvStdDevKernel() {
    return stddev_kernel_.create_or_get<InvStdDevGPU<ParamType, InputType>>();
  }

  template <typename OutputType, typename InputType>
  NormalizeGPU<OutputType, InputType> &GetNormalizeKernel() {
    return normalize_kernel_.create_or_get<NormalizeGPU<OutputType, InputType>>();
  }

  TensorListView<StorageGPU, float> BroadcastMean(KernelContext &ctx, float value) const;

  AnyKernelInstance mean_kernel_, stddev_kernel_, normalize_kernel_;
};


DALI_REGISTER_OPERATOR(Normalize, Normalize<GPUBackend>, GPU);

namespace {

template <typename ToUpdate, typename Other>
inline void MaxInPlace(ToUpdate &inout, const Other &other) {
  auto b1 = dali::begin(inout);
  auto b2 = dali::begin(other);
  auto e1 = dali::end(inout);
  auto e2 = dali::end(other);
  for (; b1 != e1 && b2 != e2; b1++, b2++) {
    if (*b1 < *b2) {
      *b1 = *b2;
    }
  }
}

template <int ndim>
int64_t MaxSampleSize(const TensorListShape<ndim> &tls) {
  int64_t max_sample_size = 0;
  for (int i = 0; i < tls.num_samples(); i++) {
    int64_t v = volume(tls.tensor_shape_span(i));
    if (v > max_sample_size)
      max_sample_size = v;
  }
  return max_sample_size;
}

template <typename T>
__global__ void Fill(T *data, size_t count, T value) {
  auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    data[i] = value;
}

}  // namespace

TensorListView<StorageGPU, float>
Normalize<GPUBackend>::BroadcastMean(KernelContext &ctx, float value) const {
  TensorListView<StorageGPU, float> mean_gpu;
  mean_gpu.shape = param_shape_;
  mean_gpu.data.resize(param_shape_.num_samples());
  // allocate enough memory to hold the largest sample...
  int64_t max_sample_size = MaxSampleSize(param_shape_);
  float *gpu_mean_data = ctx.scratchpad->AllocateGPU<float>(max_sample_size);
  int grid = div_ceil(max_sample_size, 1024);
  int block = std::min<int64_t>(max_sample_size, 1024);
  // ...fill it with given value...
  Fill<<<grid, block, 0, ctx.gpu.stream>>>(gpu_mean_data, max_sample_size, value);
  // ...and reuse the memory for all samples
  for (auto &ptr : mean_gpu.data)
    ptr = gpu_mean_data;
  return mean_gpu;
}

template <typename OutputType, typename InputType>
void Normalize<GPUBackend>::SetupTyped(const Workspace &ws) {
  auto &input = ws.Input<GPUBackend>(0);
  int nsamples = input.num_samples();

  KernelContext ctx;
  ctx.gpu.stream = ws.stream();

  int64_t param_volume = param_shape_.num_elements();

  // setup and get memory requirements from kernels

  auto &norm = GetNormalizeKernel<OutputType, InputType>();
  // if stddev is calculated internally, it's already inverse
  bool scale_is_stddev = !ShouldCalcStdDev();
  auto req = norm.Setup(ctx, data_shape_, make_span(axes_),
                        has_scalar_mean_, has_scalar_stddev_, scale_is_stddev);

  if (ShouldCalcMean()) {
    auto &mean = GetMeanKernel<float, InputType>();
    auto mean_req = mean.Setup(ctx, data_shape_, make_span(axes_), true, batch_norm_);
    assert(mean_req.output_shapes[0] == param_shape_);
  }

  if (ShouldCalcStdDev()) {
    auto &stddev = GetInvStdDevKernel<float, InputType>();
    auto stddev_req = stddev.Setup(ctx, data_shape_, make_span(axes_), true, batch_norm_);
    assert(stddev_req.output_shapes[0] == param_shape_);
  }
}

template <typename OutputType, typename InputType>
void Normalize<GPUBackend>::RunTyped(Workspace &ws) {
  auto &input = ws.Input<GPUBackend>(0);
  TensorListView<StorageGPU, const InputType> in_view = view<const InputType>(input);

  auto &output = ws.Output<GPUBackend>(0);
  TensorListView<StorageGPU, OutputType> out_view = view<OutputType>(output);
  output.SetLayout(input.GetLayout());

  int nsamples = input.num_samples();

  cudaStream_t stream = ws.stream();
  DynamicScratchpad dyn_scratchpad(stream);;
  KernelContext ctx;
  ctx.gpu.stream = stream;

  // Prepare mean and stddev

  float scalar_mean   = has_scalar_mean_   ? spec_.GetArgument<float>("mean")   : 0;
  float scalar_stddev = has_scalar_stddev_ ? spec_.GetArgument<float>("stddev") : 1;

  OutListGPU<float> mean_gpu, stddev_gpu;

  if (!has_scalar_mean_) {
    mean_gpu = dyn_scratchpad.AllocTensorList<mm::memory_kind::device, float>(param_shape_);
  } else if (ShouldCalcStdDev()) {
    ctx.scratchpad = &dyn_scratchpad;
    mean_gpu = BroadcastMean(ctx, scalar_mean);
    ctx.scratchpad = nullptr;
  }

  if (!has_scalar_stddev_) {
    stddev_gpu = dyn_scratchpad.AllocTensorList<mm::memory_kind::device, float>(param_shape_);
  }

  if (ShouldCalcMean()) {
    DynamicScratchpad scratchpad(stream);
    ctx.scratchpad = &scratchpad;
    auto &mean_kernel = GetMeanKernel<float, InputType>();
    mean_kernel.Run(ctx, mean_gpu, in_view);
    ctx.scratchpad = nullptr;
  } else if (has_tensor_mean_) {
    kernels::copy(mean_gpu, mean_input_, stream);
  }

  if (ShouldCalcStdDev()) {
    DynamicScratchpad scratchpad(stream);
    ctx.scratchpad = &scratchpad;
    auto &stddev_kernel = GetInvStdDevKernel<float, InputType>();
    stddev_kernel.Run(ctx, stddev_gpu, in_view, mean_gpu, degrees_of_freedom_, epsilon_);
    ctx.scratchpad = nullptr;
  } else if (has_tensor_stddev_) {
    kernels::copy(stddev_gpu, stddev_input_, stream);
  }

  // finally, run the normalize kernel
  {
    DynamicScratchpad scratchpad(stream);
    ctx.scratchpad = &scratchpad;
    auto &norm_kernel = GetNormalizeKernel<OutputType, InputType>();

    // if stddev is calculated internally, epsilon has already been included
    float epsilon = ShouldCalcStdDev() ? 0 : epsilon_;

    if (has_scalar_mean_) {
      if (has_scalar_stddev_) {
        norm_kernel.Run(ctx, out_view, in_view, scalar_mean, scalar_stddev,
                        scale_, shift_, epsilon);
      } else {
        norm_kernel.Run(ctx, out_view, in_view, scalar_mean, stddev_gpu,
                        scale_, shift_, epsilon);
      }
    } else {
      if (has_scalar_stddev_) {
        norm_kernel.Run(ctx, out_view, in_view, mean_gpu, scalar_stddev,
                        scale_, shift_, epsilon);
      } else {
        norm_kernel.Run(ctx, out_view, in_view, mean_gpu, stddev_gpu,
                        scale_, shift_, epsilon);
      }
    }
  }
}

}  // namespace dali

// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/operators/math/normalize/normalize_utils.h"

namespace dali {

template <>
class Normalize<GPUBackend> : public NormalizeBase<GPUBackend> {
 public:
  explicit Normalize(const OpSpec &spec) : NormalizeBase<GPUBackend>(spec) {}

 private:
  friend class NormalizeBase<GPUBackend>;

  template <typename OutputType, typename InputType>
  void SetupTyped(const DeviceWorkspace &ws);

  template <typename OutputType, typename InputType>
  void RunTyped(DeviceWorkspace &ws);

  void AllocTempStorage();
  void FoldMeans();
  void FoldStdDev();

  template <typename OutputType, typename InputType>
  kernels::MeanGPU<OutputType, InputType> &GetMeanKernel() {
    return mean_kernel_.create_or_get<kernels::MeanGPU<float, InputType>>();
  }

  template <typename OutputType, typename InputType>
  kernels::MeanGPU<OutputType, InputType> &GetStdDevKernel() {
    return mean_kernel_.create_or_get<kernels::StdDevGPU<float, InputType>>();
  }

  template <typename OutputType, typename InputType>
  kernels::MeanGPU<OutputType, InputType> &GetNormalizeKernel() {
    return normalize_kernel_.create_or_get<kernels::NormalizeGPU<OutputType, InputType>>();
  }

  kernels::AnyKernelInstance mean_kernel_, std_kernel_, normalize_kernel_;
  kernels::ScratchpadAllocator alloc_;
  TensorListView<StorageGPU, float> base_, scale_;
};


DALI_REGISTER_OPERATOR(Normalize, Normalize<GPUBackend>, GPU);

using namespace normalize;  // NOLINT
using namespace kernels;    // NOLINT

template <typename ToUpdate, typename Other>
inline void UpdateIfLess(ToUpdate &&update, const Other &other) {
  auto &b1 = begin(update);
  auto &b2 = begin(other);
  auto &e1 = end(update);
  auto &e2 = end(other);
  for (; b1 != e1 && b2 != e2; b1++, b2++) {
    if (*b1 < *b2) {
      *b1 = *b2;
    }
  }
}

template <typename OutputType, typename InputType>
void Normalize<GPUBackend>::SetupTyped(const DeviceWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  int nsamples = input.ntensor();

  kernels::KernelContext ctx;
  ctx.gpu.stream = ws.stream();

  kernels::ScratchpadEstimator se;

  int64_t param_volume = param_shape_.num_elements();

  if (!has_scalar_mean_) {
    se.add<float>(AllocType::GPU, param_volume);
  } else {
    if (ShouldCalcStdDev()) {
      // stddev kernel requires the mean to be in GPU memory, even if it's a scalar
      se.add<float>(AllocType::GPU, 1);
    }
  }

  if (!has_scalar_stddev_) {
    se.add<float>(AllocType::GPU, param_volume);
  }

  if (ShouldCalcMean()) {
    auto &mean = GetMeanKernel<OutputType, InputType>();
    auto mean_req = mean.Setup(ctx, data_shape_, make_span(axes_), true, batch_norm_);
    assert(mean_req.output_shapes[0] == param_shape_);
    UpdateIfLess(se.sizes, mean_req.scratch_sizes);
  }

  if (ShouldCalcStdDev()) {
    auto &stddev = GetStdDevKernel<OutputType, InputType>();
    auto stddev_req = stddev.Setup(ctx, data_shape_, make_span(axes_), true, batch_norm_);
    assert(stddev_req.output_shapes[0] == param_shape_);
    UpdateIfLess(se.sizes, mean_req.scratch_sizes);
  }

  auto &norm = GetNormalizeKernel<OutputType, InputType>();
  auto req = norm.Setup(ctx, data_shape_, make_span(axes_),
                        has_scalar_mean_, has_scalar_stddev_,
                        !ShouldCalcStdDev());

  UpdateIfLess(se.sizes, mean_req.scratch_sizes);

  alloc_.Reserve(se.sizes);

}

template <>
template <typename OutputType, typename InputType>
void Normalize<GPUBackend>::RunTyped(DeviceWorkspace &ws) {
  auto &input = ws.InputRef<GPUBackend>(0);
  TensorListView<StorageGPU, const InputType> in_view = view<const InputType>(input);

  auto &output = ws.OutputRef<GPUBackend>(0);
  TensorListView<StorageGPU, OutputType> out_view = view<OutputType>(output);

  int nsamples = input.ntensor();

  cudaStream_t stream = ws.stream();
  Scratchpad scratch = alloc_.GetScratchpad();
  KernelContext ctx;
  ctx.scratchpad = &scratch;
  ctx.gpu.stream = stream;

  // Prepare mean and stddev

  float scalar_mean = 0;
  float scalar_inv_stddev = 1;

  OutListGPU<float> mean_gpu, stddev_gpu

  if (has_scalar_mean_) {
    scalar_mean = spec_.GetArgument<float>("mean");
  } else {
    mean_gpu = scratch.AllocTensorList<AllocType::GPU, float>(param_shape_);
    if (ShouldCalcMean()) {
    } else {

    }
  }

  if (ShouldCalcStdDev()) {
    if (has_scalar_mean_) {
      mean_gpu = scratch.AllocTensorList<AllocType::GPU, float>({ TensorShape<>() });
      CUDA_CALL(cudaMemcpyAsync(mean_gpu.data[0], &scalar_mean, stream));
    }
  }

}

}  // namespace dali

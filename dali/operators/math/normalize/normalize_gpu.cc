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
#include "dali/kernels/common/copy.h"

namespace dali {

using namespace kernels; // NOLINT

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

  template <typename ParamType, typename InputType>
  MeanGPU<ParamType, InputType> &GetMeanKernel() {
    return mean_kernel_.create_or_get<MeanGPU<ParamType, InputType>>();
  }

  template <typename ParamType, typename InputType>
  InvStdDevGPU<ParamType, InputType> &GetInvStdDevKernel() {
    return mean_kernel_.create_or_get<InvStdDevGPU<ParamType, InputType>>();
  }

  template <typename OutputType, typename InputType>
  NormalizeGPU<OutputType, InputType> &GetNormalizeKernel() {
    return normalize_kernel_.create_or_get<NormalizeGPU<OutputType, InputType>>();
  }

  AnyKernelInstance mean_kernel_, std_kernel_, normalize_kernel_;
  ScratchpadAllocator alloc_;
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

using scratch_sizes_t = std::array<size_t, static_cast<size_t>(AllocType::Count)>;

scratch_sizes_t GetScratchpadSnapshot(const PreallocatedScratchpad &s) {
  scratch_sizes_t ss;
  for (size_t i = 0; i < ss.size(); i++)
    ss[i] = s.allocs[i].used();
  return ss;
}

void RestoreScratchpadSnapshot(PreallocatedScratchpad &s, const scratch_sizes_t &ss) {
  s.Clear();  // this doesn't clear the memory - just resets the usage counter to 0
  for (size_t i = 0; i < ss.size(); i++) {
    s.allocs[i].alloc(ss[i]);
  }
}

}  // namespace

template <typename OutputType, typename InputType>
void Normalize<GPUBackend>::SetupTyped(const DeviceWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  int nsamples = input.ntensor();

  KernelContext ctx;
  ctx.gpu.stream = ws.stream();

  ScratchpadEstimator se;

  int64_t param_volume = param_shape_.num_elements();

  // estimate memory requirements for intermediate buffers

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
    MaxInPlace(req.scratch_sizes, mean_req.scratch_sizes);
  }

  if (ShouldCalcStdDev()) {
    auto &stddev = GetInvStdDevKernel<float, InputType>();
    auto stddev_req = stddev.Setup(ctx, data_shape_, make_span(axes_), true, batch_norm_);
    assert(stddev_req.output_shapes[0] == param_shape_);
    MaxInPlace(req.scratch_sizes, stddev_req.scratch_sizes);
  }

  MaxInPlace(se.sizes, req.scratch_sizes);

  alloc_.Reserve(se.sizes);
}

template <typename OutputType, typename InputType>
void Normalize<GPUBackend>::RunTyped(DeviceWorkspace &ws) {
  auto &input = ws.InputRef<GPUBackend>(0);
  TensorListView<StorageGPU, const InputType> in_view = view<const InputType>(input);

  auto &output = ws.OutputRef<GPUBackend>(0);
  TensorListView<StorageGPU, OutputType> out_view = view<OutputType>(output);

  int nsamples = input.ntensor();

  cudaStream_t stream = ws.stream();
  PreallocatedScratchpad scratch = alloc_.GetScratchpad();
  KernelContext ctx;
  ctx.scratchpad = &scratch;
  ctx.gpu.stream = stream;

  // Prepare mean and stddev

  float scalar_mean = 0;
  float scalar_stddev = 1;

  OutListGPU<float> mean_gpu, stddev_gpu;

  if (!has_scalar_mean_) {
    mean_gpu = scratch.AllocTensorList<AllocType::GPU, float>(param_shape_);
  } else if (ShouldCalcStdDev()) {
    mean_gpu.shape.resize(param_shape_.num_samples(), 0);
    float *gpu_scalar_mean = scratch.Allocate<float>(AllocType::GPU, 1);
    for (auto &ptr : mean_gpu.data)
      ptr = gpu_scalar_mean;
    CUDA_CALL(cudaMemcpyAsync(gpu_scalar_mean, &scalar_mean, sizeof(scalar_mean),
                              cudaMemcpyHostToDevice, stream));
  }

  if (!has_scalar_stddev_) {
    stddev_gpu = scratch.AllocTensorList<AllocType::GPU, float>(param_shape_);
  }

  // We can't just Clear() the scratchpad to reuse it, because temporary buffers are also
  // stored there - so let's make a snapshot of current allocation state and restore it
  // before launching each kernel.
  auto scratch_snap = GetScratchpadSnapshot(scratch);

  if (has_scalar_mean_) {
    scalar_mean = spec_.GetArgument<float>("mean");
  } else if (ShouldCalcMean()) {
    auto &mean_kernel = GetMeanKernel<float, InputType>();

    RestoreScratchpadSnapshot(scratch, scratch_snap);
    mean_kernel.Run(ctx, mean_gpu, in_view);
  } else {
    kernels::copy(mean_gpu, mean_input_, stream);
  }

  if (has_scalar_stddev_) {
    scalar_stddev = spec_.GetArgument<float>("stddev");
  } else if (ShouldCalcStdDev()) {
    auto &stddev_kernel = GetInvStdDevKernel<float, InputType>();

    RestoreScratchpadSnapshot(scratch, scratch_snap);
    stddev_kernel.Run(ctx, stddev_gpu, in_view, mean_gpu, degrees_of_freedom_, epsilon_);
  } else {
    kernels::copy(stddev_gpu, stddev_input_, stream);
  }

  // finally, run the normalize kernel
  {
    auto &norm_kernel = GetNormalizeKernel<OutputType, InputType>();

    RestoreScratchpadSnapshot(scratch, scratch_snap);

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

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

#ifndef DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_IMPL_H_
#define DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_IMPL_H_

#include <cuda_runtime.h>
#include "dali/kernels/imgproc/resample/separable.h"
#include "dali/kernels/imgproc/resample/resampling_setup.h"
#include "dali/kernels/imgproc/resample/resampling_batch.h"
#include "dali/kernels/common/copy.h"

namespace dali {
namespace kernels {

/// @brief Implements a separable resampling filter
///
/// This implementation can apply differnt resampling filters to each sample.
/// Resampling order is chosen based on input/output shapes and filter type and support.
/// The filter allocates memory only in `Setup` - and even there, it won't reallocate
/// if subsequent calls do not exceed previous number of samples.
template <typename OutputElement, typename InputElement,
          typename Interface = SeparableResamplingFilter<OutputElement, InputElement>>
struct SeparableResamplingGPUImpl : Interface {
  using typename Interface::Params;
  using typename Interface::Input;
  using typename Interface::Output;
  using SampleDesc = SeparableResamplingSetup::SampleDesc;

  /// Generates and stores resampling setup
  BatchResamplingSetup setup;

  using IntermediateElement = float;
  using Intermediate = OutListGPU<IntermediateElement, 3>;

  /// The intermediate tensor list
  Intermediate intermediate;

  void Initialize(KernelContext &context) {
    setup.Initialize();
  }

  virtual KernelRequirements Setup(KernelContext &context, const Input &in, const Params &params) {
    Initialize(context);
    setup.SetupBatch(in.shape, params);
    // this will allocate and calculate offsets
    intermediate = { nullptr, setup.intermediate_shape };

    KernelRequirements req;
    ScratchpadEstimator se;

    // Sample descriptions need to be delivered to the GPU - hence, the storage
    se.add<SampleDesc>(AllocType::GPU, setup.sample_descs.size());

    // CPU block2sample lookup may change in size and is large enough
    // to mandate declaring it as a requirement for external allocator.
    size_t num_blocks = setup.total_blocks.pass[0] + setup.total_blocks.pass[1];
    se.add<SampleBlockInfo>(AllocType::GPU, num_blocks);
    se.add<SampleBlockInfo>(AllocType::Host, num_blocks);

    // Request memory for intermediate storage.
    se.add<IntermediateElement>(AllocType::GPU, setup.intermediate_size);

    req.scratch_sizes = se.sizes;
    req.output_shapes = { setup.output_shape };
    return req;
  }

  template <int which_pass, typename PassOutputElement, typename PassInputElement>
  void RunPass(
      const OutListGPU<PassOutputElement, 3> &out,
      const InListGPU<PassInputElement, 3> &in,
      const SampleDesc *descs_gpu,
      const InTensorGPU<SampleBlockInfo, 1> &block2sample,
      cudaStream_t stream) {
    BatchedSeparableResample<which_pass>(
        out.data, in.data, descs_gpu, in.num_samples(),
        block2sample.data, block2sample.shape[0],
        setup.block_size,
        stream);
  }

  /// @remarks This function shall not allocate memory by ano other means
  ///          than through `context.scratchpad`
  virtual void
  Run(KernelContext &context, const Output &out, const Input &in, const Params &params) {
    cudaStream_t stream = context.gpu.stream;

    SampleDesc *descs_gpu = context.scratchpad->Allocate<SampleDesc>(
        AllocType::GPU, setup.sample_descs.size());

    int total_blocks = setup.total_blocks.pass[0] + setup.total_blocks.pass[1];

    OutTensorCPU<SampleBlockInfo, 1> sample_lookup_cpu = {
      context.scratchpad->Allocate<SampleBlockInfo>(AllocType::Host, total_blocks),
      { total_blocks }
    };
    OutTensorGPU<SampleBlockInfo, 1> sample_lookup_gpu = {
      context.scratchpad->Allocate<SampleBlockInfo>(AllocType::GPU, total_blocks),
      { total_blocks }
    };
    setup.InitializeSampleLookup(sample_lookup_cpu);
    copy(sample_lookup_gpu, sample_lookup_cpu, stream);  // NOLINT (it thinks it's std::copy)

    cudaMemcpyAsync(
        descs_gpu,
        setup.sample_descs.data(),
        setup.sample_descs.size()*sizeof(SampleDesc),
        cudaMemcpyHostToDevice,
        stream);

    InTensorGPU<SampleBlockInfo, 1> first_pass_lookup = make_tensor_gpu<1>(
        sample_lookup_gpu.data,
        { setup.total_blocks.pass[0] });

    InTensorGPU<SampleBlockInfo, 1> second_pass_lookup = make_tensor_gpu<1>(
        sample_lookup_gpu.data + setup.total_blocks.pass[0],
        { setup.total_blocks.pass[1] });

    intermediate.data = context.scratchpad->Allocate<IntermediateElement>(
        AllocType::GPU, setup.intermediate_size);

    RunPass<0, IntermediateElement, InputElement>(
      intermediate, in, descs_gpu, first_pass_lookup, stream);
    RunPass<1, OutputElement, IntermediateElement>(
      out, intermediate, descs_gpu, second_pass_lookup, stream);
  }
};

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_RESAMPLE_SEPARABLE_IMPL_H_

// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <vector>

#include "dali/core/dynlink_nvcomp.h"
#include "dali/core/backend_tags.h"
#include "dali/core/common.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/operators/decoder/inflate/inflate.h"
#include "dali/operators/decoder/inflate/inflate_gpu.h"
#include "dali/operators/decoder/inflate/inflate_params.h"
#include "dali/pipeline/data/sequence_utils.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

namespace inflate {

class InflateOpGpuLZ4Impl : public InflateOpImplBase<GPUBackend> {
 public:
  explicit InflateOpGpuLZ4Impl(const OpSpec &spec) : InflateOpImplBase<GPUBackend>{spec} {}

  void RunImpl(Workspace &ws) override {
    const auto &input = ws.template Input<GPUBackend>(0);
    auto &output = ws.template Output<GPUBackend>(0);
    output.SetLayout(params_.GetOutputLayout());
    auto total_chunks_num = params_.GetTotalChunkNum();
    // nvCOMP does not like being called with no chunks in the input
    if (total_chunks_num == 0) {
      assert(params_.GetOutputShape().num_elements() == 0);
      return;
    }
    SetupInChunks(input);
    SetupOutChunks(output);
    auto stream = ws.stream();

    kernels::DynamicScratchpad scratchpad(stream);
    size_t *actual_out_sizes = scratchpad.AllocateGPU<size_t>(total_chunks_num);

    auto [in_sizes, in, out_sizes, out] = scratchpad.ToContiguousGPU(
        stream, params_.GetInChunkSizes(), input_ptrs_, inflated_sizes_, inflated_ptrs_);

    size_t tempSize;
    CUDA_CALL(nvcompBatchedLZ4DecompressGetTempSizeAsync(total_chunks_num,
                                                         params_.GetMaxOutChunkVol(),
                                                         {}, &tempSize, params_.GetMaxOutVol()));

    void *temp = scratchpad.AllocateGPU<uint8_t>(tempSize);
    nvcompStatus_t *device_statuses = scratchpad.AllocateGPU<nvcompStatus_t>(total_chunks_num);
    CUDA_CALL(nvcompBatchedLZ4DecompressAsync(in, in_sizes, out_sizes, actual_out_sizes,
                                               total_chunks_num, temp, tempSize, out, {},
                                               device_statuses, stream));

    // nvCOMP uses ``out_sizes`` to avoid OOB accesses if the actual data size after the compression
    // is greater than what follows from the shapes reported by the user.
    // Here, we take care of the opposite case, when the allocations are bigger than really
    // necessary and thus may contain some uninitialized memory.
    FillTheTails(output.type(), total_chunks_num, out, actual_out_sizes, out_sizes, stream);
  }

 protected:
  template <typename TL>
  void SetupInChunks(const TL &input) {
    auto in_view = view<const uint8_t>(input);
    auto batch_size = in_view.shape.num_samples();
    const auto &num_chunks_per_sample = params_.GetChunksNumPerSample();
    auto total_chunks_num = params_.GetTotalChunkNum();
    const auto &offsets = params_.GetInChunkOffsets();
    input_ptrs_.clear();
    input_ptrs_.reserve(total_chunks_num);
    for (int64_t sample_idx = 0, chunk_flat_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto num_chunks = num_chunks_per_sample[sample_idx].num_elements();
      const uint8_t *sample_base_ptr = in_view.data[sample_idx];
      for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++, chunk_flat_idx++) {
        input_ptrs_.push_back(sample_base_ptr + offsets[chunk_flat_idx]);
      }
    }
  }

  template <typename TL>
  void SetupOutChunks(TL &output) {
    TYPE_SWITCH(output.type(), type2id, Out, INFLATE_SUPPORTED_TYPES, (
      BOOL_SWITCH(params_.HasChunks(), HasChunks, (
        SetupOutChunksTyped<HasChunks, Out>(output);
      ));  //NOLINT
    ), DALI_FAIL(  //NOLINT
      make_string("Unsupported output type was specified for GPU inflate LZ4 operator: `",
                  output.type(), "`.")));
  }

  template <bool has_chunks, typename Out, typename TL>
  void SetupOutChunksTyped(TL &output) {
    auto out_view = view<Out>(output);
    auto batch_size = out_view.shape.num_samples();
    auto total_chunks_num = params_.GetTotalChunkNum();
    inflated_ptrs_.clear();
    inflated_ptrs_.reserve(total_chunks_num);
    inflated_sizes_.clear();
    inflated_sizes_.reserve(total_chunks_num);
    for (int64_t sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      const int dims_to_unfold = has_chunks ? 1 : 0;
      auto sample_range = sequence_utils::unfolded_view_range<dims_to_unfold>(out_view[sample_idx]);
      auto chunk_size = sample_range.SliceSize() * sizeof(Out);
      for (auto &&chunk : sample_range) {
        inflated_sizes_.push_back(chunk_size);
        inflated_ptrs_.push_back(static_cast<void *>(chunk.data));
      }
    }
  }

  std::vector<const void *> input_ptrs_;
  std::vector<void *> inflated_ptrs_;
  std::vector<size_t> inflated_sizes_;
};

}  // namespace inflate

template <>
void Inflate<GPUBackend>::SetupOpImpl() {
  if (!impl_) {
    DALI_ENFORCE(alg_ == inflate::InflateAlg::LZ4,
                 make_string("Algorithm `", to_string(alg_),
                             "` is not supported by the GPU inflate operator."));
    impl_ = std::make_unique<inflate::InflateOpGpuLZ4Impl>(spec_);
  }
}

DALI_REGISTER_OPERATOR(decoders__Inflate, Inflate<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(experimental__Inflate, Inflate<GPUBackend>, GPU);

}  // namespace dali

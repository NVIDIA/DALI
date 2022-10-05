// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_INFLATE_INFLATE_H_
#define DALI_OPERATORS_DECODER_INFLATE_INFLATE_H_

#include <vector>

#include <nvcomp/lz4.h>

#include "dali/core/backend_tags.h"
#include "dali/core/common.h"
#include "dali/core/host_dev.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_shape.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/sequence_utils.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

namespace inflate {

constexpr static const char *shapeArgName = "shape";
constexpr static const char *offsetArgName = "frame_offset";
constexpr static const char *sizeArgName = "frame_size";

template <typename Status>
void nvcompCall(Status status) {
  if (status != nvcompSuccess) {
    throw std::runtime_error(make_string("nvComp returned non-zero status: ", status));
  }
}


// TODO instead of vector use contigious TL with templated pinned or host
// backing allocation
class InflateParams {
 public:
  InflateParams(const OpSpec &spec)
      : spec_{spec},
        shape_{shapeArgName, spec},
        frame_offset_{offsetArgName, spec},
        frame_size_{sizeArgName, spec} {}

  template <typename Workspace>
  void ProcessInputArgs(const Workspace &ws, int batch_size) {
    SetupOffsetsAndSizes(ws);
    shape_.Acquire(spec_, ws, batch_size);
    chunk_shape_ = PrepareChunkShape(shape_.get());
  }

  int64_t GetMaxOutChunkVol() const {
    return max_output_sample_vol_;
  }

  TensorListShape<> GetOutShape() const {
    if (!HasSequenceSamples()) {
      return chunk_shape_;
    }
    int batch_size = chunk_shape_.num_samples();
    TensorListShape<> shape(batch_size, chunk_shape_.sample_dim() + 1);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      int num_chunks = offset_shape_[sample_idx].num_elements();
      shape.set_tensor_shape(sample_idx,
                             shape_cat(TensorShape<>{num_chunks}, chunk_shape_[sample_idx]));
    }
    return shape;
  }

  const std::vector<int64_t> &GetInChunkOffsets() const {
    return offsets_;
  }

  const std::vector<size_t> &GetInChunkSizes() const {
    return sizes_;
  }

  const TensorListShape<1> &GetChunksNum() const {
    return offset_shape_;
  }

  bool HasSequenceSamples() const {
    return frame_offset_.HasExplicitValue();
  }

  bool HasExplicitChunkSizes() const {
    return frame_size_.HasExplicitValue();
  }

 private:
  template <typename Workspace>
  void SetupOffsetsAndSizes(const Workspace &ws) {
    const auto &in_shape = ws.GetInputShape(0);
    int batch_size = in_shape.num_samples();
    DALI_ENFORCE(HasSequenceSamples() || !HasExplicitChunkSizes(),
                 make_string("If the `", sizeArgName, "` argument is specified, the `",
                             offsetArgName, "` is required."));

    const auto copy_to_vector = [](auto &v, const auto &tlv) {
      v.clear();
      v.reserve(tlv.shape.num_elements());
      for (int sample_idx = 0; sample_idx < tlv.shape.num_samples(); sample_idx++) {
        auto num_elements = tlv.shape[sample_idx].num_elements();
        const auto *sample_data = tlv.data[sample_idx];
        v.insert(v.end(), sample_data, sample_data + num_elements);
      }
    };

    if (!HasSequenceSamples()) {
      offset_shape_ = uniform_list_shape(batch_size, TensorShape<1>{1});
      offsets_.resize(batch_size, 0);
    } else {
      frame_offset_.Acquire(spec_, ws, batch_size);
      const auto &offset_views = frame_offset_.get();
      ValidateOffsets(offset_views, in_shape, batch_size);
      offset_shape_ = offset_views.shape;
      copy_to_vector(offsets_, offset_views);
    }

    if (!HasExplicitChunkSizes()) {
      InferSizesFromOffsets(batch_size, in_shape);
    } else {
      assert(HasSequenceSamples());
      frame_size_.Acquire(spec_, ws, batch_size);
      const auto &size_views = frame_size_.get();
      ValidateSizesAgainstOffsets(size_views, batch_size, in_shape);
      copy_to_vector(sizes_, size_views);
    }
  }

  template <typename InShape>
  void ValidateOffsets(const TensorListView<StorageCPU, const int, 1> &offsets,
                       const InShape &in_shape, int batch_size) {
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto sample_num_bytes = in_shape[sample_idx].num_elements();
      const int *chunk_offsets = offsets.data[sample_idx];
      for (int chunk_idx = 0; chunk_idx < offsets.shape[sample_idx].num_elements(); chunk_idx++) {
        DALI_ENFORCE(chunk_offsets[chunk_idx] >= 0);
        DALI_ENFORCE(chunk_offsets[chunk_idx] < sample_num_bytes);
      }
    }
  }

  template <typename TLShape>
  void ValidateSizesAgainstOffsets(const TensorListView<StorageCPU, const int, 1> &sizes,
                                   int batch_size, const TLShape &in_shape) {
    for (int64_t sample_idx = 0, flat_chunk_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto num_chunks = offset_shape_[sample_idx].num_elements();
      DALI_ENFORCE(num_chunks == sizes.shape[sample_idx].num_elements(),
                   make_string("The number of chunk offsets and chunk sizes must match for "
                               "corresponding samples. However for sample of idx ",
                               sample_idx, " there are ", num_chunks, " and ",
                               sizes.shape[sample_idx].num_elements(), " for `", offsetArgName,
                               "` and for `", sizeArgName, "` respectively."));
      int64_t sample_num_bytes = in_shape[sample_idx].num_elements();
      const int *chunk_sizes = sizes.data[sample_idx];
      for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++, flat_chunk_idx++) {
        int64_t chunk_size = chunk_sizes[chunk_idx];
        DALI_ENFORCE(chunk_size >= 0);
        DALI_ENFORCE(offsets_[flat_chunk_idx] + chunk_size < sample_num_bytes);
      }
    }
  }

  template <typename TLShape>
  void InferSizesFromOffsets(int batch_size, const TLShape &in_shape) {
    // if frame sizes are not provided, they are inferred assuming that
    // the input data are densely packed and offsets describe consecutive chunks
    sizes_.clear();
    sizes_.reserve(offset_shape_.num_elements());
    for (int64_t sample_idx = 0, flat_chunk_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto num_chunks = offset_shape_[sample_idx].num_elements();
      if (num_chunks == 0) {
        continue;
      }
      auto in_chunk_size = in_shape[sample_idx].num_elements();
      for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++, flat_chunk_idx++) {
        int64_t next_chunk_offset =
            chunk_idx < num_chunks - 1 ? offsets_[flat_chunk_idx + 1] : in_chunk_size;
        int64_t chunk_size = next_chunk_offset - offsets_[flat_chunk_idx];
        DALI_ENFORCE(chunk_size > 0);  // or non-strict?
        sizes_[flat_chunk_idx] = chunk_size;
      }
    }
  }

  TensorListShape<> PrepareChunkShape(
      const TensorListView<StorageCPU, const int, 1> &provided_shape) {
    auto num_samples = provided_shape.num_samples();
    if (num_samples == 0) {
      return {};
    }
    int sample_dim = provided_shape[0].num_elements();
    DALI_ENFORCE(sample_dim >= 1,
                 make_string("The output sample cannot be a scalar, however empty shape was "
                             "provided for the output sample of idx 0."));
    TensorListShape<> shape(num_samples, sample_dim);
    for (int sample_idx = 0; sample_idx < provided_shape.num_samples(); sample_idx++) {
      DALI_ENFORCE(
          sample_dim = provided_shape.shape[sample_idx].num_elements(),
          make_string("The output shapes must have uniform dimensionality, however shapes "
                      "passed as the `",
                      shapeArgName, "` argument have different number of extents. ",
                      "The shape for sample of idx 0 has ", sample_dim,
                      " extents while the shape provided for the sample of idx ", sample_idx,
                      " has ", provided_shape.shape[sample_idx].num_elements(), " extents."));
      const int *data = provided_shape.tensor_data(sample_idx);
      TensorShape<> sample_shape(sample_dim);
      for (int d = 0; d < sample_dim; d++) {
        sample_shape[d] = data[d];
        DALI_ENFORCE(
            data[d] >= 0,
            make_string("Extents of the output shape must be non-negative integers. Please verify "
                        "the sample of idx ",
                        sample_idx, " passed as the argument `", shapeArgName, "`."));
      }
      max_output_sample_vol_ = std::max(max_output_sample_vol_, volume(sample_shape));
      shape.set_tensor_shape(sample_idx, sample_shape);
    }
    return shape;
  }

  const OpSpec &spec_;

  ArgValue<int, 1> shape_;
  ArgValue<int, 1> frame_offset_;
  ArgValue<int, 1> frame_size_;

  int64_t max_output_sample_vol_;

  TensorListShape<> chunk_shape_;
  TensorListShape<1> offset_shape_;
  std::vector<int64_t> offsets_;
  std::vector<size_t> sizes_;
};

}  // namespace inflate

template <typename Backend>
class Inflate : public Operator<Backend> {
 public:
  explicit Inflate(const OpSpec &spec) : Operator<Backend>(spec), params_{spec} {}

  bool CanInferOutputs() const override {
    return true;
  }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    output_desc.resize(1);
    auto input_type = ws.GetInputDataType(0);
    auto batch_size = ws.GetInputShape(0).num_samples();
    DALI_ENFORCE(input_type == type2id<uint8_t>::value);
    params_.ProcessInputArgs(ws, batch_size);
    output_desc[0].shape = params_.GetOutShape();
    output_desc[0].type = input_type;
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    const auto &input = ws.template Input<GPUBackend>(0);
    auto &output = ws.template Output<GPUBackend>(0);
    auto batch_size = input.num_samples();
    // output.set_order(ws.stream());

    // size_t *offs SampleDesc *samples_gpu;
    // BlockDesc *blocks_gpu;
    // std::tie(samples_gpu, blocks_gpu) = ctx.scratchpad->ToContiguousGPU(
    //     ctx.gpu.stream, make_cspan(sample_descs_), block_setup_.Blocks());
    // nvcompBatchedLZ4DecompressAsync()
    // offset_tl_dev_.set_order(ws.stream());
    // offset_tl_dev_.set_order(ws.stream());
    const auto &offsets = params_.GetInChunkOffsets();
    const auto &sizes = params_.GetInChunkSizes();
    const auto &num_chunks_per_sample = params_.GetChunksNum();
    auto total_chunks_num = num_chunks_per_sample.num_elements();
    auto in_view = view<const uint8_t>(input);
    // TODO support different output types
    auto out_view = view<uint8_t>(output);
    in_chunk_ptrs_.clear();
    in_chunk_ptrs_.reserve(total_chunks_num);
    out_chunk_ptrs_.clear();
    out_chunk_ptrs_.reserve(total_chunks_num);
    uncompressed_sizes_.clear();
    uncompressed_sizes_.reserve(total_chunks_num);
    for (int64_t sample_idx = 0, chunk_flat_idx = 0; sample_idx < batch_size; sample_idx++) {
      int num_chunks = num_chunks_per_sample[sample_idx].num_elements();
      const uint8_t *sample_base_ptr = in_view.data[sample_idx];
      // TODO support no sequence inputs
      auto sample_out_view = sequence_utils::unfolded_view_range<1>(out_view[sample_idx]);
      for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++, chunk_flat_idx++) {
        auto chunk_out_view = sample_out_view[chunk_idx];
        // TODO the params parser should return simply a proper view over the input and not any
        // other internal bs
        in_chunk_ptrs_.push_back(
            static_cast<const void *>(sample_base_ptr + offsets[chunk_flat_idx]));
        // TODO support different output types
        uncompressed_sizes_.push_back(chunk_out_view.shape.num_elements() * sizeof(uint8_t));
        out_chunk_ptrs_.push_back(static_cast<void *>(chunk_out_view.data));
        // std::string msg = make_string("sample ", sample_idx, ": (", offsets[chunk_flat_idx], ",
        // ",
        //                               sizes[chunk_flat_idx], ")");
        // std::cerr << msg << std::endl;
      }
    }

    kernels::DynamicScratchpad scratchpad({}, ws.stream());
    ctx_.gpu.stream = ws.stream();
    ctx_.scratchpad = &scratchpad;

    size_t *in_sizes;
    size_t *out_sizes;
    void const **in;
    void **out;
    std::tie(in_sizes, out_sizes, in, out) =
        ctx_.scratchpad->ToContiguousGPU(ctx_.gpu.stream, params_.GetInChunkSizes(),
                                         uncompressed_sizes_, in_chunk_ptrs_, out_chunk_ptrs_);
    size_t tempSize;
    inflate::nvcompCall(nvcompBatchedLZ4DecompressGetTempSize(
        total_chunks_num, params_.GetMaxOutChunkVol(), &tempSize));
    void *temp = ctx_.scratchpad->AllocateGPU<uint8_t>(tempSize);

    inflate::nvcompCall(nvcompBatchedLZ4DecompressAsync(in, in_sizes, out_sizes, nullptr,
                                                        total_chunks_num, temp, tempSize, out,
                                                        nullptr, ws.stream()));
  }

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

 protected:
  inflate::InflateParams params_;
  kernels::KernelContext ctx_;
  std::vector<const void *> in_chunk_ptrs_;
  std::vector<void *> out_chunk_ptrs_;
  std::vector<size_t> uncompressed_sizes_;
  // TensorList<GPUBackend> actual_sizes_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_INFLATE_INFLATE_H_

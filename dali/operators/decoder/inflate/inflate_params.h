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

#ifndef DALI_OPERATORS_DECODER_INFLATE_INFLATE_PARAMS_H_
#define DALI_OPERATORS_DECODER_INFLATE_INFLATE_PARAMS_H_

#include <string>
#include <vector>

#include "dali/core/backend_tags.h"
#include "dali/core/common.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

namespace inflate {

constexpr static const char *dTypeArgName = "dtype";
constexpr static const char *algArgName = "algorithm";
constexpr static const char *shapeArgName = "shape";
constexpr static const char *offsetArgName = "chunks_offsets";
constexpr static const char *sizeArgName = "chunks_sizes";
constexpr static const char *layoutArgName = "layout";
constexpr static const char *sequenceLayoutArgName = "sequence_extent";

enum class InflateAlg {
  LZ4
};

inline std::string to_string(const InflateAlg &alg) {
  switch (alg) {
    case InflateAlg::LZ4:
      return "LZ4";
    default:
      return "<unknown>";
  }
}

inline InflateAlg parse_inflate_alg(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), [](auto c) { return std::tolower(c); });
  if (str == "lz4") {
    return InflateAlg::LZ4;
  }
  DALI_FAIL(make_string("Unknown inflate algorithm was specified for `", algArgName,
                        "` argument: `", str, "`."));
}

template <typename Backend>
class ShapeParams {
 public:
  explicit ShapeParams(const OpSpec &spec)
      : spec_{spec},
        shape_{shapeArgName, spec},
        chunk_offsets_{offsetArgName, spec},
        chunk_sizes_{sizeArgName, spec},
        layout_{spec.GetArgument<TensorLayout>(layoutArgName)},
        sequence_extent_{spec.GetArgument<TensorLayout>(sequenceLayoutArgName)} {
    DALI_ENFORCE(sequence_extent_.size() == 1,
                 make_string("The `", sequenceLayoutArgName, "` must be a single character, got `",
                             sequence_extent_, "`."));
  }

  void ProcessInputArgs(const workspace_t<Backend> &ws, int batch_size) {
    SetupOffsetsAndSizes(ws);
    shape_.Acquire(spec_, ws, batch_size, ArgValue_EnforceUniform, ArgShapeFromSize<1>{});
    SetupOutputShape(shape_.get());
    SetupOutputLayout();
  }

  auto GetMaxOutChunkVol() const {
    return max_output_sample_vol_;
  }

  auto GetTotalChunksNum() const {
    auto total_chunks_num = sizes_.size();
    assert(total_chunks_num == offsets_.size());
    assert(total_chunks_num == chunks_per_sample_.num_elements());
    return total_chunks_num;
  }

  const auto &GetChunksNumPerSample() const {
    return chunks_per_sample_;
  }

  const TensorListShape<> &GetOutputShape() const {
    return output_shape_;
  }

  const TensorLayout &GetOutputLayout() const {
    return output_layout_;
  }

  const std::vector<int64_t> &GetInChunkOffsets() const {
    return offsets_;
  }

  const std::vector<size_t> &GetInChunkSizes() const {
    return sizes_;
  }

  bool HasChunks() const {
    return HasExplicitChunkOffsets() || HasExplicitChunkSizes();
  }

 private:
  bool HasExplicitChunkOffsets() const {
    return chunk_offsets_.HasExplicitValue();
  }

  bool HasExplicitChunkSizes() const {
    return chunk_sizes_.HasExplicitValue();
  }

  void SetupOutputLayout() {
    if (layout_.empty()) {
      return;
    }
    auto sample_dim = GetOutputShape().sample_dim() - HasChunks();
    DALI_ENFORCE(
        layout_.size() == sample_dim,
        make_string("The size of inflated chunks' layout must match the number of extents in "
                    "the output shape. However, got the layout `",
                    layout_, "` while the chunks output shape has ", sample_dim, " extent(s)."));
    if (!HasChunks()) {
      output_layout_ = layout_;
    } else {
      assert(sequence_extent_.size() == 1);
      output_layout_ = sequence_extent_ + layout_;
    }
  }

  void SetupOffsetsAndSizes(const workspace_t<Backend> &ws) {
    if (HasChunks()) {
      AcquireInferOffsetsSizes(ws);
    } else {
      const auto &in_shape = ws.GetInputShape(0);
      int batch_size = in_shape.num_samples();
      chunks_per_sample_ = uniform_list_shape(batch_size, TensorShape<1>{1});
      offsets_.resize(batch_size, 0);
      sizes_.clear();
      sizes_.reserve(batch_size);
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        sizes_.push_back(in_shape[sample_idx].num_elements());
      }
    }
  }

  void AcquireInferOffsetsSizes(const workspace_t<Backend> &ws) {
    assert(HasChunks());

    const auto copy_to_vector = [](auto &v, const auto &tlv) {
      v.clear();
      v.reserve(tlv.shape.num_elements());
      for (int sample_idx = 0; sample_idx < tlv.shape.num_samples(); sample_idx++) {
        auto num_elements = tlv.shape[sample_idx].num_elements();
        const auto *chunks_data = tlv.data[sample_idx];
        v.insert(v.end(), chunks_data, chunks_data + num_elements);
      }
    };

    const auto validate_offset = [](auto sample_idx, auto sample_num_bytes, auto chunk_offset) {
      DALI_ENFORCE(chunk_offset >= 0,
                   make_string("Input chunks offsets must be non-negative. Got ", chunk_offset,
                               " offset for sample of idx ", sample_idx, "."));
      DALI_ENFORCE(chunk_offset < sample_num_bytes,
                   make_string("Input chunks offsets cannot exceed the sample size. Got offset ",
                               chunk_offset, " while the sample size is ", sample_num_bytes,
                               " for sample of idx ", sample_idx, "."));
    };

    const auto validate_size = [](auto sample_idx, auto sample_num_bytes, auto chunk_size) {
      DALI_ENFORCE(chunk_size > 0,
                   make_string("Input chunk size must be positive. Got ", chunk_size,
                               " offset for sample of idx ", sample_idx, "."));
      DALI_ENFORCE(chunk_size <= sample_num_bytes,
                   make_string("Input chunk size cannot exceed the sample size. Got size ",
                               chunk_size, " while the sample size is ", sample_num_bytes,
                               " for sample of idx ", sample_idx, "."));
    };

    const auto &in_shape = ws.GetInputShape(0);
    int batch_size = in_shape.num_samples();

    if (HasExplicitChunkOffsets()) {
      chunk_offsets_.Acquire(spec_, ws, batch_size);
      ValidateChunks(chunk_offsets_.get(), in_shape, validate_offset);
    }
    if (HasExplicitChunkSizes()) {
      chunk_sizes_.Acquire(spec_, ws, batch_size);
      ValidateChunks(chunk_sizes_.get(), in_shape, validate_size);
    }
    if (HasExplicitChunkOffsets() && HasExplicitChunkSizes()) {
      const auto &offset_views = chunk_offsets_.get();
      const auto &size_views = chunk_sizes_.get();
      ValidateSizesAgainstOffsets(offset_views, size_views, in_shape);
      chunks_per_sample_ = offset_views.shape;
      copy_to_vector(offsets_, offset_views);
      copy_to_vector(sizes_, size_views);
    } else if (HasExplicitChunkOffsets()) {
      const auto &offset_views = chunk_offsets_.get();
      chunks_per_sample_ = offset_views.shape;
      copy_to_vector(offsets_, offset_views);
      InferSizesFromOffsets(in_shape);
    } else {
      assert(HasExplicitChunkSizes());
      const auto &size_views = chunk_sizes_.get();
      chunks_per_sample_ = size_views.shape;
      copy_to_vector(sizes_, size_views);
      InferOffsetsFromSizes(in_shape);
    }
  }

  template <typename T, typename InShape, typename ChunkValidator>
  void ValidateChunks(const TensorListView<StorageCPU, const T, 1> &samples,
                      const InShape &in_shape, ChunkValidator &&chunk_validator) {
    for (int sample_idx = 0; sample_idx < in_shape.num_samples(); sample_idx++) {
      auto sample_num_bytes = in_shape[sample_idx].num_elements();
      const auto *chunks_data = samples.data[sample_idx];
      for (int chunk_idx = 0; chunk_idx < samples.shape[sample_idx].num_elements(); chunk_idx++) {
        chunk_validator(sample_idx, sample_num_bytes, chunks_data[chunk_idx]);
      }
    }
  }

  template <typename InOffsetT, typename InSizeT, typename TLShape>
  void ValidateSizesAgainstOffsets(const TensorListView<StorageCPU, const InOffsetT, 1> &offsets,
                                   const TensorListView<StorageCPU, const InSizeT, 1> &sizes,
                                   const TLShape &in_shape) {
    auto batch_size = in_shape.num_samples();
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto num_chunks = offsets.shape[sample_idx].num_elements();
      DALI_ENFORCE(
          num_chunks == sizes.shape[sample_idx].num_elements(),
          make_string("The number of elements passed as `", offsetArgName, "` and `", sizeArgName,
                      "` must match for corresponding samples. However for sample of idx ",
                      sample_idx, " there are ", num_chunks, " offsets and ",
                      sizes.shape[sample_idx].num_elements(), " sizes."));
    }
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto num_chunks = offsets.shape[sample_idx].num_elements();
      auto sample_num_bytes = in_shape[sample_idx].num_elements();
      const auto *chunk_offsets = offsets.data[sample_idx];
      const auto *chunk_sizes = sizes.data[sample_idx];
      for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int64_t chunk_end = chunk_offsets[chunk_idx];
        chunk_end += chunk_sizes[chunk_idx];
        DALI_ENFORCE(
            chunk_end <= sample_num_bytes,
            make_string("Input chunk cannot exceed the sample size. However, for a sample of idx ",
                        sample_idx, ", got a chunk that starts at the position ",
                        chunk_offsets[chunk_idx], " and has size ", chunk_sizes[chunk_idx],
                        " while the sample contains only ", sample_num_bytes, " bytes."));
      }
    }
  }

  template <typename TLShape>
  void InferSizesFromOffsets(const TLShape &in_shape) {
    assert(offsets_.size() == chunks_per_sample_.num_elements());
    auto last_chunk_error_msg = [](auto sample_idx) {
      return make_string(
          "The last chunk's offset exceeds the size of the sample for sample of idx ", sample_idx,
          ".");
    };
    auto non_monotone_error = [](auto sample_idx) {
      return make_string("If the `", offsetArgName, "` argument is specified and the `",
                         sizeArgName, "` is not, the offsets must be strictly increasing. ",
                         "The inferred size of a chunk would be non-positive for sample of idx ",
                         sample_idx, ".");
    };
    auto batch_size = in_shape.num_samples();
    sizes_.clear();
    sizes_.reserve(chunks_per_sample_.num_elements());
    for (int64_t sample_idx = 0, flat_chunk_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto num_chunks = chunks_per_sample_[sample_idx].num_elements();
      if (num_chunks == 0) {
        continue;
      }
      auto in_chunk_size = in_shape[sample_idx].num_elements();
      for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++, flat_chunk_idx++) {
        bool is_last_chunk = chunk_idx == num_chunks - 1;
        int64_t next_chunk_offset = is_last_chunk ? in_chunk_size : offsets_[flat_chunk_idx + 1];
        int64_t chunk_size = next_chunk_offset - offsets_[flat_chunk_idx];
        DALI_ENFORCE(chunk_size > 0, is_last_chunk ? last_chunk_error_msg(sample_idx) :
                                                     non_monotone_error(sample_idx));
        sizes_.push_back(chunk_size);
      }
    }
    assert(offsets_.size() == sizes_.size());
  }

  template <typename TLShape>
  void InferOffsetsFromSizes(const TLShape &in_shape) {
    assert(sizes_.size() == chunks_per_sample_.num_elements());
    auto batch_size = in_shape.num_samples();
    offsets_.clear();
    offsets_.reserve(chunks_per_sample_.num_elements());
    for (int64_t sample_idx = 0, flat_chunk_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto num_chunks = chunks_per_sample_[sample_idx].num_elements();
      if (num_chunks == 0) {
        continue;
      }
      int64_t cum_offset = 0;
      auto sample_num_bytes = in_shape[sample_idx].num_elements();
      for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++, flat_chunk_idx++) {
        offsets_.push_back(cum_offset);
        cum_offset += sizes_[flat_chunk_idx];
      }
      DALI_ENFORCE(cum_offset <= sample_num_bytes,
                   make_string("The sum of chunk sizes for sample of idx ", sample_idx,
                               " exceeds the total size of the sample"));
    }
    assert(offsets_.size() == sizes_.size());
  }

  void SetupOutputShape(const TensorListView<StorageCPU, const int> &provided_shape) {
    DALI_ENFORCE(provided_shape.sample_dim() == 0 || provided_shape.sample_dim() == 1,
                 make_string("The shape argument must be a scalar or a 1D tensor, got tensor with ",
                             provided_shape.sample_dim(), " extents."));
    auto chunk_shapes = ParseOutputShape(provided_shape);
    if (!HasChunks()) {
      output_shape_ = chunk_shapes;
    } else {
      int batch_size = chunk_shapes.num_samples();
      output_shape_.resize(batch_size, chunk_shapes.sample_dim() + 1);
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        int num_chunks = chunks_per_sample_[sample_idx].num_elements();
        output_shape_.set_tensor_shape(
            sample_idx, shape_cat(TensorShape<>{num_chunks}, chunk_shapes[sample_idx]));
      }
    }
  }

  TensorListShape<> ParseOutputShape(const TensorListView<StorageCPU, const int> &provided_shape) {
    auto num_samples = provided_shape.num_samples();
    if (num_samples == 0) {
      return {};
    }
    int sample_dim = provided_shape[0].num_elements();
    TensorListShape<> shape(num_samples, sample_dim);
    for (int sample_idx = 0; sample_idx < provided_shape.num_samples(); sample_idx++) {
      const int *data = provided_shape.tensor_data(sample_idx);
      for (int d = 0; d < sample_dim; d++) {
        DALI_ENFORCE(
            data[d] >= 0,
            make_string("Extents of the output shape must be non-negative integers. Please verify "
                        "the sample of idx ",
                        sample_idx, " passed as the argument `", shapeArgName, "`."));
      }
      TensorShape<> sample_shape(data, data + sample_dim);
      max_output_sample_vol_ = std::max(max_output_sample_vol_, volume(sample_shape));
      shape.set_tensor_shape(sample_idx, sample_shape);
    }
    return shape;
  }

  const OpSpec &spec_;

  ArgValue<int, DynamicDimensions> shape_;
  ArgValue<int, 1> chunk_offsets_;
  ArgValue<int, 1> chunk_sizes_;
  TensorLayout layout_;
  TensorLayout sequence_extent_;

  TensorListShape<1> chunks_per_sample_;
  std::vector<int64_t> offsets_;
  std::vector<size_t> sizes_;

  int64_t max_output_sample_vol_;
  TensorListShape<> output_shape_;
  TensorLayout output_layout_ = "";
};

}  // namespace inflate
}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_INFLATE_INFLATE_PARAMS_H_

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

inline InflateAlg parse_inflate_alg(const std::string &inflate_alg_str) {
  if (inflate_alg_str == "LZ4") {
    return InflateAlg::LZ4;
  }
  DALI_FAIL(make_string("Unknown inflate algorithm was specified for `", algArgName,
                        "` argument: `", inflate_alg_str, "`."));
}

template <typename Backend>
class ShapeParams {
 public:
  explicit ShapeParams(const OpSpec &spec)
      : spec_{spec},
        shape_{shapeArgName, spec},
        frame_offset_{offsetArgName, spec},
        frame_size_{sizeArgName, spec} {}

  void ProcessInputArgs(const workspace_t<Backend> &ws, int batch_size) {
    SetupOffsetsAndSizes(ws);
    shape_.Acquire(spec_, ws, batch_size, ArgValue_EnforceUniform, ArgShapeFromSize<1>{});
    SetupOutputShape(shape_.get());
  }

  auto GetMaxOutChunkVol() const {
    return max_output_sample_vol_;
  }

  auto GetTotalChunksNum() const {
    auto total_chunks_num = sizes_.size();
    assert(total_chunks_num == offsets_.size());
    assert(total_chunks_num == offset_shape_.num_elements());
    return total_chunks_num;
  }

  const auto &GetChunksNumPerSample() const {
    return offset_shape_;
  }

  TensorListShape<> GetOutputShape() const {
    return output_shape_;
  }

  const std::vector<int64_t> &GetInChunkOffsets() const {
    return offsets_;
  }

  const std::vector<size_t> &GetInChunkSizes() const {
    return sizes_;
  }

  bool HasChunks() const {
    return frame_offset_.HasExplicitValue();
  }

 private:
  bool HasExplicitChunkSizes() const {
    return frame_size_.HasExplicitValue();
  }

  void SetupOffsetsAndSizes(const workspace_t<Backend> &ws) {
    const auto &in_shape = ws.GetInputShape(0);
    int batch_size = in_shape.num_samples();
    DALI_ENFORCE(HasChunks() || !HasExplicitChunkSizes(),
                 make_string("If the `", sizeArgName,
                             "` argument is specified for the inflate operator, the `",
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

    if (!HasChunks()) {
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
      assert(HasChunks());
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
        DALI_ENFORCE(
            chunk_offsets[chunk_idx] >= 0,
            make_string("Input chunks offsets must be non-negative. Got negative offset for ",
                        "sample of idx ", sample_idx, "."));
        DALI_ENFORCE(chunk_offsets[chunk_idx] < sample_num_bytes,
                     make_string("Input chunks offsets cannot exceed the sample size. Got offset `",
                                 chunk_offsets[chunk_idx], "` while the sample size is `",
                                 sample_num_bytes, "` for sample of idx ", sample_idx, "."));
      }
    }
  }

  template <typename TLShape>
  void ValidateSizesAgainstOffsets(const TensorListView<StorageCPU, const int, 1> &sizes,
                                   int batch_size, const TLShape &in_shape) {
    for (int64_t sample_idx = 0, flat_chunk_idx = 0; sample_idx < batch_size; sample_idx++) {
      auto num_chunks = offset_shape_[sample_idx].num_elements();
      DALI_ENFORCE(num_chunks == sizes.shape[sample_idx].num_elements(),
                   make_string("The number of `", offsetArgName, "` and `", sizeArgName,
                               "` must match for corresponding samples. However for sample of idx ",
                               sample_idx, " there are ", num_chunks, " offsets and ",
                               sizes.shape[sample_idx].num_elements(), " sizes."));
      int64_t sample_num_bytes = in_shape[sample_idx].num_elements();
      const int *chunk_sizes = sizes.data[sample_idx];
      for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++, flat_chunk_idx++) {
        int64_t chunk_size = chunk_sizes[chunk_idx];
        DALI_ENFORCE(chunk_size > 0,
                     make_string("Input chunk size must be positive. Got chunk size `", chunk_size,
                                 "` for sample of idx ", sample_idx, "."));
        DALI_ENFORCE(
            offsets_[flat_chunk_idx] + chunk_size <= sample_num_bytes,
            make_string("Input chunk cannot exceed the sample size. However, for a sample of idx ",
                        sample_idx, ", got a chunk that starts at the position `",
                        offsets_[flat_chunk_idx], "` and has size `", chunk_size,
                        "` while the sample contains only `", sample_num_bytes, "` bytes."));
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
        sizes_.push_back(chunk_size);
      }
    }
    assert(sizes_.size() == offset_shape_.num_elements());
  }

  void SetupOutputShape(const TensorListView<StorageCPU, const int> &provided_shape) {
    DALI_ENFORCE(provided_shape.sample_dim() == 0 || provided_shape.sample_dim() == 1,
                 make_string("The shape argument must be a scalar a 1D tensor, got tensor with `",
                             provided_shape.sample_dim(), "` extents."));
    auto chunk_shapes = ParseOutputShape(provided_shape);
    if (!HasChunks()) {
      output_shape_ = chunk_shapes;
    } else {
      int batch_size = chunk_shapes.num_samples();
      output_shape_.resize(batch_size, chunk_shapes.sample_dim() + 1);
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        int num_chunks = offset_shape_[sample_idx].num_elements();
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

  ArgValue<int, DynamicDimensions> shape_;
  ArgValue<int, 1> frame_offset_;
  ArgValue<int, 1> frame_size_;

  TensorListShape<1> offset_shape_;
  std::vector<int64_t> offsets_;
  std::vector<size_t> sizes_;

  int64_t max_output_sample_vol_;
  TensorListShape<> output_shape_;
};

}  // namespace inflate
}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_INFLATE_INFLATE_PARAMS_H_

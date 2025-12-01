// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_
#define DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/format.h"
#include "dali/core/geom/geom_utils.h"
#include "dali/core/geom/vec.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/imgproc/paste/paste_gpu_input.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"
#include "dali/util/crop_window.h"


namespace dali {

#define MULTIPASTE_INPUT_TYPES uint8_t, int16_t, int32_t, float
#define MULTIPASTE_OUTPUT_TYPES uint8_t, int16_t, int32_t, float

namespace multipaste_utils {

template <typename Backend>
inline void GetValidateInputTypeAndDim(int& ndim, DALIDataType &type, const Workspace &ws) {
  const auto &images = ws.Input<Backend>(0);
  ndim = images.sample_dim();
  type = images.type();
  for (int i = 1; i < ws.NumInput(); i++) {
    int sample_dim = ws.Input<Backend>(i).sample_dim();
    auto sample_type = ws.Input<Backend>(i).type();
    DALI_ENFORCE(ndim == sample_dim,
                 make_string("All input batches must have the same dimensionality. Got ", ndim,
                             " and ", sample_dim, " at indicies 0 and ", i, "."));
    DALI_ENFORCE(type == sample_type,
                 make_string("All input batches must have the same data type. Got ", type,
                             " and ", sample_type, " at indicies 0 and ", i, "."));
  }
}

}  // namespace multipaste_utils

template <typename Backend, typename Actual>
class MultiPasteOp : public SequenceOperator<Backend, StatelessOperator> {
 public:
  DISABLE_COPY_MOVE_ASSIGN(MultiPasteOp);

  Actual &This() noexcept {
    return static_cast<Actual &>(*this);
  }

  const Actual &This() const noexcept {
    return static_cast<const Actual &>(*this);
  }

  template <typename T>
  using InList = TensorListView<detail::storage_tag_map_t<Backend>, const T>;

 protected:
  using Coords = TensorView<StorageCPU, const int, 1>;

  inline void UnsupportedInputType(DALIDataType actual) {
    DALI_FAIL(make_string("Unsupported input type: ", actual,
              "\nSupported types: ", ListTypeNames<MULTIPASTE_INPUT_TYPES>()));
  }

  inline void UnsupportedOutpuType(DALIDataType actual) {
    DALI_FAIL(make_string("Unsupported output type: ", actual,
              "\nSupported types: ", ListTypeNames<MULTIPASTE_OUTPUT_TYPES>()));
  }

  explicit MultiPasteOp(const OpSpec &spec)
      : SequenceOperator<Backend, StatelessOperator>(spec)
      , output_type_(DALI_NO_TYPE)
      , input_type_(DALI_NO_TYPE)
      , output_size_("output_size", spec)
      , in_idx_("in_ids", spec)
      , in_anchors_("in_anchors", spec)
      , in_anchors_rel_("in_anchors_rel", spec)
      , shapes_("shapes", spec)
      , shapes_rel_("shapes_rel", spec)
      , out_anchors_("out_anchors", spec)
      , out_anchors_rel_("out_anchors_rel", spec) {
    spec.TryGetArgument(output_type_arg_, "dtype");
    if (std::is_same<Backend, GPUBackend>::value) {
      kernel_manager_.Resize(1);
    } else {
      kernel_manager_.Resize(max_batch_size_);
    }
  }

  bool Intersects(ivec2 anchors1, ivec2 shapes1,
                  ivec2 anchors2, ivec2 shapes2) const {
    for (int i = 0; i < 2; i++) {
      if (anchors1[i] + shapes1[i] <= anchors2[i]
          || anchors2[i] + shapes2[i] <= anchors1[i]) {
        return false;
      }
    }
    return true;
  }

  void ValidateFactor(float factor, const std::string &arg_name, int out_sample_idx,
                      int paste_idx) {
    DALI_ENFORCE(0.f <= factor && factor <= 1.f,
                 make_string("The `", arg_name, "` must be float in [0, 1] range, got ", factor,
                             " for region at index ", paste_idx,
                             " pasted into output sample at index ", out_sample_idx, "."));
  }

  void AcquireInIdxs(const Workspace &ws) {
    auto nsamples = ws.GetInputShape(0).num_samples();
    if (in_idx_.HasExplicitValue()) {
      DALI_ENFORCE(ws.NumInput() == 1,
                   make_string("If the `in_ids` is specified, the operator accepts exactly one "
                               "positional input batch. Got: ",
                               ws.NumInput(), " inputs."));
      in_idx_.Acquire(spec_, ws, nsamples);
      for (int i = 0; i < nsamples; i++) {
        auto paste_count = GetPasteCount(ws, i);
        for (int j = 0; j < paste_count; j++) {
          auto paste_idx = in_idx_[i].data[j];
          DALI_ENFORCE(
              0 <= paste_idx && paste_idx < nsamples,
              make_string("The `in_idx` must be in range [0, .., batch_size - 1]. Got in_idx: ",
                          paste_idx, ". Input batch size is: ", nsamples, "."));
        }
      }
    }
  }

  // Get, for the given output sample, the number of channels. It's inferred
  // from the pasted regions, or -1 if there are no pasted regions.
  inline int InferChannelsNum(const Workspace &ws, int out_sample_idx) {
    int num_channels = -1;
    for (int paste_idx = 0; paste_idx < GetPasteCount(ws, out_sample_idx); paste_idx++) {
      const auto &sample = GetPasteSample(ws, out_sample_idx, paste_idx);
      int input_channels = sample.shape()[spatial_ndim_];
      DALI_ENFORCE(paste_idx == 0 || num_channels == input_channels,
                   make_string("All regions pasted into given output sample must have the same "
                               "number of channels. Got different number of channels: ",
                               num_channels, ", ", input_channels, " for output sample at index ",
                               out_sample_idx, "."));
      num_channels = input_channels;
    }
    return num_channels;
  }

  void AcquireSetOutputShape(TensorListShape<> &out_shape, const Workspace &ws, int ndim) {
    const auto &in_shape = ws.GetInputShape(0);
    int nsamples = in_shape.num_samples();
    out_shape.resize(nsamples, ndim);

    if (nsamples == 0) {
      return;
    }

    TensorShape<2> ref_shape = in_shape[0].first(spatial_ndim_);
    // validate sources of the shape
    if (output_size_.HasExplicitValue()) {
      output_size_.Acquire(spec_, ws, nsamples, ArgValue_EnforceUniform);
      for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
        const auto &sample_out_shape = output_size_[sample_idx];
        // TODO(ktokarski) It should be equality, really. But we did not enforce that
        // initially. There are even some test cases in DALI codebase that specify full shape.
        DALI_ENFORCE(sample_out_shape.num_elements() >= spatial_ndim_,
                     make_string("The `output_size` should be a tuple (H, W), got ",
                                 sample_out_shape.num_elements(),
                                 " elements for a sample at index ", sample_idx, "."));
        for (int d = 0; d < spatial_ndim_; d++) {
          DALI_ENFORCE(
              sample_out_shape.data[d] >= 0,
              make_string("The `output_size` must be non-negative, ",
                          "got a shape with negative extent: ",
                          TensorShape<>(sample_out_shape.data,
                                        sample_out_shape.data + sample_out_shape.num_elements()),
                          " for a sample at index ", sample_idx, "."));
        }
      }
    } else {
      // if the output_size_ is not specified, check if the input shapes are uniform
      for (int i = 0; i < ws.NumInput(); i++) {
        const auto &in_shapes = ws.Input<Backend>(i).shape();
        for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
          auto sample_shape = in_shapes[sample_idx].first(2);
          DALI_ENFORCE(ref_shape == sample_shape,
                       make_string("If the `output_size` is not specified, all the input samples "
                                   "must have the same shape. Got samples of different shapes: ",
                                   ref_shape, " and ", sample_shape, "."));
        }
      }
    }

    // set the output shape
    for (int i = 0; i < nsamples; i++) {
      int num_channels = InferChannelsNum(ws, i);
      // The old behaviour was to use the number of channels from the i-th input sample.
      // The number of channels of the pasted regions does not have to match that.
      // Previously, it would result in broken outputs or wrong memory access. Now, we infer
      // the number from the pasted regions. We keep the old behaviour only for cases when
      // there are no pasted regions - as that had a chance to work.
      if (num_channels < 0) {
        num_channels = in_shape[i][spatial_ndim_];
      }

      if (output_size_.HasExplicitValue()) {
        TensorShape<> size(output_size_[i].data, output_size_[i].data + spatial_ndim_);
        auto out_sh = shape_cat(size, num_channels);
        out_shape.set_tensor_shape(i, out_sh);
      } else {
        out_shape.set_tensor_shape(i, shape_cat(ref_shape, num_channels));
      }
    }
  }

  void AcquireArguments(const Workspace &ws, const TensorListShape<> &out_shapes) {
    auto curr_batch_size = ws.GetRequestedBatchSize(0);
    if (curr_batch_size == 0)
      return;

    DALI_ENFORCE(!out_anchors_.HasExplicitValue() || !out_anchors_rel_.HasExplicitValue(),
                 "The `out_anchors` and `out_anchors_rel` cannot be specified together");
    if (out_anchors_.HasExplicitValue()) {
      out_anchors_.Acquire(spec_, ws, curr_batch_size);
    } else if (out_anchors_rel_.HasExplicitValue()) {
      out_anchors_rel_.Acquire(spec_, ws, curr_batch_size);
    }

    DALI_ENFORCE(!in_anchors_.HasExplicitValue() || !in_anchors_rel_.HasExplicitValue(),
                 "The `in_anchors` and `in_anchors_rel` cannot be specified together");
    if (in_anchors_.HasExplicitValue()) {
      in_anchors_.Acquire(spec_, ws, curr_batch_size);
    } else if (in_anchors_rel_.HasExplicitValue()) {
      in_anchors_rel_.Acquire(spec_, ws, curr_batch_size);
    }

    DALI_ENFORCE(!shapes_.HasExplicitValue() || !shapes_rel_.HasExplicitValue(),
                 "The `shapes` and `shapes_rel` cannot be specified together");
    if (shapes_.HasExplicitValue()) {
      shapes_.Acquire(spec_, ws, curr_batch_size);
    } else if (shapes_rel_.HasExplicitValue()) {
      shapes_rel_.Acquire(spec_, ws, curr_batch_size);
    }

    const auto validateArgShape = [spatial_ndim = spatial_ndim_](int sample_idx, int paste_count,
                                                                 const std::string &arg_name,
                                                                 const TensorShape<> arg_shape) {
      int num_args = arg_shape[0];
      int arg_len = arg_shape[1];
      DALI_ENFORCE(num_args == paste_count && arg_len == spatial_ndim,
                   make_string("Unexpected shape for argument `", arg_name,
                               "` for output sample at index ", sample_idx,
                               ". It should be a 2D tensor of shape `number of pasted regions` x 2",
                               ", i.e. ", TensorShape<2>{paste_count, spatial_ndim},
                               ". Got tensor of shape: ", TensorShape<2>{num_args, arg_len}, "."));
    };

    in_anchors_data_.clear();
    in_anchors_data_.resize(curr_batch_size);
    region_shapes_data_.clear();
    region_shapes_data_.resize(curr_batch_size);
    out_anchors_data_.clear();
    out_anchors_data_.resize(curr_batch_size);
    for (int i = 0; i < curr_batch_size; i++) {
      const auto n_paste = GetPasteCount(ws, i);
      const auto& out_shape = out_shapes[i];

      if (in_anchors_.HasExplicitValue()) {
        validateArgShape(i, n_paste, "in_anchors", in_anchors_[i].shape);
      } else if (in_anchors_rel_.HasExplicitValue()) {
        validateArgShape(i, n_paste, "in_anchors_rel", in_anchors_rel_[i].shape);
      }

      if (shapes_.HasExplicitValue()) {
        validateArgShape(i, n_paste, "shapes", shapes_[i].shape);
      } else if (shapes_rel_.HasExplicitValue()) {
        validateArgShape(i, n_paste, "shapes_rel", shapes_rel_[i].shape);
      }

      if (out_anchors_.HasExplicitValue()) {
        validateArgShape(i, n_paste, "out_anchors", out_anchors_[i].shape);
      } else if (out_anchors_rel_.HasExplicitValue()) {
        validateArgShape(i, n_paste, "out_anchors_rel", out_anchors_rel_[i].shape);
      }

      in_anchors_data_[i].resize(n_paste);
      region_shapes_data_[i].resize(n_paste);
      out_anchors_data_[i].resize(n_paste);
      for (int j = 0; j < n_paste; j++) {
        SetupShapesAndAnchors(ws, i, j, out_shape);
      }
    }
  }

  void SetupShapesAndAnchors(const Workspace &ws, int out_sample_idx, int paste_idx,
                             const TensorShape<3> &out_shape) {
    auto region_source_shape = GetInputShape(ws, out_sample_idx, paste_idx);
    auto &region_shape = region_shapes_data_[out_sample_idx][paste_idx];
    auto &in_anchor = in_anchors_data_[out_sample_idx][paste_idx];
    SetupInShapeAnchor(region_shape, in_anchor, out_sample_idx, paste_idx, region_source_shape);
    auto &out_anchor = out_anchors_data_[out_sample_idx][paste_idx];
    SetupOutAnchor(out_anchor, out_sample_idx, paste_idx, out_shape.first(spatial_ndim_));
    const auto as_shape = [spatial_ndim = spatial_ndim_](ivec2 extents) {
      return TensorShape<2>(&extents[0], &extents[0] + spatial_ndim);
    };
    for (int k = 0; k < spatial_ndim_; k++) {
      DALI_ENFORCE(
          in_anchor[k] >= 0 && region_shape[k] >= 0 &&
              in_anchor[k] + region_shape[k] <= region_source_shape[k],
          make_string("The pasted region must be within input sample. Got input anchor: ",
                      as_shape(in_anchor), ", pasted region shape: ", as_shape(region_shape),
                      ", input shape: ", as_shape(region_source_shape),
                      ", for the region at index ", paste_idx,
                      ", pasted into output sample at index ", out_sample_idx, "."));
      DALI_ENFORCE(
          out_anchor[k] >= 0 && out_anchor[k] + region_shape[k] <= out_shape[k],
          make_string("The pasted region must be within output bounds. Got output anchor: ",
                      as_shape(out_anchor), ", pasted region shape: ", as_shape(region_shape),
                      ", output shape: ", out_shape.first(spatial_ndim_),
                      ", for the region at index ", paste_idx,
                      ", pasted into output sample at index ", out_sample_idx, "."));
    }
  }

  void SetupInShapeAnchor(ivec2 &region_shape, ivec2 &anchor, int out_sample_idx, int paste_idx,
                          ivec2 source_shape) {
    // get the shape of the region to be pasted from the source
    if (shapes_.HasExplicitValue()) {  // absolute shape provided as argument
      auto sh_view = subtensor(shapes_[out_sample_idx], paste_idx);
      for (int d = 0; d < spatial_ndim_; d++) {
        region_shape[d] = sh_view.data[d];
      }
    } else if (shapes_rel_.HasExplicitValue()) {  // scale the source_shape
      auto shape_rel = subtensor(shapes_rel_[out_sample_idx], paste_idx);
      for (int d = 0; d < spatial_ndim_; d++) {
        float scale = shape_rel.data[d];
        ValidateFactor(scale, "shapes_rel", out_sample_idx, paste_idx);
        // the shape is rounded up, the anchor rounded down, so that
        // shape = 0.5, anchor = 0.5 fit into the input
        region_shape[d] = std::ceil(source_shape[d] * scale);
      }
    } else {  // use plain source_shape
      region_shape = source_shape;
    }

    // get the in anchors
    anchor = {0, 0};
    if (in_anchors_.HasExplicitValue()) {
      auto in_anchor_view = subtensor(in_anchors_[out_sample_idx], paste_idx);
      for (int d = 0; d < spatial_ndim_; d++) {
        anchor[d] = in_anchor_view.data[d];
      }
    } else if (in_anchors_rel_.HasExplicitValue()) {
      auto in_anchor_rel_view = subtensor(in_anchors_rel_[out_sample_idx], paste_idx);
      for (int d = 0; d < spatial_ndim_; d++) {
        float factor = in_anchor_rel_view.data[d];
        ValidateFactor(factor, "in_anchor_rel", out_sample_idx, paste_idx);
        // the shape is rounded up, the anchor rounded down, so that
        // shape = 0.5, anchor = 0.5 fit into the input
        anchor[d] = std::floor(source_shape[d] * in_anchor_rel_view.data[d]);
      }
    }

    // if the region shape was not set explicitly, the region_shape is now
    // full source_shape, subtract the anchor
    if (!shapes_.HasExplicitValue() && !shapes_rel_.HasExplicitValue()) {
      for (int d = 0; d < spatial_ndim_; d++) {
        region_shape[d] -= anchor[d];
      }
    }
  }

  void SetupOutAnchor(ivec2 &anchor, int out_sample_idx, int paste_idx,
                      const TensorShape<2> out_shape) {
    anchor = {0, 0};
    if (out_anchors_.HasExplicitValue()) {
      auto out_anchor_view = subtensor(out_anchors_[out_sample_idx], paste_idx);
      for (int d = 0; d < spatial_ndim_; d++) {
        anchor[d] = out_anchor_view.data[d];
      }
    } else if (out_anchors_rel_.HasExplicitValue()) {
      auto out_anchor_rel_view = subtensor(out_anchors_rel_[out_sample_idx], paste_idx);
      for (int d = 0; d < spatial_ndim_; d++) {
        float factor = out_anchor_rel_view.data[d];
        ValidateFactor(factor, "out_anchor_rel", out_sample_idx, paste_idx);
        anchor[d] = std::floor(out_shape[d] * out_anchor_rel_view.data[d]);
      }
    }
  }

  using Operator<Backend>::RunImpl;

  void RunImpl(Workspace &ws) override {
    const auto input_type_id = ws.Input<Backend>(0).type();
    TYPE_SWITCH(input_type_id, type2id, InputType, (MULTIPASTE_INPUT_TYPES), (
        TYPE_SWITCH(output_type_, type2id, OutputType, (MULTIPASTE_OUTPUT_TYPES), (
                This().template RunTyped<OutputType, InputType>(ws);
        ), UnsupportedOutpuType(output_type_))  // NOLINT
    ), UnsupportedInputType(input_type_id))  // NOLINT
    SetSourceInfo(ws);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const Workspace &ws) override {
    int ndim;
    multipaste_utils::GetValidateInputTypeAndDim<Backend>(ndim, input_type_, ws);
    DALI_ENFORCE(ndim == 3, make_string("Unsupported sample dimensionality, got ", ndim,
                                        ", expected 3 (2D data with channels - HWC)."));
    // set output type
    output_type_ = output_type_arg_ != DALI_NO_TYPE ? output_type_arg_ : input_type_;

    // Acquire in_idx_ first, the number of output channels may depend on it
    AcquireInIdxs(ws);

    output_desc.resize(1);
    output_desc[0].type = output_type_;
    auto &out_shape = output_desc[0].shape;
    AcquireSetOutputShape(out_shape, ws, ndim);

    // acquire remaining arguments
    AcquireArguments(ws, out_shape);

    TYPE_SWITCH(input_type_, type2id, InputType, (MULTIPASTE_INPUT_TYPES), (
        TYPE_SWITCH(output_type_, type2id, OutputType, (MULTIPASTE_OUTPUT_TYPES), (
            This().template SetupTyped<OutputType, InputType>(ws, out_shape);
        ), UnsupportedOutpuType(output_type_))  // NOLINT
    ), UnsupportedInputType(input_type_))  // NOLINT
    return true;
  }

  void SetSourceInfo(Workspace &ws) {
    auto &out = ws.Output<Backend>(0);
    for (int i = 0; i < out.num_samples(); i++) {
      std::string out_source_info;
      const auto append_info = [&out_source_info] (auto &in_source_info) {
        if (!in_source_info.empty()) {
          if (!out_source_info.empty())
            out_source_info += ";";
          out_source_info += in_source_info;
        }
      };
      int paste_count = GetPasteCount(ws, i);
      for (int j = 0; j < paste_count; j++) {
        if (!in_idx_.HasExplicitValue()) {
          auto &&in_source_info = ws.Input<Backend>(j).GetMeta(i).GetSourceInfo();
          append_info(in_source_info);
        } else {
          int source_sample = in_idx_[i].data[j];
          auto &&in_source_info = ws.Input<Backend>(0).GetMeta(source_sample).GetSourceInfo();
          append_info(in_source_info);
        }
      }
      out.SetSourceInfo(i, out_source_info);
    }
  }

  inline int64_t GetPasteCount(const Workspace &ws, int out_sample_idx) const {
    if (!in_idx_.HasExplicitValue()) {
      return ws.NumInput();
    } else {
      return in_idx_[out_sample_idx].shape[0];
    }
  }

  inline ConstSampleView<Backend> GetPasteSample(const Workspace &ws, int out_sample_idx,
                                                 int paste_idx) const {
    if (in_idx_.HasExplicitValue()) {
      int in_sample_idx = in_idx_[out_sample_idx].data[paste_idx];
      return ws.Input<Backend>(0)[in_sample_idx];
    } else {
      return ws.Input<Backend>(paste_idx)[out_sample_idx];
    }
  }

  inline ivec2 GetInputShape(const Workspace &ws, int out_sample_idx, int paste_idx) const {
    ivec2 shape;
    const auto &sample = GetPasteSample(ws, out_sample_idx, paste_idx);
    const auto& sample_shape = sample.shape();
    assert(sample_shape.size() >= spatial_ndim_);
    for (int d = 0; d < spatial_ndim_; d++) {
        shape[d] = sample_shape[d];
    }
    return shape;
  }

  USE_OPERATOR_MEMBERS();
  DALIDataType output_type_arg_ = DALI_NO_TYPE;
  DALIDataType output_type_ = DALI_NO_TYPE;
  DALIDataType input_type_ = DALI_NO_TYPE;

  ArgValue<int, 1> output_size_;
  ArgValue<int, 1> in_idx_;
  ArgValue<int, 2> in_anchors_;
  ArgValue<float, 2> in_anchors_rel_;
  std::vector<std::vector<ivec2>> in_anchors_data_;
  ArgValue<int, 2> shapes_;
  ArgValue<float, 2> shapes_rel_;
  std::vector<std::vector<ivec2>> region_shapes_data_;
  ArgValue<int, 2> out_anchors_;
  ArgValue<float, 2> out_anchors_rel_;
  std::vector<std::vector<ivec2>> out_anchors_data_;

  const int spatial_ndim_ = 2;
  const TensorShape<> coords_sh_ = TensorShape<>(2);

  kernels::KernelManager kernel_manager_;
};


class MultiPasteCPU : public MultiPasteOp<CPUBackend, MultiPasteCPU> {
 public:
  explicit MultiPasteCPU(const OpSpec &spec) : MultiPasteOp(spec) {}

  bool HasIntersections(const Workspace &ws, int sample_idx) {
    int paste_count = GetPasteCount(ws, sample_idx);
    for (int i = 0; i < paste_count; i++) {
      auto out_anchor_i = out_anchors_data_[sample_idx][i];
      auto shape_i = region_shapes_data_[sample_idx][i];
      for (int j = 0; j < i; j++) {
        auto out_anchor_j = out_anchors_data_[sample_idx][j];
        auto shape_j = region_shapes_data_[sample_idx][j];
        if (Intersects(out_anchor_i, shape_i, out_anchor_j, shape_j)) {
          return true;
        }
      }
    }
    return false;
  }

 private:
  template<typename OutputType, typename InputType>
  void RunTyped(Workspace &ws);

  template<typename OutputType, typename InputType>
  void SetupTyped(const Workspace &ws,
                  const TensorListShape<> &out_shape);

  template <typename OutputType, typename InputType>
  void CopyPatch(TensorListView<StorageCPU, OutputType, 3> &out_view,
                 std::vector<TensorListView<StorageCPU, const InputType, 3>> &in_views,
                 int sample_idx, int paste_idx);

  friend class MultiPasteOp<CPUBackend, MultiPasteCPU>;
};

class MultiPasteGPU : public MultiPasteOp<GPUBackend, MultiPasteGPU> {
 public:
  explicit MultiPasteGPU(const OpSpec &spec) : MultiPasteOp(spec) {}

 private:
  template<typename OutputType, typename InputType>
  void RunTyped(Workspace &ws);

  template<typename OutputType, typename InputType>
  void SetupTyped(const Workspace &ws,
                  const TensorListShape<> &out_shape);

  void InitSamples(const Workspace &ws, const TensorListShape<> &out_shape);


  vector<kernels::paste::MultiPasteSampleInput<2>> samples_;

  friend class MultiPasteOp<GPUBackend, MultiPasteGPU>;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_

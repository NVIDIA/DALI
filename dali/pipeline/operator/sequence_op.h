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

#ifndef DALI_PIPELINE_OPERATOR_SEQUENCE_OP_H_
#define DALI_PIPELINE_OPERATOR_SEQUENCE_OP_H_

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/argument.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename T>
struct is_shared_ptr : std::false_type {};
template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

struct ExpandDesc {
  ExpandDesc() = default;
  ExpandDesc(const TensorListShape<> &shape, const TensorLayout &layout, bool expand_channels) {
    if (!expand_channels) {
      if (VideoLayoutInfo::FrameDimIndex(layout) == 0) {
        frame_dim_ = 0;
      }
    } else {
      int frames_dim = VideoLayoutInfo::FrameDimIndex(layout);
      int channel_dim = ImageLayoutInfo::ChannelDimIndex(layout);
      frames_dim = frames_dim > 1 ? -1 : frames_dim;
      channel_dim = channel_dim > 1 ? -1 : channel_dim;
      if (frames_dim == 0 || channel_dim == 0) {
        frame_dim_ = frames_dim;
        channel_dim_ = channel_dim;
      }
    }
    int num_dims = NumExpandDims();
    if (num_dims > 0) {
      expanded_layout_ = layout.first(num_dims);
      initial_shape_ = shape.first(num_dims);
      num_elements_ = initial_shape_.num_elements();
    }
  }

  int NumExpandDims() const {
    return (frame_dim_ >= 0) + (channel_dim_ >= 0);
  }

  int NumElements() const {
    return num_elements_;
  }

  bool ShouldExpand() const {
    return NumExpandDims() > 0;
  }

  bool HasChannels() const {
    return (channel_dim_ >= 0);
  }

  bool HasFrames() const {
    return (frame_dim_ >= 0);
  }

  bool IsChannelFirst() const {
    return HasChannels() && channel_dim_ < frame_dim_;
  }

  template <typename Backend>
  TensorVector<Backend> Unfold(const TensorVector<Backend> &data) const {
    return Unfold(data, NumExpandDims());
  }

  template <typename Backend>
  TensorList<Backend> Unfold(const TensorList<Backend> &data) const {
    return Unfold(data, NumExpandDims());
  }

  template <typename Backend>
  TensorVector<Backend> Unfold(const TensorVector<Backend> &data, int num_unfold_dims) const {
    const auto &initial_shape = data.shape();
    auto unfold_shape = initial_shape.first(num_unfold_dims);
    const auto target_num_samples = unfold_shape.num_elements();
    TensorVector<Backend> flat_tensor(target_num_samples);
    int elem_idx = 0;
    auto type_info = data.type_info();
    auto type = type_info.id();
    auto is_pinned = data.is_pinned();
    auto order = data.order();
    flat_tensor.set_type(type);
    flat_tensor.set_pinned(is_pinned);
    flat_tensor.set_order(order);
    for (int sample_idx = 0; sample_idx < initial_shape.num_samples(); sample_idx++) {
      const auto &sample_shape = initial_shape[sample_idx];
      int num_frames = volume(sample_shape.begin(), sample_shape.begin() + num_unfold_dims);
      TensorShape<> element_shape{sample_shape.begin() + num_unfold_dims, sample_shape.end()};
      int element_volume = volume(element_shape);
      auto num_bytes = type_info.size() * element_volume;
      uint8_t *base_ptr;
      base_ptr = const_cast<uint8_t *>(static_cast<const uint8_t *>(data.raw_tensor(sample_idx)));
      for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        uint8_t *ptr = base_ptr + frame_idx * num_bytes;
        flat_tensor[elem_idx++].ShareData(ptr, num_bytes, is_pinned, element_shape, type, order);
      }
    }
    DALI_ENFORCE(elem_idx == target_num_samples, "TODO: change me to assert");
    return flat_tensor;
  }

  TensorListShape<> Unfold(const TensorListShape<> &shape, int num_unfold_dims) const {
    DALI_ENFORCE(num_unfold_dims >= 0 && num_unfold_dims < shape.sample_dim(),
                 "TODO: maybe just an assert?");
    if (num_unfold_dims == 0) {
      return shape;
    } else if (num_unfold_dims == 1) {
      return unfold_outer_dim(shape);
    } else {
      auto data_shape = collapse_dims(shape, {{0, num_unfold_dims}});
      return unfold_outer_dim(data_shape);
    }
  }

  template <typename Backend>
  TensorList<Backend> Unfold(const TensorList<Backend> &data, int num_unfold_dims) const {
    TensorList<Backend> tl;
    tl.ShareData(data);
    auto shape = Unfold(data.shape(), num_unfold_dims);
    tl.Resize(shape, data.type());
    return tl;
  }

  TensorListShape<> FoldLike(const TensorListShape<> &shape) const {
    auto num_samples = shape.num_samples();
    auto num_groups = initial_shape_.num_samples();
    DALI_ENFORCE(num_samples == NumElements());
    TensorListShape<> res(num_groups, initial_shape_.sample_dim() + shape.sample_dim());
    int sample_offset = 0;
    for (int i = 0; i < num_groups; i++) {
      const auto &frame_shape = shape[sample_offset];
      res.set_tensor_shape(i, shape_cat(initial_shape_[i], frame_shape));
      auto num_frames = volume(initial_shape_[i]);
      for (int j = 1; j < num_frames; j++) {
        DALI_ENFORCE(shape[sample_offset + j] == frame_shape);
      }
      sample_offset += num_frames;
    }
    return res;
  }

  TensorVector<CPUBackend> SpreadTensorArgumentLike(const TensorVector<CPUBackend> &tv,
                                                    const std::string &name) const {
    const auto &layout = tv.GetLayout();
    bool argument_has_frames_dim = VideoLayoutInfo::FrameDimIndex(layout) == 0;
    DALI_ENFORCE(
        !argument_has_frames_dim || HasFrames(),
        "Argument input contains frames but the input of the operator does not. How ridiculous.");
    DALI_ENFORCE(initial_shape_.sample_dim() >= NumExpandDims(),
                 "Inital shape must be greater than number of expanded dims");
    TensorVector<CPUBackend> flat_tensor(NumElements());
    int arg_sample_dim = tv.sample_dim();
    DALI_ENFORCE(!argument_has_frames_dim || arg_sample_dim >= 1,
                 "No frames found for per-frame argument");  // TODO(ktokarski) adjust the message
    int sample_offset = 0;
    for (int sample_idx = 0; sample_idx < initial_shape_.num_samples(); sample_idx++) {
      const auto &arg_tensor = tv[sample_idx];
      const auto &arg_shape = arg_tensor.shape();
      int num_input_frames = !HasFrames() ? 1 : initial_shape_[sample_idx][frame_dim_];
      int num_arg_frames = !argument_has_frames_dim ? 1 : arg_shape[0];
      auto elem_shape = argument_has_frames_dim ? arg_shape.last(arg_sample_dim - 1) : arg_shape;
      DALI_ENFORCE(
          num_arg_frames == 1 || num_input_frames == num_arg_frames,
          make_string("The tensor argument ", name, " for sample ", sample_idx,
                      " for an input that contains frames, should either be a single set "
                      "of parameters to be reused accross all frames in the sample all match the "
                      "number of frames. Got ",
                      num_arg_frames, " but there are ", num_input_frames,
                      " frames in the sample."));
      uint8_t *base_ptr =
          const_cast<uint8_t *>(static_cast<const uint8_t *>(arg_tensor.raw_data()));
      auto elem_volume = volume(elem_shape);
      auto type_info = arg_tensor.type_info();
      auto num_bytes = type_info.size() * elem_volume;
      auto type = type_info.id();
      auto is_pinned = arg_tensor.is_pinned();
      auto order = arg_tensor.order();
      int num_input_channels = !HasChannels() ? 1 : initial_shape_[sample_idx][channel_dim_];
      int num_repeat =
          num_arg_frames == 1 ? num_input_channels * num_input_frames : num_input_channels;
      int inner_stride, outer_stride;
      if (num_arg_frames == 1) {
        inner_stride = 1;
        outer_stride = 1;
      } else if (IsChannelFirst()) {
        inner_stride = num_input_frames;
        outer_stride = 1;
      } else {
        inner_stride = 1;
        outer_stride = num_input_channels;
      }
      for (int i = 0; i < num_arg_frames; i++) {  // expands argument
        uint8_t *ptr = base_ptr + i * num_bytes;
        for (int j = 0; j < num_repeat; j++) {  // repeats argument
          flat_tensor[sample_offset + i * outer_stride + j * inner_stride].ShareData(
              ptr, num_bytes, is_pinned, elem_shape, type, order);
        }
      }
      sample_offset += num_input_channels * num_input_frames;
    }
    DALI_ENFORCE(sample_offset == num_elements_,
                 make_string("TODO: change me to assert", sample_offset, " ", num_elements_, " ",
                             initial_shape_));
    return flat_tensor;
  }

  TensorLayout TrimFlattenedLayout(const TensorLayout &layout) const {
    return layout.sub(NumExpandDims());
  }

  TensorLayout AppendFlattenedLayout(const TensorLayout &layout) const {
    return expanded_layout_ + layout;
  }

  int frame_dim_ = -1;
  int channel_dim_ = -1;
  TensorLayout expanded_layout_ = {};
  TensorListShape<> initial_shape_ = {};
  int num_elements_;
};

template <typename Backend>
class SequenceOperator : public Operator<Backend> {
 public:
  inline explicit SequenceOperator(const OpSpec &spec) : Operator<Backend>{spec} {}

  using Operator<Backend>::Setup;
  using Operator<Backend>::Run;

  template <typename InputBackend>
  using ws_input_t = typename workspace_t<Backend>::template input_t<InputBackend>;
  template <typename OutputBackend>
  using ws_output_t = typename workspace_t<Backend>::template output_t<OutputBackend>;

  bool Setup(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    SetupSequenceOperator(ws);
    expand_ = ShouldExpand(ws);
    bool is_inferred;
    if (!expand_) {
      is_inferred = Operator<Backend>::Setup(output_desc, ws);
    } else {
      SetupExpandedWorkspace(expanded_, ws);
      ExpandInputs(ws);
      ExpandArguments(ws);
      is_inferred = Operator<Backend>::Setup(output_desc, expanded_);
    }
    return ProcessOutputDesc(output_desc, ws, is_inferred);
  }

  void Run(workspace_t<Backend> &ws) override {
    if (!expand_) {
      Operator<Backend>::Run(ws);
    } else {
      ExpandOutputs(ws);
      Operator<Backend>::Run(expanded_);
      SetOutputLayouts(ws);
      expanded_.Clear();
    }
  }

  bool IsExpanded() const {
    return expand_;
  }

  virtual bool ShouldExpandChannels() const {
    return true;
  }

 protected:
  void SetupExpandedWorkspace(workspace_t<CPUBackend> &expanded,
                              const workspace_t<CPUBackend> &ws) {
    expanded.SetThreadPool(&ws.GetThreadPool());
  }

  void SetupExpandedWorkspace(workspace_t<GPUBackend> &expanded,
                              const workspace_t<GPUBackend> &ws) {
    if (ws.has_stream()) {
      expanded.set_stream(ws.stream());
    }
  }

  virtual bool ShouldExpand(const workspace_t<Backend> &ws) {
    (void)ws;
    return std::any_of(input_expand_desc_.begin(), input_expand_desc_.end(),
                       [](const ExpandDesc &desc) { return desc.ShouldExpand(); });
  }

  virtual void ExpandInput(const workspace_t<Backend> &ws, int input_idx) {
    const auto &expand_desc = input_expand_desc_[input_idx];
    if (!expand_desc.ShouldExpand()) {
      AddUnchangedInputHelper(expanded_, ws, input_idx);
    } else if (ws.template InputIsType<GPUBackend>(input_idx)) {
      ExpandInputHelper<GPUBackend>(ws, input_idx, expand_desc);
    } else {
      ExpandInputHelper<CPUBackend>(ws, input_idx, expand_desc);
    }
  }

  virtual void ExpandInputs(const workspace_t<Backend> &ws) {
    for (int input_idx = 0; input_idx < ws.NumInput(); input_idx++) {
      ExpandInput(ws, input_idx);
    }
    DALI_ENFORCE(ws.NumInput() == expanded_.NumInput());
  }

  virtual const ExpandDesc &GetOutputExpandDesc(const workspace_t<Backend> &ws, int output_idx) {
    return input_expand_desc_[output_idx];
  }

  virtual void ExpandOutput(const workspace_t<Backend> &ws, int output_idx) {
    const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
    if (!expand_desc.ShouldExpand()) {
      AddUnchangedOutputHelper(expanded_, ws, output_idx);
    } else if (ws.template OutputIsType<GPUBackend>(output_idx)) {
      ExpandOutputHelper<GPUBackend>(ws, output_idx, expand_desc);
    } else {
      ExpandOutputHelper<CPUBackend>(ws, output_idx, expand_desc);
    }
  }

  virtual void ExpandOutputs(const workspace_t<Backend> &ws) {
    for (int output_idx = 0; output_idx < ws.NumInput(); output_idx++) {
      ExpandOutput(ws, output_idx);
    }
    DALI_ENFORCE(ws.NumOutput() == expanded_.NumOutput());
  }

  virtual const ExpandDesc &GetArgExpandDesc(const std::string &arg_name) {
    (void)arg_name;
    DALI_ENFORCE(input_expand_desc_.size() > 0,
                 "Cannot expand argument tensor input without matching input tensor");
    return input_expand_desc_[0];
  }

  virtual void ExpandArgment(const workspace_t<Backend> &ws, const std::string &arg_name,
                             const TensorVector<CPUBackend> &arg_tensor) {
    const auto &expand_desc = GetArgExpandDesc(arg_name);
    const auto &tv = ws.ArgumentInput(arg_name);
    auto expanded_arg = expand_desc.SpreadTensorArgumentLike(tv, arg_name);
    auto expanded_handle = std::make_shared<TensorVector<CPUBackend>>(std::move(expanded_arg));
    expanded_.AddArgumentInput(arg_name, expanded_handle);
  }

  virtual void ExpandArguments(const workspace_t<Backend> &ws) {
    for (const auto &arg_input : ws) {
      auto shared_tvec = arg_input.second.tvec;
      if (!shared_tvec) {
        continue;
      }
      ExpandArgment(ws, arg_input.first, *shared_tvec);
    }
  }

  virtual void SetOutputLayout(workspace_t<Backend> &ws, int output_idx) {
    const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
    if (!expand_desc.ShouldExpand()) {
      return;
    }
    if (expanded_.template OutputIsType<GPUBackend>(output_idx)) {
      SetOutputLayoutHelper<GPUBackend>(ws, output_idx, expand_desc);
    } else {
      SetOutputLayoutHelper<CPUBackend>(ws, output_idx, expand_desc);
    }
  }

  virtual void SetOutputLayouts(workspace_t<Backend> &ws) {
    auto num_output = ws.NumOutput();
    DALI_ENFORCE(num_output == expanded_.NumOutput());
    for (int output_idx = 0; output_idx < num_output; output_idx++) {
      SetOutputLayout(ws, output_idx);
    }
  }

  virtual void CoalesceOutputShapes(std::vector<OutputDesc> &output_desc,
                                    const workspace_t<Backend> &ws) {
    DALI_ENFORCE(output_desc.size() == static_cast<size_t>(ws.NumOutput()));
    for (size_t output_idx = 0; output_idx < output_desc.size(); output_idx++) {
      const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
      output_desc[output_idx].shape = expand_desc.FoldLike(output_desc[output_idx].shape);
    }
  }

  virtual bool ProcessOutputDesc(std::vector<OutputDesc> &output_desc,
                                 const workspace_t<Backend> &ws, bool is_inferred) {
    if (is_inferred && expand_) {
      CoalesceOutputShapes(output_desc, ws);
    }
    return is_inferred;
  }

 private:
  void SetupSequenceOperator(const workspace_t<Backend> &ws) {
    auto num_inputs = ws.NumInput();
    input_expand_desc_.resize(num_inputs);
    for (int input_idx = 0; input_idx < num_inputs; input_idx++) {
      const auto &input_shape = ws.GetInputShape(input_idx);
      const auto &layout = GetInputLayout(ws, input_idx);
      DALI_ENFORCE(layout.empty() || input_shape.sample_dim() == layout.size());
      input_expand_desc_[input_idx] = {input_shape, layout, ShouldExpandChannels()};
    }
  }

  void AddUnchangedInputHelper(workspace_t<Backend> &expanded, const workspace_t<Backend> &ws,
                               int input_idx) {
    if (ws.template InputIsType<GPUBackend>(input_idx)) {
      auto input = ws.template InputPtr<GPUBackend>(input_idx);
      expanded.AddInput(input);
    } else {
      auto input = ws.template InputPtr<CPUBackend>(input_idx);
      expanded.AddInput(input);
    }
  }

  void AddUnchangedOutputHelper(workspace_t<Backend> &expanded, const workspace_t<Backend> &ws,
                                int output_idx) {
    if (ws.template OutputIsType<GPUBackend>(output_idx)) {
      auto output = ws.template OutputPtr<GPUBackend>(output_idx);
      expanded.AddOutput(output);
    } else {
      auto output = ws.template OutputPtr<CPUBackend>(output_idx);
      expanded.AddOutput(output);
    }
  }

  template <typename InputBackend>
  void ExpandInputHelper(const workspace_t<Backend> &ws, int input_idx,
                         const ExpandDesc &expand_desc) {
    const auto &input = ws.template Input<InputBackend>(input_idx);
    auto expanded_input = expand_desc.Unfold(input);
    auto layout = expand_desc.TrimFlattenedLayout(input.GetLayout());
    expanded_input.SetLayout(layout);
    using input_handle_t = ws_input_t<InputBackend>;
    static_assert(is_shared_ptr<input_handle_t>::value,
                  "Workspace input handle expected to be shared_ptr");
    auto expaned_handle =
        std::make_shared<typename input_handle_t::element_type>(std::move(expanded_input));
    expanded_.AddInput(expaned_handle);
  }

  template <typename OutputBackend>
  void ExpandOutputHelper(const workspace_t<Backend> &ws, int output_idx,
                          const ExpandDesc &expand_desc) {
    const auto &output = ws.template Output<OutputBackend>(output_idx);
    auto expanded_output = expand_desc.Unfold(output);
    using output_handle_t = ws_output_t<OutputBackend>;
    static_assert(is_shared_ptr<output_handle_t>::value,
                  "Workspace output handle expected to be shared_ptr");
    expanded_.AddOutput(
        std::make_shared<typename output_handle_t::element_type>(std::move(expanded_output)));
  }

  template <typename OutputBackend>
  void SetOutputLayoutHelper(const workspace_t<Backend> &ws, int output_idx,
                             const ExpandDesc &expand_desc) {
    auto &expanded_output = expanded_.template Output<OutputBackend>(output_idx);
    auto &output = ws.template Output<OutputBackend>(output_idx);
    const auto &layout = expanded_output.GetLayout();
    if (layout.size() == 0) {
      output.SetLayout(layout);
    } else {
      auto expanded_layout = expand_desc.AppendFlattenedLayout(layout);
      output.SetLayout(expanded_layout);
    }
  }

  bool expand_;
  std::vector<ExpandDesc> input_expand_desc_;
  workspace_t<Backend> expanded_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_OP_H_

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
#include "dali/pipeline/operator/sequence_info.h"
#include "dali/pipeline/operator/sequence_shape.h"

namespace dali {

template <typename Backend>
class SequenceOperator : public Operator<Backend> {
 public:
  inline explicit SequenceOperator(const OpSpec &spec) : Operator<Backend>{spec} {}

  using Operator<Backend>::Setup;
  using Operator<Backend>::Run;

  template <typename T>
  struct is_shared_ptr : std::false_type {};
  template <typename T>
  struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

  template <typename InputBackend>
  using ws_input_t = typename workspace_t<Backend>::template input_t<InputBackend>;
  template <typename OutputBackend>
  using ws_output_t = typename workspace_t<Backend>::template output_t<OutputBackend>;

  bool Setup(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    expand_ = ShouldExpand(ws);
    bool is_inferred;
    if (!expand_) {
      is_inferred = Operator<Backend>::Setup(output_desc, ws);
    } else {
      SetupSequenceOperator(ws);
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
      PostprocessOutputs(ws);
      Clear();
    }
  }

  bool IsExpanded() const {
    return expand_;
  }

  const SampleFrameCtx &GetSampleFrameCtx() {
    return sample_ctx_;
  }

  virtual bool ShouldExpandChannels() const {
    return true;
  }

 protected:
  virtual bool ShouldExpand(const workspace_t<Backend> &ws) {
    auto num_inputs = ws.NumInput();
    for (int input_idx = 0; input_idx < num_inputs; input_idx++) {
      const auto &layout = GetInputLayout(ws, input_idx);
      auto layout_desc =
          ShouldExpandChannels() ? LayoutDesc::FrameAndChannel(layout) : LayoutDesc::Frame(layout);
      if (layout_desc.NumDims() > 0) {
        return true;
      }
    }
    return false;
  }

  virtual void SetupSequenceOperator(const workspace_t<Backend> &ws) {
    auto num_inputs = ws.NumInput();
    input_expand_desc_.reserve(num_inputs);
    for (int input_idx = 0; input_idx < num_inputs; input_idx++) {
      const auto &input_shape = ws.GetInputShape(input_idx);
      const auto &layout = GetInputLayout(ws, input_idx);
      auto layout_desc =
          ShouldExpandChannels() ? LayoutDesc::FrameAndChannel(layout) : LayoutDesc::Frame(layout);
      input_expand_desc_.emplace_back(input_shape, layout_desc);
    }
  }

  virtual void ExpandInputs(const workspace_t<Backend> &ws) {
    for (int input_idx = 0; input_idx < ws.NumInput(); input_idx++) {
      const auto &expand_desc = GetInputExpandDesc(input_idx);
      auto num_expand_dims = expand_desc.NumDims();
      if (ws.template InputIsType<GPUBackend>(input_idx)) {
        auto expanded_handle = ExpandInputHelper<GPUBackend>(ws, input_idx, num_expand_dims);
        AddInputHelper<GPUBackend>(expanded_handle, {ExpandDescFrameInfoFn{&expand_desc}});
      } else {
        auto expanded_handle = ExpandInputHelper<CPUBackend>(ws, input_idx, num_expand_dims);
        AddInputHelper<CPUBackend>(expanded_handle, {ExpandDescFrameInfoFn{&expand_desc}});
      }
    }
  }

  virtual void ExpandOutputs(const workspace_t<Backend> &ws) {
    for (int output_idx = 0; output_idx < ws.NumInput(); output_idx++) {
      const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
      auto num_expand_dims = expand_desc.NumDims();
      if (ws.template OutputIsType<GPUBackend>(output_idx)) {
        auto expanded_handle = ExpandOutputHelper<GPUBackend>(ws, output_idx, num_expand_dims);
        AddOutputHelper<GPUBackend>(expanded_handle);
      } else {
        auto expanded_handle = ExpandOutputHelper<CPUBackend>(ws, output_idx, num_expand_dims);
        AddOutputHelper<CPUBackend>(expanded_handle);
      }
    }
  }

  virtual void PostprocessOutputs(workspace_t<Backend> &ws) {
    auto num_output = ws.NumOutput();
    for (int output_idx = 0; output_idx < num_output; output_idx++) {
      const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
      auto layout_prefix = expand_desc.ExpandedLayout();
      if (ws.template OutputIsType<GPUBackend>(output_idx)) {
        SetOutputLayoutHelper<GPUBackend>(ws, output_idx, layout_prefix);
      } else {
        SetOutputLayoutHelper<CPUBackend>(ws, output_idx, layout_prefix);
      }
    }
  }

  virtual void CoalesceOutputShapes(std::vector<OutputDesc> &output_desc,
                                    const workspace_t<Backend> &ws) {
    for (size_t output_idx = 0; output_idx < output_desc.size(); output_idx++) {
      const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
      const auto &shape = output_desc[output_idx].shape;
      DALI_ENFORCE(
          shape.num_samples() == expand_desc.NumElements(),
          make_string("Unexpected number of samples or frames inferred in the operator for output ",
                      output_idx, ". Expected ", expand_desc.NumElements(),
                      " but the operator returned ", shape.num_samples(), "."));
      output_desc[output_idx].shape = fold_outermost_like(shape, expand_desc.ExpandedShape());
    }
  }

  virtual bool ProcessOutputDesc(std::vector<OutputDesc> &output_desc,
                                 const workspace_t<Backend> &ws, bool is_inferred) {
    if (is_inferred && expand_) {
      CoalesceOutputShapes(output_desc, ws);
    }
    return is_inferred;
  }

  virtual void ExpandArgment(const workspace_t<Backend> &ws, const std::string &arg_name,
                             const TensorVector<CPUBackend> &arg_tensor) {
    const auto &expand_desc = GetArgExpandDesc(arg_name);
    const auto &tv = ws.ArgumentInput(arg_name);
    auto expanded_arg = ExpandArgumentLike(tv, arg_name, expand_desc);
    auto expanded_handle = std::make_shared<TensorVector<CPUBackend>>(std::move(expanded_arg));
    AddArgumentInputHelper(arg_name, expanded_handle, {ExpandDescFrameInfoFn{&expand_desc}});
  }

  virtual void ExpandArguments(const workspace_t<Backend> &ws) {
    for (const auto &arg_input : ws) {
      auto shared_tvec = arg_input.second.tvec;
      assert(shared_tvec);
      ExpandArgment(ws, arg_input.first, *shared_tvec);
    }
  }

  virtual const ExpandDesc &GetOutputExpandDesc(const workspace_t<Backend> &ws, int output_idx) {
    (void)ws;
    return GetInputExpandDesc(output_idx);
  }

  virtual const ExpandDesc &GetArgExpandDesc(const std::string &arg_name) {
    (void)arg_name;
    return GetInputExpandDesc(0);
  }

  template <typename InputBackend>
  ws_input_t<InputBackend> ExpandInputHelper(const workspace_t<Backend> &ws, int input_idx,
                                             int num_expand_dims) {
    static_assert(is_shared_ptr<ws_input_t<InputBackend>>::value,
                  "Workspace input handle expected to be shared_ptr");
    const auto &input = ws.template Input<InputBackend>(input_idx);
    auto sample_dim = input.shape().sample_dim();
    DALI_ENFORCE(
        sample_dim > num_expand_dims,
        make_string("Cannot flatten the input ", input_idx,
                    ". Samples should have more dimensions (got ", sample_dim,
                    ") than requested number of dimensions to unfold: ", num_expand_dims, "."));
    // TODO(ktokarski) TODO(klecki)
    // Rework it when TensorList stops being contigious and supports "true sample" mode
    auto expanded_input = unfold_outer_dims(input, num_expand_dims);
    const auto &input_layout = input.GetLayout();
    expanded_input.SetLayout(input_layout.sub(num_expand_dims));
    return std::make_shared<typename ws_input_t<InputBackend>::element_type>(
        std::move(expanded_input));
  }

  template <typename OutputBackend>
  ws_input_t<OutputBackend> ExpandOutputHelper(const workspace_t<Backend> &ws, int output_idx,
                                               int num_expand_dims) {
    static_assert(is_shared_ptr<ws_input_t<OutputBackend>>::value,
                  "Workspace output handle expected to be shared_ptr");
    const auto &output = ws.template Output<OutputBackend>(output_idx);
    auto sample_dim = output.shape().sample_dim();
    DALI_ENFORCE(
        sample_dim > num_expand_dims,
        make_string("Cannot flatten the output ", output_idx,
                    ". Samples should have more dimensions (got ", sample_dim,
                    ") than requested number of dimensions to unfold: ", num_expand_dims, "."));
    // TODO(ktokarski) TODO(klecki)
    // Rework it when TensorList stops being contigious and supports "true sample" mode
    auto expanded_output = unfold_outer_dims(output, num_expand_dims);
    return std::make_shared<typename ws_input_t<OutputBackend>::element_type>(
        std::move(expanded_output));
  }

  template <typename OutputBackend>
  void SetOutputLayoutHelper(const workspace_t<Backend> &ws, int output_idx,
                             TensorLayout layout_prefix) {
    DALI_ENFORCE(ExpandedOutputIsType<OutputBackend>(output_idx));
    auto &expanded_output = ExpandedOutput<OutputBackend>(output_idx);
    auto &output = ws.template Output<OutputBackend>(output_idx);
    const auto &layout = expanded_output.GetLayout();
    if (layout.size() == 0) {
      output.SetLayout(layout);
    } else {
      auto expanded_layout = layout_prefix + layout;
      output.SetLayout(expanded_layout);
    }
  }

  const ExpandDesc &GetInputExpandDesc(int input_idx) {
    DALI_ENFORCE(expand_ && 0 <= input_idx &&
                 static_cast<size_t>(input_idx) < input_expand_desc_.size());
    return input_expand_desc_[input_idx];
  }

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

  template <typename InputBackend>
  void AddInputHelper(ws_input_t<InputBackend> input, SampleFrameCtx::FrameInfoFn fn) {
    expanded_.AddInput(input);
    sample_ctx_.AddInputInfoFn(fn);
  }

  void AddArgumentInputHelper(const std::string &arg_name,
                              std::shared_ptr<TensorVector<CPUBackend>> arg_input,
                              SampleFrameCtx::FrameInfoFn fn) {
    expanded_.AddArgumentInput(arg_name, arg_input);
    sample_ctx_.AddArgumentInputInfoFn(arg_name, fn);
  }

  template <typename OutputBackend>
  void AddOutputHelper(ws_input_t<OutputBackend> output) {
    expanded_.AddOutput(output);
  }

  template <typename InputBackend>
  bool ExpandedInputIsType(int input_idx) const {
    return expanded_.template InputIsType<InputBackend>(input_idx);
  }

  template <typename OutputBackend>
  bool ExpandedOutputIsType(int output_idx) const {
    return expanded_.template OutputIsType<OutputBackend>(output_idx);
  }

  template <typename InputBackend>
  const auto &ExpandedInput(int intput_idx) {
    return expanded_.template Input<InputBackend>(intput_idx);
  }

  template <typename OutputBackend>
  const auto &ExpandedOutput(int output_idx) {
    return expanded_.template Output<OutputBackend>(output_idx);
  }

  void Clear() {
    expanded_.Clear();
    sample_ctx_.Clear();
    input_expand_desc_.clear();
  }

  TensorVector<CPUBackend> ExpandArgumentLike(const TensorVector<CPUBackend> &tv,
                                              const std::string &name,
                                              const ExpandDesc &expand_desc) {
    const auto &layout = tv.GetLayout();
    bool arg_has_frames_dim = VideoLayoutInfo::FrameDimIndex(layout) == 0;
    int arg_sample_dim = tv.sample_dim();
    DALI_ENFORCE(
        !arg_has_frames_dim || expand_desc.HasFrames(),
        make_string(
            "Tensor input for argument ", name, " is specified per frame (got ", layout,
            " layout), but samples in the input batch do not contain frames (expected input "
            "layout that starts with 'F', ", ShouldExpandChannels()? " or 'CF'" : "", ")."));
    DALI_ENFORCE(!arg_has_frames_dim || arg_sample_dim >= 1,
                 make_string("The layout of tensor argument ", name,
                             "contains frames, but the tensor has 0 dimensionality."));
    auto num_elements = expand_desc.NumElements();
    int num_samples = expand_desc.NumSamples();
    TensorVector<CPUBackend> flat_tensor(num_elements);
    int sample_offset = 0;
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto &arg_tensor = tv[sample_idx];
      const auto &arg_shape = arg_tensor.shape();
      int num_input_frames = expand_desc.HasFrames() ? expand_desc.NumFrames(sample_idx) : 1;
      int num_arg_frames = !arg_has_frames_dim ? 1 : arg_shape[0];
      auto elem_shape = arg_has_frames_dim ? arg_shape.last(arg_sample_dim - 1) : arg_shape;
      DALI_ENFORCE(
          num_arg_frames == 1 || num_input_frames == num_arg_frames,
          make_string("The tensor argument ", name, " for sample ", sample_idx,
                      " should either be a single argument to be reused accross all frames in the "
                      "sample or should be specified per each frame. Got ",
                      num_arg_frames, " arguments for the sample but there are ", num_input_frames,
                      " frames in the sample."));
      uint8_t *base_ptr =
          const_cast<uint8_t *>(static_cast<const uint8_t *>(arg_tensor.raw_data()));
      auto elem_volume = volume(elem_shape);
      auto type_info = arg_tensor.type_info();
      auto num_bytes = type_info.size() * elem_volume;
      auto type = type_info.id();
      auto is_pinned = arg_tensor.is_pinned();
      auto order = arg_tensor.order();
      int num_input_channels = expand_desc.HasChannels() ? expand_desc.NumChannels(sample_idx) : 1;
      int num_repeat =
          num_arg_frames == 1 ? num_input_channels * num_input_frames : num_input_channels;
      int inner_stride, outer_stride;
      if (num_arg_frames == 1) {
        inner_stride = 1;
        outer_stride = 1;
      } else if (expand_desc.IsChannelFirst()) {
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
    assert(sample_offset == num_elements);
    return flat_tensor;
  }

  std::vector<ExpandDesc> input_expand_desc_;

 private:
  bool expand_;
  workspace_t<Backend> expanded_;
  SampleFrameCtx sample_ctx_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_OP_H_

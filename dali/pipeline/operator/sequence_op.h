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
#include "dali/pipeline/operator/sequence_workspace.h"

namespace dali {

template <typename T>
struct is_shared_ptr : std::false_type {};
template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

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
      SetupExpandedWorkspace(ws);
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
  void SetupExpandedWorkspace(const workspace_t<Backend> &ws) {
    std::vector<SampleFrameInfoFn> input_sample_info;
    for (const auto &expand_desc : input_expand_desc_) {
      ExpandDescFrameInfoFn fn{&expand_desc};
      input_sample_info.push_back({fn});
    }
    expanded_.SetFrameInfoFns(std::move(input_sample_info));
    SetupExpandedWorkspace(expanded_, ws);
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
    auto expanded_arg = SpreadTensorArgumentLike(tv, arg_name, expand_desc);
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
      output_desc[output_idx].shape = FoldLike(output_desc[output_idx].shape, expand_desc);
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
    auto expanded_input = Unfold(input, expand_desc.NumExpandDims());
    const auto &input_layout = input.GetLayout();
    expanded_input.SetLayout(input_layout.sub(expand_desc.NumExpandDims()));
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
    auto expanded_output = Unfold(output, expand_desc.NumExpandDims());
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
      auto expanded_layout = expand_desc.expanded_layout_ + layout;
      output.SetLayout(expanded_layout);
    }
  }

  bool expand_;
  std::vector<ExpandDesc> input_expand_desc_;
  SequenceWorkspaceView<Backend> expanded_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_OP_H_

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
      expanded_.Clear();
      input_expand_desc_.clear();
    }
  }

  bool IsExpanded() const {
    return expand_;
  }

  virtual bool ShouldExpandChannels() const {
    return true;
  }

 protected:
  virtual bool ShouldExpand(const workspace_t<Backend> &ws) {
    auto num_inputs = ws.NumInput();
    for (int input_idx = 0; input_idx < num_inputs; input_idx++) {
      const auto &layout = GetInputLayout(ws, input_idx);
      auto layout_desc = LayoutDesc::Create(layout, ShouldExpandChannels());
      if (layout_desc.NumDims() > 0) {
        return true;
      }
    }
    return false;
  }

  virtual void SetupSequenceOperator(const workspace_t<Backend> &ws) {
    auto num_inputs = ws.NumInput();
    input_expand_desc_.resize(num_inputs);
    for (int input_idx = 0; input_idx < num_inputs; input_idx++) {
      const auto &input_shape = ws.GetInputShape(input_idx);
      const auto &layout = GetInputLayout(ws, input_idx);
      DALI_ENFORCE(layout.empty() || input_shape.sample_dim() == layout.size());
      input_expand_desc_[input_idx] = {input_shape, layout, ShouldExpandChannels()};
    }
  }

  virtual void ExpandInputs(const workspace_t<Backend> &ws) {
    for (int input_idx = 0; input_idx < ws.NumInput(); input_idx++) {
      const auto &expand_desc = GetInputExpandDesc(input_idx);
      auto num_expand_dims = expand_desc.NumDims();
      if (ws.template InputIsType<GPUBackend>(input_idx)) {
        auto expanded_handle = ExpandInputHelper<GPUBackend>(ws, input_idx, num_expand_dims);
        expanded_.AddInput(expanded_handle, ExpandDescFrameInfoFn{&expand_desc});
      } else {
        auto expanded_handle = ExpandInputHelper<CPUBackend>(ws, input_idx, num_expand_dims);
        expanded_.AddInput(expanded_handle, ExpandDescFrameInfoFn{&expand_desc});
      }
    }
  }

  virtual const ExpandDesc &GetOutputExpandDesc(const workspace_t<Backend> &ws, int output_idx) {
    (void)ws;
    return GetInputExpandDesc(output_idx);
  }

  virtual void ExpandOutputs(const workspace_t<Backend> &ws) {
    for (int output_idx = 0; output_idx < ws.NumInput(); output_idx++) {
      const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
      auto num_expand_dims = expand_desc.NumDims();
      if (ws.template OutputIsType<GPUBackend>(output_idx)) {
        ExpandOutputHelper<GPUBackend>(ws, output_idx, num_expand_dims);
      } else {
        ExpandOutputHelper<CPUBackend>(ws, output_idx, num_expand_dims);
      }
    }
  }

  virtual void PostprocessOutputs(workspace_t<Backend> &ws) {
    auto num_output = ws.NumOutput();
    DALI_ENFORCE(num_output == expanded_.NumOutput());
    for (int output_idx = 0; output_idx < num_output; output_idx++) {
      const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
      auto layout_prefix = expand_desc.ExpandedLayout();
      if (expanded_.template OutputIsType<GPUBackend>(output_idx)) {
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
      output_desc[output_idx].shape =
          fold_outermost_like(output_desc[output_idx].shape, expand_desc);
    }
  }

  virtual bool ProcessOutputDesc(std::vector<OutputDesc> &output_desc,
                                 const workspace_t<Backend> &ws, bool is_inferred) {
    if (is_inferred && expand_) {
      CoalesceOutputShapes(output_desc, ws);
    }
    return is_inferred;
  }

  virtual const ExpandDesc &GetArgExpandDesc(const std::string &arg_name) {
    (void)arg_name;
    return GetInputExpandDesc(0);
  }

  virtual void ExpandArgment(const workspace_t<Backend> &ws, const std::string &arg_name,
                             const TensorVector<CPUBackend> &arg_tensor) {
    const auto &expand_desc = GetArgExpandDesc(arg_name);
    const auto &tv = ws.ArgumentInput(arg_name);
    auto expanded_arg = SpreadTensorArgumentLike(tv, arg_name, expand_desc);
    auto expanded_handle = std::make_shared<TensorVector<CPUBackend>>(std::move(expanded_arg));
    expanded_.AddArgumentInput(arg_name, expanded_handle, ExpandDescFrameInfoFn{&expand_desc});
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

  template <typename InputBackend>
  auto ExpandInputHelper(const workspace_t<Backend> &ws, int input_idx, int num_expand_dims) {
    using input_handle_t = ws_input_t<InputBackend>;
    static_assert(is_shared_ptr<input_handle_t>::value,
                  "Workspace input handle expected to be shared_ptr");
    const auto &input = ws.template Input<InputBackend>(input_idx);
    auto expanded_input = unfold_outer_dims(input, num_expand_dims);
    const auto &input_layout = input.GetLayout();
    expanded_input.SetLayout(input_layout.sub(num_expand_dims));
    return std::make_shared<typename input_handle_t::element_type>(std::move(expanded_input));
  }

  template <typename OutputBackend>
  void ExpandOutputHelper(const workspace_t<Backend> &ws, int output_idx, int num_expand_dims) {
    const auto &output = ws.template Output<OutputBackend>(output_idx);
    auto expanded_output = unfold_outer_dims(output, num_expand_dims);
    using output_handle_t = ws_output_t<OutputBackend>;
    static_assert(is_shared_ptr<output_handle_t>::value,
                  "Workspace output handle expected to be shared_ptr");
    expanded_.AddOutput(
        std::make_shared<typename output_handle_t::element_type>(std::move(expanded_output)));
  }

  template <typename OutputBackend>
  void SetOutputLayoutHelper(const workspace_t<Backend> &ws, int output_idx,
                             TensorLayout layout_prefix) {
    auto &expanded_output = expanded_.template Output<OutputBackend>(output_idx);
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

  bool expand_;
  std::vector<ExpandDesc> input_expand_desc_;
  SequenceWorkspaceView<Backend> expanded_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_OP_H_

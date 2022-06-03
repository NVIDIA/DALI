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

#ifndef DALI_PIPELINE_OPERATOR_SEQUENCE_OPERATOR_H_
#define DALI_PIPELINE_OPERATOR_SEQUENCE_OPERATOR_H_

#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "dali/kernels/common/scatter_gather.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/type_traits.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/argument.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_shape.h"

namespace dali {

namespace detail {
inline bool is_per_frame(const TensorVector<CPUBackend> &arg_input) {
  const auto &layout = arg_input.GetLayout();
  return layout.size() > 0 && layout[0] == 'F';
}
}  // namespace detail


// broadcasting on gpu requires some extra state, so separate broadcasting from the
// sequence operator and derive from appropriate specialization
template <typename Backend>
class SampleBroadcasting;

template <>
class SampleBroadcasting<CPUBackend> {
 protected:
  template <typename DataBackend>
  void BroadcastSamples(TensorVector<DataBackend> &expanded_batch,
                        const TensorVector<DataBackend> &batch, const ExpandDesc &expand_desc,
                        const ArgumentWorkspace &expanded) {
    (void)expanded;
    sequence_utils::broadcast_samples(expanded_batch, batch, expand_desc.NumExpanded(),
                                      expand_desc.DimsToExpand());
  }
};

template <>
class SampleBroadcasting<GPUBackend> : public SampleBroadcasting<CPUBackend> {
 protected:
  using SampleBroadcasting<CPUBackend>::BroadcastSamples;

  void BroadcastSamples(TensorList<GPUBackend> &expanded_batch, const TensorList<GPUBackend> &batch,
                        const ExpandDesc &expand_desc, const DeviceWorkspace &expanded) {
    sequence_utils::broadcast_samples(expanded_batch, batch, expand_desc.NumExpanded(),
                                      expand_desc.DimsToExpand(), scatter_gather_,
                                      expanded.stream());
  }

  void BroadcastSamples(TensorList<CPUBackend> &expanded_batch, const TensorList<CPUBackend> &batch,
                        const ExpandDesc &expand_desc, const DeviceWorkspace &expanded) {
    (void)expanded;
    sequence_utils::broadcast_samples(expanded_batch, batch, expand_desc.NumExpanded(),
                                      expand_desc.DimsToExpand());
  }

  kernels::ScatterGatherGPU scatter_gather_;
};

/**
 * @brief SequenceOperator
 * Provides generic support for sequence processing to an operator by unfolding
 * a batch of sequences into a batch of frames. Applicability of the SequenceOperator
 * is limited to operators that can process frames independently. For example, an operator
 * that applies convolution to a batch of images can be turned into the operator that supports
 * batches of sequences of frames, given that there is no need to convolve along frames dimension.
 *
 * Adds support for per-frame tensor arguments: if a tensor argument is marked in the schema as
 * supporting per-frame tensor values and the tensor argument layout starts with `F`, the
 * outermost dimension of the tensor argument will be unfolded to match the the expanded
 * input of the operator.
 *
 * The operator must infer output shapes in the setup stage and must not manually
 * resize the outputs.
 */
template <typename Backend>
class SequenceOperator : public Operator<Backend>, protected SampleBroadcasting<Backend> {
 public:
  inline explicit SequenceOperator(const OpSpec &spec) : Operator<Backend>{spec} {}

  using Operator<Backend>::Setup;
  using Operator<Backend>::Run;
  using SampleBroadcasting<Backend>::BroadcastSamples;

  template <typename InputBackend>
  using ws_input_t = typename workspace_t<Backend>::template input_t<InputBackend>;
  template <typename OutputBackend>
  using ws_output_t = typename workspace_t<Backend>::template output_t<OutputBackend>;

  bool Setup(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    CheckInputLayouts(ws, spec_);
    SetupSequenceOperator(ws);
    is_expanding_ = ShouldExpand(ws);
    bool is_inferred;
    if (!IsExpanding()) {
      is_inferred = Operator<Backend>::Setup(output_desc, ws);
    } else {
      Operator<Backend>::template EnforceUniformInputBatchSize<Backend>(ws);
      DALI_ENFORCE(IsExpandable(),
                   "Operator requested to expand the sequence-like inputs, but no expandable input "
                   "was found");
      if (!is_expanded_ws_initialized_) {
        InitializeExpandedWorkspace(ws);
        is_expanded_ws_initialized_ = true;
      }
      ExpandInputs(ws);
      ExpandArguments(ws);
      is_inferred = Operator<Backend>::Setup(output_desc, expanded_);
    }
    is_inferred = ProcessOutputDesc(output_desc, ws, is_inferred);
    DALI_ENFORCE(is_inferred, "SequenceOperator must infer the input shape");
    return is_inferred;
  }

  void Run(workspace_t<Backend> &ws) override {
    if (!IsExpanding()) {
      Operator<Backend>::Run(ws);
    } else {
      ExpandOutputs(ws);
      Operator<Backend>::Run(expanded_);
      PostprocessOutputs(ws);
    }
  }

  bool IsExpanding() const {
    return is_expanding_;
  }

  bool IsExpandable() const {
    auto expand_like_idx = GetReferenceInputIdx();
    assert(expand_like_idx == -1 || GetInputExpandDesc(expand_like_idx).NumDimsToExpand() > 0);
    return expand_like_idx >= 0;
  }

 protected:
  /**
   * @brief Controls if the SequenceOperator should expand batches in the current iteration.
   *
   * It can be used to restrict cases when the expansion is performed, but it is an error for
   * ``ShouldExpand`` to return true if none of the inputs is expandable (i.e. !IsExpandable()).
   * Overriding the method may be useful for validation of the initial input layouts.
   */
  virtual bool ShouldExpand(const workspace_t<Backend> &ws) {
    return IsExpandable();
  }

  /**
   * @brief A common path for processing inferred output shapes for expanding and non-expanding
   * cases.
   *
   * By default, the expanded output is assumed to have the same expandable layout and
   * expandable shape as the expandable input. The inferred shapes are coalesced, i.e. the
   * SequenceOperator verifies if the the shapes of frames corresponding to the same sequence are
   * equal and merges them into sequences.
   *
   * May be overriden to infer output shapes outside of the operator's SetupImpl method,
   * which may be easier or more performant.
   */
  virtual bool ProcessOutputDesc(std::vector<OutputDesc> &output_desc,
                                 const workspace_t<Backend> &ws, bool is_inferred) {
    if (is_inferred && IsExpanding()) {
      CoalesceOutputShapes(output_desc, ws);
    }
    return is_inferred;
  }

  /**
   * @brief Controls if given input should expand leading channel dimensions.
   *
   * If false, only a batch whose layout starts with ``F`` will be considered expandable
   * (for example ``FHWC -> HWC``). If true, layouts starting with ``C`` will be expandable as well,
   * i.e. ``FCHW -> HW, CFHW -> HW, CHW -> HW`` etc.
   *
   * The operator does not support specifying tensor arguments per-channel.
   */
  virtual bool ShouldExpandChannels(int input_idx) const {
    (void)input_idx;
    return false;
  }

  /**
   * @brief Gets the ExpandDesc instance that describes the expandable prefix of the output layout
   * and corresponding prefixes of the shapes in the batch. By default, the sequence shape and
   * expandable layout are assumed to be the same for the output and the expandable input. If that's
   * not the case, the method may be overriden to provide different ExpandDesc instance.
   */
  virtual const ExpandDesc &GetOutputExpandDesc(const workspace_t<Backend> &ws,
                                                int output_idx) const {
    (void)output_idx;
    return GetInputExpandDesc(GetReferenceInputIdx());
  }

  /**
   * @brief Returns the index of the operator input which is expandable and given argument input
   * should be expanded or broadcast in a manner consistent with the input.
   */
  virtual int GetArgExpandDescInputIdx(const std::string &arg_name) const {
    (void)arg_name;
    return GetReferenceInputIdx();
  }

  virtual void InitializeExpandedArgument(const workspace_t<Backend> &ws,
                                          const std::string &arg_name,
                                          const TensorVector<CPUBackend> &arg_input) {
    expanded_.AddArgumentInput(arg_name, sequence_utils::expanded_like(arg_input));
  }

  /**
   * @brief Expands the tensor arguments from ``ws`` and adds them to expanded workspace.
   * Arguments marked as supporting per-frame tensors and with leading `F` in tensor layout are
   * unfolded (and additionally broadcast if the channels in the operator's input are unfolded) to
   * match the operator's input. If the argument is unfolded, each sample of the argument must have
   * the number of frames equal to either: one (so it can be broadcast) or to the number of frames
   * in the corresponding sample of operator's expandable input.
   *
   * Tensor inputs not marked per-frame or with no leading frames in the layout are broadcast to
   * match the operator's expandable input.
   *
   * Assumes ``IsExpandble()`` is true, i.e. there is some expandable input to match the expanded
   * arguments.
   */
  virtual void ExpandArgument(const ArgumentWorkspace &ws, const std::string &arg_name,
                              const TensorVector<CPUBackend> &arg_input) {
    assert(IsExpandable());
    auto input_idx = GetArgExpandDescInputIdx(arg_name);
    ExpandArgumentLikeInput(ExpandedArg(arg_name), arg_input, arg_name, input_idx);
  }

  virtual void InitializeExpandedInput(const workspace_t<Backend> &ws, int input_idx) {
    ProcessInput(ws, input_idx, [&](const auto &batch) {
      expanded_.AddInput(sequence_utils::expanded_like(batch));
    });
  }

  // TODO(ktokarski) Treat all but main input as arguments and make positional args expansion
  // consistent with named arguments expansion. For now, if the main input has "CF/FC" expandable
  // layout, named arg with "F" or "" expandable layout is supported, while for positional arg it
  // must be either "CF/FC" or "".
  /**
   * @brief Expands the input from the provided workspace and adds to the expanded workspace.
   *
   * Assumes ``IsExpandble()`` is true, i.e. there is at least one expandable input.
   *
   * By default, for multiple-input operators, if any of the inputs is to be expanded,
   * all other inputs must be expandable in an agreeable way, i.e:
   * 1. have the same expandble layout prefix and matching outermost dimensions that are to
   *    be expanded, or
   * 2. have no expandable layout prefix, in that case the samples are going to be broadcast
   *    to match the other expanded inputs.
   */
  virtual void ExpandInput(const workspace_t<Backend> &ws, int input_idx) {
    assert(IsExpandable());
    int ref_input_idx = GetReferenceInputIdx();
    const auto &ref_expand_desc = GetInputExpandDesc(ref_input_idx);
    const auto &input_desc = GetInputExpandDesc(input_idx);
    int num_expand_dims = input_desc.NumDimsToExpand();
    if (num_expand_dims == 0) {
      ProcessInput(ws, input_idx, [&](const auto &input) {
        auto &expanded_input = ExpandedInput<batch_backend_t<decltype(input)>>(input_idx);
        BroadcastBatch(expanded_input, input, ref_expand_desc);
      });
    } else {
      if (ref_input_idx != input_idx) {
        VerifyExpansionConsistency(ref_input_idx, ref_expand_desc, input_idx, input_desc);
      }
      ProcessInput(ws, input_idx, [&](const auto &input) {
        auto &expanded_input = ExpandedInput<batch_backend_t<decltype(input)>>(input_idx);
        UnfoldBatch(expanded_input, input, input_desc);
      });
    }
  }

  virtual void InitializeExpandedOutput(const workspace_t<Backend> &ws, int output_idx) {
    ProcessOutput(ws, output_idx, [&](const auto &batch) {
      expanded_.AddOutput(sequence_utils::expanded_like(batch));
    });
  }

  /**
   * @brief Expands the output from the provided workspace and adds to the expanded workspace.
   *
   * Assumes ``IsExpandble()`` is true, i.e. there is at least one expandable input.
   */
  virtual void ExpandOutput(const workspace_t<Backend> &ws, int output_idx) {
    const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
    ProcessOutput(ws, output_idx, [&](const auto &output) {
      auto &expanded_output = ExpandedOutput<batch_backend_t<decltype(output)>>(output_idx);
      UnfoldBatch(expanded_output, output, expand_desc);
    });
  }

  /**
   * @brief If the outputs were expanded, the meta-data from the expanded outputs
   * needs to be reflected in the original workspace outputs.
   */
  virtual void PostprocessOutputs(workspace_t<Backend> &ws) {
    auto num_output = ws.NumOutput();
    for (int output_idx = 0; output_idx < num_output; output_idx++) {
      const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
      auto layout_prefix = expand_desc.ExpandedLayout();
      if (ws.template OutputIsType<GPUBackend>(output_idx)) {
        SetOutputLayout<GPUBackend>(ws, output_idx, layout_prefix);
      } else {
        SetOutputLayout<CPUBackend>(ws, output_idx, layout_prefix);
      }
    }
  }

  /**
   * @brief Coalesces batch of frames shapes into batch of sequence shapes.
   *
   * For example if in the input there is a ``FCHW`` batch of shape
   * ``[{1, 3, 20, 20}, {2, 2, 40, 20}]`` with ``FC`` expandable layout, the inferred output shape
   * (if any) is expected to be of the form
   * ``[shape_1, shape_1, shape_1, shape_2, shape_2, shape_2, shape_2]``, so that it can be
   * coalesced into ``[{1, 3, shape_1...}, {2, 2, shape_2...}]``.
   */
  virtual void CoalesceOutputShapes(std::vector<OutputDesc> &output_desc,
                                    const workspace_t<Backend> &ws) {
    for (size_t output_idx = 0; output_idx < output_desc.size(); output_idx++) {
      const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
      const auto &shape = output_desc[output_idx].shape;
      DALI_ENFORCE(shape.num_samples() == expand_desc.NumExpanded(),
                   make_string("Unexpected number of frames inferred in the operator for output ",
                               output_idx, ". Expected ", expand_desc.NumExpanded(),
                               " but the operator returned ", shape.num_samples(), "."));
      output_desc[output_idx].shape =
          sequence_utils::fold_outermost_like(shape, expand_desc.DimsToExpand());
    }
  }

  const ExpandDesc &GetInputExpandDesc(int input_idx) const {
    DALI_ENFORCE(0 <= input_idx && static_cast<size_t>(input_idx) < input_expand_desc_.size());
    return input_expand_desc_[input_idx];
  }

  void SetupSequenceOperator(const workspace_t<Backend> &ws) {
    auto num_inputs = ws.NumInput();
    input_expand_desc_.clear();
    input_expand_desc_.reserve(num_inputs);
    for (int input_idx = 0; input_idx < num_inputs; input_idx++) {
      const auto &input_shape = ws.GetInputShape(input_idx);
      const auto &layout = GetInputLayout(ws, input_idx);
      input_expand_desc_.emplace_back(input_shape, layout, ShouldExpandChannels(input_idx));
    }
    expand_like_idx_ = InferExpandableInputIdx(ws);
  }

  int InferExpandableInputIdx(const workspace_t<Backend> &ws) {
    auto num_inputs = ws.NumInput();
    for (int input_idx = 0; input_idx < num_inputs; input_idx++) {
      if (input_expand_desc_[input_idx].NumDimsToExpand() > 0) {
        return input_idx;
      }
    }
    return -1;
  }

  /**
   * @brief Returns the index of the operator input which is expandable and other
   * inputs should be expanded or broadcast in a manner consistent with the
   * given input. If no input is expandable, returns -1.
   */
  int GetReferenceInputIdx() const {
    return expand_like_idx_;
  }

  void ExpandInputs(const workspace_t<Backend> &ws) {
    for (int input_idx = 0; input_idx < ws.NumInput(); input_idx++) {
      ExpandInput(ws, input_idx);
    }
  }

  void ExpandOutputs(const workspace_t<Backend> &ws) {
    for (int output_idx = 0; output_idx < ws.NumOutput(); output_idx++) {
      ExpandOutput(ws, output_idx);
    }
  }

  void ExpandArguments(const ArgumentWorkspace &ws) {
    for (const auto &arg_input : ws) {
      auto &shared_tvec = arg_input.second.tvec;
      assert(shared_tvec);
      ExpandArgument(ws, arg_input.first, *shared_tvec);
    }
  }

  bool HasPerFrameArgInputs(const workspace_t<Backend> &ws) {
    for (const auto &arg_input : ws) {
      auto &shared_tvec = arg_input.second.tvec;
      assert(shared_tvec);
      if (IsPerFrameArg(arg_input.first, *shared_tvec)) {
        return true;
      }
    }
    return false;
  }

  bool IsPerFrameArg(const std::string &arg_name, const TensorVector<CPUBackend> &arg_input) {
    const auto &schema = Operator<Backend>::GetSpec().GetSchema();
    // Do not error out but simply ignore `F` layout of the argument input
    // if it is not marked as per-frame in schema to be consistent with operators
    // that do not support per-frame at all
    return schema.ArgSupportsPerFrameInput(arg_name) && detail::is_per_frame(arg_input);
  }

  void VerifyExpansionConsistency(int expand_idx, const ExpandDesc &expand_desc, int input_idx,
                                  const ExpandDesc &input_desc) {
    // TODO(ktokarski) consider mix of expansion and broadcasting similar to handling of
    // per-frame arguments for "FC" or "CF" layouts. Consider agreeing "FC" and "CF" inputs.
    DALI_ENFORCE(
        input_desc.ExpandedLayout() == expand_desc.ExpandedLayout(),
        make_string("Failed to match expanding of sequence-like inputs for multiple-input "
                    "operator. For the ",
                    expand_idx, " input with layout ", expand_desc.Layout(),
                    " the following outermost dimension(s) should be expanded into samples: `",
                    expand_desc.ExpandedLayout(), "`. The input ", input_idx, " with layout ",
                    input_desc.Layout(),
                    " is expected to either: have no outermost dimensions planned for expanding or "
                    "have exactly the same outermost dimensions to expand. However, got `",
                    input_desc.ExpandedLayout(), "` planned for expanding."));
    for (int sample_idx = 0; sample_idx < expand_desc.NumSamples(); ++sample_idx) {
      assert(expand_desc.NumExpanded(sample_idx) == input_desc.NumExpanded(sample_idx));
      if (expand_desc.ExpandFrames()) {
        DALI_ENFORCE(expand_desc.NumFrames(sample_idx) == input_desc.NumFrames(sample_idx),
                     make_string("Inputs ", expand_idx, " and ", input_idx,
                                 " have different number of frames for sample ", sample_idx,
                                 ", respectively: ", expand_desc.NumFrames(sample_idx), " and ",
                                 input_desc.NumFrames(sample_idx)));
      }
      if (expand_desc.ExpandChannels()) {
        DALI_ENFORCE(expand_desc.NumChannels(sample_idx) == input_desc.NumChannels(sample_idx),
                     make_string("Inputs ", expand_idx, " and ", input_idx,
                                 " have different number of channels for sample ", sample_idx,
                                 ", respectively: ", expand_desc.NumChannels(sample_idx), " and ",
                                 input_desc.NumChannels(sample_idx)));
      }
    }
  }

  void VerifyExpandedBatchSizeNumericLimit(const ExpandDesc &expand_desc) {
    // int is used in containers such as TensorVector to store the number of samples,
    // so we cannot handle total number of frames in the batch greater than that
    DALI_ENFORCE(
        expand_desc.NumExpanded() <= std::numeric_limits<int>::max(),
        make_string(
            "Cannot expand sequence-like batch into batch of samples: there are too many frames ",
            (expand_desc.ExpandChannels() ? "(or channels)" : ""), ": ", expand_desc.NumExpanded(),
            " while the limit for batch size is ", std::numeric_limits<int>::max(), "."));
  }

  template <typename OutputBackend>
  void SetOutputLayout(const workspace_t<Backend> &ws, int output_idx, TensorLayout layout_prefix) {
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

  void InitializeExpandedInputs(const workspace_t<Backend> &ws) {
    assert(expanded_.NumInput() == 0);
    for (int input_idx = 0; input_idx < ws.NumInput(); input_idx++) {
      InitializeExpandedInput(ws, input_idx);
    }
    assert(expanded_.NumInput() == ws.NumInput());
  }

  void InitializeExpandedOutputs(const workspace_t<Backend> &ws) {
    assert(expanded_.NumOutput() == 0);
    for (int output_idx = 0; output_idx < ws.NumOutput(); output_idx++) {
      InitializeExpandedOutput(ws, output_idx);
    }
    assert(expanded_.NumOutput() == ws.NumOutput());
  }

  void InitializeExpandedArguments(const workspace_t<Backend> &ws) {
    for (const auto &arg_input : ws) {
      auto &shared_tvec = arg_input.second.tvec;
      assert(shared_tvec);
      InitializeExpandedArgument(ws, arg_input.first, *shared_tvec);
    }
  }

  void InitializeExpandedWorkspace(const workspace_t<Backend> &ws) {
    InitializeExpandedWorkspace(expanded_, ws);
    InitializeExpandedInputs(ws);
    InitializeExpandedArguments(ws);
    InitializeExpandedOutputs(ws);
  }

  void InitializeExpandedWorkspace(workspace_t<CPUBackend> &expanded,
                                   const workspace_t<CPUBackend> &ws) {
    expanded.SetThreadPool(&ws.GetThreadPool());
  }

  void InitializeExpandedWorkspace(workspace_t<GPUBackend> &expanded,
                                   const workspace_t<GPUBackend> &ws) {
    if (ws.has_stream()) {
      expanded.set_stream(ws.stream());
    }
  }

  template <typename ProcessFunc>
  void ProcessInput(const workspace_t<Backend> &ws, int input_idx, ProcessFunc &&process) {
    if (ws.template InputIsType<GPUBackend>(input_idx)) {
      process(ws.template Input<GPUBackend>(input_idx));
    } else {
      process(ws.template Input<CPUBackend>(input_idx));
    }
  }

  template <typename ProcessFunc>
  void ProcessOutput(const workspace_t<Backend> &ws, int output_idx, ProcessFunc &&process) {
    if (ws.template OutputIsType<GPUBackend>(output_idx)) {
      process(ws.template Output<GPUBackend>(output_idx));
    } else {
      process(ws.template Output<CPUBackend>(output_idx));
    }
  }

  template <typename InputBackend>
  auto &ExpandedInput(int intput_idx) {
    return expanded_.template UnsafeMutableInput<InputBackend>(intput_idx);
  }

  template <typename OutputBackend>
  auto &ExpandedOutput(int output_idx) {
    return expanded_.template Output<OutputBackend>(output_idx);
  }

  TensorVector<CPUBackend> &ExpandedArg(const std::string &arg_name) {
    return expanded_.UnsafeMutableArgumentInput(arg_name);
  }

  template <typename BatchType>
  void UnfoldBatch(BatchType &expanded_batch, const BatchType &batch,
                   const ExpandDesc &expand_desc) {
    auto sample_dim = batch.shape().sample_dim();
    auto num_expand_dims = expand_desc.NumDimsToExpand();
    DALI_ENFORCE(
        sample_dim >= num_expand_dims,
        make_string(
            "Cannot flatten the sequence-like batch. Samples cannot have less dimensions (got ",
            sample_dim, ") than the requested number of dimensions to unfold: ", num_expand_dims,
            "."));
    VerifyExpandedBatchSizeNumericLimit(expand_desc);
    UnfoldOuterDims(expanded_batch, batch, expand_desc);
  }

  template <typename BatchType>
  void BroadcastBatch(BatchType &expanded_batch, const BatchType &batch,
                      const ExpandDesc &expand_desc) {
    VerifyExpandedBatchSizeNumericLimit(expand_desc);
    BroadcastSamples(expanded_batch, batch, expand_desc, expanded_);
  }

  void ExpandArgumentLikeInput(TensorVector<CPUBackend> &expanded_arg_input,
                               const TensorVector<CPUBackend> &arg_input,
                               const std::string &arg_name, int input_idx) {
    const auto &expand_desc = GetInputExpandDesc(input_idx);
    assert(expand_desc.NumDimsToExpand() > 0);
    DALI_ENFORCE(arg_input.num_samples() == expand_desc.NumSamples(),
                 make_string("Number of samples passed for argument ", arg_name, " (got ",
                             arg_input.num_samples(),
                             ") does not match the number of samples in the input (got ",
                             expand_desc.NumSamples(), ")."));
    VerifyExpandedBatchSizeNumericLimit(expand_desc);
    if (!IsPerFrameArg(arg_name, arg_input)) {
      BroadcastSamples(expanded_arg_input, arg_input, expand_desc, expanded_);
    } else {
      DALI_ENFORCE(
          expand_desc.ExpandFrames(),
          make_string(
              "Tensor input for argument ", arg_name, " is specified per frame (got ",
              arg_input.GetLayout(),
              " layout), but samples in the input batch do not contain frames (expected input "
              "layout that starts with 'F'",
              ShouldExpandChannels(input_idx) ? " or 'CF'" : "", "). Got layout ",
              expand_desc.Layout(), " for operator intput ", input_idx, "."));
      UnfoldBroadcastArgument(expanded_arg_input, arg_input, arg_name, expand_desc);
    }
  }

  void UnfoldBroadcastArgument(TensorVector<CPUBackend> &expanded_arg,
                               const TensorVector<CPUBackend> &arg_input,
                               const std::string &arg_name, const ExpandDesc &expand_desc) {
    constexpr int ndims_to_unfold = 1;
    assert((arg_input.num_samples() == expand_desc.NumSamples()) && expand_desc.ExpandFrames());
    auto tv_builder = sequence_utils::tv_builder_like(expanded_arg, arg_input,
                                                      expand_desc.NumExpanded(), ndims_to_unfold);
    for (int sample_idx = 0; sample_idx < expand_desc.NumSamples(); sample_idx++) {
      auto frames_range =
          sequence_utils::unfolded_slice_range(arg_input, sample_idx, ndims_to_unfold);
      auto num_input_frames = expand_desc.NumFrames(sample_idx);
      auto num_arg_frames = frames_range.NumSlices();
      DALI_ENFORCE(
          num_arg_frames == 1 || num_input_frames == num_arg_frames,
          make_string("The tensor argument ", arg_name, " for sample ", sample_idx,
                      " should either be a single argument to be reused accross all frames in the "
                      "sample or should be specified per each frame. Got ",
                      num_arg_frames, " arguments for the sample but there are ", num_input_frames,
                      " frames in the sample."));
      if (num_arg_frames == 1) {  // broadcast the sample
        auto slice = frames_range[0];
        for (int j = 0; j < expand_desc.NumExpanded(sample_idx); j++) {
          tv_builder.SetNext(slice);
        }
      } else if (!expand_desc.ExpandChannels()) {  // expand frames dimension
        assert(num_arg_frames == expand_desc.NumExpanded(sample_idx));
        for (auto &&slice : frames_range) {
          tv_builder.SetNext(slice);
        }
      } else {
        int num_input_channels = expand_desc.NumChannels(sample_idx);
        assert(num_arg_frames * num_input_channels == expand_desc.NumExpanded(sample_idx));
        if (expand_desc.IsChannelFirst()) {
          for (int j = 0; j < num_input_channels; j++) {
            for (auto &&slice : frames_range) {
              tv_builder.SetNext(slice);
            }
          }
        } else {
          for (auto &&slice : frames_range) {
            for (int j = 0; j < num_input_channels; j++) {
              tv_builder.SetNext(slice);
            }
          }
        }
      }
    }
    assert(tv_builder.NextSampleIdx() == expanded_arg.num_samples());
  }

  template <typename DataBackend>
  void UnfoldOuterDims(TensorVector<DataBackend> &expanded_batch,
                       const TensorVector<DataBackend> &batch, const ExpandDesc &expand_desc) {
    auto ndims_to_unfold = expand_desc.NumDimsToExpand();
    assert(expand_desc.NumSamples() == batch.num_samples());
    assert(batch.shape().first(ndims_to_unfold).num_elements() == expand_desc.NumExpanded());
    sequence_utils::unfold_outer_dims(expanded_batch, batch, expand_desc.NumDimsToExpand(),
                                      expand_desc.NumExpanded());
  }

  template <typename DataBackend>
  void UnfoldOuterDims(TensorList<DataBackend> &expanded_batch,
                       const TensorList<DataBackend> &batch, const ExpandDesc &expand_desc) {
    sequence_utils::unfold_outer_dims(expanded_batch, batch, expand_desc.NumDimsToExpand());
  }

  std::vector<ExpandDesc> input_expand_desc_;
  USE_OPERATOR_MEMBERS();

 private:
  int expand_like_idx_ = -1;
  bool is_expanding_ = false;
  bool is_expanded_ws_initialized_ = false;
  workspace_t<Backend> expanded_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_OPERATOR_H_

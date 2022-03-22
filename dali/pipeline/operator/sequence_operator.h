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
#include "dali/pipeline/operator/sequence_shape.h"

namespace dali {

namespace detail {
inline bool is_per_frame(const TensorVector<CPUBackend> &arg_tensor) {
  const auto &layout = arg_tensor.GetLayout();
  return layout.size() > 0 && layout[0] == 'F';
}
}  // namespace detail

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
 * The operator must infer input shapes in the setup stage and must not manually
 * resize the outputs.
 */
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
      SetupExpandedWorkspace(expanded_, ws);
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
      ClearExpanded();
    }
  }

  bool IsExpanding() const {
    return is_expanding_;
  }

  bool IsExpandable() const {
    auto expand_like_idx = GetReferentialInputIdx();
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
   * However, operator does not support specifing tensor arguments per-chanel.
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
    return GetInputExpandDesc(GetReferentialInputIdx());
  }

  virtual int GetArgExpandDescInputIdx(const std::string &arg_name) const {
    (void)arg_name;
    return GetReferentialInputIdx();
  }

  /**
   * @brief Expands the tensor arguments from ``ws`` and adds them to expanded workspace.
   * Arguments marked as supporting per-frame tensors and with leading `F` in tensor layout are
   * unfolded (and additionally broadcasted if the channels in the operator's input are unfolded) to
   * match the operator's input. If the argument is unfolded, each sample of the argument must have
   * the number of frames equal to either: one (so it can be broadcasted) or to the number of frames
   * in the corresponding sample of operator's expandable input.
   *
   * Tensor inputs not marked per-frame or with no leading frames in the layout are broadcasted to
   * match the operator's expandable input.
   *
   * Assumes ``IsExpandble()`` is true, i.e. there is some expandable input to match the expanded
   * arguments.
   */
  virtual void ExpandArgument(const ArgumentWorkspace &ws, const std::string &arg_name,
                              const TensorVector<CPUBackend> &arg_tensor) {
    auto input_idx = GetArgExpandDescInputIdx(arg_name);
    auto expanded_arg = ExpandArgumentLikeInput(arg_tensor, arg_name, input_idx);
    auto expanded_handle = std::make_shared<TensorVector<CPUBackend>>(std::move(expanded_arg));
    ExpandedAddArgument(arg_name, std::move(expanded_handle));
  }

  /**
   * @brief Expands the input from the provided workspace and adds to the expanded workspace.
   *
   * Assumes ``IsExpandble()`` is true, i.e. there is at least one expandable input.
   *
   * By default, for multiple-input operators, if any of the inputs is to be expanded,
   * all other inputs must be expandable in an agreeable way, i.e:
   * 1. have the same expandble layout prefix and matching outermost dimensions that are to
   *    be expanded, or
   * 2. have no expandable layout prefix, in that case the samples are going to be broadcasted
   *    to match the other expanded inputs (TODO).
   */
  virtual void ExpandInput(const workspace_t<Backend> &ws, int input_idx) {
    int ref_input_idx = GetReferentialInputIdx();
    const auto &ref_expand_desc = GetInputExpandDesc(ref_input_idx);
    const auto &input_desc = GetInputExpandDesc(input_idx);
    int num_expand_dims = input_desc.NumDimsToExpand();
    if (num_expand_dims == 0) {
      // TODO(ktokarski) Add support for broadcasting inputs in multi-input case.
      DALI_FAIL("Broadcasting of inputs for multi-input operators is not supported.")
    } else {
      if (ref_input_idx != input_idx) {
        VerifyExpanionConsistency(ref_input_idx, ref_expand_desc, input_idx, input_desc);
      }
      ExpandedAddProcessedInput(ws, input_idx, [&](const auto &input) {
        return UnfoldInput(ws, input, input_idx, num_expand_dims);
      });
    }
  }

  /**
   * @brief Expands the output from the provided workspace and adds to the expanded workspace.
   *
   * Assumes ``IsExpandble()`` is true, i.e. there is at least one expandable input.
   */
  virtual void ExpandOutput(const workspace_t<Backend> &ws, int output_idx) {
    const auto &expand_desc = GetOutputExpandDesc(ws, output_idx);
    auto num_expand_dims = expand_desc.NumDimsToExpand();
    ExpandedAddProcessedOutput(ws, output_idx, [&](const auto &output) {
      return UnfoldOutput(ws, output, output_idx, num_expand_dims);
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
      output_desc[output_idx].shape = fold_outermost_like(shape, expand_desc.DimsToExpand());
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
   * inputs should be expanded or broadcasted in a manner consistent with the
   * given input. If no input is expandable, returns -1.
   */
  int GetReferentialInputIdx() const {
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
      if (detail::is_per_frame(*shared_tvec)) {
        return true;
      }
    }
    return false;
  }

  void VerifyExpanionConsistency(int expand_idx, const ExpandDesc &expand_desc, int input_idx,
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

  template <typename ProcessFunc>
  void ExpandedAddProcessedInput(const workspace_t<Backend> &ws, int input_idx,
                                 ProcessFunc &&process) {
    if (ws.template InputIsType<GPUBackend>(input_idx)) {
      ExpandedAddInput(process(ws.template Input<GPUBackend>(input_idx)));
    } else {
      ExpandedAddInput(process(ws.template Input<CPUBackend>(input_idx)));
    }
  }

  template <typename ProcessFunc>
  void ExpandedAddProcessedOutput(const workspace_t<Backend> &ws, int output_idx,
                                  ProcessFunc &&process) {
    if (ws.template OutputIsType<GPUBackend>(output_idx)) {
      ExpandedAddOutput(process(ws.template Output<GPUBackend>(output_idx)));
    } else {
      ExpandedAddOutput(process(ws.template Output<CPUBackend>(output_idx)));
    }
  }

  template <typename InputBackend>
  void ExpandedAddInput(ws_input_t<InputBackend> &&input) {
    expanded_.AddInput(input);
  }

  template <typename OutputBackend>
  void ExpandedAddOutput(ws_input_t<OutputBackend> &&output) {
    expanded_.AddOutput(output);
  }

  void ExpandedAddArgument(const std::string &arg_name,
                           std::shared_ptr<TensorVector<CPUBackend>> &&arg_input) {
    expanded_.AddArgumentInput(arg_name, arg_input);
  }

  template <typename InputBackend>
  const auto &ExpandedInput(int intput_idx) {
    return expanded_.template Input<InputBackend>(intput_idx);
  }

  template <typename OutputBackend>
  const auto &ExpandedOutput(int output_idx) {
    return expanded_.template Output<OutputBackend>(output_idx);
  }

  void ClearExpanded() {
    expanded_.Clear();
  }

  template <typename InputType>
  auto UnfoldInput(const workspace_t<Backend> &ws, const InputType &input, int input_idx,
                   int num_expand_dims) {
    auto sample_dim = input.shape().sample_dim();
    DALI_ENFORCE(
        sample_dim > num_expand_dims,
        make_string("Cannot flatten the sequence-like input ", input_idx,
                    ". Samples must have more dimensions (got ", sample_dim,
                    ") than the requested number of dimensions to unfold: ", num_expand_dims, "."));
    auto expanded_input = unfold_outer_dims(input, num_expand_dims);
    const auto &input_layout = input.GetLayout();
    expanded_input.SetLayout(input_layout.sub(num_expand_dims));
    return std::make_shared<InputType>(std::move(expanded_input));
  }

  template <typename OutputType>
  auto UnfoldOutput(const workspace_t<Backend> &ws, const OutputType &output, int output_idx,
                    int num_expand_dims) {
    auto sample_dim = output.shape().sample_dim();
    DALI_ENFORCE(
        sample_dim > num_expand_dims,
        make_string("Cannot flatten the sequence-like output ", output_idx,
                    ". Samples must have more dimensions (got ", sample_dim,
                    ") than the requested number of dimensions to unfold: ", num_expand_dims, "."));
    auto expanded_output = unfold_outer_dims(output, num_expand_dims);
    return std::make_shared<OutputType>(std::move(expanded_output));
  }

  TensorVector<CPUBackend> ExpandArgumentLikeInput(const TensorVector<CPUBackend> &arg_tensor,
                                                   const std::string &arg_name, int input_idx) {
    const auto &expand_desc = GetInputExpandDesc(input_idx);
    DALI_ENFORCE(static_cast<ptrdiff_t>(arg_tensor.num_samples()) == expand_desc.NumSamples(),
                 make_string("Number of samples passed for argument ", arg_name, " (got ",
                             arg_tensor.num_samples(),
                             ") does not match the number of samples in the input (got ",
                             expand_desc.NumSamples(), ")."));
    const auto &schema = Operator<Backend>::GetSpec().GetSchema();
    // Do not error out but simply ignore `F` layout of the argument input
    // if it is not marked as per-frame in schema to be consistent with operators
    // that do not support per-frame at all
    if (schema.ArgSupportsPerFrameInput(arg_name) && detail::is_per_frame(arg_tensor)) {
      return UnfoldBroadcastArgument(arg_tensor, arg_name, input_idx, expand_desc);
    }
    return BroadcastArgument(arg_tensor, expand_desc);
  }

  TensorVector<CPUBackend> UnfoldBroadcastArgument(const TensorVector<CPUBackend> &arg_tensor,
                                                   const std::string &arg_name, int input_idx,
                                                   const ExpandDesc &expand_desc) {
    DALI_ENFORCE(
        expand_desc.ExpandFrames(),
        make_string(
            "Tensor input for argument ", arg_name, " is specified per frame (got ",
            arg_tensor.GetLayout(),
            " layout), but samples in the input batch do not contain frames (expected input "
            "layout that starts with 'F'",
            ShouldExpandChannels(input_idx) ? " or 'CF'" : "", "). Got layout ",
            expand_desc.Layout(), " for operator intput ", input_idx, "."));
    const auto &shape = arg_tensor.shape();
    int arg_sample_dim = shape.sample_dim();
    TensorVector<CPUBackend> flat_tensor(expand_desc.NumExpanded());
    int num_samples = expand_desc.NumSamples();
    auto type_info = arg_tensor.type_info();
    auto type = type_info.id();
    auto is_pinned = arg_tensor.is_pinned();
    auto order = arg_tensor.order();
    int slice_idx = 0;
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto &sample_shape = shape[sample_idx];
      int num_input_frames = expand_desc.NumFrames(sample_idx);
      int num_arg_frames = sample_shape[0];
      DALI_ENFORCE(
          num_arg_frames == 1 || num_input_frames == num_arg_frames,
          make_string("The tensor argument ", arg_name, " for sample ", sample_idx,
                      " should either be a single argument to be reused accross all frames in the "
                      "sample or should be specified per each frame. Got ",
                      num_arg_frames, " arguments for the sample but there are ", num_input_frames,
                      " frames in the sample."));
      auto slice_shape = sample_shape.last(arg_sample_dim - 1);
      auto slice_volume = volume(slice_shape);
      uint8_t *base_ptr =
          const_cast<uint8_t *>(static_cast<const uint8_t *>(arg_tensor.raw_tensor(sample_idx)));
      auto num_bytes = type_info.size() * slice_volume;
      if (num_arg_frames == 1) {  // broadcast the sample
        for (int j = 0; j < expand_desc.NumExpanded(sample_idx); j++) {
          flat_tensor[slice_idx++].ShareData(base_ptr, num_bytes, is_pinned, slice_shape, type,
                                             order);
        }
      } else if (!expand_desc.ExpandChannels()) {  // expand frames dimension
        assert(num_arg_frames == expand_desc.NumExpanded(sample_idx));
        for (int i = 0; i < num_arg_frames; i++, base_ptr += num_bytes) {
          flat_tensor[slice_idx++].ShareData(base_ptr, num_bytes, is_pinned, slice_shape, type,
                                             order);
        }
      } else {
        int num_input_channels = expand_desc.NumChannels(sample_idx);
        assert(num_arg_frames * num_input_channels == expand_desc.NumExpanded(sample_idx));
        int inner_stride, outer_stride;
        if (expand_desc.IsChannelFirst()) {
          inner_stride = num_input_frames;
          outer_stride = 1;
        } else {
          inner_stride = 1;
          outer_stride = num_input_channels;
        }
        for (int i = 0; i < num_arg_frames; i++, base_ptr += num_bytes) {
          for (int j = 0; j < num_input_channels; j++) {
            flat_tensor[slice_idx + i * outer_stride + j * inner_stride].ShareData(
                base_ptr, num_bytes, is_pinned, slice_shape, type, order);
          }
        }
        slice_idx += num_input_channels * num_input_frames;
      }
    }
    assert(slice_idx == expand_desc.NumExpanded());
    return flat_tensor;
  }

  TensorVector<CPUBackend> BroadcastArgument(const TensorVector<CPUBackend> &arg_tensor,
                                             const ExpandDesc &expand_desc) {
    const auto &shape = arg_tensor.shape();
    TensorVector<CPUBackend> res(expand_desc.NumExpanded());
    int num_samples = expand_desc.NumSamples();
    auto type_info = arg_tensor.type_info();
    auto type = type_info.id();
    auto is_pinned = arg_tensor.is_pinned();
    auto order = arg_tensor.order();
    int slice_idx = 0;
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
      const auto &slice_shape = shape[sample_idx];
      int num_elements = expand_desc.NumExpanded(sample_idx);
      auto num_bytes = type_info.size() * num_elements;
      uint8_t *ptr =
          const_cast<uint8_t *>(static_cast<const uint8_t *>(arg_tensor.raw_tensor(sample_idx)));
      for (int sample_slice_idx = 0; sample_slice_idx < num_elements; sample_slice_idx++) {
        res[slice_idx++].ShareData(ptr, num_bytes, is_pinned, slice_shape, type, order);
      }
    }
    assert(slice_idx == expand_desc.NumExpanded());
    return res;
  }

  std::vector<ExpandDesc> input_expand_desc_;
  USE_OPERATOR_MEMBERS();

 private:
  int expand_like_idx_ = -1;
  bool is_expanding_ = false;
  workspace_t<Backend> expanded_;
};


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_SEQUENCE_OPERATOR_H_

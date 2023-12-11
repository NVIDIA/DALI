// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CONVOLUTION_FILTER_H_
#define DALI_OPERATORS_IMAGE_CONVOLUTION_FILTER_H_

#include <memory>
#include <string>
#include <vector>

#include "dali/core/boundary.h"
#include "dali/core/common.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"
#include "dali/pipeline/util/operator_impl_utils.h"


namespace dali {

#define FILTER_INPUT_SUPPORTED_TYPES_CPU (uint8_t, uint16_t, int16_t, float)
#define FILTER_INPUT_SUPPORTED_TYPES_GPU (uint8_t, int8_t, uint16_t, int16_t, float16, float)

#define FILTER_KERNEL_SUPPORTED_TYPES (float)

#define FILTER_INPUT_SUPPORTED_SPATIAL_NDIM_GPU (2, 3)

namespace filter {
using namespace boundary;  // NOLINT(build/namespaces)

struct InputDesc {
  int num_seq_dims, axes;
  bool has_channels;
  bool is_valid_mode;
};

inline void parse_input_layout(InputDesc& input_desc, const TensorLayout& layout) {
  const auto is_seq_like = [](char extent) { return extent == 'C' || extent == 'F'; };
  int pos = 0;
  while (pos < layout.size() && is_seq_like(layout[pos])) {
    pos++;
  }
  input_desc.num_seq_dims = pos;
  while (pos < layout.size() && !is_seq_like(layout[pos])) {
    pos++;
  }
  input_desc.axes = pos - input_desc.num_seq_dims;
  input_desc.has_channels = layout.size() && layout[layout.size() - 1] == 'C';
}

inline InputDesc setup_input_desc(const TensorLayout& layout, int ndim, bool is_valid_mode) {
  if (!layout.size()) {
    DALI_ENFORCE(
        ndim == 2 || ndim == 3,
        make_string("The filter operator got intput with ", ndim,
                    " dimensions. However, only 2D and 3D convolutions are supported. If the input "
                    "has non-spatial dimensions, such as channels or frames (for video sequences), "
                    "please make sure the input batch has a proper layout set."));
    return {0, ndim, false, is_valid_mode};
  }
  InputDesc input_desc;
  input_desc.is_valid_mode = is_valid_mode;
  parse_input_layout(input_desc, layout);
  DALI_ENFORCE(
      input_desc.num_seq_dims + input_desc.axes + input_desc.has_channels == ndim,
      make_string("Filter operator encountered input with unsupported layout: `", layout, "`."));
  return input_desc;
}

inline BoundaryType parse_filter_border_type(const std::string& border_type_str) {
  try {
    auto border_type = parse(border_type_str);
    switch (border_type) {
      case BoundaryType::CONSTANT:
      case BoundaryType::CLAMP:
      case BoundaryType::REFLECT_1001:
      case BoundaryType::REFLECT_101:
      case BoundaryType::WRAP:
        return border_type;
      default:
        DALI_FAIL(
            make_string("Unsupported ``border_type`` was provided: ``", border_type_str, "``."));
    }
  } catch (const std::invalid_argument&) {
    DALI_FAIL(make_string("Unknown ``border_type`` was provided: ``", border_type_str, "``."));
  }
}

inline bool parse_is_valid_mode(std::string mode) {
  std::transform(mode.begin(), mode.end(), mode.begin(), [](auto c) { return std::tolower(c); });
  if (mode == "same") {
    return false;
  }
  if (mode == "valid") {
    return true;
  }
  DALI_FAIL(make_string("Unknown ``mode`` was provided: ``", mode,
                        "``. Supported modes are ``same`` and ``valid``."));
}

template <typename InputShapes, typename FilterShapes>
InputShapes infer_output_shape(const InputShapes& input_shapes, const FilterShapes& filter_shapes,
                               const InputDesc& input_desc) {
  if (!input_desc.is_valid_mode) {
    return input_shapes;
  }
  auto num_samples = input_shapes.num_samples();
  InputShapes output_shapes{};
  output_shapes.resize(num_samples, input_shapes.sample_dim());
  for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    auto shape = input_shapes[sample_idx];
    const auto& filter_shape = filter_shapes[sample_idx];
    for (int dim_idx = 0; dim_idx < filter_shapes.sample_dim(); dim_idx++) {
      DALI_ENFORCE(
          filter_shape[dim_idx] > 0,
          make_string(
              "Filter of volume zero is not supported, got zero-volume filter for sample of idx ",
              sample_idx, "."));
      shape[input_desc.num_seq_dims + dim_idx] -= filter_shape[dim_idx] - 1;
      DALI_ENFORCE(
          shape[input_desc.num_seq_dims + dim_idx] > 0,
          make_string(
              "Filter for sample of idx ", sample_idx,
              " is bigger than the sample. This is not allowed if ``mode`` is set to ``valid``."));
    }
    output_shapes.set_tensor_shape(sample_idx, shape);
  }
  return output_shapes;
}

// it does not handle `F` extents by the assumption that passing per-frame positional arguments
// triggers SequenceOperator unfolding
template <typename In, typename Backend>
TensorListView<detail::storage_tag_map_t<Backend>, const In, 0> get_fill_values_view(
    const TensorList<Backend>& fill_values) {
  DALI_ENFORCE(
      fill_values.type() == type2id<In>::value,
      make_string("The padding scalars (third positional argument) must be of the "
                  "same time as the input samples. Got ",
                  fill_values.type(), " for pad values while the input samples are of type ",
                  type2id<In>::value, "."));
  if (fill_values.sample_dim() == 0) {
    return view<const In, 0>(fill_values);
  }
  const auto& shape = fill_values.shape();
  int num_samples = shape.num_samples();
  for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    DALI_ENFORCE(
        shape[sample_idx].num_elements() == 1,
        make_string("Padding values must be scalars, got ", shape[sample_idx].num_elements(),
                    " elements for a sample of idx ", sample_idx, "."));
  }
  auto view_dyn = view<const In>(fill_values);
  return reshape<0>(view_dyn, uniform_list_shape<0>(num_samples, std::vector<int64_t>{}));
}

}  // namespace filter

template <typename Backend>
class Filter : public SequenceOperator<Backend, StatelessOperator> {
 public:
  using Base = SequenceOperator<Backend, StatelessOperator>;
  inline explicit Filter(const OpSpec& spec)
      : Base(spec),
        is_valid_mode_{filter::parse_is_valid_mode(spec.GetArgument<std::string>("mode"))} {
    spec.TryGetArgument(dtype_, "dtype");
  }

  DISABLE_COPY_MOVE_ASSIGN(Filter);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool ShouldExpand(const Workspace& ws) override {
    const auto& input_layout = GetInputLayout(ws, 0);
    int frame_idx = VideoLayoutInfo::FrameDimIndex(input_layout);
    DALI_ENFORCE(frame_idx == -1 || frame_idx == 0,
                 make_string("When the input is video-like (i.e. contains frames), the frames must "
                             "be an outermost dimension. However, got input with `",
                             input_layout, "` layout."));
    auto input_sample_dim = ws.GetInputShape(0).sample_dim();
    input_desc_ = filter::setup_input_desc(input_layout, input_sample_dim, is_valid_mode_);
    // Pass to the kernel less samples (i.e. sequences not split into frames)
    // when there are no per-frame arguments, to reduce the number of instances of
    // per-sample data-structure when they are not needed.
    bool should_expand =
        Base::ShouldExpand(ws) && (HasPerFramePositionalArgs(ws) || Base::HasPerFrameArgInputs(ws));
    if (should_expand && input_layout.size() && input_layout[0] == 'F') {
      assert(input_desc_.num_seq_dims >= 1);
      input_desc_.num_seq_dims--;
    }
    return should_expand;
  }

  template <typename Out, typename In, typename W>
  std::unique_ptr<OpImplBase<Backend>> GetFilterImpl(const OpSpec& spec_,
                                                     const filter::InputDesc& input_desc);

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const Workspace& ws) override {
    if (!impl_) {
      auto input_type = ws.GetInputDataType(0);
      auto filter_type = ws.GetInputDataType(1);
      auto dtype = dtype_ == DALI_NO_TYPE ? input_type : dtype_;
      DALI_ENFORCE(dtype == input_type || dtype == filter_type,
                   "Output data type must be same as input, FLOAT or skipped (defaults to "
                   "input type)");
      const auto& filter_shape = ws.GetInputShape(1);
      auto filter_ndim = filter_shape.sample_dim();
      DALI_ENFORCE(filter_ndim == input_desc_.axes,
                   make_string("Filter dimensionality must match the number of spatial dimensions "
                               "in the input samples. Got ",
                               input_desc_.axes, " spatial dimensions but the filter is ",
                               filter_ndim, " dimensional."));
      InputTypeSwitch<Backend>(input_type, [&](auto in) {
        using In = decltype(in);
        TYPE_SWITCH(filter_type, type2id, W, FILTER_KERNEL_SUPPORTED_TYPES, (
          if (dtype == input_type) {
            impl_ = GetFilterImpl<In, In, W>(spec_, input_desc_);
          } else {
            impl_ = GetFilterImpl<W, In, W>(spec_, input_desc_);
          }
        ), DALI_FAIL(make_string("Unsupported filter type: ", filter_type, ".")));  // NOLINT
      });
    }
    return impl_->SetupImpl(output_desc, ws);
  }

  void RunImpl(Workspace& ws) override {
    impl_->RunImpl(ws);
  }

  bool HasPerFrameFilters(const Workspace& ws) {
    auto filter_dim = ws.GetInputDim(1);
    const auto& filter_layout = GetInputLayout(ws, 1);
    return filter_dim == 3 && filter_layout.size() == 3 && filter_layout[0] == 'F';
  }

  bool HasPerFrameFillValues(const Workspace& ws) {
    if (ws.NumInput() < 3) {
      return false;
    }
    const auto& layout = GetInputLayout(ws, 2);
    return layout.size() == 1 && layout[0] == 'F';
  }

  bool HasPerFramePositionalArgs(const Workspace& ws) {
    return HasPerFrameFilters(ws) || HasPerFrameFillValues(ws);
  }

 private:
  template <typename Backend_, typename Cb>
  std::enable_if_t<std::is_same_v<Backend_, GPUBackend>> InputTypeSwitch(DALIDataType in_type,
                                                                         Cb cb) {
    TYPE_SWITCH(in_type, type2id, In, FILTER_INPUT_SUPPORTED_TYPES_GPU, (
      cb(In{});
    ), DALI_FAIL(make_string("Filter GPU does not support input type ", in_type, ".")));  // NOLINT
  }

  template <typename Backend_, typename Cb>
  std::enable_if_t<std::is_same_v<Backend_, CPUBackend>> InputTypeSwitch(DALIDataType in_type,
                                                                         Cb cb) {
    TYPE_SWITCH(in_type, type2id, In, FILTER_INPUT_SUPPORTED_TYPES_CPU, (
      cb(In{});
    ), DALI_FAIL(make_string("Filter CPU does not support input type ", in_type, ".")));  // NOLINT
  }

  filter::InputDesc input_desc_;
  bool is_valid_mode_;
  DALIDataType dtype_ = DALI_NO_TYPE;
  std::unique_ptr<OpImplBase<Backend>> impl_ = nullptr;
  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_FILTER_H_

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

#ifndef DALI_OPERATORS_IMAGE_CONVOLUTION_FILTER_H_
#define DALI_OPERATORS_IMAGE_CONVOLUTION_FILTER_H_

#include <memory>
#include <string>
#include <vector>

#include "dali/core/boundary.h"
#include "dali/core/common.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"
#include "dali/pipeline/util/operator_impl_utils.h"


namespace dali {

#define FILTER_INPUT_SUPPORTED_TYPES \
  (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float16, float)

#define FILTER_KERNEL_SUPPORTED_TYPES (float)

namespace filter {
using namespace boundary;  // NOLINT(build/namespaces)

struct InputLayoutDesc {
  int num_seq_dims = 0;
  bool has_channels = false;
};

inline InputLayoutDesc parse_input_layout(const TensorLayout& layout) {
  InputLayoutDesc input_desc;
  const auto is_seq_like = [](char extent) { return extent == 'C' || extent == 'F'; };
  while (input_desc.num_seq_dims < layout.size() && is_seq_like(layout[input_desc.num_seq_dims])) {
    input_desc.num_seq_dims++;
  }
  input_desc.has_channels = layout.size() && layout[layout.size() - 1] == 'C';
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

template <typename InputShapes, typename FilterShapes>
InputShapes infer_output_shape(const InputShapes& input_shapes, const FilterShapes& filter_shapes,
                               boundary::BoundaryType border_type, int spatial_dim_start) {
  if (border_type != boundary::BoundaryType::ISOLATED) {
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
      shape[spatial_dim_start + dim_idx] -= filter_shape[dim_idx] - 1;
      DALI_ENFORCE(
          shape[spatial_dim_start + dim_idx] >= 0,
          make_string(
              "Filter for sample of idx ", sample_idx,
              " is bigger than the sample. This is not allowed when ``border_type`` is set to ",
              to_string(BoundaryType::ISOLATED), "."));
    }
    output_shapes.set_tensor_shape(sample_idx, shape);
  }
  return output_shapes;
}

// it does not handle `F` extents on the assumption that passing per-frame positional arguments
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
class Filter : public SequenceOperator<Backend> {
 public:
  inline explicit Filter(const OpSpec& spec) : SequenceOperator<Backend>(spec) {
    spec.TryGetArgument(dtype_, "dtype");
  }

  DISABLE_COPY_MOVE_ASSIGN(Filter);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool ShouldExpand(const Workspace& ws) override;

  void ValidateLayouts(const Workspace& ws) {
    auto filter_dim = ws.GetInputDim(1);
    if (filter_dim == 2) {
      return;
    }
    DALI_ENFORCE(filter_dim == 3, make_string("Filter must be a 2D array or a sequence of 2D "
                                              "arrays but got filter of dimensionality: ",
                                              filter_dim));
    const auto& filter_layout = GetInputLayout(ws, 1);
    DALI_ENFORCE(
        filter_layout.size() == 3 && filter_layout[0] == 'F',
        make_string(
            "Filter must be a 2D array or a sequence of 2D arrays. To pass sequence of "
            "filters "
            "mark them per-frame with the per_frame operator. Got filters of dimensionality "
            "3 with layout: `",
            filter_layout, "`."));
  }

  template <typename Out, typename In, typename W>
  std::unique_ptr<OpImplBase<Backend>> GetFilterImpl(const OpSpec& spec_,
                                                     const filter::InputLayoutDesc& input_desc);

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const Workspace& ws) override {
    if (!impl_) {
      auto input_type = ws.GetInputDataType(0);
      auto filter_type = ws.GetInputDataType(1);
      auto dtype = dtype_ == DALI_NO_TYPE ? input_type : dtype_;
      DALI_ENFORCE(dtype == input_type || dtype == filter_type,
                   "Output data type must be same as input, FLOAT or skipped (defaults to "
                   "input type)");
      auto input_layout = filter::parse_input_layout(GetInputLayout(ws, 0));
      TYPE_SWITCH(input_type, type2id, In, FILTER_INPUT_SUPPORTED_TYPES, (
        TYPE_SWITCH(filter_type, type2id, W, FILTER_KERNEL_SUPPORTED_TYPES, (
          if (dtype == input_type) {
            impl_ = GetFilterImpl<In, In, W>(spec_, input_layout);
          } else {
            impl_ = GetFilterImpl<W, In, W>(spec_, input_layout);
          }
        ), DALI_FAIL(make_string("Unsupported filter type: ", filter_type)));  // NOLINT
      ), DALI_FAIL(make_string("Unsupported input type: ", input_type)));  // NOLINT
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
  DALIDataType dtype_ = DALI_NO_TYPE;
  USE_OPERATOR_MEMBERS();
  std::unique_ptr<OpImplBase<Backend>> impl_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CONVOLUTION_FILTER_H_

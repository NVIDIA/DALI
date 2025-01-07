// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <vector>

#include "dali/core/math_util.h"
#include "dali/core/static_switch.h"
#include "dali/core/tensor_layout.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/operators/generic/reshape.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/name_utils.h"

namespace dali {

DALI_SCHEMA(Reshape)
  .DocStr(R"code(Treats content of the input as if it had a different shape and/or layout.

The buffer contents are not copied.)code")
  .NumInput(1, 2)
  .NumOutput(1)
  .InputDox(0, "data", "TensorList", "Data to be reshaped")
  .InputDox(1, "shape_input", "1D TensorList of integers", "Same as `shape` keyword argument")
  .InputDevice(1, InputDevice::CPU)
  .PassThrough({{0, 0}})
  .AllowSequences()
  .SupportVolumetric()
  .AddOptionalArg<int>("shape", R"code(The desired shape of the output.

There can be one negative extent that receives the size that is required to match the input volume.
For example, an input of shape ``[480, 640, 3]`` and ``shape = [240, -1]``
results in the output shape ``[240, 3840]``.

.. note::
  `rel_shape` and `shape` are mutually exclusive.
)code",
                  std::vector<int>(), true)
  .AddOptionalArg<float>("rel_shape", R"code(The relative shape of the output.

The output shape is calculated by multiplying the input shape by `rel_shape`::

  out_shape[i] = in_shape[i] * rel_shape[i]

An additional argument `src_dims` may be used to alter which source dimension is used
for calculating the output shape::

  out_shape[i] = in_shape[src_dims[i]] * rel_shape[i]

There can be one negative extent that receives the size that is required to match the input volume.
For example, an input of shape ``[480, 640, 3]`` and a ``rel_shape = [0.5, -1]`` results in
the output shape ``[240, 3840]``.

The number of dimensions is subject to the following restrictions:

- if `src_dims` argument is used, the number of elements in `src_dims`
  and `rel_shape` must match
- otherwise, the length of `rel_shape` must not exceed the number of dimensions in the input
  except when the last element in `rel_shape` is negative, in which case an extra dimension at
  the end will be added

.. note::
  `rel_shape` and `shape` are mutually exclusive.
)code",
                  std::vector<float>(), true)
  .AddOptionalArg("layout", R"code(New layout for the data.

If a value is not specified, if number of dimension matches existing layout, the output
layout is preserved. If the number of dimensions does not match, the argument is reset
to empty. If a value is set, and is not empty, the layout must match the dimensionality
of the output.)code",
                  TensorLayout(""))
  .AddOptionalArg("src_dims", R"code(Indices of dimensions to keep.

This argument can be used to manipulate the order of existing dimensions or to remove or
add dimension. A special index value -1 can be used to insert new dimensions.

For example, reshaping a sample with shape ``[300, 200, 1]`` and a `src_dims`
argument ``[-1, 1, 0]`` produces an output shape ``[1, 200, 300]``. A leading dimension with
extent 1 is inserted at the beginning, followed by the first original dimensions but in reverse
order. The last dimension is removed.

The `src_dims` argument can be used together with `rel_shape`, in which case the relative
extents in `rel_shape` describe to the target dimensions. In the example above, specifying
``rel_shape = [-1, 0.5, 2]`` would result in the output shape ``[1, 100, 600]``.

All indices must be in the range of valid dimensions of the input, or -1.)code",
                  std::vector<int>(), true);

DALI_SCHEMA(Reinterpret)
  .DocStr(R"(Treats content of the input as if it had a different type, shape, and/or layout.

The buffer contents are not copied.)")
  .NumInput(1, 2)
  .NumOutput(1)
  .InputDox(0, "data", "TensorList", "Data to be reshaped")
  .InputDox(1, "shape_input", "1D TensorList of integers", "Same as `shape` keyword argument")
  .InputDevice(1, InputDevice::CPU)
  .PassThrough({{0, 0}})
  .AllowSequences()
  .SupportVolumetric()
  .AddOptionalTypeArg("dtype", R"code(Output data type.

The total size, in bytes, of the output must match the input. If no shape is provided,
the innermost dimension is adjusted accordingly. If the byte size of the innermost
dimension is not divisible by the size of the target type, an error occurs.)code")
  .AddParent("Reshape");

template <typename Backend>
Reshape<Backend>::Reshape(const OpSpec &spec) : Base(spec) {
  bool has_shape_input = spec.NumRegularInput() == 2;
  bool has_shape_arg = spec.HasArgument("shape") || spec.HasTensorArgument("shape");
  bool has_layout_arg = spec.HasArgument("layout");
  bool has_rel_shape_arg = spec.HasArgument("rel_shape") || spec.HasTensorArgument("rel_shape");
  bool has_src_dims_arg = spec.HasArgument("src_dims");

  if (has_src_dims_arg) {
    use_src_dims_ = true;
    src_dims_ = spec.GetRepeatedArgument<int>("src_dims");
  }
  if (spec.HasArgument("dtype"))
    output_type_arg_ = spec.GetArgument<DALIDataType>("dtype");
  DALI_ENFORCE(has_shape_input + has_shape_arg + has_rel_shape_arg <= 1, make_string(
    "Shape input, `shape` argument and `rel_shape` argument are mutually exclusive"));

  if (!has_shape_input && !has_shape_arg && !has_rel_shape_arg && !has_layout_arg
      && !has_src_dims_arg) {
    bool can_have_dtype = spec.GetSchema().HasArgument("dtype");
    if (can_have_dtype) {
      DALI_ENFORCE(output_type_arg_ != DALI_NO_TYPE, make_string("`", GetOpDisplayName(spec, true),
                   "` is no-op: arguments specify neither new shape, layout nor type."));
    } else {
      DALI_FAIL(make_string("`", GetOpDisplayName(spec, true),
                "` is no-op: arguments specify neither new shape nor layout."));
    }
  }
  use_layout_ = has_layout_arg;
  use_rel_shape_ = false;
  if (has_shape_arg) {
    if (spec.HasTensorArgument("shape")) {
      shape_source_ = ShapeSource::ArgInput;
    } else {
      auto shape_vec = spec.GetRepeatedArgument<int>("shape");

      uniform_shape_.resize(shape_vec.size());
      int num_negative = 0;
      for (int i = 0; i < uniform_shape_.sample_dim(); i++) {
        if (shape_vec[i] < 0) {
          DALI_ENFORCE(++num_negative == 1, make_string(
            "Only one negative extent is allowed; got: ", uniform_shape_));
          uniform_shape_[i] = 0;
          wildcard_dim_ = i;
        } else {
          DALI_ENFORCE(shape_vec[i] != 0, make_string("Extent of 0 is illegal; got: ",
              uniform_shape_));
          uniform_shape_[i] = shape_vec[i];
        }
      }
      shape_source_ = ShapeSource::Arg;
    }
  } else if (has_rel_shape_arg) {
    use_rel_shape_ = true;
    if (spec.HasTensorArgument("rel_shape")) {
      shape_source_ = ShapeSource::ArgInput;
    } else {
      rel_uniform_shape_ = spec.GetRepeatedArgument<float>("rel_shape");
      DALI_ENFORCE(!rel_uniform_shape_.empty(), make_string(
          "`rel_shape` specified as an empty list"));
      int num_negative = 0;
      int out_dims = rel_uniform_shape_.size();
      for (int i = 0; i < out_dims; i++) {
        if (rel_uniform_shape_[i] < 0) {
          DALI_ENFORCE(++num_negative == 1, make_string(
              "Only one negative extent is allowed; got: ", uniform_shape_));
          wildcard_dim_ = i;
        } else {
          DALI_ENFORCE(rel_uniform_shape_[i] != 0,
                       make_string("Extent of 0 is illegal"));
        }
      }
      shape_source_ = ShapeSource::Arg;
    }
  } else if (has_shape_input) {
    DALI_ENFORCE(spec.InputDevice(1) == StorageDevice::CPU,
                 make_string("Output shapes must be provided as a CPU input"));
    shape_source_ = ShapeSource::Input;
  }
  if (has_layout_arg) {
    layout_ = spec.GetArgument<TensorLayout>("layout");
  }
}

template <typename Backend>
bool Reshape<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  output_desc.resize(1);
  SetOutputType(ws);

  CheckSrcDims(ws);
  CalculateOutputShape(ws);
  output_desc[0].type = output_type_->id();
  output_desc[0].shape = output_shape_;
  // return false, because we don't want the executor to allocate anything
  // - this operator returns pointer to input memory
  return false;
}

template <bool relative, typename OutSampleShape, typename InSampleShape, typename ShapeArg>
void SetOutputSampleShape(OutSampleShape &&out_shape,
                          const InSampleShape &in_shape,
                          const ShapeArg &arg,
                          const int *src_dims,
                          int out_ndim) {
  for (int d = 0; d < out_ndim; d++) {
    auto e = arg[d];

    int out_e = 1;
    if (e < 0) {
      out_e = -1;
    } else if (relative) {
      int src_d = src_dims ? src_dims[d] : d;
      out_e = src_d < 0 ? 1 : round_int(e * in_shape[src_d]);
    } else {
      out_e = e;
    }

    out_shape[d] = out_e;
  }
}

template <typename Backend>
void Reshape<Backend>::ValidateRelativeShapeNDim(int ndim, bool last_dim_inferred) const {
  if (use_src_dims_) {
    if (ndim + 0_uz != src_dims_.size()) {
      DALI_FAIL(make_string(
        "`src_dims` and `rel_shape` have different lengths: ",
        src_dims_.size(), " vs ", ndim));
    }
  } else {
    if (ndim > input_shape_.sample_dim() + last_dim_inferred) {
      DALI_FAIL(make_string(
        "`rel_shape` has more elements (", ndim, ") than "
        "there are dimensions in the input (", input_shape_.sample_dim(), "). "
        "To add new dimensions, use `src_dims` to specify the mapping between "
        "input and output dimensions."));
    }
  }
}


template <typename Backend>
template <bool relative, typename Extent>
void Reshape<Backend>::ShapeFromInput(
      const TensorListView<StorageCPU, const Extent> &shape,
      std::bool_constant<relative>) {
  DALI_ENFORCE(shape.sample_dim() == 1 || (shape.sample_dim() == 2 && shape.num_samples() == 1),
    make_string("shape input must be a list of 1D tensors or a single 2D tensor"));
  if (shape.sample_dim() == 2) {
    auto shape_tensor = shape[0];
    int N = shape_tensor.shape[0];
    DALI_ENFORCE(N == input_shape_.num_samples(), make_string(
      "The new shape must have same number of samples. Got ",
      output_shape_.num_samples(), ", expected ", N));
    int dim = shape_tensor.shape[1];
    if (relative) {
      bool last_dim_negative = true;
      for (int i = 0; i < N; i++) {
        if (shape_tensor(i)[dim - 1] >= 0) {
          last_dim_negative = false;
          break;
        }
      }
      ValidateRelativeShapeNDim(dim, last_dim_negative);
    }
    output_shape_.resize(N, dim);
    for (int i = 0; i < N; i++) {
      SetOutputSampleShape<relative>(output_shape_.tensor_shape_span(i),
                                     input_shape_.tensor_shape_span(i),
                                     shape_tensor(i),
                                     use_src_dims_ ? src_dims_.data() : nullptr,
                                     dim);
    }
  } else {
    int N = shape.num_samples();
    DALI_ENFORCE(N == input_shape_.num_samples(),
      make_string("The new shape must have same number of samples. Got ",
      output_shape_.num_samples(), ", expected ", N));
    int sample_dim = 0;
    for (int i = 0; i < N; i++) {
      int current_sample_dim = shape.tensor_shape_span(i)[0];
      if (i == 0) {
        sample_dim = current_sample_dim;
        output_shape_.resize(N, sample_dim);
      } else {
        DALI_ENFORCE(current_sample_dim == sample_dim,
          make_string("All samples must have the same number of dimensions"));
      }

      if (relative) {
        bool last_dim_negative = shape.tensor_data(i)[sample_dim - 1] < 0;
        ValidateRelativeShapeNDim(sample_dim, last_dim_negative);
      }
      SetOutputSampleShape<relative>(output_shape_.tensor_shape_span(i),
                                     input_shape_.tensor_shape_span(i),
                                     shape.tensor_data(i),
                                     use_src_dims_ ? src_dims_.data() : nullptr,
                                     sample_dim);
    }
  }
}

template <typename Backend>
template <bool _relative, typename TensorListLike>
void Reshape<Backend>::ShapeFromInput(const TensorListLike &tl,
                                      std::bool_constant<_relative> relative) {
  if (relative) {
    this->ShapeFromInput(view<const float>(tl), relative);
  } else {
    TYPE_SWITCH(tl.type(), type2id, type,
      (int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t),
      (this->ShapeFromInput(view<const type>(tl), relative);),
      (DALI_FAIL(make_string("Shape input must have integral type; got: ",
                 tl.type()));)
    );  // NOLINT
  }
}

template <typename Backend>
void Reshape<Backend>::CalculateOutputShape(const Workspace &ws) {
  auto &in = ws.Input<Backend>(0);
  input_shape_ = in.shape();
  const int N = input_shape_.num_samples();
  switch (shape_source_) {
    case ShapeSource::Arg:
      if (use_rel_shape_) {
        ValidateRelativeShapeNDim(rel_uniform_shape_.size(), rel_uniform_shape_.back() < 0);
        output_shape_.resize(N, rel_uniform_shape_.size());
        for (int i = 0; i < N; i++) {
          for (int d = 0; d < output_shape_.sample_dim(); d++) {
            int src_d = use_src_dims_ ? src_dims_[d]
                                      : rel_uniform_shape_[d] < 0 ? -1 : d;
            int out_e = round_int(rel_uniform_shape_[d] *
              (src_d < 0 ? 1 : input_shape_.tensor_shape_span(i)[src_d]));
            output_shape_.tensor_shape_span(i)[d] = out_e;
          }
        }
      } else {
        if (output_shape_.num_samples() != N) {
          output_shape_ = uniform_list_shape(N, uniform_shape_);
        }
      }
      break;
    case ShapeSource::ArgInput:
      if (use_rel_shape_)
        ShapeFromInput(ws.ArgumentInput("rel_shape"), std::true_type());
      else
        ShapeFromInput(ws.ArgumentInput("shape"), std::false_type());
      break;
    case ShapeSource::Input:
      ShapeFromInput(ws.Input<CPUBackend>(1), std::false_type());
      break;
    case ShapeSource::None:
      if (!use_src_dims_) {
        output_shape_ = input_shape_;
        break;
      }

      output_shape_.resize(N, src_dims_.size());
      for (int i = 0; i < N; i++) {
        for (size_t d = 0; d < src_dims_.size(); d++) {
          const int src_d = src_dims_[d];
          output_shape_.tensor_shape_span(i)[d] =
            src_d < 0 ? 1 : input_shape_.tensor_shape_span(i)[src_d];
        }
        DALI_ENFORCE(
          output_shape_.tensor_size(i) == input_shape_.tensor_size(i),
          make_string("The volume of the new shape should match the"
          " one of the original shape. Requested a shape with ", output_shape_.tensor_size(i),
          " elements but the original shape has ", input_shape_.tensor_size(i), " elements."));
      }
      break;
  }

  int64_t input_element_size = in.type_info().size();
  int64_t output_element_size = output_type_->size();

  if (shape_source_ != ShapeSource::None) {
    for (int i = 0; i < N; i++) {
      auto actual_volume = volume(input_shape_.tensor_shape_span(i));
      auto out_sample_shape = output_shape_.tensor_shape_span(i);
      int wildcard_dim = wildcard_dim_;
      if (shape_source_ != ShapeSource::Arg) {
        for (int d = 0; d < out_sample_shape.size(); d++)
          if (out_sample_shape[d] < 0) {
            DALI_ENFORCE(wildcard_dim < 0,
                         make_string("Only one dimension can have negative extent"));
            wildcard_dim = d;
          }
      }
      if (wildcard_dim >= 0) {
        // calculate the volume in all other dimensions
        out_sample_shape[wildcard_dim] = 1;
        auto other_dims_volume = volume(out_sample_shape);
        if (other_dims_volume == 0) {
          if (wildcard_dim < input_shape_.sample_dim() &&
              output_shape_.sample_dim() > input_shape_.sample_dim()) {
            DALI_FAIL(make_string("Cannot infer the extent of dimension ", wildcard_dim,
              " for ", output_shape_.sample_dim(), "D output and ", input_shape_.sample_dim(),
              "D input. Use `src_dims` to achieve more complex "
              "mapping of input to output dimensions."));
          } else {
            DALI_FAIL(make_string("Cannot infer the extent of dimension ", wildcard_dim,
              " when the volume of remaining dimensions is 0. Input shape: ", input_shape_[i]));
          }
        }
        // try to make wildcard dim match the input volume - if it fails,
        // volume comparison will fail
        out_sample_shape[wildcard_dim] = (actual_volume * input_element_size) /
                                         (other_dims_volume * output_element_size);
      }

      auto requested_volume = volume(out_sample_shape);
      DALI_ENFORCE(actual_volume * input_element_size == requested_volume * output_element_size,
        make_string(
          "Input and output samples must occupy the same size in bytes."
          "\nSample index:     ", i,
          "\nActual volume:    ", actual_volume,
          "\n     in bytes:    ", actual_volume * input_element_size,
          "\nRequested volume: ", requested_volume,
          "\n     in bytes:    ", requested_volume * output_element_size,
          "\nInput shape:\t", input_shape_[i],
          "\nRequested shape:\t", output_shape_[i]));
    }
  } else if (output_element_size != input_element_size) {
    for (int i = 0; i < N; i++) {
      auto out_sample_shape = output_shape_.tensor_shape_span(i);
      auto &innermost = out_sample_shape[output_shape_.sample_dim()-1];
      DALI_ENFORCE((innermost * input_element_size) % output_element_size == 0,
        make_string("The size, in bytes, of the innermost dimension is not divisible "
        "by the sizes of the requested output type."));
      innermost = innermost * input_element_size / output_element_size;
    }
  }
}

template <typename Backend>
TensorLayout Reshape<Backend>::GetOutputLayout(const Workspace &ws) const {
  if (!layout_.empty()) {
    DALI_ENFORCE(output_shape_.sample_dim() == layout_.ndim(), make_string(
      "Requested layout '", layout_, "' is not compatible with a ",
      output_shape_.sample_dim(), "D shape"));
    return layout_;
  } else if (use_layout_) {
    // layout was explicitly cleared
    return TensorLayout();
  }
  auto &in = ws.Input<Backend>(0);
  auto in_layout = in.GetLayout();
  return in_layout.ndim() == output_shape_.sample_dim() ? in_layout : TensorLayout();
}

template <>
void Reshape<CPUBackend>::RunImpl(Workspace &ws) {
  auto &out = ws.Output<CPUBackend>(0);
  auto &in = ws.Input<CPUBackend>(0);
  TensorLayout layout = GetOutputLayout(ws);
  out.ShareData(in);
  out.Resize(output_shape_, output_type_->id());
  int N = output_shape_.num_samples();
  for (int i = 0; i < N; i++) {
    assert(out.raw_tensor(i) == in.raw_tensor(i));
    assert(out.tensor_shape(i) == output_shape_[i]);
  }
  out.SetLayout(layout);
}

template <>
void Reshape<GPUBackend>::RunImpl(Workspace &ws) {
  auto &out = ws.Output<GPUBackend>(0);
  auto &in = ws.Input<GPUBackend>(0);
  TensorLayout layout = GetOutputLayout(ws);
  out.Reset();
  out.ShareData(in);
  out.Resize(output_shape_, output_type_->id());
  int N = output_shape_.num_samples();
  for (int i = 0; i < N; i++) {
    assert(out.raw_tensor(i) == in.raw_tensor(i));
  }
  out.SetLayout(layout);
}

template <typename Backend>
void Reshape<Backend>::CheckSrcDims(const Workspace &ws) {
  if (!use_src_dims_) {
    return;
  }

  const auto &in = ws.Input<Backend>(0);
  const auto &input_shape = in.shape();
  const int ndim = input_shape.sample_dim();
  for (size_t d = 0; d < src_dims_.size(); d++) {
    DALI_ENFORCE(-1 <= src_dims_[d] && src_dims_[d] < ndim,
      make_string("`src_dims[", d, "] == ", src_dims_[d], "` is out of bounds.\n"
        "The indices in `src_dims` should be either valid dimension indices in "
        "range [0..", ndim-1, "] or -1 to insert a new dimension."));
  }
}

DALI_REGISTER_OPERATOR(Reshape, Reshape<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Reshape, Reshape<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(Reinterpret, Reshape<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Reinterpret, Reshape<GPUBackend>, GPU);

}  // namespace namespace dali

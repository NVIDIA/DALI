// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

namespace dali {

DALI_SCHEMA(Reshape)
  .DocStr(R"(Treats content of the input as if it had a different shape and/or layout.
  The buffer contents are not copied.)")
  .NumInput(1, 2)
  .NumOutput(1)
  .InputDox(0, "data", "TensorList", "Data to be reshaped")
  .InputDox(1, "shape_input", "1D TensorList of integers", "Same as `shape` keyword argument")
  .PassThrough({{0, 0}})
  .AllowSequences()
  .SupportVolumetric()
  .AddOptionalArg<int>("shape", "The desired shape of the output. Number of dimensions "
                  "must not exceed the number of dimensions of the input. There can be one "
                  "negative extent which receives the size required to match the input volume, "
                  "e.g. input of shape `[480, 640, 3]` and `shape = [240, -1]` would get the "
                  "shape [240, 3840].\n"
                  "NOTE: `rel_shape` and `shape` are mutually exclusive.",
                  std::vector<int>(), true)
  .AddOptionalArg<float>("rel_shape", "The relative shape of the output. Number of dimensions "
                  "must not exceed the number of dimensions of the input. There can be one "
                  "negative extent which receives the size required to match the input volume, "
                  "e.g. input of shape `[480, 640, 3]` and `rel_shape = [0.5, -1]` would get the "
                  "shape [240, 3840].\n"
                  "NOTE: `rel_shape` and `shape` are mutually exclusive.",
                  std::vector<float>(), true)
  .AddOptionalArg("layout", "New layout for the data. If not specified, the output layout is "
                  "preserved if number of dimension matches existing layout or reset to empty "
                  "otherwise. If set and not empty, the layout must match the dimensionality of "
                  "the output.",
                  TensorLayout(""));

DALI_SCHEMA(Reinterpret)
  .DocStr(R"(Treats content of the input as if it had a different type, shape and/or layout.
  The buffer contents are not copied.)")
  .NumInput(1, 2)
  .NumOutput(1)
  .InputDox(0, "data", "TensorList", "Data to be reshaped")
  .InputDox(1, "shape_input", "1D TensorList of integers", "Same as `shape` keyword argument")
  .PassThrough({{0, 0}})
  .AllowSequences()
  .SupportVolumetric()
  .AddOptionalArg("dtype", "The desired output type. The total size in bytes of the output "
                  "must match the input. If no shape is provided, the innermost dimension is "
                  "adjusted accordingly. If the byte size of the innermost dimension is not "
                  "divisible by the size of the target type, an error is thrown.", DALI_NO_TYPE)
  .AddParent("Reshape");

template <typename Backend>
Reshape<Backend>::Reshape(const OpSpec &spec) : Base(spec) {
  bool has_shape_input = spec.NumRegularInput() == 2;
  bool has_shape_arg = spec.HasArgument("shape") || spec.HasTensorArgument("shape");
  bool has_layout_arg = spec.HasArgument("layout");
  bool has_rel_shape_arg = spec.HasArgument("rel_shape") || spec.HasTensorArgument("rel_shape");
  if (spec.HasArgument("dtype"))
    output_type_id_ = spec.GetArgument<DALIDataType>("dtype");
  DALI_ENFORCE(has_shape_input + has_shape_arg + has_rel_shape_arg <= 1, make_string(OpName(),
    ": shape input, `shape` argument and `rel_shape` argument are mutually exclusive"));

  bool does_reshape = has_shape_input || has_shape_arg || has_rel_shape_arg || has_layout_arg;
  if (!has_shape_input && !has_shape_arg && !has_rel_shape_arg && !has_layout_arg) {
    bool can_have_dtype = spec.GetSchema().HasArgument("dtype");
    if (can_have_dtype) {
      DALI_ENFORCE(output_type_id_ != DALI_NO_TYPE, make_string(OpName(),
                   " is no-op: arguments specify neither new shape, layout nor type."));
    } else {
      DALI_FAIL(make_string(OpName(),
                " is no-op: arguments specify neither new shape nor layout."));
    }
  }
  use_layout_ = has_layout_arg;
  use_rel_shape_ = false;
  if (has_shape_arg) {
    if (spec.HasTensorArgument("shape")) {
      shape_source_ = ShapeSource::ArgInput;
    } else {
      auto shape_vec = spec.GetRepeatedArgument<int>("shape");
      DALI_ENFORCE(!shape_vec.empty(),
                   make_string(OpName(), ": `shape` specified as an empty list"));

      uniform_shape_.resize(shape_vec.size());
      int num_negative = 0;
      for (int i = 0; i < uniform_shape_.sample_dim(); i++) {
        if (shape_vec[i] < 0) {
          DALI_ENFORCE(++num_negative == 1, make_string(OpName(),
            ": Only one negative extent is allowed; got: ", uniform_shape_));
          uniform_shape_[i] = 0;
          wildcard_dim_ = i;
        } else {
          DALI_ENFORCE(shape_vec[i] != 0, make_string(OpName(), ": extent of 0 is illegal; got: ",
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
      DALI_ENFORCE(!rel_uniform_shape_.empty(), make_string(OpName(),
                   ": `rel_shape` specified as an empty list"));
      int num_negative = 0;
      int out_dims = rel_uniform_shape_.size();
      for (int i = 0; i < out_dims; i++) {
        if (rel_uniform_shape_[i] < 0) {
          DALI_ENFORCE(++num_negative == 1, make_string(OpName(),
            ": Only one negative extent is allowed; got: ", uniform_shape_));
          wildcard_dim_ = i;
        } else {
          DALI_ENFORCE(rel_uniform_shape_[i] != 0,
                       make_string(OpName(), ": zero extent is illegal"));
        }
      }
      shape_source_ = ShapeSource::Arg;
    }
  } else if (has_shape_input) {
    DALI_ENFORCE(spec.InputDevice(1) == "cpu",
                 make_string(OpName(), ": Output shapes must be provided as a CPU input"));
    shape_source_ = ShapeSource::Input;
  }
  if (has_layout_arg) {
    layout_ = spec.GetArgument<TensorLayout>("layout");
  }
}

template <typename Backend>
bool Reshape<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  output_desc.resize(1);

  output_type_ = output_type_id_ != DALI_NO_TYPE
      ? &TypeTable::GetTypeInfo(output_type_id_)
      : &ws.template InputRef<Backend>(0).type();

  CalculateOutputShape(ws);
  output_desc[0].type = *output_type_;
  output_desc[0].shape = output_shape_;
  // return false, because we don't want the executor to allocate anything
  // - this operator returns pointer to input memory
  return false;
}

template <typename Backend>
template <typename Extent>
void Reshape<Backend>::ShapeFromInput(
      const TensorListView<StorageCPU, Extent> &shape) {
  constexpr bool relative = std::is_floating_point<Extent>::value;
  DALI_ENFORCE(shape.sample_dim() == 1 || (shape.sample_dim() == 2 && shape.num_samples() == 1),
    make_string(OpName(), ": shape input must be a list of 1D tensors or a single 2D tensor"));
  if (shape.sample_dim() == 2) {
    auto shape_tensor = shape[0];
    int N = shape_tensor.shape[0];
    DALI_ENFORCE(N == input_shape_.num_samples(), make_string(OpName(),
      ": the new shape mush have same number of samples. Got ",
      output_shape_.num_samples(), ", expected ", N));
    int dim = shape_tensor.shape[1];
    output_shape_.resize(N, dim);
    for (int i = 0; i < N; i++) {
      for (int d = 0; d < dim; d++) {
        Extent e = *shape_tensor(i, d);
        int out_e = e < 0 ? -1 : relative ? round_int(e * input_shape_.tensor_shape_span(i)[d]) : e;
        output_shape_.tensor_shape_span(i)[d] = out_e;
      }
    }
  } else {
    int N = shape.num_samples();
    DALI_ENFORCE(N == input_shape_.num_samples(),
      make_string(OpName(), ": the new shape mush have same number of samples. Got ",
      output_shape_.num_samples(), ", expected ", N));
    int sample_dim;
    for (int i = 0; i < N; i++) {
      int current_sample_dim = shape.tensor_shape_span(i)[0];
      if (i == 0) {
        sample_dim = current_sample_dim;
        output_shape_.resize(N, sample_dim);
      } else {
        DALI_ENFORCE(current_sample_dim == sample_dim,
          make_string(OpName(), ": all samples must have the same number of dimensions"));
      }

      for (int d = 0; d < sample_dim; d++) {
        Extent e = shape.tensor_data(i)[d];
        int out_e = e < 0 ? -1 : relative ? round_int(e * input_shape_.tensor_shape_span(i)[d]) : e;
        output_shape_.tensor_shape_span(i)[d] = out_e;
      }
    }
  }
}

template <typename Backend>
template <typename TensorListLike>
void Reshape<Backend>::ShapeFromInput(const TensorListLike &tl, bool relative) {
  if (relative) {
    this->ShapeFromInput(view<const float>(tl));
  } else {
    TYPE_SWITCH(tl.type().id(), type2id, type,
      (int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t),
      (this->ShapeFromInput(view<const type>(tl));),
      (DALI_FAIL(make_string(OpName(), ": shape input must have integral type; got: ",
                 tl.type().name(), " (id = ", static_cast<int>(tl.type().id()), ")"));)
    );  // NOLINT
  }
}

template <typename Backend>
void Reshape<Backend>::CalculateOutputShape(const Workspace &ws) {
  auto &in = ws.template InputRef<Backend>(0);
  input_shape_ = in.shape();
  const int N = input_shape_.num_samples();
  switch (shape_source_) {
    case ShapeSource::Arg:
      if (use_rel_shape_) {
        output_shape_.resize(N, rel_uniform_shape_.size());
        for (int i = 0; i < N; i++) {
          for (int d = 0; d < output_shape_.sample_dim(); d++) {
            int out_e = round_int(rel_uniform_shape_[d] * input_shape_.tensor_shape_span(i)[d]);
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
        ShapeFromInput(ws.ArgumentInput("rel_shape"), true);
      else
        ShapeFromInput(ws.ArgumentInput("shape"), false);
      break;
    case ShapeSource::Input:
      ShapeFromInput(ws.template InputRef<CPUBackend>(1), false);
      break;
    case ShapeSource::None:
      output_shape_ = input_shape_;
      break;
  }

  int64_t input_element_size = in.type().size();
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
                         make_string(OpName(), ": Only one dimension can have negative extent"));
            wildcard_dim = d;
          }
      }
      if (wildcard_dim >= 0) {
        // calculate the volume in all other dimensions
        out_sample_shape[wildcard_dim] = 1;
        auto other_dims_volume = volume(out_sample_shape);
        // try to make wildcard dim match the input volume - if it fails,
        // volume comparison will fail
        out_sample_shape[wildcard_dim] = (actual_volume * input_element_size) /
                                         (other_dims_volume * output_element_size);
      }

      auto requested_volume = volume(out_sample_shape);
      DALI_ENFORCE(actual_volume * input_element_size == requested_volume * output_element_size,
        make_string(OpName(),
          ": Input and output samples must occupy the same size in bytes."
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
        make_string(OpName(), ": the size, in bytes, of the innermost dimension is not divisible "
        "by the sizes of the requested output type."));
      innermost = innermost * input_element_size / output_element_size;
    }
  }
}

template <typename Backend>
TensorLayout Reshape<Backend>::GetOutputLayout(const Workspace &ws) const {
  if (!layout_.empty()) {
    DALI_ENFORCE(output_shape_.sample_dim() == layout_.ndim(), make_string(OpName(),
      ": requested layout '", layout_, "' is not compatible with a ",
      output_shape_.sample_dim(), "D shape"));
    return layout_;
  } else if (use_layout_) {
    // layout was explicitly cleared
    return TensorLayout();
  }
  auto in_layout = this->InputLayout(ws, 0);
  return in_layout.ndim() == output_shape_.sample_dim() ? in_layout : TensorLayout();
}

template <>
void Reshape<CPUBackend>::RunImpl(HostWorkspace &ws) {
  auto &out = ws.OutputRef<CPUBackend>(0);
  auto &in = ws.InputRef<CPUBackend>(0);
  TensorLayout layout = GetOutputLayout(ws);
  out.ShareData(&in);
  out.Resize(output_shape_, *output_type_);
  int N = output_shape_.num_samples();
  for (int i = 0; i < N; i++) {
    assert(out[i].raw_data() == in[i].raw_data());
    assert(out[i].shape() == output_shape_[i]);
  }
  out.SetLayout(layout);
}

template <>
void Reshape<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  auto &out = ws.OutputRef<GPUBackend>(0);
  auto &in = ws.InputRef<GPUBackend>(0);
  TensorLayout layout = GetOutputLayout(ws);
  out.Reset();
  out.ShareData(&in);
  out.Resize(output_shape_, *output_type_);
  int N = output_shape_.num_samples();
  for (int i = 0; i < N; i++) {
    assert(out.raw_tensor(i) == in.raw_tensor(i));
  }
  out.SetLayout(layout);
}

DALI_REGISTER_OPERATOR(Reshape, Reshape<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Reshape, Reshape<GPUBackend>, GPU);
DALI_REGISTER_OPERATOR(Reinterpret, Reshape<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Reinterpret, Reshape<GPUBackend>, GPU);

}  // namespace namespace dali

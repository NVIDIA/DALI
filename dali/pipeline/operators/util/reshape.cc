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

#include "dali/pipeline/operators/util/reshape.h"

#include <vector>
#include "dali/pipeline/data/views.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/tensor_shape_print.h"

namespace dali {

DALI_SCHEMA(Reshape)
  .DocStr(R"code(Treats content of the input as if it had a different shape and layout.)code")
  .NumInput(1, 2)
  .NumOutput(1)
  .AddOptionalArg<int>("shape", "The desired shape of the output. Number of elements in "
                                "each sample must match that of the input sample.",
                                std::vector<int>(), true)
  .AddOptionalArg("layout",     "New layout for the data. If not specified, the output layout "
                                "is preserved if number of dimension matches existing layout "
                                "or reset to empty otherwise",
                                "");


template <typename Backend>
Reshape<Backend>::Reshape(const OpSpec &spec) : Base(spec) {
  bool has_shape_input = spec.NumRegularInput() == 2;
  bool has_shape_arg = spec.HasArgument("shape");
  bool has_layout_arg = spec.HasArgument("layout");
  DALI_ENFORCE(!(has_shape_input && has_shape_arg),
    "Reshape: use either shape input or shape argument, not both");
  DALI_ENFORCE(has_shape_input || has_shape_arg || has_layout_arg,
    "Reshape is no-op: arguments specify neither new shape nor layout.");
  if (has_shape_arg) {
    if (spec.HasTensorArgument("shape")) {
      shape_source_ = ShapeSource::ArgInput;
    } else {
      auto shape_vec = spec.GetRepeatedArgument<int>("shape");
      DALI_ENFORCE(!shape_vec.empty(), "Reshape: `shape` specified as empty list");
      uniform_shape_.resize(shape_vec.size());
      for (int i = 0; i < uniform_shape_.sample_dim(); i++) {
        DALI_ENFORCE(shape_vec[i] > 0, "Reshape: all extents must be positive; got: " +
            std::to_string(uniform_shape_));
        uniform_shape_[i] = shape_vec[i];
      }
      shape_source_ = ShapeSource::Arg;
    }
  }
  if (has_shape_input) {
    DALI_ENFORCE(spec.InputDevice(1) == "cpu", "Output shapes must be provided as a CPU input");
    shape_source_ = ShapeSource::Input;
  }
  if (has_layout_arg) {
    layout_ = spec.GetArgument<TensorLayout>("layout");
  }
}

template <typename Backend>
bool Reshape<Backend>::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  output_desc.resize(1);
  CalculateOutputShape(ws);
  output_desc[0].type = ws.template InputRef<Backend>(0).type();
  output_desc[0].shape = output_shape_;
  return false;
}

template <typename Backend>
template <typename Integer>
void Reshape<Backend>::ShapeFromInput(
      const kernels::TensorListView<kernels::StorageCPU, Integer> &shape) {
  DALI_ENFORCE(shape.sample_dim() == 1 || (shape.sample_dim() == 2 && shape.num_samples() == 1),
    "Reshape: shape input must be a list of 1D tensors or a single 2D tensor");
  if (shape.sample_dim() == 2) {
    auto shape_tensor = shape[0];
    int N = shape_tensor.shape[0];
    int dim = shape_tensor.shape[1];
    output_shape_.resize(N, dim);
    for (int i = 0; i < N; i++) {
      for (int d = 0; d < dim; d++) {
        output_shape_.tensor_shape_span(i)[d] = *shape_tensor(i, d);
      }
    }
  } else {
    int N = shape.num_samples();
    int sample_dim;
    for (int i = 0; i < N; i++) {
      int current_sample_dim = shape.tensor_shape_span(i)[0];
      if (i) {
        DALI_ENFORCE(current_sample_dim == sample_dim,
          "Reshape: all samples must have the same number of dimensions");
      } else {
        sample_dim = current_sample_dim;
        output_shape_.resize(N, sample_dim);
      }

      for (int d = 0; d < sample_dim; d++) {
        output_shape_.tensor_shape_span(i)[d] = shape.tensor_data(i)[d];
      }
    }
  }
}

template <typename Backend>
template <typename TensorListLike>
void Reshape<Backend>::ShapeFromInput(const TensorListLike &tl) {
  TYPE_SWITCH(tl.type().id(), type2id, type, (int32_t),
    (this->ShapeFromInput(view<const type>(tl));),
    (DALI_FAIL("Reshape: shape input must have integral type; got: " + tl.type().name());)
  );  // NOLINT
}

template <typename Backend>
void Reshape<Backend>::CalculateOutputShape(const Workspace &ws) {
  input_shape_ = ws.template InputRef<Backend>(0).shape();
  const int N = input_shape_.num_samples();
  switch (shape_source_) {
    case ShapeSource::Arg:
      if (output_shape_.num_samples() != N) {
        output_shape_ = kernels::uniform_list_shape(N, uniform_shape_);
      }
      break;
    case ShapeSource::ArgInput:
      ShapeFromInput(ws.ArgumentInput("shape"));
      break;
    case ShapeSource::Input:
      ShapeFromInput(ws.template InputRef<CPUBackend>(1));
      break;
    case ShapeSource::None:
      output_shape_ = input_shape_;
      return;
  }

  DALI_ENFORCE(output_shape_.num_samples() == N,
    "Reshape: the new shape mush have same number of samples. Got " +
    to_string(output_shape_.num_samples()) + ", expected " + to_string(N));

  for (int i = 0; i < N; i++) {
    auto actual_volume = volume(input_shape_.tensor_shape_span(i));
    auto requested_volume = volume(output_shape_.tensor_shape_span(i));
    DALI_ENFORCE(actual_volume == requested_volume,
      "Input and output samples should have the same number of elements");
  }
}

template <typename Backend>
TensorLayout Reshape<Backend>::GetOutputLayout(const Workspace &ws) const {
  if (!layout_.empty()) {
    DALI_ENFORCE(output_shape_.sample_dim() == layout_.ndim(), "Reshape: requested layout '" +
      layout_.str() + "' not compatible with " + to_string(output_shape_.sample_dim()) + "D shape");
    return layout_;
  }
  auto in_layout = this->InputLayout(ws, 0);
  return in_layout.ndim() == output_shape_.sample_dim() ? in_layout : TensorLayout();
}

template <>
void Reshape<CPUBackend>::RunImpl(HostWorkspace &ws) {
  auto &out = ws.OutputRef<CPUBackend>(0);
  auto &in = ws.InputRef<CPUBackend>(0);
  out.Resize(output_shape_);
  out.SetContiguous(false);
  int N = output_shape_.num_samples();
  for (int i = 0; i < N; i++) {
    out[i].ShareData(&in[i]);
    out[i].Resize(output_shape_[i]);
    assert(out[i].raw_data() == in[i].raw_data());
  }
}

template <>
void Reshape<GPUBackend>::RunImpl(DeviceWorkspace &ws) {
  auto &out = ws.OutputRef<GPUBackend>(0);
  auto &in = ws.InputRef<GPUBackend>(0);
  out.ShareData(&in);
  out.Resize(output_shape_);
  int N = output_shape_.num_samples();
  for (int i = 0; i < N; i++) {
    assert(out.raw_tensor(i) == in.raw_tensor(i));
  }
}

DALI_REGISTER_OPERATOR(Reshape, Reshape<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(Reshape, Reshape<GPUBackend>, GPU);


}  // namespace namespace dali

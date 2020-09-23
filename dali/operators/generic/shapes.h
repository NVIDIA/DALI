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

#ifndef DALI_OPERATORS_GENERIC_SHAPES_H_
#define DALI_OPERATORS_GENERIC_SHAPES_H_

#include <memory>
#include <vector>
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/static_switch.h"

namespace dali {

template <typename Backend>
class Shapes : public Operator<Backend> {
 public:
  Shapes(const Shapes &) = delete;
  explicit Shapes(const OpSpec &spec) : Operator<Backend>(spec) {
    output_type_ = spec.GetArgument<DALIDataType>("dtype");
    switch (output_type_) {
    case DALI_INT32:
    case DALI_UINT32:
    case DALI_INT64:
    case DALI_UINT64:
    case DALI_FLOAT:
    case DALI_FLOAT64:
      break;
    default:
      {
        auto &name = TypeTable::GetTypeInfo(output_type_).name();
        DALI_FAIL("Operator Shapes can return the output as one of the following:\n"
          "int32, uint32, int64, uint64, float or double;\n"
          "requested: " + name);
        break;
      }
    }
  }
  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    output_desc.resize(1);
    output_desc[0].type = TypeTable::GetTypeInfo(output_type_);
    decltype(auto) shape = GetInputShape(ws);
    output_desc[0].shape = ShapeShape(shape);
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override {
    RunBackend(ws);
  }

  template <typename type>
  void ConvertShape(TensorList<CPUBackend> &out, const TensorListShape<> &shape) {
    int n = out.ntensor();
    assert(n == shape.num_samples());
    for (int i = 0; i < n; i++) {
      type *data = out.mutable_tensor<type>(i);
      auto sample_shape = shape.tensor_shape_span(i);
      for (int j = 0; j < shape.sample_dim(); j++)
        data[j] = sample_shape[j];
    }
  }

  template <typename type>
  void ConvertShape(TensorVector<CPUBackend> &out, const TensorListShape<> &shape) {
    int n = out.size();
    assert(n == shape.num_samples());
    for (int i = 0; i < n; i++) {
      type *data = out[i].mutable_data<type>();
      auto sample_shape = shape.tensor_shape_span(i);
      for (int j = 0; j < shape.sample_dim(); j++)
        data[j] = sample_shape[j];
    }
  }

  template <typename CPUTensorListOrVector>
  void ConvertShape(CPUTensorListOrVector &out, const TensorListShape<> &shape) {
    TYPE_SWITCH(output_type_, type2id, type,
                (int32_t, uint32_t, int64_t, uint64_t, float, double),
      (ConvertShape<type>(out, shape);),
      (DALI_FAIL(make_string("Unsupported type for Shapes: ", output_type_))));
  }

  void RunBackend(DeviceWorkspace &ws) {
    if (!tmp_.raw_data()) {
      auto &type = TypeTable::GetTypeInfo(output_type_);
      tmp_.set_type(type);
      tmp_.set_pinned(true);
    }

    auto &output = ws.OutputRef<GPUBackend>(0);
    tmp_.Resize(output.shape());
    ConvertShape(tmp_, GetInputShape(ws));
    output.Copy(tmp_, ws.stream());
  }

  void RunBackend(HostWorkspace &ws) {
    ConvertShape(ws.OutputRef<CPUBackend>(0), GetInputShape(ws));
  }

  static TensorListShape<1> ShapeShape(const TensorListShape<> &shape) {
    return uniform_list_shape<1>(shape.num_samples(), { shape.sample_dim() });
  }

  static const TensorListShape<> &GetInputShape(const DeviceWorkspace &ws) {
    if (ws.InputIsType<GPUBackend>(0)) {
      return ws.InputRef<GPUBackend>(0).shape();
    } else {
      return ws.InputRef<CPUBackend>(0).shape();
    }
  }

  static auto GetInputShape(const HostWorkspace &ws) {
    return ws.InputRef<CPUBackend>(0).shape();
  }

 private:
  TensorList<CPUBackend> tmp_;
  DALIDataType output_type_ = DALI_INT64;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_SHAPES_H_

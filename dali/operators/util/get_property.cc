// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/util/get_property.h"
#include "dali/pipeline/data/types.h"

namespace dali {

DALI_SCHEMA(GetProperty)
    .DocStr(
        R"code(Returns a property of the tensor passed as an input.
The type of the output will depend on the ``key`` of the requested property.

Some of the properties are returned as a byte-array. The easiest way to convert them to string
would be to )code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg("key",
            R"code(Specifies, which property is requested. The following properties are supported:

* ``"source_info"``: Returned type: byte-array.
                     Information about the origin of the sample. The actual origin may differ according
                     to the source of the data. For example, when the Tensor is read via :meth:`fn.readers.file`,
                     the ``source_info`` property will contain full path of that file. When the Tensor
                     is read by :meth:`fn.readers.webdataset`, the property will contain full path of the tar
                     archive and the index of the component in that archive. Lastly, when the Tensor
                     is loaded via :meth:`fn.external_source`, the ``source_info`` will be empty.
* ``"layout"``: Returned type: byte-array
                Data layout in the given Tensor.
)code",
            DALI_STRING);


DALI_REGISTER_OPERATOR(GetProperty, GetProperty<CPUBackend>, CPU)
DALI_REGISTER_OPERATOR(GetProperty, GetProperty<GPUBackend>, GPU)

namespace detail {
namespace {

template <typename Backend, typename BatchContainer = batch_container_t<Backend>>
struct SourceInfo : public Property<Backend> {
  TensorListShape<> GetShape(const BatchContainer& input) override {
    TensorListShape<> ret{static_cast<int>(input.num_samples()), 1};
    for (int i = 0; i < ret.size(); i++) {
      ret.set_tensor_shape(i, {static_cast<int>(GetSourceInfo(input, i).length())});
    }
    return ret;
  }

  DALIDataType GetType(const BatchContainer&) override {
    return DALI_UINT8;
  }

  void FillOutput(workspace_t<Backend>& ws) override {
    const auto& input = ws.template Input<Backend>(0);
    auto& output = ws.template Output<Backend>(0);
    for (size_t sample_id = 0; sample_id < input.num_samples(); sample_id++) {
      auto si = GetSourceInfo(input, sample_id);
      output[sample_id].Copy(make_cspan((const uint8_t*)si.c_str(), si.length()), nullptr);
    }
  }

 private:
  std::string GetSourceInfo(const BatchContainer& input, size_t idx) {
    return input[idx].GetMeta().GetSourceInfo();
  }
};

template <>
std::string SourceInfo<GPUBackend>::GetSourceInfo(const TensorList<GPUBackend>& input, size_t idx) {
  return input.GetMeta(static_cast<int>(idx)).GetSourceInfo();
}

template <>
void SourceInfo<GPUBackend>::FillOutput(workspace_t<GPUBackend>& ws) {
  const auto& input = ws.template Input<GPUBackend>(0);
  auto& output = ws.template Output<GPUBackend>(0);
  for (size_t sample_id = 0; sample_id < input.num_samples(); sample_id++) {
    auto si = GetSourceInfo(input, sample_id);
    auto output_ptr = output.raw_mutable_tensor(static_cast<int>(sample_id));
    assert(ws.has_stream());  // GPU operator must have CUDA stream assigned
    cudaMemcpyAsync(output_ptr, si.c_str(), si.length(), cudaMemcpyDefault, ws.stream());
  }
}


template <typename Backend, typename BatchContainer = batch_container_t<Backend>>
struct Layout : public Property<Backend> {
  TensorListShape<> GetShape(const BatchContainer& input) override {
    // Every tensor in the output has the same number of dimensions
    return uniform_list_shape(input.num_samples(), {GetLayout(input, 0).size()});
  }

  DALIDataType GetType(const BatchContainer&) override {
    return DALI_UINT8;
  }

  void FillOutput(workspace_t<Backend>& ws) override {
    const auto& input = ws.template Input<Backend>(0);
    auto& output = ws.template Output<Backend>(0);
    for (size_t sample_id = 0; sample_id < input.num_samples(); sample_id++) {
      auto layout = GetLayout(input, sample_id);
      output[sample_id].Copy(make_cspan((const uint8_t*)layout.c_str(), layout.size()), nullptr);
    }
  }

 private:
  const TensorLayout& GetLayout(const BatchContainer& input, int idx) {
    return input[idx].GetMeta().GetLayout();
  }
};

template <>
const TensorLayout& Layout<GPUBackend>::GetLayout(const TensorList<GPUBackend>& input, int idx) {
  return input.GetMeta(idx).GetLayout();
}

template <>
void Layout<GPUBackend>::FillOutput(workspace_t<GPUBackend>& ws) {
  const auto& input = ws.template Input<GPUBackend>(0);
  auto& output = ws.template Output<GPUBackend>(0);
  for (size_t sample_id = 0; sample_id < input.num_samples(); sample_id++) {
    auto layout = GetLayout(input, sample_id);
    auto output_ptr = output.raw_mutable_tensor(static_cast<int>(sample_id));
    assert(ws.has_stream());  // GPU operator must have CUDA stream assigned
    cudaMemcpyAsync(output_ptr, layout.c_str(), layout.size(), cudaMemcpyDefault, ws.stream());
  }
}

}  // namespace
}  // namespace detail

template <typename Backend>
GetProperty<Backend>::GetProperty(const OpSpec& spec) : Operator<Backend>(spec) {
  auto property_name = spec.template GetArgument<std::string>("key");
  PropertyFactory(property_name);
}

template <typename Backend>
void GetProperty<Backend>::PropertyFactory(const std::string& property_name) {
  if (property_name == "source_info") {
    property_ = std::make_unique<detail::SourceInfo<Backend>>();
  } else if (property_name == "layout") {
    property_ = std::make_unique<detail::Layout<Backend>>();
  } else {
    DALI_FAIL(make_string("Unknown property key: ", property_name));
  }
}

}  // namespace dali

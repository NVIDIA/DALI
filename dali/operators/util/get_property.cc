// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

The type of the output will depend on the ``key`` of the requested property.)code")
    .NumInput(1)
    .InputDevice(0, InputDevice::Metadata)
    .NumOutput(1)
    .AddArg("key",
            R"code(Specifies, which property is requested.

The following properties are supported:

* ``"source_info"``: Returned type: byte-array.
                     String-like byte array, which contains information about the origin of the sample.
                     Fox example, :meth:`fn.get_property` called on tensor loaded via :meth:`fn.readers.file`
                     returns full path of the file, from which the tensor comes from.
* ``"layout"``: Returned type: byte-array
                :ref:`Data layout<layout_str_doc>` in the given Tensor.
)code",
            DALI_STRING);

template <typename Backend, typename SampleShapeFunc, typename CopySampleFunc>
void GetPerSample(TensorList<CPUBackend> &out, const TensorList<Backend> &in,
                  SampleShapeFunc &&sample_shape, CopySampleFunc &&copy_sample) {
  int N = in.num_samples();
  TensorListShape<> tls;
  for (int i = 0; i < N; i++) {
    auto shape = sample_shape(in, i);
    if (i == 0)
      tls.resize(N, shape.sample_dim());
    tls.set_tensor_shape(i, shape);
  }
  out.Resize(tls, DALI_UINT8);
  for (int i = 0; i < N; i++) {
    copy_sample(out, in, i);
  }
}

template <typename Backend>
void SourceInfoToTL(TensorList<CPUBackend> &out, const TensorList<Backend> &in) {
  GetPerSample(out, in,
    [](auto &in, int idx) {
      auto &info = in.GetMeta(idx).GetSourceInfo();
      return TensorShape<1>(info.length());
    },
    [](auto &out, auto &in, int idx) {
      auto &info = in.GetMeta(idx).GetSourceInfo();
      std::memcpy(out.raw_mutable_tensor(idx), info.c_str(), info.length());
    });
}

template <typename Backend>
void SourceInfoToTL(TensorList<GPUBackend> &out, const TensorList<Backend> &in) {
  TensorList<CPUBackend> tmp;
  tmp.set_pinned(true);
  SourceInfoToTL(tmp, in);
  tmp.set_order(out.order());
  out.Copy(tmp);
}

template <typename OutputBackend>
void SourceInfoToTL(TensorList<OutputBackend> &out, const Workspace &ws) {
  ws.Output<OutputBackend>(0).set_order(ws.output_order());
  if (ws.InputIsType<CPUBackend>(0))
    return SourceInfoToTL(out, ws.Input<CPUBackend>(0));
  else if (ws.InputIsType<GPUBackend>(0))
    return SourceInfoToTL(out, ws.Input<GPUBackend>(0));
  else
    DALI_FAIL("Internal error - input 0 is neither CPU nor GPU.");
}

template <typename Backend>
void RepeatTensor(TensorList<Backend> &tl, const Tensor<Backend> &t, int N) {
  tl.Reset();
  tl.set_device_id(t.device_id());
  tl.SetSize(N);
  tl.set_sample_dim(t.ndim());
  tl.set_type(t.type());
  tl.SetLayout(t.GetLayout());
  for (int i = 0; i < N; i++)
    tl.SetSample(i, t);
}

template <typename Backend>
void RepeatFirstSample(TensorList<Backend> &tl, int N) {
  Tensor<Backend> t;
  TensorShape<> shape = tl[0].shape();
  t.ShareData(unsafe_sample_owner(tl, 0), shape.num_elements(), tl.is_pinned(),
              shape, tl.type(), tl.device_id(), tl.order());
  t.SetMeta(tl.GetMeta(0));
  RepeatTensor(tl, t, N);
}

void LayoutToTL(TensorList<CPUBackend> &out, const Workspace &ws) {
  TensorLayout l = ws.GetInputLayout(0);
  out.Resize(uniform_list_shape(1, { l.size() }), DALI_UINT8);
  memcpy(out.raw_mutable_tensor(0), l.data(), l.size());
  RepeatFirstSample(out, ws.GetInputBatchSize(0));
}

void LayoutToTL(TensorList<GPUBackend> &out, const Workspace &ws) {
  TensorLayout l = ws.GetInputLayout(0);
  Tensor<CPUBackend> tmp_cpu;
  Tensor<GPUBackend> tmp_gpu;
  tmp_cpu.Resize(TensorShape<1>(l.size()), DALI_UINT8);
  memcpy(tmp_cpu.raw_mutable_data(), l.data(), l.size());
  tmp_cpu.set_order(ws.output_order());
  tmp_gpu.set_order(ws.output_order());
  tmp_gpu.Copy(tmp_cpu);

  RepeatTensor(out, tmp_gpu, ws.GetInputBatchSize(0));
}

template <typename Backend>
auto GetProperty<Backend>::GetPropertyReader(std::string_view key) -> PropertyReader {
  if (key == "source_info") {
    return static_cast<PropertyReaderFunc &>(SourceInfoToTL<Backend>);
  } else if (key == "layout") {
    return static_cast<PropertyReaderFunc &>(LayoutToTL);
  } else {
    DALI_FAIL(make_string("Unsupported property key: ", key));
  }
}


DALI_REGISTER_OPERATOR(GetProperty, GetProperty<CPUBackend>, CPU)
DALI_REGISTER_OPERATOR(GetProperty, GetProperty<GPUBackend>, GPU)

}  // namespace dali

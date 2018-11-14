// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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


#include <half.h>
#include <fstream>
#include <sstream>
#include <string>
#include "dali/pipeline/operators/tensorRTInfer/tensorRTInfer.h"
#include "dali/common.h"
namespace dali {
DALI_REGISTER_OPERATOR(TensorRTInfer, TensorRTInfer<GPUBackend>, GPU);

DALI_SCHEMA(TensorRTInfer)
    .DocStr(R"code(Perform inference over the TensorRT engine.)code")
    .OutputFn([](const OpSpec &spec) { return spec.GetArgument<int>("num_outputs"); })
    .NumInput(1, MAX_ALLOWED_TRT_OP_INPUT)
    .AddOptionalArg("num_outputs",
        R"code(Number of outputs.)code", 1)
    .AddArg("engine",
        R"code(TensorRT engine file to run inference)code", DALI_STRING, false)
    .AddArg("input_blobs",
        R"code(Input blob in the engine)code", DALI_STRING_VEC, false)
    .AddArg("output_blobs",
        R"code(Output blob in the engine)code", DALI_STRING_VEC, false)
    .AddOptionalArg("plugins",
        R"code(Plugin library to load)code", std::vector<std::string>({""}), false)
    .AddOptionalArg("trt_batch_size",
        R"code(Batch size to run inference)code", 1, false)
    .AddOptionalArg("use_dla_core",
        R"code(DLA core to run inference upon)code", -1, false);

template <>
void TensorRTInfer<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  int num_bindings = engine_->getNbBindings();
  std::string blob_name;
  std::vector<const void *> input_buffers(input_blobs_.size());

  // Input output buffer
  std::vector<void *> io_buffers(num_bindings);

  // Copy the address of data for each input bindings from engine
  for (size_t i = 0; i < input_blobs_.size(); i++) {
    // Assign the first tensor address as the binding address to run for >1 batch size
    input_buffers[binding_param_[input_blobs_[i]].binding_index] =
           ws->Input<GPUBackend>(i).raw_tensor(0);
  }

  // Copy the input bindings for TensorRT
  std::memcpy(io_buffers.data(), input_buffers.data(), sizeof(void *) * input_blobs_.size());

  for (size_t i = 0; i < output_blobs_.size(); i++) {
      blob_name = output_blobs_[i];

      // Output tensorlist freom DALI pipeline
      auto output = ws->Output<GPUBackend>(i);
      // Allocate memory for output tensorrlist and set datatype
      output->Resize(binding_param_[blob_name].dali_dimensions);
      TypeInfo type;
      switch (binding_param_[blob_name].data_type) {
      case DALI_FLOAT16:
          type = TypeInfo::Create<half_float::half>();
          break;
      case DALI_INT32:
          type = TypeInfo::Create<int32>();
          break;
      default:
          type = TypeInfo::Create<float>();
      }
      output->set_type(type);
      // Binding address points to the address of the output binding batch
      io_buffers[binding_param_[blob_name].binding_index] =
          output->raw_mutable_tensor(0);
  }

  // Run inference
  context_->enqueue(trt_batch_size_, io_buffers.data(), ws->stream(), nullptr);
}

}  // namespace dali

// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include "dali/c_api_2/pipeline_test_utils.h"
#include "dali/core/common.h"
#include "dali/test/dali_test_config.h"
#include "dali/pipeline/pipeline.h"
#include "dali/c_api_2/data_objects.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/static_switch.h"

namespace dali::c_api::test {

template <typename Backend>
auto &Unwrap(daliTensorList_h h) {
  return static_cast<ITensorList *>(h)->Unwrap<Backend>();
}

std::unique_ptr<Pipeline>
ReaderDecoderPipe(std::string_view decoder_device, StorageDevice output_device) {
  std::string file_root = testing::dali_extra_path() + "/db/single/jpeg/";
  std::string file_list = file_root + "image_list.txt";
  auto pipe = std::make_unique<Pipeline>(4, 1, 0, 12345, true, 2, true, true);
  pipe->AddOperator(OpSpec("FileReader")
    .AddArg("device", "cpu")
    .AddArg("file_root", file_root)
    .AddArg("file_list", file_list)
    .AddOutput("compressed_images", StorageDevice::CPU)
    .AddOutput("labels", StorageDevice::CPU));

  pipe->AddOperator(OpSpec("ImageDecoder")
    .AddArg("device", decoder_device)
    .AddArg("output_type", DALI_RGB)
    .AddInput("compressed_images", StorageDevice::CPU)
    .AddOutput("decoded", decoder_device == "cpu" ? StorageDevice::CPU : StorageDevice::GPU));

  auto out_dev_str = to_string(output_device);
  pipe->SetOutputDescs({{ "decoded", out_dev_str }, { "labels", out_dev_str }});
  return pipe;
}

void CompareTensorList(const TensorList<CPUBackend> &a, const TensorList<CPUBackend> &b) {
  ASSERT_EQ(a.type(), b.type());
  ASSERT_EQ(a.sample_dim(), b.sample_dim());
  ASSERT_EQ(a.num_samples(), b.num_samples());
  TYPE_SWITCH(a.type(), type2id, T,
    (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double),
    (
      CheckEqual(view<const T>(a), view<const T>(b));
    ), (GTEST_FAIL() << "Unsupported type " << a.type();));
}

void CompareTensorList(const TensorList<CPUBackend> &a, const TensorList<CPUBackend> &b) {


void ComparePipelineOutput(Pipeline &ref, PipelineHandle test) {
  Workspace ws;
  ref.Outputs(&ws);
  auto outs = PopOutputs(test);
  int out_count;
  ASSERT_EQ(daliPipelineGetOutputCount(test, &out_count), DALI_SUCCESS);
  ASSERT_EQ(ws.NumOutput(), out_count) << "The pipelines have a different number of outputs.";
  for (int i = 0; i < out_count; i++) {
    auto test_tl = GetOutput(outs, i);
    daliBufferPlacement_t placement{};
    CHECK_DALI(daliTensorListGetBufferPlacement(test_tl, &placement));
    if (ws.OutputIsType<CPUBackend>(i)) {
      ASSERT_EQ(placement.device_type, DALI_STORAGE_CPU);
      CompareTensorList(ws.Output<CPUBackend>(i), *Unwrap<CPUBackend>(test_tl));
    } else if (ws.OutputIsType<GPUBackend>(i)) {
      ASSERT_EQ(placement.device_type, DALI_STORAGE_GPU);
      CompareTensorList(ws.Output<GPUBackend>(i), *Unwrap<GPUBackend>(test_tl));
    }

  }
  ref.ReleaseOutputs();
}

void ComparePipelineOutputs(Pipeline &ref, PipelineHandle test, int iters = 5) {
  for (int iter = 0; iter < iters; iter++) {
    if (iter == 0) {
      ref.Prefetch();
      CHECK_DALI(daliPipelinePrefetch(test));
    } else {
      ref.Run();
      CHECK_DALI(daliPipelineRun(test));
    }
    ComparePipelineOutput(ref, test);
  }
}

void TestReaderDecoder(std::string_view decoder_device, StorageDevice output_device) {
  auto ref_pipe = ReaderDecoderPipe(decoder_device, output_device);
  auto proto = ref_pipe->SerializeToProtobuf();
  ref_pipe->Build();

  daliPipelineParams_t params{};
  params.exec_type_present = true;
  params.exec_type = DALI_EXEC_DYNAMIC;
  ComparePipelineOutputs(*ref_pipe, Deserialize(proto, params));
}

TEST(CAPI2_PipelineTest, ReaderDecoderCPU) {
  TestReaderDecoder("cpu", StorageDevice::CPU);
}

}  // namespace dali::c_api::test
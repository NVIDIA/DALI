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

void TestReaderDecoder(std::string_view decoder_device, StorageDevice output_device) {
  auto ref_pipe = ReaderDecoderPipe(decoder_device, output_device);
  auto proto = ref_pipe->SerializeToProtobuf();
  ref_pipe->Build();

  daliPipelineParams_t params{};
  params.exec_type_present = true;
  params.exec_type = DALI_EXEC_DYNAMIC;
  auto pipe = Deserialize(proto, params);
  CHECK_DALI(daliPipelineBuild(pipe));
  ComparePipelineOutputs(*ref_pipe, pipe);
}

TEST(CAPI2_PipelineTest, ReaderDecoderCPU) {
  TestReaderDecoder("cpu", StorageDevice::CPU);
}

TEST(CAPI2_PipelineTest, ReaderDecoderGPU) {
  TestReaderDecoder("mixed", StorageDevice::GPU);
}

}  // namespace dali::c_api::test

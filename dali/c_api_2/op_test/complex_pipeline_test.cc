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
ReaderDecoderPipe(
      std::string_view decoder_device,
      StorageDevice output_device,
      PipelineParams params = {}) {
  std::string file_root = testing::dali_extra_path() + "/db/single/jpeg/";
  std::string file_list = file_root + "image_list.txt";
  if (!params.max_batch_size) params.max_batch_size = 4;
  if (!params.num_threads) params.num_threads = 1;
  if (!params.seed) params.seed = 12345;
  if (!params.executor_type) params.executor_type = ExecutorType::Dynamic;
  auto pipe = std::make_unique<Pipeline>(params);
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

TEST(CAPI2_PipelineTest, ReaderDecoderCPU2GPU) {
  TestReaderDecoder("cpu", StorageDevice::GPU);
}

TEST(CAPI2_PipelineTest, ReaderDecoderMixed) {
  if (!MixedOperatorRegistry::Registry().IsRegistered("ImageDecoder")) {
    GTEST_SKIP() << "ImageDecoder for mixed backend is not available";
  }
  TestReaderDecoder("mixed", StorageDevice::GPU);
}

TEST(CAPI2_PipelineTest, ReaderDecoderMixed2CPU) {
  if (!MixedOperatorRegistry::Registry().IsRegistered("ImageDecoder")) {
    GTEST_SKIP() << "ImageDecoder for mixed backend is not available";
  }
  TestReaderDecoder("mixed", StorageDevice::CPU);
}

TEST(CAPI2_PipelineTest, Checkpointing) {
  // This test creates three pipelines - a C++ pipeline (ref) and two C pipelines (pipe1, pipe2),
  // created by deserializing the serialized representation of the C++ pipeline.
  //
  // (pipe1) advances 3 iterations and then a checkpoint is taken and restored in (ref),
  // after which 5 iterations of outputs are compared.
  // Then a checkpoint is taken in (ref) and restored in (pipe2), after which another 5 iterations
  // are compared.
  PipelineParams params{};
  params.enable_checkpointing = true;
  params.seed = 1234;
  auto ref = ReaderDecoderPipe("cpu", StorageDevice::GPU, params);
  ref->Build();
  auto pipe_str = ref->SerializeToProtobuf();  // serialize the ref...
  auto pipe1 = Deserialize(pipe_str, {});  // ...and create pipe1
  auto pipe2 = Deserialize(pipe_str, {});  // ...and pipe2 from serialized ref
  CHECK_DALI(daliPipelineBuild(pipe1));
  CHECK_DALI(daliPipelineBuild(pipe2));

  // Advance a few iterations...
  CHECK_DALI(daliPipelinePrefetch(pipe1));
  daliPipelineOutputs_h out1_h{};
  CHECK_DALI(daliPipelinePopOutputs(pipe1, &out1_h));
  CHECK_DALI(daliPipelineOutputsDestroy(out1_h));
  CHECK_DALI(daliPipelineRun(pipe1));
  CHECK_DALI(daliPipelinePopOutputs(pipe1, &out1_h));
  CHECK_DALI(daliPipelineOutputsDestroy(out1_h));
  CHECK_DALI(daliPipelineRun(pipe1));
  CHECK_DALI(daliPipelinePopOutputs(pipe1, &out1_h));
  CHECK_DALI(daliPipelineOutputsDestroy(out1_h));

  const char pipeline_data[] = "A rose by any other name would smell as sweet";
  size_t pipeline_data_size = strlen(pipeline_data);

  daliCheckpointExternalData_t ext{};
  ext.iterator_data.data = "ITER";
  ext.iterator_data.size = 4;
  ext.pipeline_data.data = pipeline_data;
  ext.pipeline_data.size = strlen(ext.pipeline_data.data);

  daliCheckpoint_h checkpoint_h{};
  // Take a checkpoint...
  CHECK_DALI(daliPipelineGetCheckpoint(pipe1, &checkpoint_h, &ext));
  CheckpointHandle checkpoint(checkpoint_h);

  const char *data = nullptr;
  size_t size = 0;
  CHECK_DALI(daliPipelineSerializeCheckpoint(pipe1, checkpoint, &data, &size));
  ASSERT_NE(data, nullptr);

  // ...restore...
  ref->RestoreFromSerializedCheckpoint(std::string(data, size));
  // ...run and compare.
  ComparePipelineOutputs(*ref, pipe1, 5, false);

  // Now take another checkpoint...
  auto chk_str = ref->GetSerializedCheckpoint({ ext.pipeline_data.data, ext.iterator_data.data });

  // ...deserialize...
  CHECK_DALI(daliPipelineDeserializeCheckpoint(
        pipe2, &checkpoint_h, chk_str.data(), chk_str.length()));
  CheckpointHandle checkpoint2(checkpoint_h);

  daliCheckpointExternalData_t ext2{};
  CHECK_DALI(daliCheckpointGetExternalData(checkpoint2, &ext2));
  EXPECT_EQ(ext2.iterator_data.size, 4);
  EXPECT_STREQ(ext2.iterator_data.data, "ITER");
  EXPECT_EQ(ext2.pipeline_data.size, pipeline_data_size);
  EXPECT_STREQ(ext2.pipeline_data.data, pipeline_data);
  // ...restore...
  CHECK_DALI(daliPipelineRestoreCheckpoint(pipe2, checkpoint2));
  // ...run and compare.
  ComparePipelineOutputs(*ref, pipe2, 5, false);
}

}  // namespace dali::c_api::test

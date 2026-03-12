// Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/dali.h"
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

PipelineHandle
ReaderDecoderCApiPipe(
      std::string_view decoder_device,
      StorageDevice output_device,
      daliPipelineParams_t params = {}) {
  std::string file_root = testing::dali_extra_path() + "/db/single/jpeg/";
  std::string file_list = file_root + "image_list.txt";

  if (!params.max_batch_size_present) {
    params.max_batch_size_present = true;
    params.max_batch_size = 4;
  }
  if (!params.num_threads_present) {
    params.num_threads_present = true;
    params.num_threads = 1;
  }
  if (!params.seed_present) {
    params.seed_present = true;
    params.seed = 12345;
  }
  if (!params.exec_type_present) {
    params.exec_type_present = true;
    params.exec_type = DALI_EXEC_DYNAMIC;
  }

  daliPipeline_h h = nullptr;
  CHECK_DALI(daliPipelineCreate(&h, &params));
  PipelineHandle pipe(h);

  // FileReader: no inputs, outputs "compressed_images" and "labels" on CPU
  daliArgDesc_t reader_args[3];
  reader_args[0].arg_name = "device";
  reader_args[0].dtype    = DALI_STRING;
  reader_args[0].str      = "cpu";
  reader_args[1].arg_name = "file_root";
  reader_args[1].dtype    = DALI_STRING;
  reader_args[1].str      = file_root.c_str();
  reader_args[2].arg_name = "file_list";
  reader_args[2].dtype    = DALI_STRING;
  reader_args[2].str      = file_list.c_str();

  daliIODesc_t reader_out[2];
  reader_out[0].name        = "compressed_images";
  reader_out[0].device_type = DALI_STORAGE_CPU;
  reader_out[1].name        = "labels";
  reader_out[1].device_type = DALI_STORAGE_CPU;

  daliOperatorDesc_t reader_op{};
  reader_op.schema_name = "FileReader";
  reader_op.backend     = DALI_BACKEND_CPU;
  reader_op.num_inputs  = 0;
  reader_op.num_outputs = 2;
  reader_op.num_args    = 3;
  reader_op.outputs     = reader_out;
  reader_op.args        = reader_args;
  CHECK_DALI(daliPipelineAddOperator(h, &reader_op));

  // ImageDecoder: input "compressed_images" (CPU), output "decoded" (CPU or GPU)
  bool is_mixed = (decoder_device == "mixed");
  daliBackend_t decoder_backend   = is_mixed ? DALI_BACKEND_MIXED : DALI_BACKEND_CPU;
  daliStorageDevice_t decoded_dev = is_mixed ? DALI_STORAGE_GPU   : DALI_STORAGE_CPU;
  std::string decoder_device_str(decoder_device);

  daliArgDesc_t decoder_args[2];
  decoder_args[0].arg_name = "device";
  decoder_args[0].dtype    = DALI_STRING;
  decoder_args[0].str      = decoder_device_str.c_str();
  decoder_args[1].arg_name = "output_type";
  decoder_args[1].dtype    = DALI_INT32;
  decoder_args[1].ivalue   = DALI_RGB;

  daliIODesc_t decoder_in[1];
  decoder_in[0].name        = "compressed_images";
  decoder_in[0].device_type = DALI_STORAGE_CPU;

  daliIODesc_t decoder_out[1];
  decoder_out[0].name        = "decoded";
  decoder_out[0].device_type = decoded_dev;

  daliOperatorDesc_t decoder_op{};
  decoder_op.schema_name = "ImageDecoder";
  decoder_op.backend     = decoder_backend;
  decoder_op.num_inputs  = 1;
  decoder_op.num_outputs = 1;
  decoder_op.num_args    = 2;
  decoder_op.inputs      = decoder_in;
  decoder_op.outputs     = decoder_out;
  decoder_op.args        = decoder_args;
  CHECK_DALI(daliPipelineAddOperator(h, &decoder_op));

  daliStorageDevice_t out_dev =
      output_device == StorageDevice::GPU ? DALI_STORAGE_GPU : DALI_STORAGE_CPU;
  daliPipelineIODesc_t out_descs[2];
  out_descs[0] = {};
  out_descs[0].name   = "decoded";
  out_descs[0].device = out_dev;
  out_descs[1] = {};
  out_descs[1].name   = "labels";
  out_descs[1].device = out_dev;
  CHECK_DALI(daliPipelineSetOutputs(h, 2, out_descs));

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

TEST(CAPI2_SerializedPipelineTest, ReaderDecoderCPU) {
  TestReaderDecoder("cpu", StorageDevice::CPU);
}

TEST(CAPI2_SerializedPipelineTest, ReaderDecoderCPU2GPU) {
  TestReaderDecoder("cpu", StorageDevice::GPU);
}

TEST(CAPI2_SerializedPipelineTest, ReaderDecoderMixed) {
  if (!MixedOperatorRegistry::Registry().IsRegistered("ImageDecoder")) {
    GTEST_SKIP() << "ImageDecoder for mixed backend is not available";
  }
  TestReaderDecoder("mixed", StorageDevice::GPU);
}

TEST(CAPI2_SerializedPipelineTest, ReaderDecoderMixed2CPU) {
  if (!MixedOperatorRegistry::Registry().IsRegistered("ImageDecoder")) {
    GTEST_SKIP() << "ImageDecoder for mixed backend is not available";
  }
  TestReaderDecoder("mixed", StorageDevice::CPU);
}

void TestReaderDecoderBuilder(std::string_view decoder_device, StorageDevice output_device) {
  auto ref_pipe = ReaderDecoderPipe(decoder_device, output_device);
  ref_pipe->Build();
  auto test_pipe = ReaderDecoderCApiPipe(decoder_device, output_device);
  CHECK_DALI(daliPipelineBuild(test_pipe));
  ComparePipelineOutputs(*ref_pipe, test_pipe);
}

TEST(CAPI2_PipelineBuilderTest, ReaderDecoderCPU) {
  TestReaderDecoderBuilder("cpu", StorageDevice::CPU);
}

TEST(CAPI2_PipelineBuilderTest, ReaderDecoderCPU2GPU) {
  TestReaderDecoderBuilder("cpu", StorageDevice::GPU);
}

TEST(CAPI2_PipelineBuilderTest, ReaderDecoderMixed) {
  if (!MixedOperatorRegistry::Registry().IsRegistered("ImageDecoder")) {
    GTEST_SKIP() << "ImageDecoder for mixed backend is not available";
  }
  TestReaderDecoderBuilder("mixed", StorageDevice::GPU);
}

TEST(CAPI2_PipelineBuilderTest, ReaderDecoderMixed2CPU) {
  if (!MixedOperatorRegistry::Registry().IsRegistered("ImageDecoder")) {
    GTEST_SKIP() << "ImageDecoder for mixed backend is not available";
  }
  TestReaderDecoderBuilder("mixed", StorageDevice::CPU);
}

void RunCheckpointingTest(Pipeline &ref, daliPipeline_h pipe1, daliPipeline_h pipe2) {
  // Advance a few iterations in pipe1...
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
  // Take a checkpoint from pipe1...
  CHECK_DALI(daliPipelineGetCheckpoint(pipe1, &checkpoint_h, &ext));
  CheckpointHandle checkpoint(checkpoint_h);

  const char *data = nullptr;
  size_t size = 0;
  CHECK_DALI(daliPipelineSerializeCheckpoint(pipe1, checkpoint, &data, &size));
  ASSERT_NE(data, nullptr);

  // ...restore into ref...
  ref.RestoreFromSerializedCheckpoint(std::string(data, size));
  // ...run and compare.
  ComparePipelineOutputs(ref, pipe1, 5, false);

  // Now take another checkpoint from ref...
  auto chk_str = ref.GetSerializedCheckpoint({ ext.pipeline_data.data, ext.iterator_data.data });

  // ...deserialize into pipe2...
  CHECK_DALI(daliPipelineDeserializeCheckpoint(
        pipe2, &checkpoint_h, chk_str.data(), chk_str.length()));
  CheckpointHandle checkpoint2(checkpoint_h);

  daliCheckpointExternalData_t ext2{};
  CHECK_DALI(daliCheckpointGetExternalData(checkpoint2, &ext2));
  EXPECT_EQ(ext2.iterator_data.size, 4);
  EXPECT_STREQ(ext2.iterator_data.data, "ITER");
  EXPECT_EQ(ext2.pipeline_data.size, pipeline_data_size);
  EXPECT_STREQ(ext2.pipeline_data.data, pipeline_data);
  // ...restore and compare.
  CHECK_DALI(daliPipelineRestoreCheckpoint(pipe2, checkpoint2));
  ComparePipelineOutputs(ref, pipe2, 5, false);
}

TEST(CAPI2_SerializedPipelineTest, Checkpointing) {
  // This test creates three pipelines - a C++ pipeline (ref) and two C pipelines (pipe1, pipe2),
  // created by deserializing the serialized representation of the C++ pipeline.
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
  RunCheckpointingTest(*ref, pipe1, pipe2);
}

TEST(CAPI2_PipelineBuilderTest, Checkpointing) {
  // This test creates three pipelines - a C++ pipeline (ref) and two C API builder pipelines
  // (pipe1, pipe2) with identical parameters, then verifies checkpoint save/restore.
  PipelineParams cpp_params{};
  cpp_params.enable_checkpointing = true;
  cpp_params.seed = 1234;
  auto ref = ReaderDecoderPipe("cpu", StorageDevice::GPU, cpp_params);
  ref->Build();

  daliPipelineParams_t c_params{};
  c_params.enable_checkpointing_present = true;
  c_params.enable_checkpointing = true;
  c_params.seed_present = true;
  c_params.seed = 1234;
  auto pipe1 = ReaderDecoderCApiPipe("cpu", StorageDevice::GPU, c_params);
  auto pipe2 = ReaderDecoderCApiPipe("cpu", StorageDevice::GPU, c_params);
  CHECK_DALI(daliPipelineBuild(pipe1));
  CHECK_DALI(daliPipelineBuild(pipe2));
  RunCheckpointingTest(*ref, pipe1, pipe2);
}

}  // namespace dali::c_api::test

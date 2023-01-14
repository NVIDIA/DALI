// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include "dali/c_api.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_test_config.h"

namespace dali::test {

using namespace dali;  // NOLINT

struct OperatorTraceTestParam {
  int cpu_queue_depth, gpu_queue_depth;
  bool exec_async;
};

OperatorTraceTestParam operator_trace_test_params_simple_executor[] = {
        {1, 1, false},
};


OperatorTraceTestParam operator_trace_test_params_pipelined_executor_uniform_queue[] = {
        {2, 2, false},
        {7, 7, false},
        {2, 2, true},
        {7, 7, true},
};

OperatorTraceTestParam operator_trace_test_params_pipelined_executor_separate_queue[] = {
        {2, 3, false},
        {7, 5, false},
        {2, 3, true},
        {7, 5, true},
};

std::array<std::string, 2> operator_under_test_names = {
        "PassthroughCpu", "PassthroughGpu"
};


class OperatorTraceTest : public ::testing::TestWithParam<OperatorTraceTestParam> {
 protected:
  void SetUp() final {
    auto parameters = GetParam();
    cpu_queue_depth_ = parameters.cpu_queue_depth;
    gpu_queue_depth_ = parameters.gpu_queue_depth;
    exec_async_ = parameters.exec_async;
    exec_pipelined_ = cpu_queue_depth_ > 1 || gpu_queue_depth_ > 1;
    exec_separated_ = cpu_queue_depth_ != gpu_queue_depth_;
    cout << "TEST: cpu_queue_depth=" << cpu_queue_depth_ << ", gpu_queue_depth=" << gpu_queue_depth_
         << ", exec_async=" << std::boolalpha << exec_async_ << ", exec_pipelined="
         << exec_pipelined_ << endl;
    pipeline_ = std::make_unique<Pipeline>(batch_size_, num_threads_, device_id_, -1,
                                           exec_pipelined_, cpu_queue_depth_, exec_async_);

    pipeline_->SetExecutionTypes(exec_pipelined_, exec_separated_, exec_async_);

    std::string file_root = testing::dali_extra_path() + "/db/single/jpeg/";
    std::string file_list = file_root + "image_list.txt";
    pipeline_->AddOperator(OpSpec("FileReader")
                                   .AddArg("device", "cpu")
                                   .AddArg("file_root", file_root)
                                   .AddArg("file_list", file_list)
                                   .AddOutput("compressed_images", "cpu")
                                   .AddOutput("labels", "cpu"));
    pipeline_->AddOperator(OpSpec("PassthroughOp")
                                   .AddArg("device", "cpu")
                                   .AddInput("compressed_images", "cpu")
                                   .AddOutput("PT_CPU", "cpu"),
                           operator_under_test_names[0]);
    pipeline_->AddOperator(OpSpec("PassthroughOp")
                                   .AddArg("device", "gpu")
                                   .AddInput("compressed_images", "gpu")
                                   .AddOutput("PT_GPU", "gpu"),
                           operator_under_test_names[1]);

    std::vector<std::pair<std::string, std::string>> outputs = {
            {"PT_CPU", "cpu"},
            {"PT_GPU", "gpu"}
    };

    pipeline_->SetOutputDescs(outputs);

    serialized_pipeline_ = pipeline_->SerializeToProtobuf();
  }


  int batch_size_ = 3;
  int num_threads_ = 2;
  int device_id_ = 0;
  std::string output_name_ = "OUTPUT";
  int n_iterations_ = 50;
  std::unique_ptr<Pipeline> pipeline_;

  bool exec_pipelined_, exec_async_, exec_separated_;
  int cpu_queue_depth_, gpu_queue_depth_;
  std::string serialized_pipeline_;
};


TEST_P(OperatorTraceTest, OperatorTraceTest) {
  daliPipelineHandle h;
  daliCreatePipeline2(&h, serialized_pipeline_.c_str(), serialized_pipeline_.length(), batch_size_,
                      num_threads_, device_id_, exec_pipelined_ ? 0 : 1, exec_async_ ? 0 : 1,
                      exec_separated_ ? 0 : 1, cpu_queue_depth_, cpu_queue_depth_, gpu_queue_depth_,
                      0);
  for (int iteration = 0; iteration < n_iterations_; iteration++) {
    auto prefetch_depth = std::min(cpu_queue_depth_, gpu_queue_depth_);
    daliPrefetchUniform(&h, prefetch_depth);
    for (int i = 0; i < prefetch_depth; i++) {
      daliShareOutput(&h);

      for (const auto & operator_name : operator_under_test_names) {
        EXPECT_NE(daliHasOperatorTrace(&h, operator_name.c_str(), "this_trace_does_not_exist"), 0);
        ASSERT_EQ(daliHasOperatorTrace(&h, operator_name.c_str(), "test_trace"), 0);

        EXPECT_EQ(std::string(daliGetOperatorTrace(&h, operator_name.c_str(), "test_trace")),
                  make_string("test_value", iteration * prefetch_depth + i));
      }

      daliOutputRelease(&h);
    }
  }
}


INSTANTIATE_TEST_SUITE_P(OperatorTraceTestSimpleExecutor, OperatorTraceTest,
                         ::testing::ValuesIn(
                                 operator_trace_test_params_simple_executor));

INSTANTIATE_TEST_SUITE_P(OperatorTraceTestPipelinedExecutorUniformQueue, OperatorTraceTest,
                         ::testing::ValuesIn(
                                 operator_trace_test_params_pipelined_executor_uniform_queue));

INSTANTIATE_TEST_SUITE_P(OperatorTraceTestPipelinedExecutorSeparateQueue, OperatorTraceTest,
                         ::testing::ValuesIn(
                                 operator_trace_test_params_pipelined_executor_separate_queue));


}  // namespace dali::test

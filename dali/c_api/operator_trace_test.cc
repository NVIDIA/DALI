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
  int prefetch_queue_depth;
  bool exec_pipelined;
  bool exec_async;
};

OperatorTraceTestParam operator_trace_test_params_simple_executor[] = {
        {1, false, false},
};


OperatorTraceTestParam operator_trace_test_params_pipelined_executor[] = {
        {7, true,  false},
        {7, false, true},
        {7, true,  true}
};


class OperatorTraceTest : public ::testing::TestWithParam<OperatorTraceTestParam> {
 protected:
  void SetUp() final {
    auto parameters = GetParam();
    prefetch_queue_depth_ = parameters.prefetch_queue_depth;
    auto exec_async = parameters.exec_async;
    auto exec_pipelined = parameters.exec_pipelined;
    cout << "TEST: prefetch_queue_depth=" << prefetch_queue_depth_ << ", exec_async="
         << std::boolalpha << exec_async << ", exec_pipelined=" << exec_pipelined << endl;
    pipeline_ = std::make_unique<Pipeline>(batch_size_, num_threads_, device_id_, -1,
                                           exec_pipelined, prefetch_queue_depth_, exec_async);

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
                           "PassthroughCpu");
    pipeline_->AddOperator(OpSpec("PassthroughOp")
                                   .AddArg("device", "gpu")
                                   .AddInput("compressed_images", "gpu")
                                   .AddOutput("PT_GPU", "gpu"),
                           "PassthroughGpu");

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
  int prefetch_queue_depth_ = 1;
  int n_iterations_ = 50;
  std::unique_ptr<Pipeline> pipeline_;

  std::string serialized_pipeline_;
};


TEST_P(OperatorTraceTest, OperatorTraceTest) {
  daliPipelineHandle h;
  daliDeserializeDefault(&h, serialized_pipeline_.c_str(), serialized_pipeline_.size());
  for (int iteration = 0; iteration < n_iterations_; iteration++) {
    daliPrefetchUniform(&h, prefetch_queue_depth_);
    for (int i = 0; i < prefetch_queue_depth_; i++) {
      daliShareOutput(&h);
      ASSERT_EQ(daliGetNumOperatorTraces(&h, "PassthroughCpu"), 1);
      EXPECT_EQ(std::string(daliGetOperatorTrace(&h, "PassthroughCpu", "test_trace")),
                make_string("test_value", iteration * prefetch_queue_depth_ + i));
      daliOutputRelease(&h);
    }
  }
}


INSTANTIATE_TEST_SUITE_P(OperatorTraceTestSimpleExecutor, OperatorTraceTest,
                         ::testing::ValuesIn(operator_trace_test_params_simple_executor));


}  // namespace dali::test

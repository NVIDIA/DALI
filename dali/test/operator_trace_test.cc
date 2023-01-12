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
#include "dali/test/dali_test_config.cc"

namespace dali::test {

using namespace dali;  // NOLINT

TEST(OperatorTraceTest, OperatorTraceTest) {
  auto batch_size = 3;
  auto num_threads = 2;
  auto device_id = 0;
  auto output_name="OUTPUT";
  auto prefetch_queue_depth=2;
  std::string operator_under_test_name = "Passthrough";

  auto pipe = std::make_unique<Pipeline>(batch_size, num_threads, device_id);

  std::string file_root = testing::dali_extra_path() + "/db/single/jpeg/";
  std::string file_list = file_root + "image_list.txt";
  pipe->AddOperator(OpSpec("FileReader")
                           .AddArg("device", "cpu")
                           .AddArg("file_root", file_root)
                           .AddArg("file_list", file_list)
                           .AddOutput("compressed_images", "cpu")
                           .AddOutput("labels", "cpu"));
  pipe->AddOperator(OpSpec("PassthroughOp")
                            .AddArg("device", "cpu")
                            .AddInput("compressed_images", "cpu")
                            .AddOutput(output_name, "cpu"),
                    operator_under_test_name);

  std::vector<std::pair<std::string, std::string>> outputs = {{output_name, "cpu"}};

  pipe->SetOutputDescs(outputs);

  auto serialized = pipe->SerializeToProtobuf();
  daliPipelineHandle h;
  daliDeserializeDefault(&h, serialized.c_str(), serialized.size());
  daliPrefetchUniform(&h, prefetch_queue_depth);
  for (int i=0;i<prefetch_queue_depth;i++) {
    daliShareOutput(&h);
    ASSERT_EQ(daliGetNumOperatorTraces(&h, operator_under_test_name.c_str()), 1);
    EXPECT_STREQ(daliGetOperatorTrace(&h, operator_under_test_name.c_str(), "test_trace"),
                 "test_value");
    daliOutputRelease(&h);
  }

}


}  // namespace dali::test

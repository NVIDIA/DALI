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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dali/c_api.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_test_config.h"

namespace dali::test {


struct InputOperatorMixedTestParam {
  int cpu_queue_depth, gpu_queue_depth;
  bool exec_pipelined, exec_async;
  bool cpu_input;
};


namespace {

InputOperatorMixedTestParam input_operator_test_params_simple_executor[] = {
        {1, 1, false, false, true},
        {2, 2, false, false, false},
};


InputOperatorMixedTestParam input_operator_test_params_pipelined_executor_uniform_queue[] = {
        {2, 2, true, false, true},
        {3, 3, true, false, false},
        {2, 2, true, true,  true},
        {3, 3, true, true,  false},
};

InputOperatorMixedTestParam input_operator_test_params_pipelined_executor_separate_queue[] = {
        {2, 3, true, false, true},
        {3, 2, true, false, false},
        {2, 3, true, true,  true},
        {3, 2, true, true,  false},
};


template<typename T>
thrust::host_vector<T> random_vector_cpu(std::mt19937 &mt, size_t size) {
  thrust::host_vector<T> cpu(size);
  std::uniform_int_distribution<T> dist{0, 255};
  auto gen = [&]() { return dist(mt); };
  thrust::generate(cpu.begin(), cpu.end(), gen);
  return cpu;
}

}  // namespace



class InputOperatorMixedTest : public ::testing::TestWithParam<InputOperatorMixedTestParam> {
 protected:
  void SetUp() final {
    auto parameters = GetParam();
    cpu_queue_depth_ = parameters.cpu_queue_depth;
    gpu_queue_depth_ = parameters.gpu_queue_depth;
    exec_async_ = parameters.exec_async;
    exec_pipelined_ = parameters.exec_pipelined;
    exec_separated_ = cpu_queue_depth_ != gpu_queue_depth_;
    cpu_input_ = parameters.cpu_input;

    std::ios_base::fmtflags f = cout.flags();
    cout << "TEST: cpu_queue_depth=" << cpu_queue_depth_ << ", gpu_queue_depth=" << gpu_queue_depth_
         << ", exec_async=" << std::boolalpha << exec_async_ << ", exec_pipelined="
         << exec_pipelined_ << endl;
    cout.flags(f);

    pipeline_ = std::make_unique<Pipeline>(batch_size_, num_threads_, device_id_, -1,
                                           exec_pipelined_, cpu_queue_depth_, exec_async_);
    pipeline_->SetExecutionTypes(exec_pipelined_, exec_async_);

    PutTogetherDaliGraph();

    serialized_pipeline_ = pipeline_->SerializeToProtobuf();
  }


  int batch_size_ = 3;
  int num_threads_ = 2;
  int device_id_ = 0;
  int n_iterations_ = 50;
  std::unique_ptr<Pipeline> pipeline_;
  std::string operator_name_ = "IDIN";

  bool exec_pipelined_, exec_async_, exec_separated_;
  int cpu_queue_depth_, gpu_queue_depth_;
  bool cpu_input_;
  std::string serialized_pipeline_;

 private:
  void PutTogetherDaliGraph() {
    pipeline_->AddOperator(OpSpec("IdentityInput")
                                   .AddArg("device", "mixed")
                                   .AddArg("name", operator_name_)
                                   .AddArg("cpu_input", cpu_input_)
                                   .AddOutput(operator_name_, "gpu"),
                           operator_name_);
    std::vector<std::pair<std::string, std::string>> outputs = {
            {operator_name_, "gpu"},
    };
    pipeline_->SetOutputDescs(outputs);
  }
};


TEST_P(InputOperatorMixedTest, InputOperatorMixedTest) {
  daliPipelineHandle h;
  daliCreatePipeline2(&h, serialized_pipeline_.c_str(), serialized_pipeline_.length(), batch_size_,
                      num_threads_, device_id_, exec_pipelined_, exec_async_, exec_separated_,
                      cpu_queue_depth_, cpu_queue_depth_, gpu_queue_depth_, 0);
  for (int iteration = 0; iteration < n_iterations_; iteration++) {
    auto prefetch_depth = std::min(cpu_queue_depth_, gpu_queue_depth_);
    size_t sample_size = 42;
    thrust::host_vector<int32_t> in_data(sample_size * batch_size_, 2137);
    thrust::device_vector<int32_t> ref_data = in_data;

    // Feed CPU input data.
    for (int i = 0; i < prefetch_depth; i++) {
      std::vector<int64_t> shapes(batch_size_, sample_size);
      daliSetExternalInput(&h, operator_name_.c_str(), device_type_t::CPU, in_data.data(),
                           dali_data_type_t::DALI_INT32, shapes.data(), 1, nullptr,
                           DALI_ext_force_copy);
    }

    daliPrefetchUniform(&h, prefetch_depth);
    for (int i = 0; i < prefetch_depth; i++) {
      daliShareOutput(&h);
      auto sz = daliNumElements(&h, 0);
      thrust::device_vector<int32_t> out_data(sz);
      daliOutputCopy(&h, thrust::raw_pointer_cast(out_data.data()), 0, device_type_t::GPU, 0,
                     DALI_ext_force_sync);

      EXPECT_EQ(out_data, ref_data);

      daliOutputRelease(&h);
    }
  }
  daliDeletePipeline(&h);
}


INSTANTIATE_TEST_SUITE_P(
        InputOperatorMixedTestSimpleExecutor,
        InputOperatorMixedTest,
        ::testing::ValuesIn(input_operator_test_params_simple_executor)
);


INSTANTIATE_TEST_SUITE_P(
        InputOperatorMixedTestPipelinedExecutorUniformQueue,
        InputOperatorMixedTest,
        ::testing::ValuesIn(input_operator_test_params_pipelined_executor_uniform_queue)
);


INSTANTIATE_TEST_SUITE_P(
        InputOperatorMixedTestPipelinedExecutorSeparateQueue,
        InputOperatorMixedTest,
        ::testing::ValuesIn(input_operator_test_params_pipelined_executor_separate_queue)
);


}  // namespace dali::test

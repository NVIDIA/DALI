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
#include "dali/c_api.h"
#include "dali/pipeline/pipeline.h"
#include "dali/test/dali_test_config.h"

namespace dali::test {

using namespace dali;  // NOLINT

struct OperatorTraceTestParam {
  int cpu_queue_depth, gpu_queue_depth;
  bool exec_pipelined, exec_async;
};

namespace {

OperatorTraceTestParam operator_trace_test_params_simple_executor[] = {
        {1, 1, false, false},
        {2, 2, false, false},
};


OperatorTraceTestParam operator_trace_test_params_pipelined_executor_uniform_queue[] = {
        {2, 2, true, false},
        {3, 3, true, false},
        {2, 2, true, true},
        {3, 3, true, true},
};

OperatorTraceTestParam operator_trace_test_params_pipelined_executor_separate_queue[] = {
        {2, 3, true, false},
        {3, 2, true, false},
        {2, 3, true, true},
        {3, 2, true, true},
};

std::array<std::string, 2> operator_under_test_names = {
        "PassthroughWithTraceOpCpu", "PassthroughWithTraceOpGpu"
};

}  // namespace


class OperatorTraceTest : public ::testing::TestWithParam<OperatorTraceTestParam> {
 protected:
  void SetUp() final {
    auto parameters = GetParam();
    cpu_queue_depth_ = parameters.cpu_queue_depth;
    gpu_queue_depth_ = parameters.gpu_queue_depth;
    exec_async_ = parameters.exec_async;
    exec_pipelined_ = parameters.exec_pipelined;
    exec_separated_ = cpu_queue_depth_ != gpu_queue_depth_;

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
  std::string output_name_ = "OUTPUT";
  int n_iterations_ = 50;
  std::unique_ptr<Pipeline> pipeline_;

  bool exec_pipelined_, exec_async_, exec_separated_;
  int cpu_queue_depth_, gpu_queue_depth_;
  std::string serialized_pipeline_;

 private:
  virtual void PutTogetherDaliGraph() {
    std::string file_root = testing::dali_extra_path() + "/db/single/jpeg/";
    std::string file_list = file_root + "image_list.txt";
    pipeline_->AddOperator(OpSpec("FileReader")
                                   .AddArg("device", "cpu")
                                   .AddArg("file_root", file_root)
                                   .AddArg("file_list", file_list)
                                   .AddOutput("compressed_images", "cpu")
                                   .AddOutput("labels", "cpu"));
    pipeline_->AddOperator(OpSpec("PassthroughWithTraceOp")
                                   .AddArg("device", "cpu")
                                   .AddInput("compressed_images", "cpu")
                                   .AddOutput("PT_CPU", "cpu"),
                           operator_under_test_names[0]);
    pipeline_->AddOperator(OpSpec("PassthroughWithTraceOp")
                                   .AddArg("device", "gpu")
                                   .AddInput("compressed_images", "gpu")
                                   .AddOutput("PT_GPU", "gpu"),
                           operator_under_test_names[1]);

    std::vector<std::pair<std::string, std::string>> outputs = {
            {"PT_CPU", "cpu"},
            {"PT_GPU", "gpu"}
    };

    pipeline_->SetOutputDescs(outputs);
  }
};


TEST_P(OperatorTraceTest, OperatorTraceTest) {
  daliPipelineHandle h;
  daliCreatePipeline2(&h, serialized_pipeline_.c_str(), serialized_pipeline_.length(), batch_size_,
                      num_threads_, device_id_, exec_pipelined_, exec_async_, exec_separated_,
                      cpu_queue_depth_, cpu_queue_depth_, gpu_queue_depth_, 0);
  for (int iteration = 0; iteration < n_iterations_; iteration++) {
    auto prefetch_depth = std::min(cpu_queue_depth_, gpu_queue_depth_);
    daliPrefetchUniform(&h, prefetch_depth);
    for (int i = 0; i < prefetch_depth; i++) {
      daliShareOutput(&h);

      for (const auto & operator_name : operator_under_test_names) {
        EXPECT_EQ(daliHasOperatorTrace(&h, operator_name.c_str(), "this_trace_does_not_exist"), 0);
        ASSERT_NE(daliHasOperatorTrace(&h, operator_name.c_str(), "test_trace"), 0);

        EXPECT_EQ(std::string(daliGetOperatorTrace(&h, operator_name.c_str(), "test_trace")),
                  make_string("test_value", iteration * prefetch_depth + i));
      }

      daliOutputRelease(&h);
    }
  }
  daliDeletePipeline(&h);
}


INSTANTIATE_TEST_SUITE_P(
        OperatorTraceTestSimpleExecutor,
        OperatorTraceTest,
        ::testing::ValuesIn(operator_trace_test_params_simple_executor)
);


INSTANTIATE_TEST_SUITE_P(
        OperatorTraceTestPipelinedExecutorUniformQueue,
        OperatorTraceTest,
        ::testing::ValuesIn(operator_trace_test_params_pipelined_executor_uniform_queue)
);


INSTANTIATE_TEST_SUITE_P(
        OperatorTraceTestPipelinedExecutorSeparateQueue,
        OperatorTraceTest,
        ::testing::ValuesIn(operator_trace_test_params_pipelined_executor_separate_queue)
);


/**
 * Tests the Operator Traces with data provided by daliSetExternalInput.
 */
class OperatorTraceTestExternalInput : public OperatorTraceTest {
 private:
  void PutTogetherDaliGraph() override {
    pipeline_->AddExternalInput("OP_TRACE_IN_CPU", "cpu");
    pipeline_->AddExternalInput("OP_TRACE_IN_GPU", "gpu");
    pipeline_->AddOperator(OpSpec("PassthroughWithTraceOp")
                                   .AddArg("device", "cpu")
                                   .AddInput("OP_TRACE_IN_CPU", "cpu")
                                   .AddOutput("PT_CPU", "cpu"),
                           operator_under_test_names[0]);
    pipeline_->AddOperator(OpSpec("PassthroughWithTraceOp")
                                   .AddArg("device", "gpu")
                                   .AddInput("OP_TRACE_IN_GPU", "gpu")
                                   .AddOutput("PT_GPU", "gpu"),
                           operator_under_test_names[1]);

    std::vector<std::pair<std::string, std::string>> outputs = {
            {"PT_CPU", "cpu"},
            {"PT_GPU", "gpu"}
    };

    pipeline_->SetOutputDescs(outputs);
  }
};


namespace {

template<typename T>
thrust::host_vector<T> random_vector_cpu(std::mt19937 &mt, size_t size) {
  std::vector<T> ret(size);
  std::uniform_int_distribution<T> dist{0, 255};
  auto gen = [&]() { return dist(mt); };
  std::generate(ret.begin(), ret.end(), gen);
  return ret;
}


template<typename T>
thrust::device_vector<T> random_vector_gpu(std::mt19937 &mt, size_t size) {
  thrust::device_vector<T> ret = random_vector_cpu<T>(mt, size);
  return ret;
}

}  // namespace


TEST_P(OperatorTraceTestExternalInput, OperatorTraceTestExternalInput) {
  daliPipelineHandle h;
  daliCreatePipeline2(&h, serialized_pipeline_.c_str(), serialized_pipeline_.length(), batch_size_,
                      num_threads_, device_id_, exec_pipelined_, exec_async_, exec_separated_,
                      cpu_queue_depth_, cpu_queue_depth_, gpu_queue_depth_, 0);
  std::mt19937 rng(42);
  for (int iteration = 0; iteration < n_iterations_; iteration++) {
    auto prefetch_depth = std::min(cpu_queue_depth_, gpu_queue_depth_);

    // Feed CPU input data.
    for (int i = 0; i < prefetch_depth; i++) {
      size_t sample_size = 42;
      auto in_data = random_vector_cpu<uint8_t>(rng, sample_size * batch_size_);
      std::vector<int64_t> shapes(batch_size_, sample_size);
      daliSetExternalInput(&h, "OP_TRACE_IN_CPU", device_type_t::CPU, in_data.data(),
                           dali_data_type_t::DALI_UINT8, shapes.data(), 1, nullptr,
                           DALI_ext_default);
    }

    // Feed GPU input data.
    for (int i = 0; i < prefetch_depth; i++) {
      int sample_size = 42;
      auto in_data = random_vector_gpu<uint8_t>(rng, sample_size * batch_size_);
      std::vector<int64_t> shapes(batch_size_, sample_size);
      daliSetExternalInput(&h, "OP_TRACE_IN_GPU", device_type_t::GPU,
                           thrust::raw_pointer_cast(in_data.data()), dali_data_type_t::DALI_UINT8,
                           shapes.data(), 1, nullptr, DALI_ext_default);
    }

    daliPrefetchUniform(&h, prefetch_depth);
    for (int i = 0; i < prefetch_depth; i++) {
      daliShareOutput(&h);

      for (const auto &operator_name : operator_under_test_names) {
        EXPECT_EQ(daliHasOperatorTrace(&h, operator_name.c_str(), "this_trace_does_not_exist"), 0);
        ASSERT_NE(daliHasOperatorTrace(&h, operator_name.c_str(), "test_trace"), 0);

        EXPECT_EQ(std::string(daliGetOperatorTrace(&h, operator_name.c_str(), "test_trace")),
                  make_string("test_value", iteration * prefetch_depth + i));
      }

      daliOutputRelease(&h);
    }
  }
  daliDeletePipeline(&h);
}


INSTANTIATE_TEST_SUITE_P(
        OperatorTraceTestExternalInputSimpleExecutor,
        OperatorTraceTestExternalInput,
        ::testing::ValuesIn(operator_trace_test_params_simple_executor)
);


INSTANTIATE_TEST_SUITE_P(
        OperatorTraceTestExternalInputPipelinedExecutorUniformQueue,
        OperatorTraceTestExternalInput,
        ::testing::ValuesIn(operator_trace_test_params_pipelined_executor_uniform_queue)
);


INSTANTIATE_TEST_SUITE_P(
        OperatorTraceTestExternalInputPipelinedExecutorSeparateQueue,
        OperatorTraceTestExternalInput,
        ::testing::ValuesIn(operator_trace_test_params_pipelined_executor_separate_queue)
);

}  // namespace dali::test

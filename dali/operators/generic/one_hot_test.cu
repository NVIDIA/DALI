// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream.h"
#include "dali/test/dali_operator_test.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/operators/generic/one_hot.cuh"


namespace dali {

template <typename OutputType, typename InputType, int batch_size_, int64_t num_classes_, int axis_,
          int64_t... Dims>
struct OneHotTestParams {
  using Out = OutputType;
  using In = InputType;
  static constexpr int batch_size = batch_size_;
  static constexpr int64_t num_classes = num_classes_;
  static constexpr int axis = axis_;
  static constexpr int ndim = sizeof...(Dims);
  TensorShape<ndim> shape = {Dims...};
};

template <typename TestConfig>
struct OneHotOpGpuPerfTest : public ::testing::Test {
  using Out = typename TestConfig::Out;
  using In = typename TestConfig::In;
  void SetUp() override {
    stream_ = CUDAStream::Create(true);

    auto &input_shape = config_.shape;
    sample_descs_tensor_.reshape(uniform_list_shape<1>(1, {config_.batch_size}));
    auto samples_cpu = sample_descs_tensor_.cpu()[0];
    auto outer = input_shape.first(config_.axis);
    auto inner = input_shape.last(input_shape.size() - config_.axis);
    auto out_shape = shape_cat(shape_cat(outer, config_.num_classes), inner);
    auto input_list_shape = uniform_list_shape<TestConfig::ndim>(config_.batch_size, input_shape);
    auto out_list_shape = uniform_list_shape<TestConfig::ndim + 1>(config_.batch_size, out_shape);
    input_.reshape(input_list_shape);
    output_.reshape(out_list_shape);
    int num;
    auto seq_gen = [&num]() { return num = (num + 1) % TestConfig::num_classes; };
    Fill(input_.cpu(), seq_gen);

    auto outer_vol = volume(input_shape.begin(), input_shape.begin() + config_.axis);
    auto inner_vol = volume(input_shape.begin() + config_.axis, input_shape.end());
    auto inner_vol_classes = inner_vol * config_.num_classes;
    auto output_vol = outer_vol * inner_vol_classes;

    memset(samples_cpu.data, 0, TestConfig::batch_size * sizeof(one_hot::SampleDesc));
    auto input_gpu = input_.gpu(stream_);
    auto output_gpu = output_.gpu(stream_);
    for (int sample_id = 0; sample_id < config_.batch_size; ++sample_id) {
      samples_cpu(sample_id)->inner_vol = inner_vol;
      samples_cpu(sample_id)->inner_vol_classes = inner_vol_classes;
      samples_cpu(sample_id)->output_vol = output_vol;
      samples_cpu(sample_id)->out = output_gpu[sample_id].data;
      samples_cpu(sample_id)->in = input_gpu[sample_id].data;
    }
    samples_gpu_ = sample_descs_tensor_.gpu(stream_)[0].data;
  }

  void MeasurePerf() {
    auto input_vol = volume(config_.shape);
    auto output_vol = input_vol * config_.num_classes;

    const int block = 256;
    auto grid = one_hot::gridHelper(output_vol, config_.batch_size, block);

    Out on_value = 1, off_value = 0;
    one_hot::PopulateOneHot<Out, In>
    <<<grid, block, 0, stream_>>>(on_value, off_value, samples_gpu_);

    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);

    CUDA_CALL(cudaEventRecord(start, stream_));
    constexpr int kIters = 100;
    for (int i = 0; i < kIters; i++) {
      one_hot::PopulateOneHot<Out, In><<<grid, block, 0, stream_>>>(
        on_value, off_value, samples_gpu_);
    }
    CUDA_CALL(cudaEventRecord(end, stream_));
    CUDA_CALL(cudaDeviceSynchronize());
    float time;
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));

    time *= (1e+6f / kIters);  // convert to nanoseconds / 100 samples
    int64_t data_size = (input_vol * sizeof(In) + output_vol * sizeof(Out)) * config_.batch_size;
    std::cerr << "Throughput: " << data_size / time << " GB/s" << std::endl;
  }

  TestConfig config_{};
  CUDAStream stream_;

  kernels::TestTensorList<In, TestConfig::ndim> input_;
  kernels::TestTensorList<Out, TestConfig::ndim + 1> output_;
  kernels::TestTensorList<typename one_hot::SampleDesc, 1> sample_descs_tensor_;

  const one_hot::SampleDesc *samples_gpu_;
};

TYPED_TEST_SUITE_P(OneHotOpGpuPerfTest);

TYPED_TEST_P(OneHotOpGpuPerfTest, Perf) {
  std::cerr << "batch_size: " << this->config_.batch_size << ", num_classes: "
            << this->config_.num_classes << ", sample_dim: " << this->config_.shape
            << std::endl;
  this->MeasurePerf();
}

REGISTER_TYPED_TEST_SUITE_P(OneHotOpGpuPerfTest, Perf);

using TestConfigs = ::testing::Types<
OneHotTestParams<int, int, 1, 256, 0, 1024, 1024>,
OneHotTestParams<int, int, 1, 256, 1, 1024, 1024>,
OneHotTestParams<int, int, 1, 256, 2, 1024, 1024>,
OneHotTestParams<int64_t, int, 1, 256, 0, 1024, 1024>,
OneHotTestParams<int64_t, int, 1, 256, 1, 1024, 1024>,
OneHotTestParams<int64_t, int, 1, 256, 2, 1024, 1024>,
OneHotTestParams<int64_t, int64_t, 1, 256, 0, 1024, 1024>,
OneHotTestParams<int64_t, int64_t, 1, 256, 1, 1024, 1024>,
OneHotTestParams<int64_t, int64_t, 1, 256, 2, 1024, 1024>,
OneHotTestParams<int8_t, int8_t, 1, 256, 0, 1024, 1024>,
OneHotTestParams<int8_t, int8_t, 1, 256, 1, 1024, 1024>,
OneHotTestParams<int8_t, int8_t, 1, 256, 2, 1024, 1024>,
OneHotTestParams<int, int, 4, 8, 0, 32, 32, 32, 32, 32>,
OneHotTestParams<int, int, 4, 8, 1, 32, 32, 32, 32, 32>,
OneHotTestParams<int, int, 4, 8, 2, 32, 32, 32, 32, 32>,
OneHotTestParams<int, int, 4, 8, 3, 32, 32, 32, 32, 32>,
OneHotTestParams<int, int, 4, 8, 4, 32, 32, 32, 32, 32>,
OneHotTestParams<int, int, 4, 8, 5, 32, 32, 32, 32, 32>,
OneHotTestParams<int, int, 4, 1024 * 256, 0, 1024>,
OneHotTestParams<int, int, 4, 1024 * 256, 1, 1024>,
OneHotTestParams<int, int, 16, 64, 0, 1024, 128>,
OneHotTestParams<int, int, 16, 64, 1, 1024, 128>,
OneHotTestParams<int, int, 16, 64, 2, 1024, 128>
>;

INSTANTIATE_TYPED_TEST_SUITE_P(OneHotOpGpu, OneHotOpGpuPerfTest, TestConfigs);

}  // namespace dali

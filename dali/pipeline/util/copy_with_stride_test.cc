// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/util/copy_with_stride.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include "dali/core/dev_buffer.h"
#include "dali/pipeline/data/dltensor.h"

namespace dali {

TEST(CopyWithStrideTest, OneDim) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T data[] = {1, 2, 3, 4, 5, 6};
  TensorShape<1> stride{2};
  TensorShape<1> shape{3};
  constexpr int vol = 3;
  ASSERT_EQ(vol, volume(shape));
  std::array<T, vol> out;
  DLTensorResource resource(shape);
  resource.strides = stride;
  auto dl_tensor =
      MakeDLTensor(data, dtype, false, -1, std::make_unique<DLTensorResource>(resource));
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, vol>{1, 3, 5}));
}

TEST(CopyWithStrideTest, TwoDims)  {
  const auto dtype = DALI_INT64;
  using T = int64_t;
  T data[] = {11, 12, 13, 14,
              21, 22, 23, 24,
              31, 32, 33, 34,
              41, 42, 43, 44};
  TensorShape<2> stride{8, 1};
  TensorShape<2> shape{2, 4};
  constexpr int vol = 8;
  ASSERT_EQ(vol, volume(shape));
  std::array<T, vol> out;
  DLTensorResource resource(shape);
  resource.strides = stride;
  auto dl_tensor =
      MakeDLTensor(data, dtype, false, -1, std::make_unique<DLTensorResource>(resource));
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, vol>{11, 12, 13, 14,
                                         31, 32, 33, 34}));
}

TEST(CopyWithStrideTest, SimpleCopy) {
  const auto dtype = DALI_UINT8;
  using T = uint8_t;
  T data[] = {1, 2,
              3, 4,

              5, 6,
              7, 8};
  TensorShape<3>  stride{4, 2, 1};
  TensorShape<3> shape{2, 2, 2};
  constexpr int vol = 8;
  ASSERT_EQ(vol, volume(shape));
  std::array<T, vol> out;
  DLTensorResource resource(shape);
  resource.strides = stride;
  auto dl_tensor =
      MakeDLTensor(data, dtype, false, -1, std::make_unique<DLTensorResource>(resource));
  CopyDlTensorCpu(out.data(), dl_tensor);
  ASSERT_TRUE((out == std::array<T, vol>{1, 2,
                                         3, 4,

                                         5, 6,
                                         7, 8}));
}

DLMTensorPtr AsDlTensor(void* data, DALIDataType dtype, TensorShape<> shape, TensorShape<> stride) {
  DLTensorResource resource(shape);
  resource.strides = stride;
  return MakeDLTensor(data, dtype, true, 0, std::make_unique<DLTensorResource>(resource));
}

std::vector<DLMTensorPtr> DlTensorSingletonBatch(DLMTensorPtr dl_tensor) {
  std::vector<DLMTensorPtr> dl_tensors;
  dl_tensors.push_back(std::move(dl_tensor));
  return dl_tensors;
}

TensorList<GPUBackend> SingletonTL(TensorShape<> shape, DALIDataType dtype) {
  TensorList<GPUBackend> output;
  TensorListShape tls(1, shape.sample_dim());
  tls.set_tensor_shape(0, shape);
  output.Resize(tls, dtype);
  return output;
}

TEST(CopyWithStrideTest, OneDimGPU) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[] = {1, 2, 3, 4, 5, 6};
  DeviceBuffer<T> data;
  data.from_host(h_data);
  TensorShape<1> stride{2};
  TensorShape<1> shape{3};
  constexpr int vol = 3;
  ASSERT_EQ(vol, volume(shape));
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 3, 5}));
}

TEST(CopyWithStrideTest, TwoDimsGPU) {
  const auto dtype = DALI_INT64;
  using T = int64_t;
  T h_data[] = {11, 12, 13, 14,
                21, 22, 23, 24,
                31, 32, 33, 34,
                41, 42, 43, 44};
  TensorShape<2> stride{8, 1};
  TensorShape<2> shape{2, 4};
  constexpr int vol = 8;
  ASSERT_EQ(vol, volume(shape));
  DeviceBuffer<int64> data;
  data.from_host(h_data);
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{11, 12, 13, 14, 31, 32, 33, 34}));
}

TEST(CopyWithStrideTest, TwoDimsGPUOdd) {
  const auto dtype = DALI_UINT8;
  using T = uint8_t;
  T h_data[] = {1,  2,  3,  4,  5,
                6,  7,  8,  9,  10,
                11, 12, 13, 14, 15,
                16, 17, 18, 19, 20,
                21, 22, 23, 24, 25,
                26, 27, 28, 29, 30};
  TensorShape<2> stride{15, 1};
  TensorShape<2> shape{2, 4};
  constexpr int vol = 8;
  ASSERT_EQ(vol, volume(shape));
  DeviceBuffer<T> data;
  data.from_host(h_data);
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 2, 3, 4, 16, 17, 18, 19}));
}

TEST(CopyWithStrideTest, TwoDimsInnerStride) {
  const auto dtype = DALI_UINT8;
  using T = uint8_t;
  T h_data[] = {1,  2,  3,  4,  5,
                6,  7,  8,  9,  10,
                11, 12, 13, 14, 15,
                16, 17, 18, 19, 20,
                21, 22, 23, 24, 25,
                26, 27, 28, 29, 30};
  TensorShape<2> stride{15, 5};
  TensorShape<2> shape{2, 3};
  constexpr int vol = 6;
  ASSERT_EQ(vol, volume(shape));
  DeviceBuffer<T> data;
  data.from_host(h_data);
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1, 6, 11, 16, 21, 26}));
}

TEST(CopyWithStrideTest, TwoDimsTransposed) {
  const auto dtype = DALI_UINT16;
  using T = uint16_t;
  T h_data[] = {1,  2,  3,  4,  5,
                6,  7,  8,  9,  10,
                11, 12, 13, 14, 15,
                16, 17, 18, 19, 20,
                21, 22, 23, 24, 25,
                26, 27, 28, 29, 30};
  TensorShape<2> stride{1, 5};
  TensorShape<2> shape{5, 6};
  constexpr int vol = 30;
  ASSERT_EQ(vol, volume(shape));
  DeviceBuffer<T> data;
  data.from_host(h_data);
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  std::array<T, vol> ref = {
    1,  6, 11, 16, 21, 26,
    2,  7, 12, 17, 22, 27,
    3,  8, 13, 18, 23, 28,
    4,  9, 14, 19, 24, 29,
    5, 10, 15, 20, 25, 30};
  ASSERT_TRUE(h_out == ref);
}

TEST(CopyWithStrideTest, SimpleCopyGPU) {
  const auto dtype = DALI_FLOAT;
  using T = float;
  T h_data[] = {1,  2,  3,
                4,  5,  6,

                7,  8,  9,
                10, 11, 12};
  TensorShape<3> stride{6, 3, 1};
  TensorShape<3> shape{2, 2, 3};
  constexpr int vol = 12;
  ASSERT_EQ(vol, volume(shape));
  DeviceBuffer<T> data;
  data.from_host(h_data);
  auto dl_tensors = DlTensorSingletonBatch(AsDlTensor(data, dtype, shape, stride));
  auto output_tl = SingletonTL(shape, dtype);
  CopyDlTensorBatchGpu(output_tl, dl_tensors, 0);
  std::array<T, vol> h_out;
  CUDA_CALL(cudaMemcpy(h_out.data(), output_tl.raw_mutable_tensor(0), vol * sizeof(T),
                       cudaMemcpyDeviceToHost));
  ASSERT_TRUE((h_out == std::array<T, vol>{1,  2,  3,
                                           4,  5,  6,

                                           7,  8,  9,
                                           10, 11, 12}));
}

}  // namespace dali

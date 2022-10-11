// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/imgproc/geom/remap_npp.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>
#include "dali/core/cuda_stream_pool.h"
#include "dali/kernels/context.h"
#include "dali/pipeline/data/backend.h"
#include "dali/test/cv_mat_utils.h"
#include "dali/test/mat2tensor.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "include/dali/core/tensor_shape_print.h"

namespace dali::kernels::remap::test {

using namespace std;  // NOLINT

namespace {

/**
 * Allocate test data in managed memory (to be accessible both from CPU and GPU).
 *
 * Might be used to generate input data as well as remap parameters (i.e. maps).
 *
 * This function returns 3 entities:
 * 1. TensorView, that wraps a buffer (3.)
 * 2. cv::Mat, that wraps a buffer (3.)
 * 3. A buffer, which contains test data. The caller is responsible for calling `cudaFree`
 *    on this buffer.
 *
 * All three returned entities correspond to a single test case.
 *
 * @tparam Backend StorageBackend. Typically StorageUnified.
 * @tparam T Type of the test data.
 * @tparam ndims Number of dimensions in the generated data.
 * @tparam Generator Function, that suffices signature `T g();` and returns a random number.
 * @param shape Shape of the test data.
 * @param stream The stream, on which the copy will be conducted.
 * @return
 */
template<typename Backend = StorageUnified, typename T, int ndims, typename Generator>
tuple<TensorView<Backend, T, ndims>, cv::Mat, T *>
alloc_test_data(TensorShape<ndims> shape, Generator g, cudaStream_t stream = 0) {
  using U = remove_const_t<T>;
  vector<U> source_data(volume(shape));
  generate(source_data.begin(), source_data.end(), g);
  remove_const_t<U> *ret;
  CUDA_CALL(cudaMallocManaged(&ret, volume(shape) * sizeof(T)));
  CUDA_CALL(cudaMemcpyAsync(ret, source_data.data(), volume(shape) * sizeof(T), cudaMemcpyDefault,
                            stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
  auto nchannels = shape.sample_dim() < 3 ? 1 : shape[2];
  cv::Mat cv_mat(shape[0], shape[1], CV_MAKETYPE(cv::DataDepth<U>::value, nchannels), ret);
  return make_tuple(TensorView<Backend, T, ndims>(ret, shape), cv_mat, ret);
}


/**
 * Function that counts outlying pixels. This function is used for verifying the remap result.
 *
 * Outlying pixels for given two images are the pixels, that do not match the corresponding one.
 * In other words, these are the pixels that are not (0,0,0) in the difference of the images.
 *
 * @param diff The difference of two images (img1 - img2).
 * @return Number of outlying pixels.
 */
size_t count_outlying_pixels(const cv::Mat &diff) {
  const auto &p = diff.data;
  int ret = 0;
  for (int i = 0; i < diff.rows * diff.cols * diff.channels(); i += 3) {
    if (p[i] + p[i + 1] + p[i + 2] != 0) ret++;
  }
  return ret;
}

}  // namespace

template<typename T>
class NppRemapTest : public ::testing::Test {
  /*
   * The idea of this test is to:
   * 1. Prepare test data,
   * 2. Run NppRemapKernel,
   * 3. Run cv::remap,
   * 4. Compare results.
   */
 protected:
  using StorageType = StorageUnified;


  /**
   * Prepares test data. A sample data is a buffer allocated in managed memory
   * (to be accessible both from CPU and GPU). This buffer is then wrapped into
   * Tensor(List)View and cv::Mat. Remap parameters (maps) are prepared likewise,
   * but with different type and shape.
   */
  void SetUp() final {
    // Prepare random number generators
    uniform_int_distribution<> imgdist{0, 255};
    uniform_real_distribution<> wdist{0, static_cast<double>(width_)};
    uniform_real_distribution<> hdist{0, static_cast<double>(height_)};
    auto imgrng = [&]() { return imgdist(mt_); };
    auto wrng = [&]() { return wdist(mt_); };
    auto hrng = [&]() { return hdist(mt_); };

    for (size_t sample_idx = 0; sample_idx < batch_size_; sample_idx++) {
      cv::Mat _redundant;  // This won't be used

      // Convenient aliases
      auto &rit = remap_in_tv_[sample_idx];
      auto &ric = remap_in_cv_[sample_idx];
      auto &rid = remap_in_data_[sample_idx];
      auto &rot = remap_out_tv_[sample_idx];
      auto &roc = remap_out_cv_[sample_idx];
      auto &rod = remap_out_data_[sample_idx];
      auto &mxt = mapx_tv_[sample_idx];
      auto &mxc = mapx_cv_[sample_idx];
      auto &mxd = mapx_data_[sample_idx];
      auto &myt = mapy_tv_[sample_idx];
      auto &myc = mapy_cv_[sample_idx];
      auto &myd = mapy_data_[sample_idx];

      // Allocating test data and returning the buffer together with the containers
      tie(rit, ric, rid) = alloc_test_data<StorageType, const T, -1>(img_shape_, imgrng);
      tie(rot, roc, rod) = alloc_test_data<StorageType, T, -1>(img_shape_, []() { return 0; });
      tie(mxt, mxc, mxd) = alloc_test_data<StorageType, const MapType, 2>(map_shape_, wrng);
      tie(myt, myc, myd) = alloc_test_data<StorageType, const MapType, 2>(map_shape_, hrng);

      rit.shape = rot.shape = img_shape_;
      mxt.shape = myt.shape = map_shape_;
    }

    ctx_.gpu.stream = 0;
  }


  /**
   * Clean up the test data.
   */
  void TearDown() final {
    for (size_t sample_idx = 0; sample_idx < batch_size_; sample_idx++) {
      cudaFree(const_cast<void *>(reinterpret_cast<const void *>(remap_in_data_[sample_idx])));
      cudaFree(remap_out_data_[sample_idx]);
      cudaFree(const_cast<void *>(reinterpret_cast<const void *>(mapx_data_[sample_idx])));
      cudaFree(const_cast<void *>(reinterpret_cast<const void *>(mapy_data_[sample_idx])));
    }
  }


  /**
   * Dimensions of the test images.
   */
  int width_ = 1920;
  int height_ = 1080;
  size_t batch_size_ = 8;
  TensorShape<> img_shape_ = {height_, width_, 3};
  TensorShape<> map_shape_ = {height_, width_};

  using MapType = std::conditional_t<std::is_same_v<T, double>, double, float>;

  /**
   * Buffers, that store the test data.
   * These are filled after SetUp() call.
   */
  vector<const T *> remap_in_data_{batch_size_};
  vector<T *> remap_out_data_{batch_size_};
  vector<const MapType *> mapx_data_{batch_size_}, mapy_data_{batch_size_};

  /**
   * Containers (TensorViews and cv::Mats), that wrap the buffers with the test data.
   * They are initialized after SetUp() call. `vector<TensorView<>>` is used instead of
   * `TensorListView<>`, because the vector is more convenient to work together with cv::Mat
   * and converting the vector to TensorListView is easy.
   */
  vector<TensorView<StorageType, const T, -1>> remap_in_tv_{batch_size_};
  vector<TensorView<StorageType, T, -1>> remap_out_tv_{batch_size_};
  vector<TensorView<StorageType, const MapType, 2>> mapx_tv_{batch_size_};
  vector<TensorView<StorageType, const MapType, 2>> mapy_tv_{batch_size_};
  vector<cv::Mat> remap_in_cv_{batch_size_}, remap_out_cv_{batch_size_}, mapx_cv_{
          batch_size_}, mapy_cv_{batch_size_};

  mt19937 mt_;

  KernelContext ctx_;
};

using NppRemapTypes = ::testing::Types<uint8_t, uint16_t, int16_t, float, double>;
TYPED_TEST_SUITE(NppRemapTest, NppRemapTypes);


TYPED_TEST(NppRemapTest, RemapVsOpencvTest) {
  using T = TypeParam;

  NppRemapKernel<typename NppRemapTest<T>::StorageType, T> kernel;
  vector<DALIInterpType> interps(this->batch_size_, DALI_INTERP_NN);
  kernel.Run(this->ctx_, make_tensor_list(this->remap_out_tv_),
             make_tensor_list(this->remap_in_tv_), make_tensor_list(this->mapx_tv_),
             make_tensor_list(this->mapy_tv_), {}, {}, make_span(interps));
  CUDA_CALL(cudaStreamSynchronize(this->ctx_.gpu.stream));

  vector<cv::Mat> remap_out_cv(this->batch_size_);

  for (size_t sample_idx = 0; sample_idx < this->batch_size_; sample_idx++) {
    cv::remap(this->remap_in_cv_[sample_idx], remap_out_cv[sample_idx], this->mapx_cv_[sample_idx],
              this->mapy_cv_[sample_idx], cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

    auto npp_remap = testing::tensor_to_mat(this->remap_out_tv_[sample_idx], true, false);
    ASSERT_EQ(npp_remap.rows, this->height_);
    ASSERT_EQ(npp_remap.cols, this->width_);
    ASSERT_EQ(this->height_, remap_out_cv[sample_idx].rows);
    ASSERT_EQ(this->width_, remap_out_cv[sample_idx].cols);
    EXPECT_LE(static_cast<float>(count_outlying_pixels(npp_remap - remap_out_cv[sample_idx])) /
              this->width_ / this->height_,
              .01);  // Expect that there's less than 1% of outlying pixels
  }
}


TYPED_TEST(NppRemapTest, RemapVsOpencvUnifiedParametersTest) {
  // This test runs another overload of NppRemapKernel::Run
  using T = TypeParam;

  NppRemapKernel<typename NppRemapTest<T>::StorageType, T> kernel;
  vector<DALIInterpType> interps(this->batch_size_, DALI_INTERP_NN);
  kernel.Run(this->ctx_, make_tensor_list(this->remap_out_tv_),
             make_tensor_list(this->remap_in_tv_), this->mapx_tv_[0], this->mapy_tv_[0], {}, {},
             DALI_INTERP_NN);
  CUDA_CALL(cudaStreamSynchronize(this->ctx_.gpu.stream));

  vector<cv::Mat> remap_out_cv(this->batch_size_);

  for (size_t sample_idx = 0; sample_idx < this->batch_size_; sample_idx++) {
    cv::remap(this->remap_in_cv_[sample_idx], remap_out_cv[sample_idx], this->mapx_cv_[0],
              this->mapy_cv_[0], cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

    auto npp_remap = testing::tensor_to_mat(this->remap_out_tv_[sample_idx], true, false);
    ASSERT_EQ(npp_remap.rows, this->height_);
    ASSERT_EQ(npp_remap.cols, this->width_);
    ASSERT_EQ(this->height_, remap_out_cv[sample_idx].rows);
    ASSERT_EQ(this->width_, remap_out_cv[sample_idx].cols);
    EXPECT_LE(static_cast<float>(count_outlying_pixels(npp_remap - remap_out_cv[sample_idx])) /
              this->width_ / this->height_,
              .01);  // Expect that there's less than 1% of outlying pixels
  }
}

}  // namespace dali::kernels::remap::test

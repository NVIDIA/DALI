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
#include <cmath>
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
 * Generates test data in the provided CPU TensorView and creates a cvMat
 * and copies the generated test data to it. All data correspond to a single test case.
 *
 * @tparam T Type of the test data.
 * @tparam ndims Number of dimensions in the generated data.
 * @tparam Generator Function, that suffices signature `T g();` and returns a random number.
 * @param tv TensorView for the test data
 * @param cv_mat cvMat that is created based on the shape of tv
 * @param g Test data generator.
 * @return
 */
template<typename T, int ndims, typename Generator>
void fill_test_data(const TensorView<StorageCPU, T, ndims> &tv, cv::Mat &cv_mat, Generator g) {
  auto shape = tv.shape;

  using U = remove_const_t<T>;

  vector<U> source_data(volume(shape));
  generate(const_cast<U*>(tv.data), const_cast<U*>(tv.data) + tv.num_elements(), g);

  auto nchannels = shape.sample_dim() < 3 ? 1 : shape[2];
  cv_mat = cv::Mat(shape[0], shape[1], CV_MAKETYPE(cv::DataDepth<U>::value, nchannels));
  EXPECT_TRUE(cv_mat.isContinuous());
  std::copy(tv.data, tv.data + cv_mat.total() * nchannels, cv_mat.ptr<U>());
}

/**
 * Function that counts outlying pixels. This function is used for verifying the remap result.
 *
 * Outlying pixels for given two images are the pixels, that do not match the corresponding one.
 * In other words, these are the pixels that are not (0,0,0) in the difference of the images.
 *
 * @tparam T Type of the input data.
 * @param diff The difference of two images (img1 - img2). This cv::Mat mustn't be strided.
 * @return Number of outlying pixels.
 */
template<typename T>
size_t count_outlying_pixels(const cv::Mat &diff) {
  assert(diff.isContinuous());
  const auto &p = reinterpret_cast<const T *>(diff.data);
  int ret = 0;
  for (int i = 0; i < diff.rows * diff.cols * diff.channels(); i += diff.channels()) {
    // Effectively compare to 0, but secure from float precision.
    auto res = std::accumulate(p, p + diff.channels(), 0.f);
    if (abs(res) > 1e-4) ret++;
  }
  return ret;
}


/**
 * Picks random number, avoiding the situation it is halfway (i.e. .5f, -2.5f, etc...).
 *
 * Since halfway rounding is different in DALI and in OpenCV, we have to
 * avoid such values for this test.
 */
template<typename T>
T pick_avoiding_halves(uniform_real_distribution<T> &dist, mt19937 &engine) {
  while (true) {
    auto rand = dist(engine);
    auto halfway_dist = abs(rand) - std::floor(abs(rand)) - .5f;
    if (abs(halfway_dist) > 1e-4) return rand;
  }
}

}  // namespace

/**
 * Test of the NppRemapKernel.
 * @tparam InputT Type of the input data to the kernel.
 * @tparam nchannels Number of channels in the input image.
 */
template<typename InputT, int nchannels>
class NppRemapTest : public ::testing::Test {
 protected:
  using StorageType = StorageGPU;
  using Kernel = NppRemapKernel<StorageType, InputT>;


  /**
   * Prepares test data. A sample data is a buffer allocated in managed memory
   * (to be accessible both from CPU and GPU). This buffer is then wrapped into
   * Tensor(List)View and cv::Mat. Remap parameters (maps) are prepared likewise,
   * but with different type and shape.
   */
  void SetUp() final {
    // Prepare random number generators
    conditional_t<
            is_floating_point_v<InputT>,
            uniform_real_distribution<float>,
            uniform_int_distribution<>
    > imgdist{0, 255};
    uniform_real_distribution<float> wdist{0, static_cast<float>(width_)};
    uniform_real_distribution<float> hdist{0, static_cast<float>(height_)};
    auto imgrng = [&]() { return imgdist(mt_); };
    auto wrng = [&]() { return pick_avoiding_halves(wdist, mt_); };
    auto hrng = [&]() { return pick_avoiding_halves(hdist, mt_); };

    remap_in_ttv_.reshape(uniform_list_shape<-1>(batch_size_, img_shape_));
    remap_out_ttv_.reshape(uniform_list_shape<-1>(batch_size_, img_shape_));
    mapx_ttv_.reshape(uniform_list_shape<2>(batch_size_, map_shape_));
    mapy_ttv_.reshape(uniform_list_shape<2>(batch_size_, map_shape_));

    for (size_t sample_idx = 0; sample_idx < batch_size_; sample_idx++) {
      auto &ric = remap_in_cv_[sample_idx];
      auto &mxc = mapx_cv_[sample_idx];
      auto &myc = mapy_cv_[sample_idx];

      fill_test_data(remap_in_ttv_.cpu()[sample_idx], ric, imgrng);
      fill_test_data(mapx_ttv_.cpu()[sample_idx], mxc, wrng);
      fill_test_data(mapy_ttv_.cpu()[sample_idx], myc, hrng);
    }
    ctx_.gpu.stream = 0;
  }

  /**
   * Function that conducts the test. It's extracted from the test body to
   * overcome GTest limitation, that it's not possible to parameterize a typed test.
   * (It's because we need to parameterize the test with both I/O type and number
   * of channels).
   * @tparam unified The NppRemapKernel has a Run convenient overload for the case
   *                 when the transformation parameters are the same for every sample
   *                 in the output batch. If true, this overload will be invoked.
   */
  template<bool unified>
  void DoTest() {
    int device_id = -1;
    CUDA_CALL(cudaGetDevice(&device_id));
    Kernel kernel(device_id);
    invoke_kernel<unified>(kernel);
    auto stream = this->ctx_.gpu.stream;
    remap_out_ttv_.cpu(stream);
    CUDA_CALL(cudaStreamSynchronize(stream));

    vector<cv::Mat> remap_out_cv(this->batch_size_);

    for (size_t sample_idx = 0; sample_idx < this->batch_size_; sample_idx++) {
      cv::remap(this->remap_in_cv_[sample_idx], remap_out_cv[sample_idx],
                this->mapx_cv_[unified ? 0 : sample_idx], this->mapy_cv_[unified ? 0 : sample_idx],
                cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

      auto npp_remap = testing::tensor_to_mat(this->remap_out_ttv_.cpu(stream)[sample_idx], true,
                                              false);
      ASSERT_EQ(npp_remap.rows, this->height_);
      ASSERT_EQ(npp_remap.cols, this->width_);
      ASSERT_EQ(this->height_, remap_out_cv[sample_idx].rows);
      ASSERT_EQ(this->width_, remap_out_cv[sample_idx].cols);
      EXPECT_LE(static_cast<float>(count_outlying_pixels<InputT>(
              npp_remap - remap_out_cv[sample_idx])) /
                this->width_ / this->height_,
                .01);  // Expect that there's less than 1% of outlying pixels
    }
  }


  template<bool unified>
  void invoke_kernel(Kernel &kernel) {
    auto stream = this->ctx_.gpu.stream;
    if (unified) {
      kernel.Run(this->ctx_, this->remap_out_ttv_.gpu(stream),
                 this->remap_in_ttv_.gpu(stream), this->mapx_ttv_.gpu(stream)[0],
                 this->mapy_ttv_.gpu(stream)[0], {}, {}, DALI_INTERP_NN);
    } else {
      vector<DALIInterpType> interps(this->batch_size_, DALI_INTERP_NN);
      kernel.Run(this->ctx_, this->remap_out_ttv_.gpu(stream),
                 this->remap_in_ttv_.gpu(stream), this->mapx_ttv_.gpu(stream),
                 this->mapy_ttv_.gpu(stream), {}, {}, make_span(interps));
    }
  }


  /**
   * Dimensions of the test images.
   */
  int width_ = 1920;
  int height_ = 1080;
  size_t batch_size_ = 8;
  TensorShape<> img_shape_ = {height_, width_, nchannels};
  TensorShape<> map_shape_ = {height_, width_};

  using MapType = float;

  /**
   * Buffers, that store the test data.
   * These are filled after SetUp() call.
   */
  vector<const InputT *> remap_in_data_{batch_size_};
  vector<InputT *> remap_out_data_{batch_size_};
  vector<const MapType *> mapx_data_{batch_size_}, mapy_data_{batch_size_};

  /**
   * Containers (TestTensorList and cv::Mats), that wrap the buffers with the test data.
   * They are initialized after SetUp() call.
   */
  TestTensorList<const InputT, -1> remap_in_ttv_;
  TestTensorList<InputT, -1> remap_out_ttv_;
  TestTensorList<const MapType, 2> mapx_ttv_;
  TestTensorList<const MapType, 2> mapy_ttv_;

  vector<cv::Mat> remap_in_cv_{batch_size_}, mapx_cv_{
          batch_size_}, mapy_cv_{batch_size_};

  mt19937 mt_;

  KernelContext ctx_;
};

using NppRemapInputTypes = ::testing::Types<uint8_t, uint16_t, int16_t, float>;


template<typename T>
class NppRemapTest1Channel : public NppRemapTest<T, 1> {
};

TYPED_TEST_SUITE(NppRemapTest1Channel, NppRemapInputTypes);

TYPED_TEST(NppRemapTest1Channel, RemapVsOpencvTest) {
  this->template DoTest<false>();
}


TYPED_TEST(NppRemapTest1Channel, RemapVsOpencvUnifiedParametersTest) {
  // This test runs another overload of NppRemapKernel::Run
  this->template DoTest<true>();
}


template<typename T>
class NppRemapTest3Channel : public NppRemapTest<T, 3> {
};

TYPED_TEST_SUITE(NppRemapTest3Channel, NppRemapInputTypes);

TYPED_TEST(NppRemapTest3Channel, RemapVsOpencvTest) {
  this->template DoTest<false>();
}


TYPED_TEST(NppRemapTest3Channel, RemapVsOpencvUnifiedParametersTest) {
  // This test runs another overload of NppRemapKernel::Run
  this->template DoTest<true>();
}

}  // namespace dali::kernels::remap::test

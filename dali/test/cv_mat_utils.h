// Copyright (c) 2018-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_CV_MAT_UTILS_H_
#define DALI_TEST_CV_MAT_UTILS_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <utility>
#include <vector>
#include <tuple>
#include "dali/core/geom/box.h"
#include "dali/test/mat2tensor.h"

namespace dali {
namespace testing {

template <int _ndim, class T>
cv::Mat tensor_to_mat(const TensorView<StorageCPU, T, _ndim> &tensor,
                      bool has_channels = true, bool convert_to_BGR = true) {
  int ndim = tensor.dim();
  int nch = has_channels ? tensor.shape[ndim - 1] : 1;
  int spatial_ndim = ndim - has_channels;
  vector<int> sizes(spatial_ndim);
  for (int i = 0; i < spatial_ndim; i++)
    sizes[i] = tensor.shape[i];
  cv::Mat mat(sizes, CV_MAKETYPE(cv::DataDepth<std::remove_const_t<T>>::value, nch),
              const_cast<std::remove_const_t<T> *>(tensor.data));
  if (convert_to_BGR) {
    if (nch != 3)
      throw std::invalid_argument("Expected a 3-channel input");
    cv::Mat tmp;
    cv::cvtColor(mat, tmp, cv::COLOR_RGB2BGR);
    return tmp;
  }
  return mat;
}

template <int _ndim, class T>
void SaveImage(const char *name, const TensorView<StorageCPU, T, _ndim> &tensor) {
  auto mat = tensor_to_mat(tensor, tensor.dim() > 2 && tensor.shape[tensor.dim()-1] <= 3);
  cv::imwrite(name, mat);
}

/**
 * Creates cv::Mat based on provided arguments.
 * This mat is for roi-testing purposes only. Particularly, it doesn't care
 * about image type (i.e. number of channels), so don't try to imshow it.
 *
 * @param roi
 * @param base Base pointer of the rows x cols Mat; can't be const, due to cv::Mat API
 * @param rows height of the input image
 * @param cols width of the input image
 */
template <int nchannels, class T>
cv::Mat_<cv::Vec<T, nchannels>> copy_to_mat(Box<2, int> roi, T *base, int rows, int cols) {
  auto roi_w = roi.extent().x;
  auto roi_h = roi.extent().y;
  assert(roi.hi.x <= cols && roi.hi.y <= rows);  // Roi overflows the image
  cv::Mat mat(rows, cols, CV_MAKETYPE(cv::DataDepth<std::remove_const_t<T>>::value, nchannels),
              const_cast<T *>(base));
  cv::Rect rect(roi.lo.x, roi.lo.y, roi_w, roi_h);
  cv::Mat_<cv::Vec<T, nchannels>> out_copy;
  mat(rect).copyTo(out_copy);
  assert(out_copy.isContinuous());
  return out_copy;
}


/**
 * @brief Creates a copy of cv::Mat, accessible as TensorView. Main reason to use this function,
 *        is when you want to transfer cv::Mat to the GPU.
 *
 * Outputs a pair, which has ownership of the data underneath. The first element of the pair
 * shall be used to access the data. The second is just an ownership entity, should be kept
 * in scope as long as the first is used.
 *
 * In most of the cases you won't need to overload default template arguments.
 * Be mindful, that it was created for 1- or 3- channel images,
 * so the `ndims` values are restricted to those.
 *
 * @tparam DstAlloc Type of allocation. Default value guarantees access from both CPU and GPU
 * @tparam T Data type of the image.
 * @tparam ndims Number of channels in the image. Typically 3 or 1
 * @param mat
 * @return A tuple, which has ownership of the data underneath
 */
template<kernels::AllocType DstAlloc = kernels::AllocType::Unified,
        typename T = uint8_t, int ndims = 3>
// I <3 function declarations in C++
std::tuple<
        TensorView<kernels::AllocBackend<DstAlloc>, T, ndims>,
        kernels::memory::KernelUniquePtr<T>
          >
mat_to_tensor(const cv::Mat &mat) {
  static_assert(ndims == 1 || ndims == 3, "`ndims` is restricted to 1 or 3");
  auto tvcpu = kernels::view_as_tensor<T, ndims>(mat);
  auto mem = kernels::memory::alloc_unique<T>(DstAlloc, mat.cols * mat.rows * mat.channels());
  auto tvgpu = make_tensor_gpu<ndims>(mem.get(), {mat.rows, mat.cols, mat.channels()});
  kernels::copy(tvgpu, tvcpu);
  return std::forward_as_tuple(tvgpu, std::move(mem));
}

}  // namespace testing
}  // namespace dali

#endif  // DALI_TEST_CV_MAT_UTILS_H_

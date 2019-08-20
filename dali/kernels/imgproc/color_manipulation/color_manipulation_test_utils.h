// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_MANIPULATION_TEST_UTILS_H_
#define DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_MANIPULATION_TEST_UTILS_H_

#include <opencv2/opencv.hpp>

namespace dali {
namespace kernels {
namespace color_manipulation {
namespace test {


/**
 * Creates cv::Mat based on provided arguments.
 * This mat is for roi-testing purposes only. Particularly, it doesn't care
 * about image type (i.e. number of channels), so don't try to imshow it.
 *
 * @param ptr Can't be const, due to cv::Mat API
 * @param roi
 * @param rows height of the input image
 * @param cols width of the input image
 */
template <int nchannels, class T>
cv::Mat_<cv::Vec<T, nchannels>> to_mat(T *ptr, Box<2, int> roi, int rows, int cols) {
  auto roi_w = roi.extent().x;
  auto roi_h = roi.extent().y;
  assert(roi.hi.x < cols && roi.hi.y < rows);  // Roi overflows the image
  cv::Mat mat(rows, cols, CV_MAKETYPE(cv::DataDepth<std::remove_const_t<T>>::value, nchannels),
              const_cast<T *>(ptr));
  cv::Rect rect(roi.lo.x, roi.lo.y, roi_w, roi_h);
  cv::Mat_<cv::Vec<T, nchannels>> out_copy;
  mat(rect).copyTo(out_copy);
  assert(out_copy.isContinuous());
  return out_copy;
}


}  // namespace test
}  // namespace color_manipulation
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_IMGPROC_COLOR_MANIPULATION_COLOR_MANIPULATION_TEST_UTILS_H_

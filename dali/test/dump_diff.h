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

#ifndef DALI_TEST_DUMP_DIFF_H_
#define DALI_TEST_DUMP_DIFF_H_

#include <opencv2/imgcodecs.hpp>
#include <string>

namespace dali {
namespace testing {

inline void DumpDiff(const std::string &base_name,
                     const cv::Mat &actual,
                     const cv::Mat &reference,
                     bool abs_diff = false) {
  cv::imwrite(base_name+"_out.png", actual);
  cv::imwrite(base_name+"_ref.png", reference);
  cv::Mat diff;
  if (abs_diff) {
    cv::absdiff(actual, reference, diff);
  } else {
    cv::Mat tmpA, tmpR;
    actual.convertTo(tmpA, CV_32F);
    reference.convertTo(tmpR, CV_32F);
    tmpA -= tmpR;
    tmpA.convertTo(diff, CV_8U, 1.0, 128);
  }
  cv::imwrite(base_name+"_diff.png", diff);
}

}  // namespace testing
}  // namespace dali

#endif  // DALI_TEST_DUMP_DIFF_H_

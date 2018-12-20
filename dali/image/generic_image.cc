// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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


#include "dali/image/generic_image.h"
#include "dali/image/png.h"
#include "dali/util/ocv.h"

namespace dali {

GenericImage::GenericImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
        Image(encoded_buffer, length, image_type) {
}


std::pair<std::shared_ptr<uint8_t>, Image::ImageDims>
GenericImage::DecodeImpl(DALIImageType image_type, const uint8_t *encoded_buffer,
                         size_t length) const {
  // Decode image to tmp cv::Mat
  cv::Mat decoded_image = cv::imdecode(
          cv::Mat(1, length, CV_8UC1, (void *) (encoded_buffer)),          //NOLINT
          IsColor(image_type) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

  if (decoded_image.data == nullptr) {
     DALI_FAIL("Unsupported image type.");
  }
  // if different image type needed (e.g. RGB), permute from BGR
  if (IsColor(image_type) && image_type != DALI_BGR) {
    OpenCvColorConversion(DALI_BGR, decoded_image, image_type, decoded_image);
  }

  const int c = IsColor(image_type) ? 3 : 1;
  const int W = decoded_image.cols;
  const int H = decoded_image.rows;

  std::shared_ptr<uint8_t> decoded_img_ptr(
          decoded_image.ptr(),
          [decoded_image](decltype(decoded_image.ptr()) ptr) {
              // This is an empty lambda, which is a custom deleter for
              // std::shared_ptr.
              // While instantiating shared_ptr, also lambda is instantiated,
              // making a copy of cv::Mat. This way, reference counter of cv::Mat
              // is incremented. Therefore, for the duration of life cycle of
              // underlying memory in shared_ptr, cv::Mat won't free its memory.
              // It will be freed, when last shared_ptr is deleted.
          });

  return std::make_pair(decoded_img_ptr, std::make_tuple(H, W, c));
}


Image::ImageDims GenericImage::PeekDims(const uint8_t *encoded_buffer, size_t length) const {
  DALI_FAIL("Cannot peek dims for Generic image (of unknown format)");
}

}  // namespace dali

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

namespace dali {

GenericImage::GenericImage(const uint8_t *encoded_buffer, size_t length, DALIImageType image_type) :
        Image(encoded_buffer, length, image_type) {
}


std::pair<uint8_t *, Image::ImageDims>
GenericImage::DecodeImpl(DALIImageType image_type, const uint8_t *encoded_buffer, size_t length) {

  // Decode image to tmp cv::Mat
  decoded_image_ = cv::imdecode(
          cv::Mat(1, length, CV_8UC1, (void *) (encoded_buffer)),
          IsColor(image_type) ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

  // if RGB needed, permute from BGR
  if (image_type == DALI_RGB) {
    cv::cvtColor(decoded_image_, decoded_image_, cv::COLOR_BGR2RGB
    );
  }

  auto c = IsColor(image_type) ? 3 : 1;
  // Resize actual storage
  const int W = decoded_image_.cols;
  const int H = decoded_image_.rows;

  return std::make_pair(decoded_image_.ptr(), std::make_tuple(H, W, c));

}


Image::ImageDims GenericImage::PeekDims(const uint8_t *encoded_buffer, size_t length) {
  throw std::runtime_error("Cannot peek dims for Generic image (of unknown format)");
}

}  // namespace dali

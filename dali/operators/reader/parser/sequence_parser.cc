// Copyright (c) 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include <memory>
#include <utility>
#include <opencv2/imgcodecs.hpp>
#include "dali/operators/reader/parser/sequence_parser.h"
#include "dali/util/ocv.h"

namespace dali {

std::pair<cv::Mat, TensorShape<>>
DecodeOpenCV(DALIImageType image_type, const uint8_t *encoded_buffer, size_t length) {
  int flags = 0;
  if (image_type == DALI_ANY_DATA) {
    // Note: IMREAD_UNCHANGED always ignores orientation
    flags |= cv::IMREAD_UNCHANGED;
  } else if (image_type == DALI_GRAY) {
    flags |= cv::IMREAD_GRAYSCALE | cv::IMREAD_IGNORE_ORIENTATION;
  } else {
    flags |= cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION;
  }

  // Decode image to tmp cv::Mat
  cv::Mat decoded_image = cv::imdecode(
    cv::Mat(1, length, CV_8UC1, const_cast<unsigned char*>(encoded_buffer)), flags);

  int W = decoded_image.cols;
  int H = decoded_image.rows;
  int C = decoded_image.channels();
  DALI_ENFORCE(decoded_image.data != nullptr, "Unsupported image type.");
  TensorShape<> decoded_shape{H, W, C};

  if (image_type == DALI_ANY_DATA && C == 4) {
    // Special case for ANY_DATA and 4 channels -> Convert to RGBA
    // Note: ANY_DATA with 1 or 3 channels is already forced to DALI_GRAY and DALI_RGB respectively.
    cv::cvtColor(decoded_image, decoded_image, cv::COLOR_BGRA2RGBA);
  } else if (image_type == DALI_RGB || image_type == DALI_YCbCr) {
    // if different image type needed (e.g. RGB), permute from BGR
    OpenCvColorConversion(DALI_BGR, decoded_image, image_type, decoded_image);
  }
  return {decoded_image, decoded_shape};
}


void SequenceParser::Parse(const TensorSequence& data, SampleWorkspace* ws) {
  Index seq_length = data.tensors.size();
  auto file_name = data.tensors[0].GetSourceInfo();
  cv::Mat decoded_img;
  TensorShape<> decoded_shape;
  try {
    std::tie(decoded_img, decoded_shape) =
        DecodeOpenCV(image_type_, const_cast<unsigned char *>(data.tensors[0].data<uint8_t>()),
                     data.tensors[0].size());
  } catch (std::exception &e) {
    DALI_FAIL(e.what(), ". File: ", file_name);
  }
  TensorShape<> seq_shape{seq_length, decoded_shape[0], decoded_shape[1], decoded_shape[2]};
  const auto frame_size = volume(decoded_shape);
  auto& sequence = ws->Output<CPUBackend>(0);
  sequence.SetLayout("FHWC");
  sequence.Resize(seq_shape, DALI_UINT8);
  auto view_0 = sequence.SubspaceTensor(0);
  std::memcpy(view_0.raw_mutable_data(), decoded_img.data, frame_size * sizeof(uint8_t));

  // Decode and copy rest of the frames
  for (int64_t frame_idx = 1; frame_idx < seq_length; frame_idx++) {
    auto view_i = sequence.SubspaceTensor(frame_idx);
    auto file_name = data.tensors[frame_idx].GetSourceInfo();
    try {
      TensorShape<> decoded_shape_i;
      std::tie(decoded_img, decoded_shape_i) = DecodeOpenCV(
          image_type_, const_cast<unsigned char *>(data.tensors[frame_idx].data<uint8_t>()),
          data.tensors[frame_idx].size());
      DALI_ENFORCE(decoded_shape == decoded_shape_i,
                   make_string("Frames should have same dimensions: ", decoded_shape,
                               " != ", decoded_shape_i));
    } catch (std::exception &e) {
      DALI_FAIL(e.what(), ". File: ", file_name);
    }
    std::memcpy(view_i.raw_mutable_data(), decoded_img.data, frame_size * sizeof(uint8_t));
  }
}

}  // namespace dali

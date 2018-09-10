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

#include "tiff.h"

namespace dali {

namespace {

cv::Mat TiffDecode(const unsigned char *tiff, int size, DALIImageType image_type) {
    std::vector<char> img_buf(tiff, tiff + size);
    auto ret = cv::imdecode(img_buf, image_type == DALI_GRAY ? 0 : 1);
    return ret;
}

} // namespace

bool CheckIsTiff(const unsigned char *tiff) {
    assert(tiff);

    std::vector<int> header_intel = {77, 77, 0, 42};
    std::vector<int> header_motorola = {73, 73, 42, 0};

    auto check_header =
            [&](const std::vector<int> &header) -> bool {
                for (unsigned int i = 0; i < header.size(); i++) {
                    if (header[i] != tiff[i]) {
                        return false;
                    }
                }
                return true;
            };

    return check_header(header_intel) || check_header(header_motorola);
}


DALIError_t GetTiffImageDims(const unsigned char *tiff, int size, int *h, int *w) {
    auto mat = TiffDecode(tiff, size, DALI_RGB);
    if (mat.empty()) {
        return DALIError;
    }
    *w = mat.cols;
    *h = mat.rows;
    return DALISuccess;
}


DALIError_t DecodeTiffHost(const unsigned char *tiff, int size, DALIImageType image_type, Tensor<CPUBackend> *output) {
    int channels = (image_type == DALI_GRAY) ? 1 : 3;
    auto tiff_mat = TiffDecode(tiff, size, image_type);
    if (tiff_mat.empty()) {
        return DALIError;
    }
    auto width = tiff_mat.cols;
    auto height = tiff_mat.rows;

    // if RGB needed, permute from BGR
    if (image_type == DALI_RGB) {
        cv::cvtColor(tiff_mat, tiff_mat, cv::COLOR_BGR2RGB);
    }

    // resize the output tensor
    output->Resize({height, width, channels});
    // force allocation
    output->mutable_data<unsigned char>();

    std::memcpy(output->raw_mutable_data(), tiff_mat.ptr(),
                static_cast<size_t>(height) * width * channels);

    return DALISuccess;
}

}  // namespace dali

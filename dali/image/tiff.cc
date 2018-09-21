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

#include "dali/image/tiff.h"

namespace dali {

namespace {

constexpr std::array<int, 4> header_intel = {77, 77, 0, 42};
constexpr std::array<int, 4> header_motorola = {73, 73, 42, 0};
constexpr int COUNT_SIZE = 2;
constexpr int ENTRY_SIZE = 12;
constexpr int WIDTH_TAG = 256;
constexpr int HEIGHT_TAG = 257;
constexpr int TYPE_WORD = 3;
constexpr int TYPE_DWORD = 4;


cv::Mat DecodeTiff(const unsigned char *tiff, int size, DALIImageType image_type) {
    DALI_ENFORCE(tiff);
    std::vector<char> buf(tiff, tiff + size);
    return cv::imdecode(buf, image_type == DALI_GRAY ? 0 : 1);
}


bool check_header(const unsigned char *tiff, const std::array<int, 4> &header) {
    DALI_ENFORCE(tiff);
    for (unsigned int i = 0; i < header.size(); i++) {
        if (tiff[i] != header[i]) {
            return false;
        }
    }
    return true;
}


bool is_little_endian(const unsigned char *tiff) {
    DALI_ENFORCE(tiff);
    return check_header(tiff, header_intel);
}

}  // namespace

bool CheckIsTiff(const unsigned char *tiff) {
    DALI_ENFORCE(tiff);
    return check_header(tiff, header_intel) || check_header(tiff, header_motorola);
}


DALIError_t GetTiffImageDims(const unsigned char *tiff, int size, int *h, int *w) {
    DALI_ENFORCE(h && w && tiff);
    DALI_ENFORCE(CheckIsTiff(tiff));

    TiffBuffer buffer(std::string(reinterpret_cast<const char *>(tiff), static_cast<size_t>(size)),
                       is_little_endian(tiff));

    const auto ifd_offset = buffer.Read<uint32_t>(4);
    const auto entry_count = buffer.Read<uint16_t>(ifd_offset);
    bool width_read = false, height_read = false;

    for (int entry_idx = 0;
         entry_idx < entry_count && !(width_read && height_read);
         entry_idx++) {
        const auto entry_offset = ifd_offset + COUNT_SIZE + entry_idx * ENTRY_SIZE;
        const auto tag_id = buffer.Read<uint16_t>(entry_offset);
        if (tag_id == WIDTH_TAG || tag_id == HEIGHT_TAG) {
            const auto value_type = buffer.Read<uint16_t>(entry_offset + 2);
            const auto value_count = buffer.Read<uint32_t>(entry_offset + 4);
            DALI_ENFORCE(value_count == 1);

            int value;
            if (value_type == TYPE_WORD) {
                value = buffer.Read<uint16_t>(entry_offset + 8);
            } else if (value_type == TYPE_DWORD) {
                value = buffer.Read<uint32_t>(entry_offset + 8);
            } else {
                return DALIError;
            }

            if (tag_id == WIDTH_TAG) {
                *w = value;
                width_read = true;
            } else {
                *h = value;
                height_read = true;
            }
        }
    }
    if (!(width_read && height_read)) {
        return DALIError;
    }
    return DALISuccess;
}


DALIError_t DecodeTiffHost(const unsigned char *tiff, int size, DALIImageType image_type,
                           Tensor<CPUBackend> *output) {
    DALI_ENFORCE(tiff && output);
    DALI_ENFORCE(CheckIsTiff(tiff));
    auto tiff_mat = DecodeTiff(tiff, size, image_type);
    const auto height = tiff_mat.rows;
    const auto width = tiff_mat.cols;
    const auto channels = (image_type == DALI_GRAY) ? 1 : 3;

    // if RGB needed, permute from BGR
    if (image_type == DALI_RGB) {
        cv::cvtColor(tiff_mat, tiff_mat, cv::COLOR_BGR2RGB);
    }

    // resize the output tensor
    output->Resize({height, width, channels});
    // force allocation
    output->mutable_data<uint8>();

    std::memcpy(output->raw_mutable_data(), tiff_mat.ptr(),
                static_cast<size_t>(height) * width * channels);

    return DALISuccess;
}

}  // namespace dali

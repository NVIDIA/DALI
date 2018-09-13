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

#ifndef DALI_TIFF_H
#define DALI_TIFF_H

#include <opencv2/opencv.hpp>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {


class tiff_buffer {
public:

    tiff_buffer(std::string buffer, bool little_endian = false) :
            little_endian_(little_endian) {
        stream_ = std::istringstream(buffer);
        buffer_size_ = buffer.length();
    }


    template<typename ValueType>
    ValueType read(unsigned int offset = 0) {
        assert(stream_.good());
        assert(offset + sizeof(ValueType) < buffer_size_);
        static_assert(std::is_integral<ValueType>::value, "Only integral values supported");

        stream_.seekg(offset);
        ValueType ret;
        stream_.read(reinterpret_cast<char *>(&ret), sizeof(ret));
        if (little_endian_) {
            convert_le(&ret);
        }
        assert(stream_.good());
        return ret;
    }


private:

    template<typename T>
    void convert_le(T *value) {
        static_assert(std::is_integral<T>::value, "Converting floating point value unsupported");

        char *value_bytes = reinterpret_cast<char *>(value);
        std::vector<char> value_copy(value_bytes, value_bytes + sizeof(T));

        {
            int i = 0;
            for (auto it = value_copy.rbegin(); it != value_copy.rend(); ++it, ++i) {
                value_bytes[i] = *it;
            }
        }
    }


    std::istringstream stream_;
    size_t buffer_size_;
    bool little_endian_;
};

/**
 * @brief Returns 'true' if input compressed image is a tiff
 */
extern bool CheckIsTiff(const unsigned char *tiff);


/**
 * @brief Get dimensions of tiff encoded image
 */
extern DALIError_t GetTiffImageDims(const unsigned char *tiff, int size, int *h, int *w);


/**
 * @brief Decodes 'tiff' into the buffer pointed to by 'image'
 */
extern DALIError_t
DecodeTiffHost(const unsigned char *tiff, int size, DALIImageType image_type, Tensor<CPUBackend> *output);

} // namespace dali

#endif //DALI_TIFF_H

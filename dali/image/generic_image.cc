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

#include <cstdlib>

#include "dali/image/generic_image.h"
#include "dali/image/png.h"

namespace dali {

bool CheckIsGIF(const uint8_t *gif, int size) {
    DALI_ASSERT(gif);
    return (size >= 10 && gif[0] == 'G' && gif[1] == 'I' && gif[2] == 'F' && gif[3] == '8' &&
    (gif[4] == '7' || gif[4] == '9') && gif[5] == 'a');
}

// OpenCV doesn't handle gif images so we don't need it now
#if 0
DALIError_t GetGIFImageDims(const uint8 *gif, int size, int *h, int *w) {
    DALIError_t ret = DALIError;
    DALI_ASSERT(gif);

    if (size >= 10) {
        *w = (unsigned int)(gif[6] | gif[7] << 4) & 0xFFFF;
        *h = (unsigned int)(gif[8] | gif[9]<< 4) & 0xFFFF;
        ret = DALISuccess;
    }

    return ret;
}
#endif

bool CheckIsBMP(const uint8_t *bmp, int size) {
    return (size > 2 && bmp[0] == 'B' && bmp[1] == 'M');
}

DALIError_t GetBMPImageDims(const uint8 *bmp, int size, int *h, int *w) {
    DALIError_t ret = DALIError;
    DALI_ASSERT(bmp);

    unsigned head_size = bmp[14] | bmp[15] << 4 | bmp[16] << 8 | bmp[17] << 12;
    *h = 0;
    *w = 0;
    if (size >= 22 && head_size == 12) {
        *w = (unsigned int)(bmp[18] | bmp[19] << 4) && 0xFFFF;
        *h = (unsigned int)(bmp[20] | bmp[21] << 4) && 0xFFFF;
        ret = DALISuccess;
    } else if (size >= 26 && head_size >= 40) {
        *w = (unsigned int)(static_cast<int>(bmp[18] | bmp[19] << 4 |
             bmp[20] << 8 | bmp[21] << 12));
        *h = (unsigned int)abs(static_cast<int>(bmp[22] | bmp[23] << 4 |
             bmp[24] << 8 | bmp[25] << 12));
        ret = DALISuccess;
    }

    return ret;
}

DALIError_t GetImageDims(const uint8 *data, int size, int *h, int *w) {
    DALI_ASSERT(data);
    if (CheckIsPNG(data, size)) {
        return GetPNGImageDims(data, size, h, w);
    } else if (CheckIsGIF(data, size)) {
        // OpenCV doesn't handle gif images
    #if 0
        return GetGIFImageDims(data, size, h, w);
    #else
        return DALIError;
    #endif
    } else if (CheckIsBMP(data, size)) {
        return GetBMPImageDims(data, size, h, w);
    }
    // Not supported
    return DALIError;
}

}  // namespace dali

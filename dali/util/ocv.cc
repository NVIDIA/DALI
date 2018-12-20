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

#include <map>
#include "dali/util/ocv.h"
#include "dali/error_handling.h"

namespace dali {

int OCVInterpForDALIInterp(DALIInterpType type, int *ocv_type) {
  switch (type) {
  case DALI_INTERP_NN:
    *ocv_type =  cv::INTER_NEAREST;
    break;
  case DALI_INTERP_LINEAR:
    *ocv_type =  cv::INTER_LINEAR;
    break;
  case DALI_INTERP_CUBIC:
    *ocv_type =  cv::INTER_CUBIC;
    break;
  default:
    return DALIError;
  }
  return DALISuccess;
}

int GetOpenCvChannelType(size_t c) {
  return ( c == 3 ) ? CV_8UC3 : CV_8UC1;
}

cv::ColorConversionCodes GetOpenCvColorConversionCode(DALIImageType input_type, DALIImageType output_type) {
    using ColorConversionPair = std::pair<DALIImageType, DALIImageType>;
    using ColorConversionMap = std::map< ColorConversionPair, cv::ColorConversionCodes >;
    static const ColorConversionMap color_conversion_map = {
        { {DALI_RGB, DALI_BGR},   cv::COLOR_RGB2BGR },
        { {DALI_RGB, DALI_GRAY},  cv::COLOR_RGB2GRAY },
        { {DALI_RGB, DALI_YCbCr}, cv::COLOR_RGB2YCrCb }, // TODO(janton): Cr and Cb are interchanged
        
        { {DALI_BGR, DALI_RGB},   cv::COLOR_BGR2RGB },
        { {DALI_BGR, DALI_GRAY},  cv::COLOR_BGR2GRAY },
        { {DALI_BGR, DALI_YCbCr}, cv::COLOR_BGR2YCrCb }, // TODO(janton): Cr and Cb are interchanged

        { {DALI_GRAY, DALI_RGB},  cv::COLOR_GRAY2RGB },
        { {DALI_GRAY, DALI_BGR},  cv::COLOR_GRAY2BGR },
//        { {DALI_GRAY, DALI_YCbCr}, /* ? */ }, // TODO(janton): not supported?

        { {DALI_YCbCr, DALI_RGB}, cv::COLOR_YCrCb2RGB },
        { {DALI_YCbCr, DALI_BGR}, cv::COLOR_YCrCb2BGR },
//        { {DALI_YCbCr, DALI_GRAY}, /* ? */ }, // TODO(janton): not supported?
    };  // TODO(janton): add all cases

    const ColorConversionPair color_conversion_pair{ input_type, output_type };
    const auto it = color_conversion_map.find(color_conversion_pair);
    DALI_ENFORCE( it != color_conversion_map.end(), "Color conversion not supported (from " + std::to_string(input_type) + " to " + std::to_string(output_type) + ")" );
    return it->second;
}

}  // namespace dali

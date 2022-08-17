// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/imgcodec/image_decoder.h"

namespace dali {
namespace imgcodec {

bool ImageDecoder::CanDecode(ImageSource *in, DecodeParams opts, const ROI &roi = {}) {
  return true;
}

std::vector<bool> ImageDecoder::CanDecode(cspan<ImageSource *> in,
                                          DecodeParams opts,
                                          cspan<ROI> rois = {}) {
    return std::vector<bool>(in.size(), true);
}

bool SetParam(const char *key, const any &value) override {

}

any ImageDecoder::GetParam(const char *key) const {
  auto it = params_.find(key);
  return it != params_.end() ? it->second : {};
}

}  // namespace dali
}  // namespace imgcodec

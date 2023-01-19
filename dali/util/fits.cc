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

#include "dali/util/fits.h"
#include <string>
#include <vector>
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"

#define MAX_AXIS 999

namespace dali {
namespace fits {

void ParseHeader(HeaderData &parsed_header, fitsfile *src) {
  int hdu_type, status, type, n_dims;
  long dims[MAX_AXIS];
  fits_get_hdu_type(src, &hdu_type, &status);

  if (hdu_type == IMAGE_HDU) {
    fits_get_img_param(src, MAX_DIMS, &type, &n_dims, dims, &status);
  }


  parsed_header.hdu_type = hdu_type;
}

DALIDataType HeaderData::type() const {
  return type_info ? type_info->id() : DALI_NO_TYPE;
}

size_t HeaderData::size() const {
  return volume(shape);
}

size_t HeaderData::nbytes() const {
  return type_info ? type_info->size() * size() : 0_uz;
}

}  // namespace fits
}  // namespace dali

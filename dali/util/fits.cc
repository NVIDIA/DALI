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
#include <fitsio.h>
#include <string>
#include <vector>
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"

namespace dali {
namespace fits {

void HandleFitsError(int status) {
  std::string status_str;
  status_str.reserve(FLEN_STATUS);

  if (status) {
    fits_get_errstatus(status, &status_str[0]); /* get the error description */
    DALI_FAIL(status_str);
  }
}

const TypeInfo &TypeFromFitsImageType(int imgtype) {
  if (imgtype == BYTE_IMG)
    return TypeTable::GetTypeInfo<uint8_t>();
  if (imgtype == SHORT_IMG)
    return TypeTable::GetTypeInfo<int16_t>();
  if (imgtype == LONG_IMG)
    return TypeTable::GetTypeInfo<int32_t>();
  if (imgtype == LONGLONG_IMG)
    return TypeTable::GetTypeInfo<int64_t>();
  if (imgtype == ULONG_IMG)
    return TypeTable::GetTypeInfo<uint64_t>();
  if (imgtype == ULONGLONG_IMG)
    return TypeTable::GetTypeInfo<uint64_t>();
  if (imgtype == FLOAT_IMG)
    return TypeTable::GetTypeInfo<float>();
  if (imgtype == DOUBLE_IMG)
    return TypeTable::GetTypeInfo<double>();
  DALI_FAIL("Unknown FITS image type type string");
}

void ParseHeader(HeaderData &parsed_header, fitsfile *src) {
  int hdu_type, img_type, n_dims, status = 0;

  fits_get_hdu_type(src, &hdu_type, &status);
  bool is_image = (hdu_type == IMAGE_HDU);
  DALI_ENFORCE(is_image, "Only IMAGE_HDUs are supported!");

  fits_get_img_type(src, &img_type, &status);        /* get BITPIX value */
  fits_get_img_dim(src, &n_dims, &status);           /* get NAXIS value */
  std::vector<long> dims(n_dims, 0);                 /* create vector for storing img dims*/
  fits_get_img_size(src, n_dims, &dims[0], &status); /* get NAXISn values */

  parsed_header.hdu_type = hdu_type;
  parsed_header.type_info = &TypeFromFitsImageType(img_type);
  parsed_header.compressed = (fits_is_compressed_image(src, &status) == 1);

  for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
    parsed_header.shape.shape.push_back(static_cast<int64_t>(*it));
  }

  if (status)
    HandleFitsError(status);
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

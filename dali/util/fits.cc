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


std::string GetFitsErrorMessage(int status) {
  std::string status_str;
  status_str.resize(FLEN_STATUS);

  fits_get_errstatus(status, &status_str[0]); /* get the error description */

  return status_str;
}

void HandleFitsError(int status) {
  if (status) {
    DALI_FAIL(GetFitsErrorMessage(status));
  }
}

int ImgTypeToDatatypeCode(int img_type) {
  switch (img_type) {
    case SBYTE_IMG:
      return TSBYTE;
    case BYTE_IMG:
      return TBYTE;
    case SHORT_IMG:
      return TSHORT;
    case USHORT_IMG:
      return TUSHORT;
    case LONG_IMG:
      return TINT;
    case ULONG_IMG:
      return TUINT;
    case LONGLONG_IMG:
      return TLONGLONG;
    case ULONGLONG_IMG:
      return TULONGLONG;
    case FLOAT_IMG:
      return TFLOAT;
    case DOUBLE_IMG:
      return TDOUBLE;
    default:
      DALI_FAIL("Unknown BITPIX value! Refer to the CFITSIO documentation.");
  }
}

const TypeInfo &TypeFromFitsDatatypeCode(int datatype) {
  switch (datatype) {
    case TSBYTE:
      return TypeTable::GetTypeInfo<int8_t>();
    case TBYTE:
      return TypeTable::GetTypeInfo<uint8_t>();
    case TSHORT:
      return TypeTable::GetTypeInfo<int16_t>();
    case TUSHORT:
      return TypeTable::GetTypeInfo<uint16_t>();
    case TINT:
      return TypeTable::GetTypeInfo<int32_t>();
    case TUINT:
      return TypeTable::GetTypeInfo<uint32_t>();
    case TLONGLONG:
      return TypeTable::GetTypeInfo<int64_t>();
    case TULONGLONG:
      return TypeTable::GetTypeInfo<uint64_t>();
    case TFLOAT:
      return TypeTable::GetTypeInfo<float>();
    case TDOUBLE:
      return TypeTable::GetTypeInfo<double>();
    default:
      DALI_FAIL("Unknown datatype code value! Refer to the CFITSIO documentation.");
  }
}

void ParseHeader(HeaderData &parsed_header, fitsfile *src) {
  int32_t hdu_type, img_type, n_dims, status = 0;

  fits_get_hdu_type(src, &hdu_type, &status);
  bool is_image = (hdu_type == IMAGE_HDU);
  DALI_ENFORCE(is_image, "Only IMAGE_HDUs are supported!");

  fits_get_img_equivtype(src, &img_type, &status); /* get IMG_TYPE code value */
  fits_get_img_dim(src, &n_dims, &status);         /* get NAXIS value */

  DALI_ENFORCE(n_dims > 0, "NAXIS (image dimensions) value for each HDU has to be greater than 0!");
  std::vector<int64_t> dims(n_dims, 0); /* create vector for storing img dims*/

  fits_get_img_size(src, n_dims, &dims[0], &status); /* get NAXISn values */

  HandleFitsError(status);

  parsed_header.hdu_type = hdu_type;
  parsed_header.datatype_code = ImgTypeToDatatypeCode(img_type);
  parsed_header.type_info = &TypeFromFitsDatatypeCode(parsed_header.datatype_code);
  parsed_header.compressed = (fits_is_compressed_image(src, &status) == 1);

  for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
    parsed_header.shape.shape.push_back(static_cast<int64_t>(*it));
  }
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

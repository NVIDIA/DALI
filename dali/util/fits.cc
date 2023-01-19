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

#define MAX_DIMS 999

namespace dali {
namespace fits {

const TypeInfo &TypeFromFitsImageType(int imgtype) {
  if (imgtype == BYTE_IMG)
    return TypeTable::GetTypeInfo<uint8_t>();
  if (imgtype == SHORT_IMG)
    return TypeTable::GetTypeInfo<uint16_t>();
  if (imgtype == LONG_IMG)
    return TypeTable::GetTypeInfo<uint32_t>();
  if (imgtype == LONGLONG_IMG)
    return TypeTable::GetTypeInfo<uint64_t>();
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
  int hdu_type, img_type, n_dims, status = 0, i = 0;
  long dims[MAX_DIMS] = {0};

  fits_get_hdu_type(src, &hdu_type, &status);
  bool is_image = (hdu_type == IMAGE_HDU);
  DALI_ENFORCE(is_image, "Only IMAGE_HDUs are supported!");
  parsed_header.hdu_type = hdu_type;

  fits_get_img_param(src, MAX_DIMS, &img_type, &n_dims, dims, &status);
  parsed_header.type_info = &TypeFromFitsImageType(img_type);
  parsed_header.compressed = (fits_is_compressed_image(src, &status) == 1);

  while (i < MAX_DIMS && dims[i] > 0) {
    parsed_header.shape.shape.push_back(static_cast<int64_t>(dims[i]));
    i++;
  }

  if (status)
    fits_report_error(stderr, status);
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

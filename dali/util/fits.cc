// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fitsio.h>

#include <functional>
#include <string>
#include <vector>

#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"
#include "dali/util/fits.h"

namespace dali {
namespace fits {

namespace {

std::string GetFitsErrorMessage(int status) {
  std::string status_str;
  status_str.resize(FLEN_STATUS);

  fits_get_errstatus(status, &status_str[0]); /* get the error description */

  return status_str;
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

const TypeInfo& TypeFromFitsDatatypeCode(int datatype) {
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

std::vector<int64_t> GetTileSizes(fitsfile* fptr, int32_t n_dims) {
  std::vector<int64_t> tileSizes(n_dims, 1);
  int32_t status = 0;

  for (int32_t i = 0; i < n_dims; i++) {
    std::string keyword = make_string("ZTILE", i + 1);
    FITS_CALL(fits_read_key(fptr, TLONG, keyword.c_str(), &tileSizes[i], NULL, &status));
    DALI_ENFORCE(tileSizes[i] > 0,
                 make_string("All ZTILE{i} values must be greater than 0! Actual: ", tileSizes[i],
                             " at index i=", i));
  }

  return tileSizes;
}

template <unsigned level>
inline void ExtractData(fitsfile* fptr, std::vector<std::vector<uint8_t>>& raw_data,
                        std::vector<int64_t>& tile_offset, std::vector<int64_t>& tile_size,
                        int64_t* ftile, int64_t* ltile, int64_t* tilesize, int64_t* thistilesize,
                        int64_t* rowdim, int64_t* naxis, int64_t* offset, int* size,
                        int64_t* sum_nelemll, int* status, unsigned max_level = level) {
  for (int i = ftile[level]; i <= ltile[level]; ++i) {
    // first and last image pixels along each dimension of the compression tile
    int64_t tfpixel = (i - 1) * tilesize[level] + 1;
    int64_t tlpixel = minvalue(tfpixel + tilesize[level] - 1, naxis[level]);
    if (level == max_level) {
      thistilesize[level] = tlpixel - tfpixel + 1;
      offset[level] = (i - 1) * rowdim[level];
    } else {
      thistilesize[level] = thistilesize[level + 1] * (tlpixel - tfpixel + 1);
      offset[level] = (i - 1) * rowdim[level] + offset[level + 1];
    }
    ExtractData<level - 1>(fptr, raw_data, tile_offset, tile_size, ftile, ltile, tilesize,
                           thistilesize, rowdim, naxis, offset, size, sum_nelemll, status,
                           max_level);
  }
}

template <>
inline void ExtractData<0>(fitsfile* fptr, std::vector<std::vector<uint8_t>>& raw_data,
                           std::vector<int64_t>& tile_offset, std::vector<int64_t>& tile_size,
                           int64_t* ftile, int64_t* ltile, int64_t* tilesize, int64_t* thistilesize,
                           int64_t* rowdim, int64_t* naxis, int64_t* offset, int* size,
                           int64_t* sum_nelemll, int* status, unsigned max_level) {
  for (int i = ftile[0]; i <= ltile[0]; ++i) {
    int64_t tfpixel = (i - 1) * tilesize[0] + 1;
    int64_t tlpixel = minvalue(tfpixel + tilesize[0] - 1, naxis[0]);
    thistilesize[0] = thistilesize[1] * (tlpixel - tfpixel + 1);
    int64_t irow = i + offset[1];

    LONGLONG nelemll = 0, noffset = 0;
    ffgdesll(fptr, (fptr->Fptr)->cn_compressed, irow, &nelemll, &noffset, status);

    tile_offset[*size] = *sum_nelemll;
    tile_size[*size] = static_cast<int64_t>(thistilesize[0]);

    unsigned char charnull = 0;
    raw_data[*size].resize(nelemll / sizeof(unsigned char));
    FITS_CALL(fits_read_col(fptr, TBYTE, (fptr->Fptr)->cn_compressed, irow, 1,
                            static_cast<int64_t>(nelemll),
                            &charnull, raw_data[*size].data(), nullptr, status));

    ++(*size);
    *sum_nelemll += nelemll;
  }
}

}  // namespace

void ParseHeader(HeaderData& parsed_header, fitsfile* src) {
  int32_t hdu_type, img_type, n_dims, status = 0;

  FITS_CALL(fits_get_hdu_type(src, &hdu_type, &status));
  bool is_image = (hdu_type == IMAGE_HDU);
  DALI_ENFORCE(is_image, "Only IMAGE_HDUs are supported!");

  FITS_CALL(fits_get_img_equivtype(src, &img_type, &status)); /* get IMG_TYPE code value */
  FITS_CALL(fits_get_img_dim(src, &n_dims, &status));         /* get NAXIS value */

  DALI_ENFORCE(n_dims > 0, "NAXIS (image dimensions) value for each HDU has to be greater than 0!");
  std::vector<int64_t> dims(n_dims, 0); /* create vector for storing img dims*/

  FITS_CALL(fits_get_img_size(src, n_dims, &dims[0], &status)); /* get NAXISn values */

  parsed_header.hdu_type = hdu_type;
  parsed_header.datatype_code = ImgTypeToDatatypeCode(img_type);
  parsed_header.type_info = &TypeFromFitsDatatypeCode(parsed_header.datatype_code);
  {
    FitsLock lock;
    parsed_header.compressed = (fits_is_compressed_image(src, &status) == 1);
  }

  if (parsed_header.compressed) {
    FITS_CALL(fits_get_num_rows(src, &parsed_header.rows, &status)); /*get NROW value */
    parsed_header.bscale = (src->Fptr)->cn_bscale;
    parsed_header.bzero = (src->Fptr)->cn_bzero;
    parsed_header.bytepix = (src->Fptr)->rice_bytepix;
    parsed_header.zbitpix = (src->Fptr)->zbitpix;
    parsed_header.blocksize = (src->Fptr)->rice_blocksize;
    parsed_header.maxtilelen = (src->Fptr)->maxtilelen;
    parsed_header.tile_sizes = GetTileSizes(src, n_dims);
    parsed_header.tiles = std::accumulate(parsed_header.tile_sizes.begin(),
                                          parsed_header.tile_sizes.end(), 1, std::multiplies<>());
  }

  for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
    parsed_header.shape.shape.push_back(static_cast<int64_t>(*it));
  }
}

int ExtractUndecodedData(fitsfile* fptr, std::vector<uint8_t>& data,
                         std::vector<int64_t>& tile_offset, std::vector<int64_t>& tile_size,
                         int64_t rows, int* status) {
  std::vector<std::vector<uint8_t>> raw_data;
  raw_data.resize(rows);
  tile_offset.resize(rows + 1);
  tile_size.resize(rows);

  // number of first and last pixel in each dimension
  LONGLONG fpixel[MAX_COMPRESS_DIM], lpixel[MAX_COMPRESS_DIM];
  for (int i = 0; i < (fptr->Fptr)->zndim; ++i) {
    fpixel[i] = 1;
    lpixel[i] = (fptr->Fptr)->znaxis[i];
  }

  // length of each axis
  int64_t naxis[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  // number of tiles covering given axis
  int64_t tiledim[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  // size of compression tiles
  int64_t tilesize[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  // tile containing the first pixel we want to read in each dimension
  int64_t ftile[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  // tile containing the last pixel we want to read in each dimension
  int64_t ltile[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  // total tiles in each dimension
  int64_t rowdim[MAX_COMPRESS_DIM] = {1, 1, 1, 1, 1, 1};
  int64_t offset[MAX_COMPRESS_DIM], thistilesize[MAX_COMPRESS_DIM];
  int64_t mfpixel[MAX_COMPRESS_DIM], mlpixel[MAX_COMPRESS_DIM];
  int64_t ntemp, sum_nelemll;
  int ndim, size;

  ndim = (fptr->Fptr)->zndim;
  ntemp = 1;
  for (int i = 0; i < ndim; ++i) {
    if (fpixel[i] <= lpixel[i]) {
      mfpixel[i] = static_cast<int64_t>(fpixel[i]);
      mlpixel[i] = static_cast<int64_t>(lpixel[i]);
    } else {
      mfpixel[i] = static_cast<int64_t>(lpixel[i]);
      mlpixel[i] = static_cast<int64_t>(fpixel[i]);
    }

    naxis[i] = (fptr->Fptr)->znaxis[i];
    tilesize[i] = (fptr->Fptr)->tilesize[i];
    tiledim[i] = (naxis[i] - 1) / tilesize[i] + 1;
    ftile[i] = (mfpixel[i] - 1) / tilesize[i] + 1;
    ltile[i] = minvalue((mlpixel[i] - 1) / tilesize[i] + 1, tiledim[i]);
    rowdim[i] = ntemp;
    ntemp *= tiledim[i];
  }

  size = 0;
  sum_nelemll = 0;
  ExtractData<5>(fptr, raw_data, tile_offset, tile_size, ftile, ltile, tilesize, thistilesize,
                 rowdim, naxis, offset, &size, &sum_nelemll, status);

  tile_offset[size] = sum_nelemll;

  data.clear();
  for (const auto& tile : raw_data) {
    data.insert(data.end(), tile.begin(), tile.end());
  }

  return (*status);
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

void HandleFitsError(int status) {
  if (status) {
    DALI_FAIL(GetFitsErrorMessage(status));
  }
}

FitsLock::FitsLock() : lock_(mutex(), std::defer_lock) {
  if (!fits_is_reentrant()) {
    DALI_WARN_ONCE("Loaded instance of CFITSIO library does not support multithreading. "
                  "Please recompile CFITSIO in reentrant mode (--enable-reentrant) "
                  "or use CFITSIO delivered in DALI_deps. Using non-reentrant version "
                  "of CFITSIO may degrade the performance.");
    lock_.lock();
  }
}

std::mutex& fits::FitsLock::mutex()  {
  static std::mutex mutex = {};
  return mutex;
}

}  // namespace fits
}  // namespace dali

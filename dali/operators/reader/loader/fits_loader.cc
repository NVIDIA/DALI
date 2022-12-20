// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dirent.h>
#include <errno.h>
#include <cstdlib>
#include <memory>

#include <fitsio.h>
#include "dali/core/common.h"
#include "dali/operators/reader/loader/fits_loader.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/util/file.h"

namespace dali {

void FitsLoader::ReadSample(FitsFileWrapper& target) {
  auto filename = files_[current_index_++];
  fitsfile* infptr;
  int status = 0, hdupos;
  int hdutype, bitpix, bytepix, naxis = 0, nkeys, datatype = 0, anynul;
  long first, totpix = 0, npix;
  long naxes[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};


  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(filename);
  meta.SetSkipSample(false);

  auto path = filesystem::join_path(file_root_, filename);
  std::string processed_uri;

  if (uri.find("file://") == 0) {
    processed_uri = path.substr(std::string("file://").size());
  } else {
    processed_uri = path;
  }

  fits_open_file(&infptr, processed_uri, READONLY, &status);
  fits_get_hdu_type(infptr, &hdutype, &status);

  if (status != 0) {
    fits_report_error(stderr, status);
  }

  if (hdutype == IMAGE_HDU) {
    /* get image dimensions and total number of pixels in image */
    for (int ii = 0; ii < 9; ii++)
      naxes[ii] = 1;

    fits_get_img_param(infptr, 9, &bitpix, &naxis, naxes, &status);

    totpix = naxes[0] * naxes[1] * naxes[2] * naxes[3] * naxes[4] * naxes[5] * naxes[6] * naxes[7] *
             naxes[8];
  }

  if (hdutype != IMAGE_HDU || naxis == 0 || totpix == 0) {
    DALI_FAIL("Not an image!" + ". File: " + filename);
  }


  switch (bitpix) {
    case BYTE_IMG:
      datatype = TBYTE;
      break;
    case SHORT_IMG:
      datatype = TSHORT;
      break;
    case LONG_IMG:
      datatype = TINT;
      break;
    case FLOAT_IMG:
      datatype = TFLOAT;
      break;
    case DOUBLE_IMG:
      datatype = TDOUBLE;
      break;
  }

  bytepix = abs(bitpix) / 8;
  // should do sth like  that before
  // target.data.Resize(header.shape, header.type());
  fits_read_img(infptr, datatype, first, totpix, &nulval, target.data.raw_mutable_data(), &anynul,
                &status);

  // close the file handle
  infptr->Close();

  // set metadata
  target.data.SetMeta(meta);

  // set file path
  target.filename = std::move(path);
}

}  // namespace dali

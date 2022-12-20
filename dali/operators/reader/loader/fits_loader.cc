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

  npix = totpix;
  iteration = 0;

  // here data copying happens
  // we want to copy data to tensor instead of another fits file so yeah;

  /* try to allocate memory for the entire image */
  /* use double type to force memory alignment */
  array = (double*)calloc(npix, bytepix);

  /* if allocation failed, divide size by 2 and try again */
  while (!array && iteration < 10) {
    iteration++;
    npix = npix / 2;
    array = (double*)calloc(npix, bytepix);
  }

  if (!array) {
    printf("Memory allocation error\n");
    return (0);
  }

  /* turn off any scaling so that we copy the raw pixel values */
  fits_set_bscale(infptr, bscale, bzero, &status);
  fits_set_bscale(outfptr, bscale, bzero, &status);

  first = 1;
  while (totpix > 0 && !status) {
    /* read all or part of image then write it back to the output file */
    fits_read_img(infptr, datatype, first, npix, &nulval, array, &anynul, &status);

    fits_write_img(outfptr, datatype, first, npix, array, &status);
    totpix = totpix - npix;
    first = first + npix;
  }
  free(array);


  // hey ho


  // read the header
  // get it to work!
  fits::HeaderData header;
  try {
    fits::ParseHeader(header, current_file.get());
  } catch (const std::runtime_error& e) {
    DALI_FAIL(e.what() + ". File: " + filename);
  }


  Index nbytes = header.nbytes();


  if (copy_read_data_) {
    if (target.data.shares_data()) {
      target.data.Reset();
    }
    target.data.Resize(header.shape, header.type());
    // copy the image
    Index ret = current_file->Read(static_cast<uint8_t*>(target.data.raw_mutable_data()), nbytes);
    DALI_ENFORCE(ret == nbytes, make_string("Failed to read file: ", filename));
  } else {
    auto p = current_file->Get(nbytes);
    DALI_ENFORCE(p != nullptr, make_string("Failed to read file: ", filename));
    // Wrap the raw data in the Tensor object.
    target.data.ShareData(p, nbytes, false, {nbytes}, header.type(), CPU_ONLY_DEVICE_ID);
    target.data.Resize(header.shape, header.type());
  }

  // close the file handle
  current_file->Close();

  // set metadata
  target.data.SetMeta(meta);

  // set file path
  target.filename = std::move(path);

  // set meta
  target.fortran_order = header.fortran_order;
}

}  // namespace dali

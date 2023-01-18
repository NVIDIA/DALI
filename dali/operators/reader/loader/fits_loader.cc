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
#include "dali/pipeline/data/types.h"
#include "dali/util/file.h"

#define max_number_of_axes 999

namespace dali {

void FitsLoader::ReadSample(FitsFileWrapper& target) {
  auto filename = files_[current_index_++];
  fitsfile* infptr;
  int status = 0, hdupos;
  int hdutype, bitpix, bytepix, naxis = 0, nkeys, datatype = 0, ;
  double nulval = 0.0;
  // FIXME ? first changed to 1 instead of 0 because pixel indexing in fits file starts at 1 instead of 0
  long first = 1, totpix = 0, anynul, npix;
  // FIXME ? naxes converted to arrays of size max_number_of_axes instead of 9
  long* naxes = NULL;
  naxes = new long[max_number_of_axes];
  TensorShape<> shape;

  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(filename);
  meta.SetSkipSample(false);

  auto path = filesystem::join_path(file_root_, filename);
  std::string processed_uri;

  // remove the "file://" preffix in the path
  if (path.find("file://") == 0) {
    processed_uri = path.substr(std::string("file://").size());
  } else {
    processed_uri = path;
  }

  fits_open_file(&infptr, processed_uri, READONLY, &status);
  if (status != 0) {
    fits_report_error(stderr, status);
  }

  // TODO add header extensions
  fits_get_hdu_type(infptr, &hdutype, &status);
  if (status != 0) {
    fits_report_error(stderr, status);
  }

  // read the header (check if type is image)
  // get image dimensions and total number of pixels in image
  if (hdutype == IMAGE_HDU) {
    // get image dimensions
    fits_get_img_param(infptr, max_number_of_axes, &bitpix, &naxis, naxes, &status);

    // get total number of pixels in image
    // check if all naxes are > 0
    totpix = 1;
    for (int i = 0; i < naxis; i++) {
      // FIXME ? if the dimension is <= 0, do we consider the file to be incorrect?
      if (naxes[i] <= 0) {
        DALI_FAIL("Invalid image dimension " + std::to_string(naxes[i]));
      }
      totpix *= naxes[i];
    }

    // fill the tensor with the shape of the img
    for (int i = 0; i < naxis; i++) {
      shape.shape.push_back(naxes[i]);
    }
  }

  if (hdutype != IMAGE_HDU || naxis == 0 || totpix == 0) {
    DALI_FAIL("Not an image! File: " + filename);
  }

  // function from utils
  datatype = RecognizeTypeFromCfitsCode(bitpix);

  // function from utils
  TypeInfo typeInfo = TypeFromCfitsCode(datatype);

  bytepix = abs(bitpix) / 8;
  // should do sth like that before  // TODO what does it mean ?
  target.data.Resize(shape, typeInfo->id);
  fits_read_img(infptr, datatype, first, totpix, &nulval, target.data.raw_mutable_data(), &anynul,
                &status);

  // close the file handle
  infptr->Close();

  // set metadata
  target.data.SetMeta(meta);

  // set file path
  target.filename = std::move(path);

  delete[] naxes;
}

}  // namespace dali

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
#include "dali/util/fits.h"

#define max_number_of_axes 999

namespace dali {

void FitsLoader::ReadSample(FitsFileWrapper& target) {
  auto filename = files_[current_index_];
  fitsfile* current_file;
  int status = 0, hdunum;
  Index pastIndex = current_index_;

  // hanlde moving through HDUs, assuming that primary hdu is for meta data
  auto path = filesystem::join_path(file_root_, filename);
  fits_open_file(&current_file, path.c_str(), READONLY, &status);
  fits_get_num_hdus(current_file, &hdunum, &status);
  fits_movabs_hdu(current_file, hdu_index_, NULL, &status);

  if (hdu_index_ < hdunum) {
    hdu_index_++;
  } else {
    hdu_index_ = 2;    // skiping primary hdu
    current_index_++;  // moving to next file
  }

  // handle wrap-around, TODO: overwrite move to next shard
  MoveToNextShard(current_index_);
  if (pastIndex != current_index_)
    hdu_index_ = 2;

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(filename);
  meta.SetSkipSample(false);


  if (status != 0) {
    fits_report_error(stderr, status);
  }

  // read the header
  fits::HeaderData header;
  try {
    fits::ParseHeader(header, current_file);
  } catch (const std::runtime_error& e) {
    DALI_FAIL(e.what() + ". File: " + filename);
  }

  // copy the image
  int anynul = 0, nulval = 0;
  Index nbytes = header.nbytes();

  target.data.Resize(header.shape, header.type());
  fits_read_img(current_file, TBYTE, 1, nbytes, &nulval,
                static_cast<uint8_t*>(target.data.raw_mutable_data()), &anynul, &status);

  // close the file handle
  fits_close_file(current_file, &status);

  // set metadata
  target.data.SetMeta(meta);

  // set file path
  target.filename = std::move(path);
}

}  // namespace dali

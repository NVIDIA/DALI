// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <fitsio.h>
#include <cstdlib>
#include <memory>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/fits_loader.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/pipeline/data/types.h"
#include "dali/util/file.h"
#include "dali/util/fits.h"

namespace dali {

void FitsLoader::ReadSample(FitsFileWrapper& target) {
  auto filename = files_[current_index_++];
  int status = 0, num_hdus = 0;

  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  // meta.SetSourceInfo(filename); // it adds ./before a filename for some reason
  meta.SetSkipSample(false);

  // set file path
  target.filename = filesystem::join_path(file_root_, filename);
  auto current_file = fits::FitsHandle::OpenFile(target.filename.c_str(), READONLY);
  fits::FITS_CALL(fits_get_num_hdus(current_file, &num_hdus, &status));

  // resize ouput vector according to the number of HDUs
  target.data.resize(hdu_indices_.size());

  for (size_t output_idx = 0; output_idx < hdu_indices_.size(); output_idx++) {
    // move to appropiate hdu
    fits::FITS_CALL(fits_movabs_hdu(current_file, hdu_indices_[output_idx], NULL, &status));

    // read the header
    fits::HeaderData header;
    try {
      fits::ParseHeader(header, current_file);
    } catch (const std::runtime_error& e) {
      DALI_FAIL(e.what() + ". File: " + filename);
    }

    int anynul = 0, nulval = 0;
    Index nelem = header.size();

    // reset, resize specific output in target
    if (target.data[output_idx].shares_data()) {
      target.data[output_idx].Reset();
    }
    target.data[output_idx].Resize(header.shape, header.type());

    // copy the image
    fits::FITS_CALL(fits_read_img(current_file, header.datatype_code, 1, nelem, &nulval,
                                  static_cast<uint8_t*>(target.data[output_idx].raw_mutable_data()),
                                  &anynul, &status));

    // set metadata
    target.data[output_idx].SetMeta(meta);
  }
}

}  // namespace dali

// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/reader/loader/fits_loader_gpu.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/pipeline/data/types.h"
#include "dali/util/file.h"
#include "dali/util/fits.h"

namespace dali {

void FitsLoaderGPU::ReadDataFromHDU(const fits::FitsHandle& current_file,
                                    const fits::HeaderData& header, FitsFileWrapperGPU& target,
                                    size_t output_idx) {
  int status = 0, anynul = 0, nulval = 0;
  Index nelem = header.size();

  if (header.compressed) {
    vector<uint8_t> raw_data;
    dali::TensorShape<-1> shape;

    fits::ExtractUndecodedData(current_file, raw_data, target.tile_offset[output_idx],
                               target.tile_size[output_idx], header.rows, &status);

    shape.shape.push_back(raw_data.size());
    target.data[output_idx].Resize(shape, DALI_UINT8);
    memcpy(static_cast<uint8_t*>(target.data[output_idx].raw_mutable_data()), raw_data.data(),
           raw_data.size());
  } else {
    target.data[output_idx].Resize(header.shape, header.type());

    // copy the image to host memory
    FITS_CALL(fits_read_img(current_file, header.datatype_code, 1, nelem, &nulval,
                                  static_cast<uint8_t*>(target.data[output_idx].raw_mutable_data()),
                                  &anynul, &status));
  }
}

void FitsLoaderGPU::ResizeTarget(FitsFileWrapperGPU& target, size_t new_size) {
  target.data.resize(new_size);
  target.header.resize(new_size);
  target.tile_offset.resize(new_size);
  target.tile_size.resize(new_size);
}

}  // namespace dali

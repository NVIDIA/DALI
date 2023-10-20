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

#ifndef DALI_OPERATORS_READER_LOADER_FITS_LOADER_GPU_H_
#define DALI_OPERATORS_READER_LOADER_FITS_LOADER_GPU_H_

#include <dirent.h>
#include <errno.h>
#include <fitsio.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/fits_loader.h"
#include "dali/pipeline/data/types.h"
#include "dali/util/file.h"
#include "dali/util/fits.h"

namespace dali {

struct FitsFileWrapperGPU {
  std::vector<fits::HeaderData> header;
  std::vector<Tensor<CPUBackend>> data;
  std::vector<std::vector<int64_t>> tile_offset, tile_size;
  std::string filename;
};

class FitsLoaderGPU : public FitsLoader<GPUBackend, FitsFileWrapperGPU> {
 public:
  explicit FitsLoaderGPU(const OpSpec& spec, bool shuffle_after_epoch = false)
      : FitsLoader<GPUBackend, FitsFileWrapperGPU>(spec, shuffle_after_epoch) {}

 protected:
  void ReadDataFromHDU(const fits::FitsHandle& current_file, const fits::HeaderData& header,
                       FitsFileWrapperGPU& target, size_t output_idx) override;

  void ResizeTarget(FitsFileWrapperGPU& target, size_t new_size) override;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_FITS_LOADER_GPU_H_

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

#ifndef DALI_OPERATORS_READER_LOADER_FITS_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_FITS_LOADER_H_

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
#include "dali/operators/reader/loader/file_loader.h"
#include "dali/pipeline/data/types.h"
#include "dali/util/file.h"
#include "dali/util/fits.h"

namespace dali {

struct FitsFileWrapper {
  std::vector<Tensor<CPUBackend>> data;
  std::string filename;
};

class FitsLoader : public FileLoader<CPUBackend, FitsFileWrapper> {
 public:
  explicit inline FitsLoader(const OpSpec& spec, bool shuffle_after_epoch = false)
      : FileLoader(spec, shuffle_after_epoch),
        hdu_indices_(spec.GetRepeatedArgument<int>("hdu_indices")) {
    // default to DALI_UINT8, if argument dtypes not provided
    dtypes_ = spec.HasArgument("dtypes") ?
                  spec.GetRepeatedArgument<DALIDataType>("dtypes") :
                  std::vector<DALIDataType>(hdu_indices_.size(), DALI_UINT8);

    // verify if provided types are supported
    for (DALIDataType dtype : dtypes_) {
      DALI_ENFORCE(fits::supportedTypes.count(dtype),
                   make_string("Unsupported output dtype ", dtype,
                               ". Supported types are: ", fits::SupportedTypesListGen()));
    }

    DALI_ENFORCE(hdu_indices_.size() == dtypes_.size(),
                 "Number of extensions does not match the number of provided types");
  }

  void PrepareEmpty(FitsFileWrapper& target) override {
    target = {};
  }
  void ReadSample(FitsFileWrapper&) override;

 private:
  std::vector<int> hdu_indices_;
  std::vector<DALIDataType> dtypes_;
};


}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_FITS_LOADER_H_

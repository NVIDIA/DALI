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

#include <fitsio.h>
#include "dali/core/common.h"
#include "dali/operators/reader/loader/file_loader.h"
#include "dali/pipeline/data/types.h"
#include "dali/util/file.h"
#include "dali/util/fits.h"

namespace dali {

struct FitsFileWrapper {
  Tensor<CPUBackend> data;
  std::string filename;

  DALIDataType get_type() const {
    return data.type();
  }

  const TensorShape<>& get_shape() const {
    return data.shape();
  }

  const DALIMeta& get_meta() const {
    return data.GetMeta();
  }
};


class FitsLoader : public FileLoader<CPUBackend, FitsFileWrapper> {
 public:
  explicit inline FitsLoader(
    const OpSpec& spec, 
    bool shuffle_after_epoch = false)
      : FileLoader(spec, shuffle_after_epoch) {}

  void PrepareEmpty(FitsFileWrapper& target) override {
    target = {};
  }

  /**
   *  Type From Cfits Code
   */
  const TypeInfo& TypeFromCfitsCode(const int fitsDataType);

  /**
   * Recognize Type From Cfits Code
   */
  int RecognizeTypeFromCfitsCode(int bitpix);

  // we want to make it possible to override this function as well
  void ReadSample(FitsFileWrapper& target) override;
};


}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_FITS_LOADER_H_

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

#ifndef DALI_OPERATORS_READER_LOADER_FITS_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_FITS_LOADER_H_

#include <dirent.h>
#include <errno.h>
#include <fitsio.h>
#include <sys/stat.h>

#include <string>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/file_loader.h"
#include "dali/pipeline/data/types.h"
#include "dali/util/file.h"
#include "dali/util/fits.h"

namespace dali {

struct FitsFileWrapper {
  std::vector<fits::HeaderData> header;
  std::vector<Tensor<CPUBackend>> data;
  std::string filename;
};

template <typename Backend, typename Target>
class FitsLoader : public FileLoader<Backend, Target> {
 public:
  explicit FitsLoader(const OpSpec& spec, bool shuffle_after_epoch = false)
      : FileLoader<Backend, Target>(spec, shuffle_after_epoch),
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

  void PrepareEmpty(Target& target) override {
    target = {};
  }

  void ReadSample(Target& target) override {
    auto filename = files_[current_index_++];
    int status = 0, num_hdus = 0;

    // handle wrap-around
    MoveToNextShard(current_index_);

    // metadata info
    DALIMeta meta;
    // meta.SetSourceInfo(filename); // it adds ./before a filename for some reason
    meta.SetSkipSample(false);

    auto path = filesystem::join_path(file_root_, filename);
    auto current_file = fits::FitsHandle::OpenFile(path.c_str(), READONLY);
    FITS_CALL(fits_get_num_hdus(current_file, &num_hdus, &status));

    // resize ouput vector according to the number of HDUs
    ResizeTarget(target, hdu_indices_.size());

    for (size_t output_idx = 0; output_idx < hdu_indices_.size(); output_idx++) {
      // move to appropiate hdu
      FITS_CALL(fits_movabs_hdu(current_file, hdu_indices_[output_idx], NULL, &status));

      // read the header
      fits::HeaderData header;
      try {
        fits::ParseHeader(header, current_file);
        target.header[output_idx] = header;
      } catch (const std::runtime_error& e) {
        DALI_FAIL(make_string(e.what(), ". File: ", filename));
      }

      // reset, resize specific output in target
      if (target.data[output_idx].shares_data()) {
        target.data[output_idx].Reset();
      }
      target.data[output_idx].Resize(header.shape, header.type());

      ReadDataFromHDU(current_file, header, target, output_idx);

      // set metadata
      target.data[output_idx].SetMeta(meta);

      // set file path
      target.filename = std::move(path);
    }
  }

 protected:
  virtual void ReadDataFromHDU(const fits::FitsHandle& current_file, const fits::HeaderData& header,
                               Target& target, size_t output_idx) = 0;

  virtual void ResizeTarget(Target& target, size_t new_size) = 0;

 private:
  using FileLoader<Backend, Target>::MoveToNextShard;
  using FileLoader<Backend, Target>::files_;
  using FileLoader<Backend, Target>::current_index_;
  using FileLoader<Backend, Target>::file_root_;
  std::vector<int> hdu_indices_;
  std::vector<DALIDataType> dtypes_;
};

class FitsLoaderCPU : public FitsLoader<CPUBackend, FitsFileWrapper> {
 public:
  explicit FitsLoaderCPU(const OpSpec& spec, bool shuffle_after_epoch = false)
      : FitsLoader<CPUBackend, FitsFileWrapper>(spec, shuffle_after_epoch) {}

 protected:
  void ReadDataFromHDU(const fits::FitsHandle& current_file, const fits::HeaderData& header,
                       FitsFileWrapper& target, size_t output_idx) override;
  void ResizeTarget(FitsFileWrapper& target, size_t new_size) override;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_FITS_LOADER_H_

// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_NUMPY_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_NUMPY_LOADER_H_

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>
#include <map>
#include <regex>
#include <memory>

#include "dali/core/common.h"
#include "dali/pipeline/data/types.h"
#include "dali/operators/reader/loader/file_loader.h"
#include "dali/util/file.h"

namespace dali {

TypeInfo TypeFromNumpyStr(const std::string &format);

class NumpyParseTarget{
 public:
  std::vector<int64_t> shape;
  TypeInfo type_info;
  bool fortran_order;
  int64_t data_offset;

  size_t size() const {
    return volume(shape);
  }

  size_t nbytes() const {
    return type_info.size() * size();
  }
};

struct NumpyFileWrapper {
  Tensor<CPUBackend> data;
  std::string filename;
  bool fortran_order;

  const TypeInfo& get_type() const {
    return data.type();
  }

  const TensorShape<>& get_shape() const {
    return data.shape();
  }

  const DALIMeta& get_meta() const {
    return data.GetMeta();
  }
};

namespace detail {

DLL_PUBLIC void ParseHeaderMetadata(NumpyParseTarget& target, const std::string &header);

// parser function, only for internal use
void ParseHeader(FileStream *file, NumpyParseTarget& target);

class NumpyHeaderCache {
 public:
  explicit NumpyHeaderCache(bool cache_headers) : cache_headers_(cache_headers) {}
  bool GetFromCache(const string &file_name, NumpyParseTarget &target);
  void UpdateCache(const string &file_name, const NumpyParseTarget &value);

 private:
  // helper for header caching
  std::mutex cache_mutex_;
  bool cache_headers_;
  std::map<string, NumpyParseTarget> header_cache_;
};

}  // namespace detail

class NumpyLoader : public FileLoader<CPUBackend, NumpyFileWrapper> {
 public:
  explicit inline NumpyLoader(
    const OpSpec& spec,
    bool shuffle_after_epoch = false)
    : FileLoader(spec, shuffle_after_epoch),
    header_cache_(spec.GetArgument<bool>("cache_header_information")) {}

  void PrepareEmpty(NumpyFileWrapper &target) override {
    target = {};
  }

  // we want to make it possible to override this function as well
  void ReadSample(NumpyFileWrapper& target) override;

 private:
  detail::NumpyHeaderCache header_cache_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_NUMPY_LOADER_H_

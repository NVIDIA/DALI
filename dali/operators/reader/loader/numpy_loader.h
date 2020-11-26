// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

namespace detail {

DLL_PUBLIC void ParseHeaderMetadata(NumpyParseTarget& target, const std::string &header);

// parser function, only for internal use
template<typename FileStreamType, typename ReadCPUFun>
void ParseHeader(FileStreamType *file, NumpyParseTarget& target, ReadCPUFun readCPU) {
  // check if the file is actually a numpy file
  std::vector<uint8_t> token(11);
  int64_t nread = (file->*readCPU)(token.data(), 10);
  DALI_ENFORCE(nread == 10, "Can not read header.");
  token[nread] = '\0';

  // check if heqder is too short
  std::string header = std::string(reinterpret_cast<char*>(token.data()));
  DALI_ENFORCE(header.find_first_of("NUMPY") != std::string::npos,
               "File is not a numpy file.");

  // extract header length
  uint16_t header_len = 0;
  memcpy(&header_len, &token[8], 2);
  DALI_ENFORCE((header_len + 10) % 16 == 0,
               "Error extracting header length.");

  // read header: the offset is a magic number
  int64 offset = 6 + 1 + 1 + 2;
  // the header_len can be 4GiB according to the NPYv2 file format
  // specification: https://numpy.org/neps/nep-0001-npy-format.html
  // while this allocation could be sizable, it is performed on the host.
  token.resize(header_len+1);
  file->Seek(offset);
  nread = (file->*readCPU)(token.data(), header_len);
  DALI_ENFORCE(nread == header_len, "Can not read header.");
  token[header_len] = '\0';
  header = std::string(reinterpret_cast<const char*>(token.data()));
  DALI_ENFORCE(header.find("{") != std::string::npos, "Header is corrupted.");
  offset += header_len;
  file->Seek(offset);  // prepare file for later reads

  detail::ParseHeaderMetadata(target, header);
}

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

class NumpyLoader : public FileLoader<> {
 public:
  explicit inline NumpyLoader(
    const OpSpec& spec,
    bool shuffle_after_epoch = false)
    : FileLoader(spec, shuffle_after_epoch),
    header_cache_(spec.GetArgument<bool>("cache_header_information")) {}

  // we want to make it possible to override this function as well
  void ReadSample(ImageFileWrapper& tensor) override;
 private:
  detail::NumpyHeaderCache header_cache_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_NUMPY_LOADER_H_

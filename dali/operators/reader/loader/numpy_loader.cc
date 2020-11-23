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

#include <dirent.h>
#include <errno.h>
#include <cstdlib>
#include <memory>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/numpy_loader.h"
#include "dali/util/file.h"
#include "dali/operators/reader/loader/utils.h"

namespace dali {
TypeInfo TypeFromNumpyStr(const std::string &format) {
  if (format == "u1") return TypeInfo::Create<uint8_t>();
  if (format == "u2") return TypeInfo::Create<uint16_t>();
  if (format == "u4") return TypeInfo::Create<uint32_t>();
  if (format == "u8") return TypeInfo::Create<uint64_t>();
  if (format == "i1") return TypeInfo::Create<int8_t>();
  if (format == "i2") return TypeInfo::Create<int16_t>();
  if (format == "i4") return TypeInfo::Create<int32_t>();
  if (format == "i8") return TypeInfo::Create<int64_t>();
  if (format == "f2") return TypeInfo::Create<float16>();
  if (format == "f4") return TypeInfo::Create<float>();
  if (format == "f8") return TypeInfo::Create<double>();
  DALI_FAIL("Unknown Numpy type string");
}

inline void SkipSpaces(const char*& ptr) {
  while (::isspace(*ptr))
    ptr++;
}

template <size_t N>
void Skip(const char*& ptr, const char (&what)[N]) {
  DALI_ENFORCE(!strncmp(ptr, what, N - 1),
               make_string("Expected \"", what, "\" but got \"", ptr, "\""));
  ptr += N - 1;
}

template <size_t N>
bool TrySkip(const char*& ptr, const char (&what)[N]) {
  if (!strncmp(ptr, what, N-1)) {
    ptr += N - 1;
    return true;
  } else {
    return false;
  }
}


template <size_t N>
void SkipFieldName(const char*& ptr, const char (&name)[N]) {
  SkipSpaces(ptr);
  Skip(ptr, "'");
  Skip(ptr, name);
  Skip(ptr, "'");
  SkipSpaces(ptr);
  Skip(ptr, ":");
  SkipSpaces(ptr);
}

template <typename T = int64_t>
T ParseInteger(const char*& ptr) {
  char *out_ptr = const_cast<char*>(ptr);  // strtol takes a non-const pointer
  T value = static_cast<T>(strtol(ptr, &out_ptr, 10));
  DALI_ENFORCE(out_ptr != ptr, "Parse error: expected a number.");
  ptr = out_ptr;
  return value;
}

std::string ParseStringValue(const char*& input, char delim_start = '\'', char delim_end = '\'') {
  DALI_ENFORCE(*input++ == delim_start, make_string("Expected \'", delim_start, "\'"));
  std::string out;
  for (; *input != '\0'; input++) {
    if (*input == '\\') {
      switch (*++input) {
        case '\\':
          out += '\\';
          break;
        case '\'':
          out += '\'';
          break;
        case '\t':
          out += '\t';
          break;
        case '\n':
          out += '\n';
          break;
        case '\"':
          out += '\"';
          break;
        default:
          out += '\\';
          out += *input;
          break;
      }
    } else if (*input == delim_end) {
      break;
    } else {
      out += *input;
    }
  }
  DALI_ENFORCE(*input++ == delim_end, make_string("Expected \'", delim_end, "\'"));
  return out;
}

namespace detail {

void ParseHeaderMetadata(NumpyParseTarget& target, const std::string &header) {
  target.shape.clear();
  const char* hdr = header.c_str();
  SkipSpaces(hdr);
  Skip(hdr, "{");
  SkipFieldName(hdr, "descr");
  auto typestr = ParseStringValue(hdr);
    // < means LE, | means N/A, = means native. In all those cases, we can read
  bool little_endian = (typestr[0] == '<' || typestr[0] == '|' || typestr[0] == '=');
  DALI_ENFORCE(little_endian, "Big Endian files are not supported.");
  target.type_info = TypeFromNumpyStr(typestr.substr(1));

  SkipSpaces(hdr);
  Skip(hdr, ",");
  SkipFieldName(hdr, "fortran_order");
  if (TrySkip(hdr, "True")) {
    target.fortran_order = true;
  } else if (TrySkip(hdr, "False")) {
    target.fortran_order = false;
  } else {
    DALI_FAIL("Failed to parse fortran_order field.");
  }
  SkipSpaces(hdr);
  Skip(hdr, ",");
  SkipFieldName(hdr, "shape");
  Skip(hdr, "(");
  SkipSpaces(hdr);
  while (*hdr != ')') {
    // ParseInteger already skips the leading spaces (strtol does).
    target.shape.push_back(ParseInteger<int64_t>(hdr));
    SkipSpaces(hdr);
    DALI_ENFORCE(TrySkip(hdr, ",") || target.shape.size() > 1,
                 "The first number in a tuple must be followed by a comma.");
  }
  if (target.fortran_order) {
    // cheapest thing to do is to define the tensor in an reversed way
    std::reverse(target.shape.begin(), target.shape.end());
  }
}

bool NumpyHeaderCache::GetFromCache(const string &file_name, NumpyParseTarget &target) {
  if (!cache_headers_) {
    return false;
  }
  std::unique_lock<std::mutex> cache_lock(cache_mutex_);
  auto it = header_cache_.find(file_name);
  if (it == header_cache_.end()) {
    return false;
  } else {
    target = it->second;
    return true;
  }
}

void NumpyHeaderCache::UpdateCache(const string &file_name, const NumpyParseTarget &value) {
  if (cache_headers_) {
    std::unique_lock<std::mutex> cache_lock(cache_mutex_);
    header_cache_[file_name] = value;
  }
}

}  // namespace detail

void NumpyLoader::ReadSample(ImageFileWrapper& imfile) {
  auto image_file = images_[current_index_++];

  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(image_file);
  meta.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(image_file)) {
    meta.SetSkipSample(true);
    imfile.image.Reset();
    imfile.image.SetMeta(meta);
    imfile.image.set_type(TypeInfo::Create<uint8_t>());
    imfile.image.Resize({0});
    imfile.filename = "";
    return;
  }

  auto current_image = FileStream::Open(file_root_ + "/" + image_file, read_ahead_,
                                        !copy_read_data_);

  // read the header
  NumpyParseTarget target;
  auto ret = header_cache_.GetFromCache(image_file, target);
  if (ret) {
    current_image->Seek(target.data_offset);
  } else {
    detail::ParseHeader(current_image.get(), target, &FileStream::Read);
    header_cache_.UpdateCache(image_file, target);
  }

  Index image_bytes = target.nbytes();

  if (copy_read_data_) {
    if (imfile.image.shares_data()) {
      imfile.image.Reset();
    }
    imfile.image.Resize(target.shape, target.type_info);
    // copy the image
    Index ret = current_image->Read(static_cast<uint8_t*>(imfile.image.raw_mutable_data()),
                                    image_bytes);
    DALI_ENFORCE(ret == image_bytes, make_string("Failed to read file: ", image_file));
  } else {
    auto p = current_image->Get(image_bytes);
    DALI_ENFORCE(p != nullptr, make_string("Failed to read file: ", image_file));
    // Wrap the raw data in the Tensor object.
    imfile.image.ShareData(p, image_bytes, {image_bytes});
    imfile.image.Resize(target.shape, target.type_info);
  }

  // close the file handle
  current_image->Close();

  // set metadata
  imfile.image.SetMeta(meta);

  // set file path
  imfile.filename = file_root_ + "/" + image_file;

  // set meta
  imfile.meta = (target.fortran_order ? "transpose:true" : "transpose:false");
}

}  // namespace dali

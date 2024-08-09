// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/util/numpy.h"
#include <string>
#include <vector>
#include <memory>
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/views.h"
#include "dali/util/odirect_file.h"
#include "dali/core/mm/memory.h"

namespace dali {
namespace numpy {

const TypeInfo &TypeFromNumpyStr(const std::string &format) {
  if (format == "b1") return TypeTable::GetTypeInfo<bool>();
  if (format == "u1") return TypeTable::GetTypeInfo<uint8_t>();
  if (format == "u2") return TypeTable::GetTypeInfo<uint16_t>();
  if (format == "u4") return TypeTable::GetTypeInfo<uint32_t>();
  if (format == "u8") return TypeTable::GetTypeInfo<uint64_t>();
  if (format == "i1") return TypeTable::GetTypeInfo<int8_t>();
  if (format == "i2") return TypeTable::GetTypeInfo<int16_t>();
  if (format == "i4") return TypeTable::GetTypeInfo<int32_t>();
  if (format == "i8") return TypeTable::GetTypeInfo<int64_t>();
  if (format == "f2") return TypeTable::GetTypeInfo<float16>();
  if (format == "f4") return TypeTable::GetTypeInfo<float>();
  if (format == "f8") return TypeTable::GetTypeInfo<double>();
  DALI_FAIL(make_string("Unknown Numpy type string: ", format));
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

void ParseHeaderContents(HeaderData& target, const std::string &header) {
  target.shape = {};
  const char* hdr = header.c_str();
  SkipSpaces(hdr);
  Skip(hdr, "{");
  SkipFieldName(hdr, "descr");
  auto typestr = ParseStringValue(hdr);
    // < means LE, | means N/A, = means native. In all those cases, we can read
  bool little_endian = (typestr[0] == '<' || typestr[0] == '|' || typestr[0] == '=');
  DALI_ENFORCE(little_endian, "Big Endian files are not supported.");
  target.type_info = &TypeFromNumpyStr(typestr.substr(1));

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
    target.shape.shape.push_back(ParseInteger<int64_t>(hdr));
    SkipSpaces(hdr);
    DALI_ENFORCE(TrySkip(hdr, ",") || target.shape.size() > 1,
                 "The first number in a tuple must be followed by a comma.");
  }
  if (target.fortran_order) {
    // cheapest thing to do is to define the tensor in an reversed way
    std::reverse(target.shape.begin(), target.shape.end());
  }
}

void CheckNpyVersion(char *token) {
  int api_version = token[6];
  if (api_version != 1) {
    static int unrecognized_version = [&]() {
      std::string actual_version =
          (api_version == 2 || api_version == 3) ? "higher" : "unrecognized";
      DALI_WARN(make_string(
          "Expected file in NPY file format version 1. The provided file uses ", actual_version,
          " version. Please note, DALI does not support structured NumPy arrays."));
      return 0;
    }();
  }
}

uint16_t GetHeaderLen(char *data) {
  std::string header = std::string(data);
  DALI_ENFORCE(header.find_first_of("NUMPY") != std::string::npos,
               "File is not a numpy file.");

  // extract header length
  uint16_t header_len = 0;
  memcpy(&header_len, &data[8], 2);
  DALI_ENFORCE((header_len + 10) % 16 == 0,
               "Error extracting header length.");
  return header_len;
}

void ParseHeaderItself(HeaderData &parsed_header, char *data, size_t header_len) {
  auto header = std::string(data);
  DALI_ENFORCE(header.find('{') != std::string::npos, "Header is corrupted.");
  int64_t offset = 6 + 1 + 1 + 2;
  offset += header_len;

  ParseHeaderContents(parsed_header, header);
  parsed_header.data_offset = offset;
}

void ParseODirectHeader(HeaderData &parsed_header, InputStream *src, size_t o_direct_alignm,
                        size_t o_direct_read_len_alignm) {
  const size_t token_len = 6 + 1 + 1 + 2;
  size_t token_read_len = align_up(token_len + 1, o_direct_read_len_alignm);
  auto token_mem =
      mm::alloc_raw_shared<char, mm::memory_kind::host>(token_read_len, o_direct_alignm);
  char *token = token_mem.get();
  auto file = dynamic_cast<ODirectFileStream *>(src);
  DALI_ENFORCE(
      file,
      "Could not read the numpy file header: expected file stream opened with O_DIRECT flag.");
  int64_t nread = file->ReadAt(token, token_read_len, 0);
  DALI_ENFORCE(nread <= static_cast<Index>(token_read_len) &&
                   nread >= static_cast<Index>(std::min(src->Size(), token_read_len)),
               make_string("Can not read header: ",
                           static_cast<Index>(std::min(src->Size(), token_read_len)), " <= ", nread,
                           " <= ", token_read_len));
  auto char_tmp = token[token_len];
  token[token_len] = '\0';

  CheckNpyVersion(token);
  auto header_len = GetHeaderLen(token);

  // The header_len can have up to 2**16 - 1 bytes. We do not support V2 headers
  // (with up to 4GB - 4 byte header len), as those are used by numpy to save structured
  // arrays (where dtype can be different for each column and the columns have arbitrary names).
  // Parsing such a dtype in the header will fail.
  // https://numpy.org/neps/nep-0001-npy-format.html
  size_t aligned_token_header_len =
      align_up(token_len + header_len + 1, std::max(o_direct_alignm, o_direct_read_len_alignm));
  // if header_len goes beyond the previously allocated and read memory reallocate and read again
  // otherwise reuse
  if (token_read_len != aligned_token_header_len) {
    token_mem = mm::alloc_raw_shared<char, mm::memory_kind::host>(aligned_token_header_len,
                                                                  o_direct_alignm);
    nread = file->ReadAt(token_mem.get(), aligned_token_header_len, 0);
    DALI_ENFORCE(nread <= static_cast<Index>(aligned_token_header_len) &&
                     nread >= static_cast<Index>(std::min(src->Size(), aligned_token_header_len)),
                 make_string("Can not read header: ",
                             static_cast<Index>(std::min(src->Size(), aligned_token_header_len)),
                             " <= ", nread, " <= ", aligned_token_header_len));
  } else {
    // restore overriden character
    token[token_len] = char_tmp;
  }
  char *header = token_mem.get() + token_len;
  header[header_len] = '\0';
  ParseHeaderItself(parsed_header, header, header_len);
}

void ParseHeader(HeaderData &parsed_header, InputStream *src) {
  // check if the file is actually a numpy file
  SmallVector<char, 128> token;
  int64_t nread = src->Read(token.data(), 10);
  DALI_ENFORCE(nread == 10, "Can not read header.");
  token[nread] = '\0';

  CheckNpyVersion(token.data());
  auto header_len = GetHeaderLen(token.data());

  // read header: the offset is a magic number
  int64_t offset = 6 + 1 + 1 + 2;
  // The header_len can have up to 2**16 - 1 bytes. We do not support V2 headers
  // (with up to 4GB - 4 byte header len), as those are used by numpy to save structured
  // arrays (where dtype can be different for each column and the columns have arbitrary names).
  // Parsing such a dtype in the header will fail.
  // https://numpy.org/neps/nep-0001-npy-format.html
  token.resize(header_len+1);
  src->SeekRead(offset);
  nread = src->Read(token.data(), header_len);
  DALI_ENFORCE(nread == static_cast<Index>(header_len), "Can not read header.");
  token[header_len] = '\0';

  ParseHeaderItself(parsed_header, token.data(), header_len);
}

void FromFortranOrder(SampleView<CPUBackend> output, ConstSampleView<CPUBackend> input) {
  int n_dims = input.shape().sample_dim();
  SmallVector<int, 6> perm;
  perm.resize(n_dims);
  for (int i = 0; i < n_dims; ++i)
    perm[i] = n_dims - i - 1;
  TYPE_SWITCH(input.type(), type2id, T, NUMPY_ALLOWED_TYPES, (
    kernels::TransposeGrouped(view<T>(output), view<const T>(input), make_cspan(perm));
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type())));  // NOLINT
}

DALIDataType HeaderData::type() const {
  return type_info ? type_info->id() : DALI_NO_TYPE;
}

size_t HeaderData::size() const {
  return volume(shape);
}

size_t HeaderData::nbytes() const {
  return type_info ? type_info->size() * size() : 0_uz;
}

  Tensor<CPUBackend> ReadTensor(InputStream *src, bool pinned) {
  numpy::HeaderData header;
  numpy::ParseHeader(header, src);
  src->SeekRead(header.data_offset, SEEK_SET);

  Tensor<CPUBackend> data;
  data.set_pinned(pinned);
  data.Resize(header.shape, header.type());
  auto ret = src->Read(static_cast<uint8_t*>(data.raw_mutable_data()), header.nbytes());
  DALI_ENFORCE(ret == header.nbytes(), "Failed to read numpy file");

  if (header.fortran_order) {
    Tensor<CPUBackend> transposed;
    transposed.Resize(data.shape(), data.type());
    SampleView<CPUBackend> input(data.raw_mutable_data(), data.shape(), data.type());
    SampleView<CPUBackend> output(transposed.raw_mutable_data(), transposed.shape(),
                                  transposed.type());
    numpy::FromFortranOrder(output, input);
    return transposed;
  }
  return data;
}

}  // namespace numpy
}  // namespace dali

// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_UTIL_PYBIND_H_
#define DALI_UTIL_PYBIND_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include "dali/pipeline/data/types.h"

namespace dali {

namespace py = pybind11;

static std::string FormatStrFromType(const TypeInfo &type) {
  if (IsType<int8_t>(type)) {
    return "=b";
  } else if (IsType<uint8_t>(type)) {
    return "=B";
  } else if (IsType<char>(type)) {
    return "=c";
  } else if (IsType<int16_t>(type)) {
    return "=h";
  } else if (IsType<uint16_t>(type)) {
    return "=H";
  } else if (IsType<int32_t>(type) ||
            (IsType<long>(type) && sizeof(long) == sizeof(int32_t))) {  // NOLINT
    return "=i";
  } else if (IsType<uint32_t>(type) ||
            (IsType<unsigned long>(type) && sizeof(unsigned long) == sizeof(uint32_t))) {  // NOLINT
    return "=I";
  } else if (IsType<int64_t>(type) ||
            (IsType<long>(type) && sizeof(long) == sizeof(int64_t)) ||  // NOLINT
            (IsType<long long>(type) && sizeof(long long) == sizeof(int64_t))) {  // NOLINT
    return "=q";
  } else if (IsType<uint64_t>(type) ||
            (IsType<unsigned long>(type) && sizeof(unsigned long) == sizeof(uint64_t)) ||  // NOLINT
            (IsType<unsigned long long>(type) && sizeof(unsigned long long) == sizeof(uint64_t))) {  // NOLINT
    return "=Q";
  } else if (IsType<float>(type)) {
    return "=f";
  } else if (IsType<double>(type)) {
    return "=d";
  } else if (IsType<bool>(type)) {
    return "=?";
  } else if (IsType<float16>(type)) {
    return "=e";
  } else if (IsType<ssize_t>(type)) {
    return "=n";
  } else if (IsType<size_t>(type)) {
    return "=N";
  } else {
    DALI_FAIL("Cannot convert type " + type.name() +
        " to format descriptor string");
  }
}

static TypeInfo TypeFromFormatStr(const std::string &format) {
  std::string format_letter = format;
  std::string modificator_letter = "@";
  if (format.size() > 1) {
    format_letter = std::string(format, 1, 1);
    modificator_letter = std::string(format, 0, 1);
  }

  if (modificator_letter == "=") {
    // create tensors with well defined element sizes under the hood
    if (format_letter == "c") {
      return TypeInfo::Create<char>();
    } else if (format_letter == "b") {
      return TypeInfo::Create<int8_t>();
    } else if (format_letter == "B") {
      return TypeInfo::Create<uint8_t>();
    } else if (format_letter == "h") {
      return TypeInfo::Create<int16_t>();
    } else if (format_letter == "H") {
      return TypeInfo::Create<uint16_t>();
    } else if (format_letter == "i" || format_letter == "l") {
      return TypeInfo::Create<int32_t>();
    } else if (format_letter == "I") {
      return TypeInfo::Create<uint32_t>();
    } else if (format_letter == "q") {
      return TypeInfo::Create<int64_t>();
    } else if (format_letter == "Q") {
      return TypeInfo::Create<uint64_t>();
    } else if (format_letter == "f") {
      return TypeInfo::Create<float>();
    } else if (format_letter == "d") {
      return TypeInfo::Create<double>();
    } else if (format_letter == "?") {
      return TypeInfo::Create<bool>();
    } else if (format_letter == "e") {
      return TypeInfo::Create<float16>();
    } else if (format_letter == "n") {
      return TypeInfo::Create<ssize_t>();
    } else if (format_letter == "N") {
      return TypeInfo::Create<size_t>();
    } else {
      DALI_FAIL("Cannot create type for unknown format string: " + format);
    }
  } else {  // for '@' or any other case as we don't care about endianess, at least now
    // create tensor with elements of whatever size there is under the hood
    if (format_letter == "c") {
      return TypeInfo::Create<char>();
    } else if (format_letter == "b") {
      return TypeInfo::Create<signed char>();
    } else if (format_letter == "B") {
      return TypeInfo::Create<unsigned char>();
    } else if (format_letter == "h") {
      return TypeInfo::Create<short>();  // NOLINT
    } else if (format_letter == "H") {
      return TypeInfo::Create<unsigned short>();  // NOLINT
    } else if (format_letter == "i" ||
              (format_letter == "l" && sizeof(long) == sizeof(int))) { // NOLINT
      // long size may differ depending on the platform, special case for that here
      return TypeInfo::Create<int>();
    } else if (format_letter == "I" ||
              (format_letter == "L" && sizeof(long) == sizeof(int))) { // NOLINT
      return TypeInfo::Create<unsigned int>();
    } else if (format_letter == "q" ||
              (format_letter == "l" && sizeof(long) == sizeof(long long))) { // NOLINT
      // long size may differ depending on the platform, special case for that here
      return TypeInfo::Create<long long>();  // NOLINT
    } else if (format_letter == "Q" ||
              (format_letter == "L" && sizeof(long) == sizeof(long long))) { // NOLINT
      // long size may differ depending on the platform, special case for that here
      return TypeInfo::Create<unsigned long long>();  // NOLINT
    } else if (format_letter == "f") {
      return TypeInfo::Create<float>();
    } else if (format_letter == "d") {
      return TypeInfo::Create<double>();
    } else if (format_letter == "?") {
      return TypeInfo::Create<bool>();
    } else if (format_letter == "e") {
      return TypeInfo::Create<float16>();
    } else {
      DALI_FAIL("Cannot create type for unknown format string: " + format);
    }
  }
}

}  // namespace dali
#endif  // DALI_UTIL_PYBIND_H_

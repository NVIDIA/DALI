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
  if (IsType<uint8>(type)) {
    return py::format_descriptor<uint8>::format();
  } else if (IsType<int16>(type)) {
    return py::format_descriptor<int16>::format();
  } else if (IsType<int>(type)) {
    return py::format_descriptor<int>::format();
  } else if (IsType<long>(type)) { // NOLINT
    return py::format_descriptor<long>::format(); // NOLINT
  } else if (IsType<int64>(type)) { // NOLINT
    return py::format_descriptor<long long>::format(); // NOLINT
  } else if (IsType<float>(type)) {
    return py::format_descriptor<float>::format();
  } else if (IsType<double>(type)) {
    return py::format_descriptor<double>::format();
  } else if (IsType<bool>(type)) {
    return py::format_descriptor<bool>::format();
  } else if (IsType<float16>(type)) {
    return "f2";
  } else {
    DALI_FAIL("Cannot convert type " + type.name() +
        " to format descriptor string");
  }
}

static TypeInfo TypeFromFormatStr(const std::string &format) {
  if (format == py::format_descriptor<uint8>::format()) {
    return TypeInfo::Create<uint8>();
  } else if (format == py::format_descriptor<int16>::format()) {
    return TypeInfo::Create<int16>();
  } else if (format == py::format_descriptor<int>::format()) {
    return TypeInfo::Create<int>();
  } else if (format == py::format_descriptor<long>::format()) { // NOLINT
    return TypeInfo::Create<long>(); // NOLINT
  } else if (format == py::format_descriptor<long long>::format()) { // NOLINT
    return TypeInfo::Create<int64>(); // NOLINT
  } else if (format == py::format_descriptor<float>::format()) {
    return TypeInfo::Create<float>();
  } else if (format == py::format_descriptor<double>::format()) {
    return TypeInfo::Create<double>();
  } else if (format == py::format_descriptor<bool>::format()) {
    return TypeInfo::Create<bool>();
  } else if (format == "f2") {
    return TypeInfo::Create<float16>();
  } else {
    DALI_FAIL("Cannot create type for unknown format string: " + format);
  }
}

}  // namespace dali
#endif  // DALI_UTIL_PYBIND_H_

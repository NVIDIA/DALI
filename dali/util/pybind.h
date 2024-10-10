// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>
#include <string>
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/data/dltensor.h"
#include "dali/pipeline/operator/error_reporting.h"

namespace dali {

namespace py = pybind11;

static std::string FormatStrFromType(DALIDataType type) {
  // handle common types in a switch
  switch (type) {
    case DALI_INT8:
      return "=b";
    case DALI_UINT8:
      return "=B";
    case DALI_INT16:
      return "=h";
    case DALI_UINT16:
      return "=H";
    case DALI_INT32:
      return "=i";
    case DALI_UINT32:
      return "=I";
    case DALI_INT64:
      return "=q";
    case DALI_UINT64:
      return "=Q";
    case DALI_FLOAT:
      return "=f";
    case DALI_FLOAT16:
      return "=e";
    case DALI_FLOAT64:
      return "=d";
    case DALI_BOOL:
     return "=?";
    case DALI_DATA_TYPE:
    case DALI_IMAGE_TYPE:
    case DALI_INTERP_TYPE:
     throw DaliTypeError(
         "DALI enum types cannot be used with buffer protocol "
         "when they are returned as Tensors or TensorLists from DALI pipeline. "
         "You can use `nvidia.dali.fn.cast` to convert those values to an integral type.");
     // As an alternative, to allow the usage of tensors containing DALI enums (printing, use with
     // buffer protocol, numpy conversion etc), we can return format specifier for the underlying
     // type here. This would allow access to the actual numeric values, for example:
     // case DALI_DATA_TYPE:
     //  return
     //  FormatStrFromType(TypeTable::GetTypeInfo<std::underlying_type_t<DALIDataType>>().id());
    default:
      break;
  }

  // fall back to 'if' ladder

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
    DALI_FAIL(make_string("Cannot convert type ", type, " to a format descriptor string"));
  }
}

static const TypeInfo &TypeFromFormatStr(const std::string &format) {
  char format_letter;
  int format_number = -1;
  if (format.size() == 1) {
    format_letter = format[0];
  } else if (format.size() > 1) {
    format_letter = format[1];
    if (format.size() > 2) {
      format_number = std::stoi(format.substr(2));
    }
  } else {
    DALI_FAIL("Cannot create type for unknown format string: " + format);
  }

  using sized_long = std::conditional_t<sizeof(long) == 4, int32_t, int64_t>;  // NOLINT
  using sized_ulong = std::make_unsigned_t<sized_long>;
  static_assert(sizeof(sized_long) == sizeof(long),  // NOLINT
    "This code requires `long` to be 32 or 64 bit");

  switch (format_letter) {
    case 'c':
      return TypeTable::GetTypeInfo<char>();
    // type supported by cupy
    case 'u':
      if (format_number == -1 || format_number == sizeof(uint8_t)) {
        return TypeTable::GetTypeInfo<uint8_t>();
      } else if (format_number == sizeof(uint16_t)) {
        return TypeTable::GetTypeInfo<uint16_t>();
      } else if (format_number == sizeof(uint32_t)) {
        return TypeTable::GetTypeInfo<uint32_t>();
      } else if (format_number == sizeof(uint64_t)) {
        return TypeTable::GetTypeInfo<uint64_t>();
      }
    case 'b':
      return TypeTable::GetTypeInfo<int8_t>();
    case 'B':
      return TypeTable::GetTypeInfo<uint8_t>();
    case 'h':
      return TypeTable::GetTypeInfo<int16_t>();
    case 'H':
      return TypeTable::GetTypeInfo<uint16_t>();
    case 'i':
      if (format_number == -1 || format_number == sizeof(int32_t)) {
        return TypeTable::GetTypeInfo<int32_t>();
      } else if (format_number == sizeof(int64_t)) {
        return TypeTable::GetTypeInfo<int64_t>();
      } else if (format_number == sizeof(int16_t)) {
        return TypeTable::GetTypeInfo<int16_t>();
      } else if (format_number == sizeof(int8_t)) {
        return TypeTable::GetTypeInfo<int8_t>();
      }
    case 'I':
      return TypeTable::GetTypeInfo<uint32_t>();
    case 'l':
      return TypeTable::GetTypeInfo<sized_long>();
    case 'L':
      return TypeTable::GetTypeInfo<sized_ulong>();
    case 'q':
      return TypeTable::GetTypeInfo<int64_t>();
    case 'Q':
      return TypeTable::GetTypeInfo<uint64_t>();
    case 'f':
      if (format_number == -1 || format_number == sizeof(float)) {
        return TypeTable::GetTypeInfo<float>();
      } else if (format_number == sizeof(double)) {
        return TypeTable::GetTypeInfo<double>();
      } else if (format_number == sizeof(float16)) {
        return TypeTable::GetTypeInfo<float16>();
      }
    case 'd':
      return TypeTable::GetTypeInfo<double>();
    case '?':
      return TypeTable::GetTypeInfo<bool>();
    case 'e':
      return TypeTable::GetTypeInfo<float16>();
    case 'n':
      return TypeTable::GetTypeInfo<ssize_t>();
    case 'N':
      return TypeTable::GetTypeInfo<size_t>();
    default:
      DALI_FAIL("Cannot create type for unknown format string: " + format);
  }
}

constexpr const char *DLTENSOR_NAME = "dltensor";
constexpr const char *USED_DLTENSOR_NAME = "used_dltensor";

static void DLTensorCapsuleDestructor(PyObject *capsule) {
  // run the destructor only for unused capsules (those which keep the original name)
  if (std::string(PyCapsule_GetName(capsule)) == DLTENSOR_NAME) {
    auto *ptr = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, DLTENSOR_NAME));
    DLMTensorPtrDeleter(ptr);
  }
}

// Steal a DLPack Tensor from the passed pointer and wrap it into Python capsule.
static py::capsule DLTensorToCapsule(DLMTensorPtr dl_tensor) {
  auto caps = py::capsule(dl_tensor.release(), DLTENSOR_NAME, &DLTensorCapsuleDestructor);
  return caps;
}

template <typename Backend>
py::capsule TensorToDLPackView(SampleView<Backend> tensor, bool pinned, int device_id) {
  DLMTensorPtr dl_tensor = GetDLTensorView(tensor, pinned, device_id);
  return DLTensorToCapsule(std::move(dl_tensor));
}

template <typename Backend>
py::list TensorListToDLPackView(TensorList<Backend> &tensors) {
  py::list result;
  auto dl_tensors = GetDLTensorListView(tensors);
  for (DLMTensorPtr &dl_tensor : dl_tensors) {
    result.append(DLTensorToCapsule(std::move(dl_tensor)));
  }
  return result;
}

static DLManagedTensor* DLMTensorRawPtrFromCapsule(py::capsule &capsule, bool consume = true) {
  DALI_ENFORCE(std::string(capsule.name()) == DLTENSOR_NAME,
      "Invalid DLPack tensor capsule. Notice that a dl tensor can be consumed only once");
  if (consume) {
    PyCapsule_SetName(capsule.ptr(), USED_DLTENSOR_NAME);
  }
  return static_cast<DLManagedTensor*>(capsule);
}

static DLMTensorPtr DLMTensorPtrFromCapsule(py::capsule &capsule) {
  return {DLMTensorRawPtrFromCapsule(capsule), DLMTensorPtrDeleter};
}


/**
 * @see to_struct
 */
template <typename Struct, std::size_t... Idx, typename Tuple>
Struct to_struct_helper(std::index_sequence<Idx...>, Tuple &&tuple) {
  return {std::get<Idx>(std::forward<Tuple>(tuple))...};
}


/**
 * Converts Tuple to Struct by calling a Struct's brace-initializer
 * with every Tuple's argument, retaining its order. E.g.
 *
 *
 * tuple<int, string, float> tt;
 * struct Foo {
 *     int a; string b; float c;
 * }
 *
 * auto f = to_struct<Foo>(tt)  <=>  Foo f = {get<0>(tt), get<1>(tt), get<2>(tt)}
 *
 * @return
 */
template <typename Struct, typename Tuple>
Struct to_struct(Tuple &&tuple) {
  using T = std::remove_reference_t<Tuple>;
  return to_struct_helper<Struct>(std::make_index_sequence<std::tuple_size<T>{}>{},
                                  std::forward<Tuple>(tuple));
}

}  // namespace dali
#endif  // DALI_UTIL_PYBIND_H_

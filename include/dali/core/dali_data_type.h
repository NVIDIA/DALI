// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_DALI_DATA_TYPE_H_
#define DALI_CORE_DALI_DATA_TYPE_H_

#ifdef __cplusplus
#define DALI_CONSTEXPR constexpr

#include <cstdint>
#include <type_traits>

using daliBool = std::conditional_t<sizeof(bool) == sizeof(uint8_t), bool, uint8_t>;

extern "C" {
#else
#define DALI_CONSTEXPR inline
#include <stdbool.h>
#include <stdint.h>

typedef uint8_t daliBool;
#endif

/**
 * @brief Enum identifiers for the different data types that
 * the pipeline can output.
 *
 * IMPORTANT: This enum is used for serialization of DALI Pipeline. Therefore, any change made to
 * this enum must retain backward compatibility. If the backward compatibility is broken
 * (e.g. values of enumerations are shuffled), the already serialized pipelines
 * around the globe will stop working correctly.
 */
typedef enum _DALIDataType {
  DALI_NO_TYPE           = -1,
  DALI_UINT8             =  0,
  DALI_UINT16            =  1,
  DALI_UINT32            =  2,
  DALI_UINT64            =  3,
  DALI_INT8              =  4,
  DALI_INT16             =  5,
  DALI_INT32             =  6,
  DALI_INT64             =  7,
  DALI_FLOAT16           =  8,
  DALI_FLOAT             =  9,
  DALI_FLOAT64           = 10,
  DALI_BOOL              = 11,
  DALI_STRING            = 12,
  DALI_BOOL_VEC          = 13,
  DALI_INT_VEC           = 14,
  DALI_STRING_VEC        = 15,
  DALI_FLOAT_VEC         = 16,
  DALI_TF_FEATURE        = 17,
  DALI_TF_FEATURE_VEC    = 18,
  DALI_TF_FEATURE_DICT   = 19,
  DALI_IMAGE_TYPE        = 20,
  DALI_DATA_TYPE         = 21,
  DALI_INTERP_TYPE       = 22,
  DALI_TENSOR_LAYOUT     = 23,
  DALI_PYTHON_OBJECT     = 24,
  DALI_TENSOR_LAYOUT_VEC = 25,
  DALI_DATA_TYPE_VEC     = 26,
  DALI_NUM_BUILTIN_TYPES,
  DALI_CUSTOM_TYPE_START = 1001,
  DALI_DATA_TYPE_FORCE_INT32 = 0x7fffffff
} daliDataType_t;

/** Returns a display name of a DALI built-in type.
 *
 * @return A pointer to a string constant containing the name or NULL if the type is unknown.
 */
inline const char *daliDataTypeName(daliDataType_t t) {
  switch (t) {
    case DALI_NO_TYPE:
      return "<no_type>";
      break;
    case DALI_UINT8:
      return "uint8";
      break;
    case DALI_UINT16:
      return "uint16";
      break;
    case DALI_UINT32:
      return "uint32";
      break;
    case DALI_UINT64:
      return "uint64";
      break;
    case DALI_INT8:
      return "int8";
      break;
    case DALI_INT16:
      return "int16";
      break;
    case DALI_INT32:
      return "int32";
      break;
    case DALI_INT64:
      return "int64";
      break;
    case DALI_FLOAT16:
      return "float16";
      break;
    case DALI_FLOAT:
      return "float";
      break;
    case DALI_FLOAT64:
      return "double";
      break;
    case DALI_BOOL:
      return "bool";
      break;
    case DALI_STRING:
      return "string";
      break;
    case DALI_BOOL_VEC:
      return "list of bool";
      break;
    case DALI_INT_VEC:
      return "list of int";
      break;
    case DALI_STRING_VEC:
      return "list of string";
      break;
    case DALI_FLOAT_VEC:
      return "list of float";
      break;
    case DALI_TF_FEATURE:
      return "TFUtil::Feature";
      break;
    case DALI_TF_FEATURE_VEC:
      return "list of TFUtil::Feature";
      break;
    case DALI_TF_FEATURE_DICT:
      return "dictionary of TFUtil::Feature";
      break;
    case DALI_IMAGE_TYPE:
      return "DALIImageType";
      break;
    case DALI_DATA_TYPE:
      return "DALIDataType";
      break;
    case DALI_INTERP_TYPE:
      return "DALIInterpType";
      break;
    case DALI_TENSOR_LAYOUT:
      return "TensorLayout";
      break;
    case DALI_PYTHON_OBJECT:
      return "Python object";
      break;
    case DALI_TENSOR_LAYOUT_VEC:
      return "list of TensorLayout";
      break;
    case DALI_DATA_TYPE_VEC:
      return "list of DALIDataType";
    default:
      return 0;
  }
}

/** Returns `true` if the `type` is a floating point type. */
DALI_CONSTEXPR bool daliDataTypeIsFloatingPoint(daliDataType_t type) {
  switch (type) {
    case DALI_FLOAT16:
    case DALI_FLOAT:
    case DALI_FLOAT64:
      return true;
    default:
      return false;
  }
}

/** Returns `true` if the `type` is an integral type. */
DALI_CONSTEXPR bool daliDataTypeIsIntegral(daliDataType_t type) {
  switch (type) {
    case DALI_BOOL:
    case DALI_UINT8:
    case DALI_UINT16:
    case DALI_UINT32:
    case DALI_UINT64:
    case DALI_INT8:
    case DALI_INT16:
    case DALI_INT32:
    case DALI_INT64:
      return true;
    default:
      return false;
  }
}

/** Returns `true` if the `type` has a sign (includes floating point types). */
DALI_CONSTEXPR bool daliDataTypeIsSigned(daliDataType_t type) {
  switch (type) {
    case DALI_FLOAT16:
    case DALI_FLOAT:
    case DALI_FLOAT64:
    case DALI_INT8:
    case DALI_INT16:
    case DALI_INT32:
    case DALI_INT64:
      return true;
    default:
      return false;
  }
}

/** Returns `true` if the `type` is an unsigned integer (includes boolean). */
DALI_CONSTEXPR bool daliDataTypeIsUnsigned(daliDataType_t type) {
  switch (type) {
    case DALI_BOOL:
    case DALI_UINT8:
    case DALI_UINT16:
    case DALI_UINT32:
    case DALI_UINT64:
      return true;
    default:
      return false;
  }
}

/** Returns `true` if the `type` is an enumerated type. */
DALI_CONSTEXPR bool daliDataTypeIsEnum(daliDataType_t type) {
  switch (type) {
    case DALI_DATA_TYPE:
    case DALI_IMAGE_TYPE:
    case DALI_INTERP_TYPE:
      return true;
    default:
      return false;
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DALI_CORE_DALI_DATA_TYPE_H_

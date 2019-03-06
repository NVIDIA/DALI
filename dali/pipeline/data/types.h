// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_DATA_TYPES_H_
#define DALI_PIPELINE_DATA_TYPES_H_


#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <functional>
#include <mutex>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "dali/common.h"
#include "dali/util/cuda_utils.h"
#include "dali/error_handling.h"

// Workaround missing "is_trivially_copyable" in libstdc++ for g++ < 5.0.
// We have to first include some standard library headers, so to have __GLIBCXX__ symbol,
// and we have to exclude the specific version used in manylinux for our CI, because
// clang always defines __GNUC__ to 4. __GLIBCXX__ is not a linear ordering based on
// version of library but the date of the release so we can have 4.9.4 > 5.1.
#if __GLIBCXX__ == 20150212 || (__cplusplus && __GNUC__ < 5 && !__clang__)
#include <boost/type_traits/has_trivial_copy.hpp>
#define IS_TRIVIALLY_COPYABLE(T) ::boost::has_trivial_copy<T>::value
#else
#define IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif

#ifdef DALI_BUILD_PROTO3
#include "dali/pipeline/operators/reader/parser/tf_feature.h"
#endif  // DALI_BUILD_PROTO3

#ifndef DALI_TYPENAME_REGISTERER
#define DALI_TYPENAME_REGISTERER(...)
#endif

#ifndef DALI_TYPEID_REGISTERER
#define DALI_TYPEID_REGISTERER(...)
#endif

namespace dali {

namespace detail {

typedef void (*Copier)(void *, const void*, Index);

template <typename T>
inline typename std::enable_if<IS_TRIVIALLY_COPYABLE(T)>::type
CopyFunc(void *dst, const void *src, Index n) {
  // T is trivially copyable, we can copy using raw memcopy
  std::memcpy(dst, src, n*sizeof(T));
}

template <typename T>
inline typename std::enable_if<!IS_TRIVIALLY_COPYABLE(T)>::type
CopyFunc(void *dst, const void *src, Index n) {
  T *typed_dst = static_cast<T*>(dst);
  const T* typed_src = static_cast<const T*>(src);
  for (Index i = 0; i < n; ++i) {
    // T is not trivially copyable, iterate and
    // call the copy-assignment operator
    typed_dst[i] = typed_src[i];
  }
}

template <typename T>
inline Copier GetCopier() {
  return &CopyFunc<T>;
}

}  // namespace detail

/**
 * @brief Enum identifiers for the different data types that
 * the pipeline can output.
 */
enum DALIDataType {
  DALI_NO_TYPE = -1,
  DALI_UINT8 = 0,
  DALI_INT16 = 1,
  DALI_INT32 = 2,
  DALI_INT64 = 3,
  DALI_FLOAT16 = 4,
  DALI_FLOAT = 5,
  DALI_FLOAT64 = 6,
  DALI_BOOL = 7,
  DALI_STRING = 8,
  DALI_BOOL_VEC = 9,
  DALI_INT_VEC = 10,
  DALI_STRING_VEC = 11,
  DALI_FLOAT_VEC = 12,
#ifdef DALI_BUILD_PROTO3
  DALI_TF_FEATURE = 13,
  DALI_TF_FEATURE_VEC = 14,
  DALI_TF_FEATURE_DICT = 15,
#endif  // DALI_BUILD_PROTO3
  DALI_IMAGE_TYPE = 16,
  DALI_DATA_TYPE = 17,
  DALI_INTERP_TYPE = 18,
  DALI_TENSOR_LAYOUT = 19,
  DALI_DATATYPE_END = 1000
};

// Dummy type to represent the invalid default state of dali types.
struct NoType {};

template <typename T>
struct NormalizedType { using type = T; };

// This way SetType<float16_cpu> will set SetType<float16>
// As these types are compatible python wont need a special
// case for float16_cpu
template <>
struct NormalizedType<float16_cpu> { using type = float16; };

template <typename T>
using normalize_t = typename NormalizedType<T>::type;


// Stores the unqiue ID for a type and its size in bytes
class DLL_PUBLIC TypeInfo {
 public:
  DLL_PUBLIC inline TypeInfo() {
    SetType<NoType>();
  }

  template <typename T>
  DLL_PUBLIC static inline TypeInfo Create() {
    TypeInfo type;
    type.SetType<T>();
    return type;
  }

  template <typename T, typename U = normalize_t<T> >
  DLL_PUBLIC inline void SetType(DALIDataType dtype = DALI_NO_TYPE);

  template <typename DstBackend, typename SrcBackend>
  DLL_PUBLIC void Copy(void *dst, const void *src, Index n, cudaStream_t stream);

  DLL_PUBLIC inline DALIDataType id() const {
    return id_;
  }

  DLL_PUBLIC inline size_t size() const {
    return type_size_;
  }

  DLL_PUBLIC inline const string &name() const {
    return name_;
  }

  DLL_PUBLIC inline bool operator==(const TypeInfo &rhs) const {
    if ((rhs.id_ == id_) &&
        (rhs.type_size_ == type_size_) &&
        (rhs.name_ == name_)) {
      return true;
    }
    return false;
  }

 private:
  detail::Copier copier_;

  DALIDataType id_;
  size_t type_size_;
  string name_;
};

template <typename T>
struct TypeNameHelper {
  static string GetTypeName() {
    return typeid(T).name();
  }
};

/**
 * @brief Keeps track of mappings between types and unique identifiers.
 */
class DLL_PUBLIC TypeTable {
 public:
  template <typename T>
  DLL_PUBLIC static DALIDataType GetTypeID() {
    std::lock_guard<std::mutex> lock(mutex_);
    static DALIDataType type_id = TypeTable::RegisterType<T>(static_cast<DALIDataType>(++index_));
    return type_id;
  }

  template <typename T>
  DLL_PUBLIC static string GetTypeName() {
    return TypeNameHelper<T>::GetTypeName();
  }

  DLL_PUBLIC static const TypeInfo& GetTypeInfo(DALIDataType dtype) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto id_it = type_info_map_.find(dtype);
    DALI_ENFORCE(id_it != type_info_map_.end(),
        "Type with id " + to_string((size_t)dtype) + " was not registered.");
    return id_it->second;
  }

 private:
  // TypeTable should only be referenced through its static members
  TypeTable();

  template <typename T>
  static DALIDataType RegisterType(DALIDataType dtype) {
    // Use only when already guarded by the mutex to
    // avoid races when used from multiple threads

    // Check the map for this types id
    auto id_it = type_map_.find(typeid(T));

    if (id_it == type_map_.end()) {
      type_map_[typeid(T)] = dtype;
      TypeInfo t;
      t.SetType<T>(dtype);
      type_info_map_[dtype] = t;
      return dtype;
    } else {
      return id_it->second;
    }
  }

  static std::mutex mutex_;
  static std::unordered_map<std::type_index, DALIDataType> type_map_;
  // Unordered maps do not work with enums,
  // so we need to use size_t instead of DALIDataType
  static std::unordered_map<size_t, TypeInfo> type_info_map_;
  static int index_;
};


template <typename T, typename A>
struct TypeNameHelper<std::vector<T, A> > {
  static string GetTypeName() {
    return "list of " + TypeTable::GetTypeName<T>();
  }
};

template <typename T, size_t N>
struct TypeNameHelper<std::array<T, N> > {
  static string GetTypeName() {
    return "list of " + TypeTable::GetTypeName<T>();
  }
};

template <typename, typename T>
void TypeInfo::SetType(DALIDataType dtype) {
  // Note: We enforce the fact that NoType is invalid by
  // explicitly setting its type size as 0
  type_size_ = std::is_same<T, NoType>::value ? 0 : sizeof(T);
  if (!std::is_same<T, NoType>::value) {
    id_ = dtype != DALI_NO_TYPE ? dtype : TypeTable::GetTypeID<T>();
  } else {
    id_ = DALI_NO_TYPE;
  }
  name_ = TypeTable::GetTypeName<T>();

  // Get copier for this type
  copier_ = detail::GetCopier<T>();
}

inline std::string to_string(const DALIDataType& dtype) {
  return TypeTable::GetTypeInfo(dtype).name();
}

/**
 * @brief Utility to check types
 */
template <typename T>
DLL_PUBLIC inline bool IsType(TypeInfo type) {
  return type.id() == TypeTable::GetTypeID<T>();
}

/**
 * @brief Utility to check for valid type
 */
DLL_PUBLIC inline bool IsValidType(TypeInfo type) {
  return !IsType<NoType>(type);
}

// Used to define a type for use in dali. Inserts the type into the
// TypeTable w/ a unique id and creates a method to get the name of
// the type as a string. This does not work for non-fundamental types,
// as we do not have any mechanism for calling the constructor of the
// type when the buffer allocates the memory.
#define DALI_REGISTER_TYPE(Type, dtype)                           \
  template <> DLL_PUBLIC string TypeTable::GetTypeName<Type>()               \
    DALI_TYPENAME_REGISTERER(Type);                               \
  template <> DLL_PUBLIC DALIDataType TypeTable::GetTypeID<Type>()           \
    DALI_TYPEID_REGISTERER(Type, dtype);

// Instantiate some basic types
DALI_REGISTER_TYPE(NoType, DALI_NO_TYPE);
DALI_REGISTER_TYPE(uint8, DALI_UINT8);
DALI_REGISTER_TYPE(int16, DALI_INT16);
DALI_REGISTER_TYPE(int32, DALI_INT32);
DALI_REGISTER_TYPE(int64, DALI_INT64);
DALI_REGISTER_TYPE(float16, DALI_FLOAT16);
DALI_REGISTER_TYPE(float, DALI_FLOAT);
DALI_REGISTER_TYPE(double, DALI_FLOAT64);
DALI_REGISTER_TYPE(bool, DALI_BOOL);
DALI_REGISTER_TYPE(string, DALI_STRING);
DALI_REGISTER_TYPE(DALIImageType, DALI_IMAGE_TYPE);
DALI_REGISTER_TYPE(DALIDataType, DALI_DATA_TYPE);
DALI_REGISTER_TYPE(DALIInterpType, DALI_INTERP_TYPE);
DALI_REGISTER_TYPE(DALITensorLayout, DALI_TENSOR_LAYOUT);

#ifdef DALI_BUILD_PROTO3
DALI_REGISTER_TYPE(TFUtil::Feature, DALI_TF_FEATURE);
DALI_REGISTER_TYPE(std::vector<TFUtil::Feature>, DALI_TF_FEATURE_VEC);
#endif
DALI_REGISTER_TYPE(std::vector<bool>, DALI_BOOL_VEC);
DALI_REGISTER_TYPE(std::vector<int>, DALI_INT_VEC);
DALI_REGISTER_TYPE(std::vector<std::string>, DALI_STRING_VEC);
DALI_REGISTER_TYPE(std::vector<float>, DALI_FLOAT_VEC);

/**
 * @brief Easily instantiate templates for all types
 * type - DALIDataType
 * DType becomes a type corresponding to given DALIDataType
 */
#define DALI_TYPE_SWITCH_WITH_FP16_GPU(type, DType, ...) \
  switch (type) {                                    \
    case DALI_NO_TYPE:                               \
      DALI_FAIL("Invalid type.");                    \
    case DALI_UINT8:                                 \
      {                                              \
        typedef uint8 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_INT16:                                 \
      {                                              \
        typedef int16 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_INT32:                                 \
      {                                              \
        typedef int32 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_INT64:                                 \
      {                                              \
        typedef int64 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_FLOAT16:                               \
      {                                              \
        typedef float16 DType;                       \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_FLOAT:                                 \
      {                                              \
        typedef float DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_FLOAT64:                               \
      {                                              \
        typedef double DType;                        \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_BOOL:                                  \
      {                                              \
        typedef bool DType;                          \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    default:                                         \
      DALI_FAIL("Unknown type");                     \
  }

/**
 * @brief Easily instantiate templates for all types
 * type - DALIDataType
 * DType becomes a type corresponding to given DALIDataType
 */
#define DALI_TYPE_SWITCH_WITH_FP16_CPU(type, DType, ...) \
  switch (type) {                                    \
    case DALI_NO_TYPE:                               \
      DALI_FAIL("Invalid type.");                    \
    case DALI_UINT8:                                 \
      {                                              \
        typedef uint8 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_INT16:                                 \
      {                                              \
        typedef int16 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_INT32:                                 \
      {                                              \
        typedef int32 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_INT64:                                 \
      {                                              \
        typedef int64 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_FLOAT16:                               \
      {                                              \
        typedef float16_cpu DType;                   \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_FLOAT:                                 \
      {                                              \
        typedef float DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_FLOAT64:                               \
      {                                              \
        typedef double DType;                        \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_BOOL:                                  \
      {                                              \
        typedef bool DType;                          \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    default:                                         \
      DALI_FAIL("Unknown type");                     \
  }


#define DALI_TYPE_SWITCH(type, DType, ...)           \
  switch (type) {                                    \
    case DALI_NO_TYPE:                               \
      DALI_FAIL("Invalid type.");                    \
    case DALI_UINT8:                                 \
      {                                              \
        typedef uint8 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_INT16:                                 \
      {                                              \
        typedef int16 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_INT32:                                 \
      {                                              \
        typedef int32 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_INT64:                                 \
      {                                              \
        typedef int64 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_FLOAT:                                 \
      {                                              \
        typedef float DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_FLOAT64:                               \
      {                                              \
        typedef double DType;                        \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case DALI_BOOL:                                  \
      {                                              \
        typedef bool DType;                          \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    default:                                         \
      DALI_FAIL("Unknown type");                     \
  }
}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TYPES_H_

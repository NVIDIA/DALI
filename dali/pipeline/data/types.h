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

// workaround missing "is_trivially_copyable" in g++ < 5.0
#if __GNUG__ && __GNUC__ < 5
#include <boost/type_traits/has_trivial_copy.hpp>
#define IS_TRIVIALLY_COPYABLE(T) ::boost::has_trivial_copy<T>::value
#else
#define IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif

#include <cstdint>
#include <cstring>
#include <string>

#include <functional>
#include <mutex>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "dali/common.h"
#include "dali/error_handling.h"

#ifndef DALI_TYPENAME_REGISTERER
#define DALI_TYPENAME_REGISTERER(...)
#endif

#ifndef DALI_TYPEID_REGISTERER
#define DALI_TYPEID_REGISTERER(...)
#endif

namespace dali {


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
  DALI_NPPI_POINT,
  DALI_NPPI_SIZE,
  DALI_NPPI_RECT_SIZE,
  DALI_UINT8_PNTR,
  DALI_UINT32,
  DALI_RESIZE_MAPPING,
  DALI_PIX_MAPPING,
  DALI_DATATYPE_END
};

inline std::string to_string(const DALIDataType& dtype) {
  switch (dtype) {
    case DALI_NO_TYPE:
      return "NO TYPE";
    case DALI_UINT8:
      return "UINT8";
    case DALI_UINT32:
      return "UINT32";
    case DALI_INT16:
      return "INT16";
    case DALI_INT32:
      return "INT32";
    case DALI_INT64:
      return "INT64";
    case DALI_FLOAT16:
      return "FLOAT16";
    case DALI_FLOAT:
      return "FLOAT";
    case DALI_FLOAT64:
      return "FLOAT64";
    case DALI_BOOL:
      return "BOOL";
    case DALI_STRING:
      return "STRING";
    case DALI_NPPI_POINT:
      return "NPPI_POINT";
    case DALI_NPPI_SIZE:
      return "NPPI_SIZE";
    case DALI_NPPI_RECT_SIZE:
      return "NPPI_RECT";
    case DALI_UINT8_PNTR:
      return "UINT8_PNTR";
    case DALI_RESIZE_MAPPING:
      return "RESIZE_MAPPING";
    case DALI_PIX_MAPPING:
      return "PIX_MAPPING";
    default:
      return "<internal>";
  }
}

// Dummy type to represent the invalid default state of dali types.
struct NoType {};

/**
 * @brief Keeps track of mappings between types and unique identifiers.
 */
class TypeTable {
 public:
  template <typename T>
  static DALIDataType GetTypeID() {
    std::lock_guard<std::mutex> lock(mutex_);
    static DALIDataType type_id = TypeTable::RegisterType<T>(static_cast<DALIDataType>(++index_));
    return type_id;
  }

  template <typename T>
  static string GetTypeName() {
    return typeid(T).name();
  }

 private:
  // TypeTable should only be referenced through its static members
  TypeTable();

  // Used by DALI_REGISTER_TYPE macros to register a type in the type table
  template <typename T>
  static DALIDataType RegisterType(DALIDataType dtype) {
    // Lock the mutex to ensure correct setup even if this
    // method is triggered from threads

    // Check the map for this types id
    auto id_it = type_map_.find(typeid(T));

    if (id_it == type_map_.end()) {
      type_map_[typeid(T)] = dtype;
      return dtype;
    } else {
      return id_it->second;
    }
  }

  static std::mutex mutex_;
  static std::unordered_map<std::type_index, DALIDataType> type_map_;
  static int index_;
};

// Stores the unqiue ID for a type and its size in bytes
class TypeInfo {
 public:
  inline TypeInfo() {
    SetType<NoType>();
  }

  template <typename T>
  static inline TypeInfo Create() {
    TypeInfo type;
    type.SetType<T>();
    return type;
  }

  template <typename T>
  inline void SetType() {
    // Note: We enforce the fact that NoType is invalid by
    // explicitly setting its type size as 0
    type_size_ = std::is_same<T, NoType>::value ? 0 : sizeof(T);
    id_ = TypeTable::GetTypeID<T>();
    name_ = TypeTable::GetTypeName<T>();

    // Get constructor/destructor/copier for this type
    constructor_ = std::bind(&TypeInfo::ConstructorFunc<T>,
        this, std::placeholders::_1, std::placeholders::_2);
    destructor_ = std::bind(&TypeInfo::DestructorFunc<T>,
        this, std::placeholders::_1, std::placeholders::_2);
    copier_ = std::bind(&TypeInfo::CopyFunc<T>,
        this, std::placeholders::_1, std::placeholders::_2,
        std::placeholders::_3);
  }

  template <typename Backend>
  void Construct(void *ptr, Index n);

  template <typename Backend>
  void Destruct(void *ptr, Index n);

  template <typename DstBackend, typename SrcBackend>
  void Copy(void *dst, const void *src, Index n, cudaStream_t stream);

  inline DALIDataType id() const {
    return id_;
  }

  inline size_t size() const {
    return type_size_;
  }

  inline string name() const {
    return name_;
  }

  inline bool operator==(const TypeInfo &rhs) const {
    if ((rhs.id_ == id_) &&
        (rhs.type_size_ == type_size_) &&
        (rhs.name_ == name_)) {
      return true;
    }
    return false;
  }

 private:
  template <typename T>
  inline void ConstructorFunc(void *ptr, Index n) {
    T *typed_ptr = static_cast<T*>(ptr);
    for (Index i = 0; i < n; ++i) {
      new (typed_ptr + i) T;
    }
  }

  template <typename T>
  inline void DestructorFunc(void *ptr, Index n) {
    T *typed_ptr = static_cast<T*>(ptr);
    for (Index i = 0; i < n; ++i) {
      typed_ptr[i].~T();
    }
  }

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

  typedef std::function<void (void*, Index)> Constructor;
  typedef std::function<void (void*, Index)> Destructor;
  typedef std::function<void (void *, const void*, Index)> Copier;

  Constructor constructor_;
  Destructor destructor_;
  Copier copier_;

  DALIDataType id_;
  size_t type_size_;
  string name_;
};

/**
 * @brief Utility to check types
 */
template <typename T>
inline bool IsType(TypeInfo type) {
  return type.id() == TypeTable::GetTypeID<T>();
}

/**
 * @brief Utility to check for valid type
 */
inline bool IsValidType(TypeInfo type) {
  return !IsType<NoType>(type);
}

// Used to define a type for use in dali. Inserts the type into the
// TypeTable w/ a unique id and creates a method to get the name of
// the type as a string. This does not work for non-fundamental types,
// as we do not have any mechanism for calling the constructor of the
// type when the buffer allocates the memory.
#define DALI_REGISTER_TYPE(Type, dtype)                           \
  template <> string TypeTable::GetTypeName<Type>()               \
    DALI_TYPENAME_REGISTERER(Type);                               \
  template <> DALIDataType TypeTable::GetTypeID<Type>()           \
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
DALI_REGISTER_TYPE(NppiPoint, DALI_NPPI_POINT);
DALI_REGISTER_TYPE(NppiSize, DALI_NPPI_SIZE);
DALI_REGISTER_TYPE(NppiRect, DALI_NPPI_RECT_SIZE);
DALI_REGISTER_TYPE(uint8 *, DALI_UINT8_PNTR);
DALI_REGISTER_TYPE(std::string, DALI_STRING);

/**
 * @brief Easily instantiate templates for all types
 * type - DALIDataType
 * DType becomes a type corresponding to given DALIDataType
 */
#define DALI_TYPE_SWITCH_WITH_FP16(type, DType, ...) \
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

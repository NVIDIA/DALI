// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <functional>
#include <mutex>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/spinlock.h"
#include "dali/core/float16.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_layout.h"

#ifdef DALI_BUILD_PROTO3
#include "dali/operators/reader/parser/tf_feature.h"
#endif  // DALI_BUILD_PROTO3

#ifndef DALI_TYPENAME_REGISTERER
#define DALI_TYPENAME_REGISTERER(...)
#endif

#ifndef DALI_TYPEID_REGISTERER
#define DALI_TYPEID_REGISTERER(...)
#endif

#ifndef DALI_REGISTER_TYPE_IMPL
#define DALI_REGISTER_TYPE_IMPL(...)
#endif

namespace dali {

class TensorLayout;

namespace detail {

void LaunchCopyKernel(void *dst, const void *src, int64_t nbytes, cudaStream_t stream);

typedef void (*Copier)(void *, const void*, Index);

template <typename T>
inline std::enable_if_t<std::is_trivially_copyable<T>::value>
CopyFunc(void *dst, const void *src, Index n) {
  // T is trivially copyable, we can copy using raw memcopy
  std::memcpy(dst, src, n*sizeof(T));
}

template <typename T>
inline std::enable_if_t<!std::is_trivially_copyable<T>::value>
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
enum DALIDataType : int {
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
#ifdef DALI_BUILD_PROTO3
  DALI_TF_FEATURE        = 17,
  DALI_TF_FEATURE_VEC    = 18,
  DALI_TF_FEATURE_DICT   = 19,
#endif  // DALI_BUILD_PROTO3
  DALI_IMAGE_TYPE        = 20,
  DALI_DATA_TYPE         = 21,
  DALI_INTERP_TYPE       = 22,
  DALI_TENSOR_LAYOUT     = 23,
  DALI_PYTHON_OBJECT     = 24,
  DALI_TENSOR_LAYOUT_VEC = 25,
  DALI_DATA_TYPE_VEC     = 26,
  DALI_DATATYPE_END      = 1000
};

inline std::ostream &operator<<(std::ostream &os, DALIDataType t) {
  switch (t) {
    case DALI_NO_TYPE:
      os << "no_type";
      break;
    case DALI_UINT8:
      os << "uint8";
      break;
    case DALI_UINT16:
      os << "uint16";
      break;
    case DALI_UINT32:
      os << "uint32";
      break;
    case DALI_UINT64:
      os << "uint64";
      break;
    case DALI_INT8:
      os << "int8";
      break;
    case DALI_INT16:
      os << "int16";
      break;
    case DALI_INT32:
      os << "int32";
      break;
    case DALI_INT64:
      os << "int64";
      break;
    case DALI_FLOAT16:
      os << "float16";
      break;
    case DALI_FLOAT:
      os << "float";
      break;
    case DALI_FLOAT64:
      os << "double";
      break;
    case DALI_BOOL:
      os << "bool";
      break;
    case DALI_STRING:
      os << "string";
      break;
    case DALI_BOOL_VEC:
      os << "list of bool";
      break;
    case DALI_INT_VEC:
      os << "list of int";
      break;
    case DALI_STRING_VEC:
      os << "list of string";
      break;
    case DALI_FLOAT_VEC:
      os << "list of float";
      break;
#ifdef DALI_BUILD_PROTO3
    case DALI_TF_FEATURE:
      os << "TFUtil::Feature";
      break;
    case DALI_TF_FEATURE_VEC:
      os << "list of TFUtil::Feature";
      break;
    case DALI_TF_FEATURE_DICT:
      os << "dictionary of TFUtil::Feature";
      break;
#endif  // DALI_BUILD_PROTO3
    case DALI_IMAGE_TYPE:
      os << "DALIImageType";
      break;
    case DALI_DATA_TYPE:
      os << "DALIDataType";
      break;
    case DALI_INTERP_TYPE:
      os << "DALIInterpType";
      break;
    case DALI_TENSOR_LAYOUT:
      os << "TensorLayout";
      break;
    case DALI_PYTHON_OBJECT:
      os << "Python object";
      break;
    case DALI_TENSOR_LAYOUT_VEC:
      os << "list of TensorLayout";
      break;
    case DALI_DATA_TYPE_VEC:
      os << "list of DALIDataType";
    case DALI_DATATYPE_END:  // fall through
    default:
      os << "Unknown type (type id " << static_cast<int>(t) << ")";
      break;
  }
  return os;
}

inline std::string to_string(DALIDataType dtype) {
  std::stringstream ss;
  ss << dtype;
  return ss.str();
}

constexpr bool IsFloatingPoint(DALIDataType type) {
  switch (type) {
    case DALI_FLOAT16:
    case DALI_FLOAT:
    case DALI_FLOAT64:
      return true;
    default:
      return false;
  }
}

constexpr bool IsIntegral(DALIDataType type) {
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

constexpr bool IsSigned(DALIDataType type) {
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

constexpr bool IsUnsigned(DALIDataType type) {
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

template <DALIDataType id>
struct id2type_helper;

/**
 * @brief Compile-time mapping from a type to DALIDataType
 *
 * @note If your compiler complains, that "Use of class template `type2id`
 * requires template arguments", include `static_swtich.h` is your file.
 */
template <typename data_type>
struct type2id;

/**
 * @brief Compile-time mapping from DALIDataType to a type
 */
template <DALIDataType id>
using id2type = typename id2type_helper<id>::type;

#define DALI_STATIC_TYPE_MAPPING(data_type, id)\
template <>\
struct type2id<data_type> : std::integral_constant<DALIDataType, id> {};\
template <>\
struct id2type_helper<id> { using type = data_type; };

// Dummy type to represent the invalid default state of dali types.
struct NoType {};

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

  template <typename T>
  DLL_PUBLIC inline void SetType(DALIDataType dtype = DALI_NO_TYPE);

  /**
   * @brief Copies from SrcBackend memory to DstBackend memory
   * @param dst destination pointer
   * @param src source pointer
   * @param n number of elements to copy
   * @param stream CUDA stream used to perform copy. Only relevant when copying from/to GPUBackend
   * @param use_copy_kernel If true, a copy kernel will be used instead of cudaMemcpyAsync when applicable
   *        (only relevant for device and host pinned memory)
   */
  template <typename DstBackend, typename SrcBackend>
  DLL_PUBLIC void Copy(void *dst, const void *src, Index n, cudaStream_t stream,
                       bool use_copy_kernel = false) const;

  /**
   * @brief Copies from scattered locations from SrcBackend to DstBackend
   * @param dst destination pointers
   * @param srcs source pointers
   * @param sizes number of elements for each of the pointers specified in srcs
   * @param n number of copies to process
   * @param stream CUDA stream used to perform copy. Only relevant when copying from/to GPUBackend
   * @param use_copy_kernel If true, a copy kernel will be used instead of cudaMemcpyAsync when applicable
   *        (only relevant for device and host pinned memory)
   */
  template <typename DstBackend, typename SrcBackend>
  DLL_PUBLIC void Copy(void **dst, const void **srcs, const Index *sizes, int n,
                       cudaStream_t stream, bool use_copy_kernel = false) const;

  /**
   * @brief Copies from SrcBackend scattered locations to a contiguous DstBackend buffer
   * @param dst destination pointer
   * @param srcs source pointers
   * @param sizes number of elements for each of the pointers specified in srcs
   * @param n number of copies to process
   * @param stream CUDA stream used to perform copy. Only relevant when copying from/to GPUBackend
   * @param use_copy_kernel If true, a copy kernel will be used instead of cudaMemcpyAsync when applicable
   *        (only relevant for device and host pinned memory)
   */
  template <typename DstBackend, typename SrcBackend>
  DLL_PUBLIC void Copy(void *dst, const void **srcs, const Index *sizes, int n, cudaStream_t stream,
                       bool use_copy_kernel = false) const;

  /**
   * @brief Copies from SrcBackend contiguous buffer to DstBackend scattered locations
   * @param dsts destination pointers
   * @param src source pointer
   * @param sizes number of elements for each of the pointers specified in dsts
   * @param n number of copies to process
   * @param stream CUDA stream used to perform copy. Only relevant when copying from/to GPUBackend
   * @param use_copy_kernel If true, a copy kernel will be used instead of cudaMemcpyAsync when applicable
   *        (only relevant for device and host pinned memory)
   */
  template <typename DstBackend, typename SrcBackend>
  DLL_PUBLIC void Copy(void **dsts, const void *src, const Index *sizes, int n, cudaStream_t stream,
                       bool use_copy_kernel = false) const;

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
    return rhs.id_ == id_ && rhs.type_size_ == type_size_;
  }

 private:
  detail::Copier copier_;

  DALIDataType id_ = DALI_NO_TYPE;
  size_t type_size_ = 0;
  string name_ = to_string(DALI_NO_TYPE);
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
    auto &inst = instance();
    static DALIDataType type_id = inst.RegisterType<T>(static_cast<DALIDataType>(++inst.index_));
    return type_id;
  }

  template <typename T>
  DLL_PUBLIC static string GetTypeName() {
    return TypeNameHelper<T>::GetTypeName();
  }

  DLL_PUBLIC static const TypeInfo& GetTypeInfo(DALIDataType dtype) {
    auto &inst = instance();
    std::lock_guard<spinlock> guard(inst.lock_);
    auto id_it = inst.type_info_map_.find(dtype);
    DALI_ENFORCE(id_it != inst.type_info_map_.end(),
        make_string("Type with id ", static_cast<int>(dtype), " was not registered."));
    return id_it->second;
  }

  template <typename T>
  DLL_PUBLIC static const TypeInfo& GetTypeInfoFromStatic() {
    return GetTypeInfo(GetTypeID<T>());
  }

 private:
  // TypeTable should only be referenced through its static members
  TypeTable() = default;

  template <typename T>
  DALIDataType RegisterType(DALIDataType dtype) {
    std::lock_guard<spinlock> guard(lock_);
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


  spinlock lock_;
  std::unordered_map<std::type_index, DALIDataType> type_map_;
  // Unordered maps do not work with enums,
  // so we need to use underlying type instead of DALIDataType
  std::unordered_map<std::underlying_type_t<DALIDataType>, TypeInfo> type_info_map_;
  int index_ = DALI_DATATYPE_END;
  DLL_PUBLIC static TypeTable &instance();
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

template <typename... Types>
auto ListTypeNames() {
  return make_string_delim(", ", TypeTable::GetTypeID<Types>()...);
}

template <typename T>
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

/**
 * @brief Utility to check types
 */
template <typename T>
DLL_PUBLIC inline bool IsType(const TypeInfo &type) {
  return type.id() == TypeTable::GetTypeID<T>();
}

/**
 * @brief Utility to check for valid type
 */
DLL_PUBLIC inline bool IsValidType(const TypeInfo &type) {
  return !IsType<NoType>(type);
}

// Used to define a type for use in dali. Inserts the type into the
// TypeTable w/ a unique id and creates a method to get the name of
// the type as a string. This does not work for non-fundamental types,
// as we do not have any mechanism for calling the constructor of the
// type when the buffer allocates the memory.
#define DALI_REGISTER_TYPE(Type, dtype)                             \
  template <> DLL_PUBLIC string TypeTable::GetTypeName<Type>()      \
    DALI_TYPENAME_REGISTERER(Type, dtype);                          \
  template <> DLL_PUBLIC DALIDataType TypeTable::GetTypeID<Type>()  \
    DALI_TYPEID_REGISTERER(Type, dtype);                            \
  DALI_STATIC_TYPE_MAPPING(Type, dtype);                            \
  DALI_REGISTER_TYPE_IMPL(Type, dtype);

// Instantiate some basic types
DALI_REGISTER_TYPE(NoType,         DALI_NO_TYPE);
DALI_REGISTER_TYPE(uint8_t,        DALI_UINT8);
DALI_REGISTER_TYPE(uint16_t,       DALI_UINT16);
DALI_REGISTER_TYPE(uint32_t,       DALI_UINT32);
DALI_REGISTER_TYPE(uint64_t,       DALI_UINT64);
DALI_REGISTER_TYPE(int8_t,         DALI_INT8);
DALI_REGISTER_TYPE(int16_t,        DALI_INT16);
DALI_REGISTER_TYPE(int32_t,        DALI_INT32);
DALI_REGISTER_TYPE(int64_t,        DALI_INT64);
DALI_REGISTER_TYPE(float16,        DALI_FLOAT16);
DALI_REGISTER_TYPE(float,          DALI_FLOAT);
DALI_REGISTER_TYPE(double,         DALI_FLOAT64);
DALI_REGISTER_TYPE(bool,           DALI_BOOL);
DALI_REGISTER_TYPE(string,         DALI_STRING);
DALI_REGISTER_TYPE(DALIImageType,  DALI_IMAGE_TYPE);
DALI_REGISTER_TYPE(DALIDataType,   DALI_DATA_TYPE);
DALI_REGISTER_TYPE(DALIInterpType, DALI_INTERP_TYPE);
DALI_REGISTER_TYPE(TensorLayout,   DALI_TENSOR_LAYOUT);


#ifdef DALI_BUILD_PROTO3
DALI_REGISTER_TYPE(TFUtil::Feature, DALI_TF_FEATURE);
DALI_REGISTER_TYPE(std::vector<TFUtil::Feature>, DALI_TF_FEATURE_VEC);
#endif
DALI_REGISTER_TYPE(std::vector<bool>, DALI_BOOL_VEC);
DALI_REGISTER_TYPE(std::vector<int>, DALI_INT_VEC);
DALI_REGISTER_TYPE(std::vector<std::string>, DALI_STRING_VEC);
DALI_REGISTER_TYPE(std::vector<float>, DALI_FLOAT_VEC);
DALI_REGISTER_TYPE(std::vector<TensorLayout>, DALI_TENSOR_LAYOUT_VEC);
DALI_REGISTER_TYPE(std::vector<DALIDataType>, DALI_DATA_TYPE_VEC);

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
      DALI_FAIL(make_string("Unknown type: ", type)); \
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
      DALI_FAIL(make_string("Unknown type: ", type)); \
  }
}  // namespace dali

#endif  // DALI_PIPELINE_DATA_TYPES_H_

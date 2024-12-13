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

#ifndef DALI_PIPELINE_DATA_TYPES_H_
#define DALI_PIPELINE_DATA_TYPES_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include "dali/core/dali_data_type.h"
#include "dali/core/util.h"
#include "dali/core/common.h"
#include "dali/core/spinlock.h"
#include "dali/core/float16.h"
#include "dali/core/cuda_error.h"
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

inline std::string to_string(daliDataType_t dtype);
inline std::ostream &operator<<(std::ostream &, daliDataType_t dtype);

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

using DALIDataType = daliDataType_t;

constexpr auto GetBuiltinTypeName = daliDataTypeName;
constexpr auto IsFloatingPoint = daliDataTypeIsFloatingPoint;
constexpr auto IsIntegral = daliDataTypeIsIntegral;
constexpr auto IsSigned = daliDataTypeIsSigned;
constexpr auto IsUnsigned = daliDataTypeIsUnsigned;
constexpr auto IsEnum = daliDataTypeIsEnum;


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
  DLL_PUBLIC inline TypeInfo() = default;

  // Workaround for a clang bug, showing up in clang-only builds
  DLL_PUBLIC inline ~TypeInfo() {}

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

  DLL_PUBLIC inline std::string_view name() const {
    return name_;
  }

  DLL_PUBLIC inline bool operator==(const TypeInfo &rhs) const {
    return rhs.id_ == id_ && rhs.type_size_ == type_size_;
  }

 private:
  detail::Copier copier_ = nullptr;

  DALIDataType id_ = DALI_NO_TYPE;
  size_t type_size_ = 0;
  std::string_view name_ = GetBuiltinTypeName(DALI_NO_TYPE);
};

template <typename T>
struct TypeNameHelper {
  static std::string_view GetTypeName() {
    return typeid(T).name();
  }
};

/**
 * @brief Keeps track of mappings between types and unique identifiers.
 */
class DLL_PUBLIC TypeTable {
 public:
  template <typename T>
  DLL_PUBLIC static DALIDataType GetTypeId() {
    static DALIDataType type_id = instance().RegisterType<T>(
        static_cast<DALIDataType>(instance().next_id_++));
    return type_id;
  }

  template <typename T>
  DLL_PUBLIC static std::string_view GetTypeName() {
    return TypeNameHelper<T>::GetTypeName();
  }

  DLL_PUBLIC static const TypeInfo *TryGetTypeInfo(DALIDataType dtype) {
    auto *types = instance().type_info_map_;
    assert(types);
    size_t idx = dtype - DALI_NO_TYPE;
    if (idx >= types->size())
      return nullptr;
    return (*types)[idx];
  }

  DLL_PUBLIC static const TypeInfo &GetTypeInfo(DALIDataType dtype) {
    auto *info = TryGetTypeInfo(dtype);
    DALI_ENFORCE(info != nullptr,
        make_string("Type with id ", static_cast<int>(dtype), " was not registered."));
    return *info;
  }

  template <typename T>
  DLL_PUBLIC static const TypeInfo &GetTypeInfo() {
    static const TypeInfo &type_info = GetTypeInfo(GetTypeId<T>());
    return type_info;
  }

 private:
  // TypeTable should only be referenced through its static members
  TypeTable() = default;

  template <typename T>
  DALIDataType RegisterType(DALIDataType dtype) {
    static DALIDataType id = [dtype, this]() {
      std::lock_guard guard(insert_lock_);
      size_t idx = dtype - DALI_NO_TYPE;
      // We need the map because this function (and the static variable) may be instantiated
      // in multiple shared objects whereas the map instance is tied to one well defined
      // instance of the TypeTable returned by `instance()`.
      auto [it, inserted] = type_map_.emplace(typeid(T), dtype);
      if (!inserted)
        return it->second;
      if (!type_info_map_ || idx >= type_info_map_->size()) {
        constexpr size_t kMinCapacity = next_pow2(DALI_CUSTOM_TYPE_START + 100);
        // we don't need to look at the previous capacity to achieve std::vector-like growth
        size_t capacity = next_pow2(idx + 1);
        if (capacity < kMinCapacity)
          capacity = kMinCapacity;
        auto &m = type_info_maps_.emplace_back();
        m.resize(capacity);
        if (type_info_map_)  // copy the old map into the new one
          std::copy(type_info_map_->begin(), type_info_map_->end(), m.begin());
        // The new map contains everything that the old map did - we can "publish" it.
        // Make sure that the compiler doesn't reorder after the "publishing".
        std::atomic_thread_fence(std::memory_order_release);
        // Publish the new map.
        type_info_map_ = &m;
      }
      TypeInfo &info = type_infos_.emplace_back();
      info.SetType<T>(dtype);
      if ((*type_info_map_)[idx] != nullptr)
        DALI_FAIL("The type id ", idx, " is already taken by type ",
                  (*type_info_map_)[idx]->name());
      (*type_info_map_)[idx] = &info;

      return dtype;
    }();
    return id;
  }

  using TypeInfoMap = std::vector<TypeInfo*>;
  // The "current" type map - it's just a vector that maps type_id (adjusted and treated as index)
  // to a TypeInfo pointer.
  TypeInfoMap *type_info_map_ = nullptr;

  std::mutex insert_lock_;
  // All type info maps - old ones are never deleted to avoid locks when only read access is needed.
  std::list<TypeInfoMap> type_info_maps_;
  // The actual type info objects. Each type has exactly one TypeInfo - even if we need to grow
  // the storage - hence, we need to store TypeInfo* in the pas (see typedef TypeInfoMap) and
  // we need to store TypeInfo instances in a container that never invalidates pointers
  // (e.g. a list).
  std::list<TypeInfo> type_infos_;
  // This is necessary because it turns out that static field in RegisterType has many instances
  // in a program built with multiple shared libraries.
  std::unordered_map<std::type_index, DALIDataType> type_map_;

  int next_id_ = DALI_CUSTOM_TYPE_START;
  DLL_PUBLIC static TypeTable &instance();
};

template <typename T, typename A>
struct TypeNameHelper<std::vector<T, A> > {
  static std::string_view GetTypeName() {
    static const std::string name = "list of " + std::string(TypeTable::GetTypeName<T>());
    return name;
  }
};

template <typename T, size_t N>
struct TypeNameHelper<std::array<T, N> > {
  static std::string_view GetTypeName() {
    static const std::string name = "list of " + std::string(TypeTable::GetTypeName<T>());
    return name;
  }
};

template <typename... Types>
auto ListTypeNames() {
  return make_string_delim(", ", TypeTable::GetTypeId<Types>()...);
}

template <typename T>
void TypeInfo::SetType(DALIDataType dtype) {
  // Note: We enforce the fact that NoType is invalid by
  // explicitly setting its type size as 0
  constexpr bool is_no_type = std::is_same_v<T, NoType>;
  type_size_ = is_no_type ? 0 : sizeof(T);
  if constexpr (!is_no_type) {
    id_ = dtype != DALI_NO_TYPE ? dtype : TypeTable::GetTypeId<T>();
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
  return type.id() == TypeTable::GetTypeId<T>();
}

/**
 * @brief Utility to check types
 */
template <typename T>
DLL_PUBLIC inline bool IsType(DALIDataType id) {
  return id == TypeTable::GetTypeId<T>();
}

/**
 * @brief Utility to check for valid type
 */
DLL_PUBLIC inline bool IsValidType(DALIDataType type) {
  return type != DALI_NO_TYPE;
}

/**
 * @brief Utility to check for valid type
 */
DLL_PUBLIC inline bool IsValidType(const TypeInfo &type) {
  return !IsType<NoType>(type);
}

inline std::string_view TypeName(DALIDataType dtype) {
  if (const char *builtin = GetBuiltinTypeName(dtype))
    return builtin;
  auto *info = TypeTable::TryGetTypeInfo(dtype);
  if (info)
    return info->name();
  return "<unknown>";
}

// Used to define a type for use in dali. Inserts the type into the
// TypeTable w/ a unique id and creates a method to get the name of
// the type as a string. This does not work for non-fundamental types,
// as we do not have any mechanism for calling the constructor of the
// type when the buffer allocates the memory.
#define DALI_REGISTER_TYPE(Type, dtype)                                   \
  template <> DLL_PUBLIC std::string_view TypeTable::GetTypeName<Type>()  \
    DALI_TYPENAME_REGISTERER(Type, dtype);                                \
  template <> DLL_PUBLIC DALIDataType TypeTable::GetTypeId<Type>()        \
    DALI_TYPEID_REGISTERER(Type, dtype);                                  \
  DALI_STATIC_TYPE_MAPPING(Type, dtype);                                  \
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

#define DALI_INTEGRAL_TYPES uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t
#define DALI_NUMERIC_TYPES DALI_INTEGRAL_TYPES, float, double
#define DALI_NUMERIC_TYPES_FP16 DALI_NUMERIC_TYPES, float16

}  // namespace dali

inline std::string to_string(daliDataType_t dtype) {
  std::string_view name = dali::TypeName(dtype);
  if (name == "<unknown>")
    return "unknown type: " + std::to_string(static_cast<int>(dtype));
  else
    return std::string(name);
}

inline std::ostream &operator<<(std::ostream &os, daliDataType_t dtype) {
  std::string_view name = dali::TypeName(dtype);
  if (name == "<unknown>") {
    // Use string concatenation so that the result is the same as in to_string, unaffected by
    // formatting & other settings in `os`.
    return os << ("unknown type: " + std::to_string(static_cast<int>(dtype)));
  } else {
    return os << name;
  }
}


#endif  // DALI_PIPELINE_DATA_TYPES_H_

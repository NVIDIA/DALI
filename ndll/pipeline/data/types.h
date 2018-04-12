// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_DATA_TYPES_H_
#define NDLL_PIPELINE_DATA_TYPES_H_

#include <cstdint>
#include <cstring>
#include <string>

#include <functional>
#include <mutex>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

/**
 * @brief Enum identifiers for the different data types that
 * the pipeline can output.
 */
enum NDLLDataType {
  NDLL_NO_TYPE = -1,
  NDLL_UINT8 = 0,
  NDLL_INT16 = 1,
  NDLL_INT32 = 2,
  NDLL_INT64 = 3,
  NDLL_FLOAT16 = 4,
  NDLL_FLOAT = 5,
  NDLL_FLOAT64 = 6,
  NDLL_BOOL = 7,
  // Internal types
  NDLL_INTERNAL_C_UINT8_P = 1000,
  NDLL_INTERNAL_PARSEDJPEG = 1001,
  NDLL_INTERNAL_DCTQUANTINV_IMAGE_PARAM = 1002,
  NDLL_INTERNAL_TEST_TYPE = 1003,
  NDLL_INTERNAL_TEST_TYPE_2 = 1004,
};

inline std::string to_string(const NDLLDataType& dtype) {
  switch (dtype) {
    case NDLL_NO_TYPE:
      return "NO TYPE";
    case NDLL_UINT8:
      return "UINT8";
    case NDLL_INT16:
      return "INT16";
    case NDLL_INT32:
      return "INT32";
    case NDLL_INT64:
      return "INT64";
    case NDLL_FLOAT16:
      return "FLOAT16";
    case NDLL_FLOAT:
      return "FLOAT";
    case NDLL_FLOAT64:
      return "FLOAT64";
    case NDLL_BOOL:
      return "BOOL";
    case NDLL_INTERNAL_C_UINT8_P:
    case NDLL_INTERNAL_PARSEDJPEG:
    case NDLL_INTERNAL_DCTQUANTINV_IMAGE_PARAM:
    case NDLL_INTERNAL_TEST_TYPE:
    case NDLL_INTERNAL_TEST_TYPE_2:
      return "<internal>";
  }

  NDLL_FAIL("Unknown datatype");
}

// Dummy type to represent the invalid default state of ndll types.
struct NoType {};

/**
 * @brief Keeps track of mappings between types and unique identifiers.
 */
class TypeTable {
 public:
  template <typename T>
  static NDLLDataType GetTypeID();

  template <typename T>
  static string GetTypeName();

 private:
  // TypeTable should only be referenced through its static members
  TypeTable();

  // Used by NDLL_REGISTER_TYPE macros to register a type in the type table
  template <typename T>
  static NDLLDataType RegisterType(NDLLDataType dtype) {
    // Lock the mutex to ensure correct setup even if this
    // method is triggered from threads
    std::lock_guard<std::mutex> lock(mutex_);

    // Check the map for this types id
    auto id_it = type_map_.find(typeid(T));

    // This method should only be called once per type. It shouldn't
    // even be possible to call this twice without compiler errors
    // because it is only called from the explicit specialization of
    // GetIDForType().
    NDLL_ENFORCE(id_it == type_map_.end(),
        "Re-registration of type, check for duplicate NDLL_REGISTER_TYPE calls");

    type_map_[typeid(T)] = dtype;
    return dtype;
  }

  static std::mutex mutex_;
  static std::unordered_map<std::type_index, NDLLDataType> type_map_;
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

  inline NDLLDataType id() const {
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
  inline typename std::enable_if<std::is_trivially_copyable<T>::value>::type
  CopyFunc(void *dst, const void *src, Index n) {
    // T is trivially copyable, we can copy using raw memcopy
    std::memcpy(dst, src, n*sizeof(T));
  }

  template <typename T>
  inline typename std::enable_if<!std::is_trivially_copyable<T>::value>::type
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

  NDLLDataType id_;
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

// Used to define a type for use in ndll. Inserts the type into the
// TypeTable w/ a unique id and creates a method to get the name of
// the type as a string. This does not work for non-fundamental types,
// as we do not have any mechanism for calling the constructor of the
// type when the buffer allocates the memory.
#define NDLL_REGISTER_TYPE(Type, dtype)                           \
  template <> string TypeTable::GetTypeName<Type>() {             \
    return #Type;                                                 \
  }                                                               \
  template <> NDLLDataType TypeTable::GetTypeID<Type>() {               \
    static NDLLDataType type_id = TypeTable::RegisterType<Type>(dtype);      \
    return type_id;                                               \
  }


/**
 * @brief Easily instantiate templates for all types
 * type - NDLLDataType
 * DType becomes a type corresponding to given NDLLDataType
 */
#define NDLL_TYPE_SWITCH_WITH_FP16(type, DType, ...) \
  switch (type) {                                     \
    case NDLL_NO_TYPE:                               \
      NDLL_FAIL("Invalid type.");                    \
    case NDLL_UINT8:                                 \
      {                                              \
        typedef uint8 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_INT16:                                 \
      {                                              \
        typedef int16 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_INT32:                                 \
      {                                              \
        typedef int32 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_INT64:                                 \
      {                                              \
        typedef int64 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_FLOAT16:                               \
      {                                              \
        typedef float16 DType;                       \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_FLOAT:                                 \
      {                                              \
        typedef float DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_FLOAT64:                               \
      {                                              \
        typedef double DType;                        \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_BOOL:                                  \
      {                                              \
        typedef bool DType;                          \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    default:                                         \
      NDLL_FAIL("Unknown type");                     \
  }

#define NDLL_TYPE_SWITCH(type, DType, ...)           \
  switch (type) {                                     \
    case NDLL_NO_TYPE:                               \
      NDLL_FAIL("Invalid type.");                    \
    case NDLL_UINT8:                                 \
      {                                              \
        typedef uint8 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_INT16:                                 \
      {                                              \
        typedef int16 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_INT32:                                 \
      {                                              \
        typedef int32 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_INT64:                                 \
      {                                              \
        typedef int64 DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_FLOAT:                                 \
      {                                              \
        typedef float DType;                         \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_FLOAT64:                               \
      {                                              \
        typedef double DType;                        \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    case NDLL_BOOL:                                  \
      {                                              \
        typedef bool DType;                          \
        {__VA_ARGS__}                                \
      }                                              \
      break;                                         \
    default:                                         \
      NDLL_FAIL("Unknown type");                     \
  }
}  // namespace ndll

#endif  // NDLL_PIPELINE_DATA_TYPES_H_

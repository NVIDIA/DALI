#ifndef NDLL_PIPELINE_DATA_TYPES_H_
#define NDLL_PIPELINE_DATA_TYPES_H_

#include <cstdint>

#include <mutex>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

typedef int64_t TypeID;

// Dummy type to represent the invalid default state of ndll types.
struct NoType {};

/**
 * @brief Keeps track of mappings between types and unique identifiers.
 */
class TypeTable {
public:
  
  template <typename T>
  static TypeID GetTypeID();
  
  template <typename T>
  static string GetTypeName();

private:
  // TypeTable should only be referenced through its static members
  TypeTable();

  // Used by NDLL_REGISTER_TYPE macros to register a type in the type table
  template <typename T>
  static TypeID RegisterType() {
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

    int new_id = id_;
    type_map_[typeid(T)] = new_id;
    ++id_;
    return new_id;
  }

  static std::mutex mutex_;
  static int id_;
  static std::unordered_map<std::type_index, TypeID> type_map_;
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
    id_ = TypeTable::GetTypeID<T>();
    type_size_ = sizeof(T);
    name_ = TypeTable::GetTypeName<T>();
  }

  inline TypeID id() const {
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
  TypeID id_;
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
#define NDLL_REGISTER_TYPE(Type)                                  \
  template <> string TypeTable::GetTypeName<Type>() {             \
    return #Type;                                                 \
  }                                                               \
  template <> TypeID TypeTable::GetTypeID<Type>() {               \
    static TypeID type_id = TypeTable::RegisterType<Type>();      \
    return type_id;                                               \
  }

/**
 * @brief Enum identifiers for the different data types that
 * the pipeline can output. Only exists to simplify users 
 * interaction with the type system, e.g. they don't have to
 * call TypeTable::GetTypeID<type>() to give an operator its
 * output data type
 */
enum NDLLDataType {
  NDLL_NO_TYPE = -1,
  NDLL_UINT8 = 0,
  NDLL_FLOAT16 = 1,
  NDLL_FLOAT = 2,
};

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_TYPES_H_

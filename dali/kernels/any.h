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

#ifndef DALI_KERNELS_ANY_H_
#define DALI_KERNELS_ANY_H_

#include <exception>
#include <utility>
#include <type_traits>
#include <typeinfo>

namespace dali {

namespace detail {


struct alignas(8)
any_placeholder {
  constexpr any_placeholder() = default;
  char data[8] = {};
};

struct any_helper_base {
  virtual void destroy(any_placeholder *placeholder) const = 0;
  virtual void free(any_placeholder *placeholder) const = 0;
  virtual void *get_void(any_placeholder *placeholder) const noexcept = 0;
  virtual const void *get_void(const any_placeholder *placeholder) const noexcept = 0;
  virtual void clone(any_placeholder *dest, const any_placeholder *src) const = 0;
  virtual void placement_clone(any_placeholder *dest, const any_placeholder *src) const = 0;
  virtual bool is_same_ex(const any_helper_base *other) const = 0;

  inline bool is_same(const any_helper_base *h) const {
    return h && (this == h || is_same_ex(h));
  }

  template <typename T>
  T *get(any_placeholder *p) const noexcept {
    return reinterpret_cast<T*>(get_void(p));
  }
  template <typename T>
  const T *get(const any_placeholder *p) const noexcept {
    return reinterpret_cast<const T*>(get_void(p));
  }
};

template <typename T, bool dynamic =
  (sizeof(T) > sizeof(any_placeholder) ||
  !std::is_pod<T>::value ||  // this could be weaker?
  alignof(T) > alignof(any_placeholder))>
struct any_helper : any_helper_base {
  template <typename... Args>
  static void create(any_placeholder *placeholder, Args&&... args) {
    new(placeholder) T(std::forward<Args>(args)...);
  }

  void destroy(any_placeholder *placeholder) const override {
    reinterpret_cast<T*>(placeholder)->~T();
  }
  void free(any_placeholder *placeholder) const override {
    reinterpret_cast<T*>(placeholder)->~T();
  }

  void *get_void(any_placeholder *p) const noexcept override {
    return reinterpret_cast<void*>(p);
  }

  const void *get_void(const any_placeholder *p) const noexcept override {
    return reinterpret_cast<const void*>(p);
  }

  void clone(any_placeholder *dest, const any_placeholder *src) const override {
    create(dest, *get<T>(src));
  }

  void placement_clone(any_placeholder *dest, const any_placeholder *src) const override {
    create(dest, *get<T>(src));
  }

  bool is_same_ex(const any_helper_base *other) const override {
    return dynamic_cast<const any_helper *>(other);
  }

  static any_helper instance;
};

template <typename T>
struct any_helper<T, true> : any_helper_base {
  template <typename... Args>
  static void create(any_placeholder *placeholder, Args&&... args) {
    *reinterpret_cast<T**>(placeholder) = new T(std::forward<Args>(args)...);
  }

  void destroy(any_placeholder *placeholder) const override {
    get<T>(placeholder)->~T();
  }

  void free(any_placeholder *placeholder) const override {
    T **pptr = reinterpret_cast<T**>(placeholder);
    delete *pptr;
    *pptr = nullptr;
  }

  void *get_void(any_placeholder *p) const noexcept override {
    return *reinterpret_cast<void**>(p);
  }

  const void *get_void(const any_placeholder *p) const noexcept override {
    return *reinterpret_cast<const void*const*>(p);
  }

  void clone(any_placeholder *dest, const any_placeholder *src) const override {
    create(dest, *get<T>(src));
  }

  void placement_clone(any_placeholder *dest, const any_placeholder *src) const override {
    new (get<T>(dest)) T(*get<T>(src));
  }

  bool is_same_ex(const any_helper_base *other) const override {
    return dynamic_cast<const any_helper *>(other);
  }

  static any_helper instance;
};

template <typename T, bool d>
any_helper<T, d> any_helper<T, d>::instance;

template <typename T>
any_helper<T, true> any_helper<T, true>::instance;

}  // namespace detail

class any {
 public:
  ~any() {
    reset();
  }

  any(any &other) {
    *this = other;
  }
  any(const any &other) {
    *this = other;
  }
  any(any &&other) noexcept {
    helper = other.helper;
    storage = other.storage;
    other.helper = nullptr;
    other.storage = storage;
  }

  constexpr any() noexcept = default;

  template <typename T>
  any(const T &value) {  // NOLINT
    assign<T>(value);
  }

  template <typename T>
  any(T &&value) {  // NOLINT
    static_assert(std::is_copy_constructible<T>::value,
      "only copy-constructible types can be stored in 'any'");
    static_assert(std::is_destructible<T>::value,
      "objects stored in 'any' must be destructible");
    assign<T>(std::forward<T>(value));
  }

  void reset() {
    if (helper) {
      helper->free(&storage);
      helper = nullptr;
    }
  }

  bool has_value() const noexcept { return helper != nullptr; }

  void swap(any &other) noexcept {
    std::swap(helper, other.helper);
    std::swap(storage, other.storage);
  }

  template <typename T, typename... Args>
  void emplace(Args&&... args) {
    static_assert(std::is_copy_constructible<T>::value,
      "only copy-constructible types can be stored in 'any'");
    static_assert(std::is_destructible<T>::value,
      "objects stored in 'any' must be destructible");
    if (is_local_type<T>()) {
      T *ptr = get<T>();
      ptr->~T();
      new (ptr) T(std::forward<Args>(args)...);
    } else {
      reset();
      assign<T>(std::forward<Args>(args)...);
    }
  }

  template <typename T>
  any &operator=(T &&value) {
    using U = typename std::remove_reference<T>::type;
    emplace<U>(std::forward<T>(value));
    return *this;
  }

  any &operator=(const any &other) {
    if (!other.has_value()) {
      reset();
    } else if (helper == other.helper) {
      // This code variant is used if and only if the pre-existing value was of exactle the same
      // type and allocated by the same module. If the type is RTTI-compatible, but the helper
      // address differs, use the ordinary free/allocate code path to avoid potential problems
      // with memory management.
      helper->destroy(&storage);
      helper->placement_clone(&storage, &other.storage);
    } else {
      if (helper)
        helper->free(&storage);
      helper = other.helper;
      helper->clone(&storage, &other.storage);
    }
    return *this;
  }

  any &operator=(any &&other) {
    swap(other);
    other.reset();
    return *this;
  }

  any &operator=(any &other) {
    *this = const_cast<const any &>(other);
    return *this;
  }

 private:
  template <typename T, typename... Args>
  void assign(Args&&... args) {
    helper = &detail::any_helper<T>::instance;
    detail::any_helper<T>::create(&storage, std::forward<Args>(args)...);
  }

  template <typename T>
  friend struct any_cast_helper;

  /// @brief True, if contained type is T, regardless of which module it comes from
  template <typename T>
  constexpr bool is_type() const { return detail::any_helper<T>::instance.is_same(helper); }

  /// @brief True, if contained type is exactly T and defined in the same module as the caller
  template <typename T>
  constexpr bool is_local_type() const { return &detail::any_helper<T>::instance == helper; }
  detail::any_helper_base *helper = nullptr;
  detail::any_placeholder storage;

  template <typename T>
  T *get() noexcept {
    return helper->get<T>(&storage);
  }

  template <typename T>
  const T *get() const noexcept {
    return helper->get<T>(&storage);
  }
};

struct bad_any_cast : std::bad_cast {
  const char *what() const noexcept override {
    return "bad_any_cast";
  }
};

template <typename T>
struct any_cast_helper {
  static T get(any &a) {
    if (!a.is_type<T>())
      throw bad_any_cast();
    return *a.get<T>();
  }

  static T get(const any &a) {
    if (!a.is_type<T>())
      throw bad_any_cast();
    return *a.get<T>();
  }

  static T *get(any *a) {
    if (!a->is_type<T>())
      return nullptr;
    return a->get<T>();
  }
};

template <typename T>
struct any_cast_helper<const T> {
  static const T get(any &a) {
    return any_cast_helper<T>::get(a);
  }
  static const T get(const any &a) {
    return any_cast_helper<T>::get(a);
  }

  static const T *get(const any *a) {
    if (!a->is_type<T>())
      return nullptr;
    return a->get<T>();
  }
};

template <typename T>
struct any_cast_helper<T &> {
  static T &get(any &a) {
    if (!a.is_type<T>())
      throw bad_any_cast();
    return *a.get<T>();
  }
};

template <typename T>
struct any_cast_helper<const T &> {
  static const T &get(const any &a) {
    if (!a.is_type<T>())
      throw bad_any_cast();
    return *a.get<T>();
  }
};


template <typename T>
struct any_cast_helper<T &&> {
  static T &&get(any &&a) {
    if (!a.is_type<T>())
      throw bad_any_cast();
    return std::move(*a.get<T>());
  }
};

template <typename T>
T any_cast(any &a) {
  return any_cast_helper<T>::get(a);
}

template <typename T>
T any_cast(any &&a) {
  return any_cast_helper<T>::get(std::move(a));
}

template <typename T>
T &any_cast(const any &a) {
  return any_cast_helper<T>::get(a);
}


template <typename T>
T *any_cast(any *a) {
  return any_cast_helper<T>::get(a);
}

template <typename T>
const T *any_cast(const any *a) {
  return any_cast_helper<const T>::get(a);
}

// based on example reference implementation
template <typename T, typename... Args>
any make_any(Args&&... args) {
  any a;
  a.emplace<T>(std::forward<Args>(args)...);
  return a;
}

// based on example reference implementation
template <typename T, typename U, typename... Args>
any make_any(std::initializer_list<U> il, Args&&... args) {
  any a;
  a.emplace<T>(il, std::forward<Args>(args)...);
  return a;
}

}  // namespace dali

#endif  // DALI_KERNELS_ANY_H_

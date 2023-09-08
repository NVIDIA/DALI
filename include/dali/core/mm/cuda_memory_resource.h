// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DALI_CORE_MM_CUDA_MEMORY_RESOURCE_H_
#define DALI_CORE_MM_CUDA_MEMORY_RESOURCE_H_


#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
// #include <version>  // C++20
#include "dali/core/mm/cuda_stream_view.h"

#define _DALI_STD_VER 17

// DALI is built with exceptions
#define _DALI_EXT_RTTI_ENABLED

#ifdef _DALI_EXT_RTTI_ENABLED
#include <typeinfo>
#endif

#if __has_include(<memory_resource>)
#include <memory_resource>
#define _DALI_STD_PMR_NS ::std::pmr
#elif __has_include(<experimental/memory_resource>)
#include <experimental/memory_resource>
#define _DALI_STD_PMR_NS ::std::experimental::pmr
#endif  // __has_include(<experimental/memory_resource>)

// no-op-define,
#define _DALI_TEMPLATE_VIS
#define _DALI_INLINE_VISIBILITY __host__ __device__

namespace dali {
namespace cuda_for_dali {

/*!
 * \brief Groups the tag types denoting the kind of memory of an allocation.
 *
 * Memory allocation kind determines where memory can be accessed and the
 * performance characteristics of accesses.
 *
 * This is not a closed set, the user code can define custom memory kinds.
 */
namespace memory_kind {
/*!
 * \brief Ordinary host memory
 */
struct host;

/*!
 * \brief Device memory, as allocated by cudaMalloc.
 */
struct device;

/*!
 * \brief Device-accessible host memory.
 */
struct pinned;

/*!
 * \brief Virtual memory that is automatically migrated between the host and devices.
 */
struct managed;
};  // namespace memory_kind

namespace detail {

template <typename...>
struct __type_pack {};

namespace __fallback_typeid {

template <class _Tp>
struct _DALI_TEMPLATE_VIS __unique_typeinfo {
  static constexpr int __id = 0;
};
template <class _Tp>
constexpr int __unique_typeinfo<_Tp>::__id;

template <class _Tp>
inline _DALI_INLINE_VISIBILITY constexpr const void *__get_fallback_typeid() {
  return &__unique_typeinfo<std::decay_t<_Tp>>::__id;
}

template <typename _Tp>
const ::std::type_info *__get_typeid() {
#ifdef _DALI_EXT_RTTI_ENABLED
  return &typeid(_Tp);
#else
  return nullptr;
#endif
}

inline bool __compare_type(const ::std::type_info *__ti1, const void *__fallback_ti1,
                           const ::std::type_info *__ti2, const void *__fallback_ti2) {
#ifdef _DALI_EXT_RTTI_ENABLED
  if (__ti1 && __ti2 && *__ti1 == *__ti2)
    return true;
#endif
  return __fallback_ti1 == __fallback_ti2;
}

template <typename _Tp>
bool __is_type(const ::std::type_info *__ti1, const void *__fallback_ti1) {
  return __compare_type(__ti1, __fallback_ti1, __get_typeid<_Tp>(), __get_fallback_typeid<_Tp>());
}

}  // namespace __fallback_typeid

}  // namespace detail

template <typename _ResourcePointer, typename... _Properties>
class basic_resource_view;

/*!
 * \brief Groups the tag types denoting the execution environment in which the memory can be
 * accessed
 *
 * This is not a closed set, the user code can define custom accessibility.
 */
namespace memory_access {
struct host;
struct device;
}  // namespace memory_access

/*!
 * \brief A memory property tag type indicating that the memory can be oversubscribed.
 *
 * Oversubscribable memory doesn't need to have backing physical storage at all times.
 */
struct oversubscribable;

/*!
 * \brief A memory property tag type indicating that the memory has a backing physical
 *        storage in the target location at all times.
 */
struct resident;

/*!
 * \brief Groups the tag types that denote the actual location of the physical storage
 *
 * Memory kinds which can be migrated between locations can define multiple locations.
 */
namespace memory_location {
/*!
 * \brief A memory property tag type indicating that the memory is located on a device
 */
struct device;

/*!
 * \brief A memory property tag type indicating that the memory is located in the host memory
 */
struct host;
}  // namespace memory_location

template <typename _MemoryKind>
class memory_resource;

namespace detail {

class memory_resource_base {
 public:
  static constexpr std::size_t default_alignment = alignof(std::max_align_t);

  /*!
   * \brief Allocates storage of size at least `__bytes` bytes.
   *
   * The returned storage is aligned to the specified `__alignment` if such
   * alignment is supported. Otherwise throws.
   *
   * Storage may be accessed immediately within the execution contexts that
   * can access the memory.
   *
   * \throws If storage of the requested size and alignment cannot be obtained.
   *
   * \param __bytes The size in bytes of the allocation
   * \param __alignment The alignment of the allocation
   * \return Pointer to the requested storage
   */
  void *allocate(size_t __bytes, size_t __alignment = default_alignment) {
    return do_allocate(__bytes, __alignment);
  }

  /*!
   * \brief Deallocates the storage pointed to by `__p`.
   *
   * `__p` must have been returned by a prior call to `allocate(__bytes,
   * __alignment)` on a `memory_resource` that compares equal to `*this`, and
   * the storage it points to must not yet have been deallocated, otherwise
   * behavior is undefined.
   *
   * \throws Nothing.
   *
   * \param __p Pointer to storage to be deallocated
   * \param __bytes The size in bytes of the allocation. This must be equal to
   * the value of `__bytes` that was specified to the `allocate` call that
   * returned `__p`.
   * \param __alignment The alignment of the allocation. This must be equal to
   * the value of `__alignment` that was specified to the `allocate` call that
   * returned `__p`.
   */
  void deallocate(void *__mem, size_t __bytes, size_t __alignment = default_alignment) {
    do_deallocate(__mem, __bytes, __alignment);
  }

  /*!
   * \brief Tries to cast the resource to a resource of given kind
   */
  template <typename _Kind>
  memory_resource<_Kind> *as_kind() noexcept {
    using __tag = detail::__type_pack<_Kind>;
    return static_cast<memory_resource<_Kind> *>(
        __do_as_kind(detail::__fallback_typeid::__get_typeid<__tag>(),
                     detail::__fallback_typeid::__get_fallback_typeid<__tag>()));
  }

  /*!
   * \brief Tries to cast the resource to a resource of given kind
   */
  template <typename _Kind>
  const memory_resource<_Kind> *as_kind() const noexcept {
    using __tag = detail::__type_pack<_Kind>;
    return static_cast<const memory_resource<_Kind> *>(
        __do_as_kind(detail::__fallback_typeid::__get_typeid<__tag>(),
                     detail::__fallback_typeid::__get_fallback_typeid<__tag>()));
  }

 protected:
  virtual void *do_allocate(size_t __bytes, size_t __alignment) = 0;
  virtual void do_deallocate(void *__mem, size_t __bytes, size_t __alignment) = 0;

  virtual bool is_equal_base(const memory_resource_base &other) const noexcept = 0;

  bool is_equal(const memory_resource_base &other) const noexcept {
    return is_equal_base(other);
  }

  template <typename _ResourcePointer, typename... _Properties>
  friend class cuda_for_dali::basic_resource_view;

  virtual void *__do_as_kind(const ::std::type_info *__tag_type_id,
                             const void *__tag_type_fallback_id) const noexcept = 0;
};

class stream_ordered_memory_resource_base : public virtual memory_resource_base {
 public:
  /*!
   * \brief Allocates storage of size at least `__bytes` bytes in stream order
   * on `__stream`.
   *
   * The returned storage is aligned to `default_alignment`.
   *
   * The returned storage may be used immediately only on `__stream`. Accessing
   * it on any other stream (or the host) requires first synchronizing with
   * `__stream`.
   *
   * \throws If the storage of the requested size and `default_alignment` cannot
   * be obtained.
   *
   * \param __bytes The size in bytes of the allocation.
   * \param __stream The stream on which to perform the allocation.
   * \return Pointer to the requested storage.
   */
  void *allocate_async(size_t bytes, stream_view stream) {
    return allocate_async(bytes, default_alignment, stream);
  }
  /*!
   * \brief Allocates storage of size at least `__bytes` bytes in stream order
   * on `__stream`.
   *
   * The returned storage is aligned to the specified `__alignment` if such
   * alignment is supported.
   *
   * The returned storage may be used immediately only on `__stream`. Using it
   * on any other stream (or the host) requires first synchronizing with
   * `__stream`.
   *
   * \throws If the storage of the requested size and alignment cannot be
   * obtained.
   *
   * \param __bytes The size in bytes of the allocation.
   * \param __alignment The alignment of the allocation
   * \param __stream The stream on which to perform the allocation.
   * \return Pointer to the requested storage.
   */
  void *allocate_async(size_t bytes, size_t alignment, stream_view stream) {
    return do_allocate_async(bytes, alignment, stream);
  }

  /*!
   * \brief Deallocates the storage pointed to by `__p` in stream order on
   * `__stream`.
   *
   * `__p` must have been returned by a prior call to
   * `allocate_async(__bytes, default_alignment)` or `allocate(__bytes,
   * default_alignment)` on a `stream_ordered_memory_resource` that compares
   * equal to `*this`, and the storage it points to must not yet have been
   * deallocated, otherwise behavior is undefined.
   *
   * Asynchronous, stream-ordered operations on `__stream` initiated before
   * `deallocate_async(__p, __bytes, __stream)` may still access the storage
   * pointed to by `__p` after `deallocate_async` returns.
   *
   * Storage deallocated on `__stream` may be reused by a future
   * call to `allocate_async` on the same stream without synchronizing
   * `__stream`. Therefore,  `__stream` is typically the last stream on which
   * `__p` was last used. It is the caller's responsibility to ensure the
   * storage pointed to by `__p` is not in use on any other stream (or the
   * host), or behavior is undefined.
   *
   * \param __p Pointer to storage to be deallocated.
   * \param __bytes The size in bytes of the allocation. This must be equal to
   * the value of `__bytes` that was specified to the `allocate` or
   * `allocate_async` call that returned `__p`.
   * \param __stream The stream on which to perform the deallocation.
   */
  void deallocate_async(void *__mem, size_t __bytes, stream_view __stream) {
    deallocate_async(__mem, __bytes, default_alignment, __stream);
  }

  /*!
   * \brief Deallocates the storage pointed to by `__p` in stream order on
   * `__stream`.
   *
   * `__p` must have been returned by a prior call to
   * `allocate_async(__bytes, __alignment)` or `allocate(__bytes,
   * __alignment)` on a `stream_ordered_memory_resource` that compares
   * equal to `*this`, and the storage it points to must not yet have been
   * deallocated, otherwise behavior is undefined.
   *
   * Asynchronous, stream-ordered operations on `__stream` initiated before
   * `deallocate_async(__p, __bytes, __stream)` may still access the storage
   * pointed to by `__p` after `deallocate_async` returns.
   *
   * Storage deallocated on `__stream` may be reused by a future
   * call to `allocate_async` on the same stream without synchronizing
   * `__stream`. Therefore,  `__stream` is typically the last stream on which
   * `__p` was last used. It is the caller's responsibility to ensure the
   * storage pointed to by `__p` is not in use on any other stream (or the
   * host), or behavior is undefined.
   *
   * \param __p Pointer to storage to be deallocated.
   * \param __bytes The size in bytes of the allocation. This must be equal to
   * the value of `__bytes` that was specified to the `allocate` or
   * `allocate_async` call that returned `__p`.
   * \param __alignment The alignment of the allocation. This must be equal to
   * the value of `__alignment` that was specified to the `allocate` or
   * `allocate_async` call that returned `__p`.
   * \param __stream The stream on which to perform the deallocation.
   */
  void deallocate_async(void *__mem, size_t __bytes, size_t __alignment, stream_view __stream) {
    do_deallocate_async(__mem, __bytes, __alignment, __stream);
  }

 protected:
  virtual void *do_allocate_async(size_t __bytes, size_t __alignment, stream_view __stream) = 0;
  virtual void do_deallocate_async(void *__mem, size_t __bytes, size_t __alignment,
                                   stream_view __stream) = 0;

  template <typename _ResourcePointer, typename... _Properties>
  friend class cuda_for_dali::basic_resource_view;
};

}  // namespace detail

/*!
 * \brief Abstract interface for memory allocation.
 *
 * \tparam _MemoryKind The kind of the allocated memory.
 */
template <typename _MemoryKind>
class memory_resource : private virtual detail::memory_resource_base {
 public:
  using memory_kind = _MemoryKind;
  static constexpr std::size_t default_alignment = memory_resource_base::default_alignment;

  virtual ~memory_resource() = default;

  /*!
   * \brief Allocates storage of size at least `__bytes` bytes.
   *
   * The returned storage is aligned to the specified `__alignment` if such
   * alignment is supported. Otherwise throws.
   *
   * Storage may be accessed immediately within the execution contexts that
   * can access the memory.
   *
   * \throws If storage of the requested size and alignment cannot be obtained.
   *
   * \param __bytes The size in bytes of the allocation
   * \param __alignment The alignment of the allocation
   * \return Pointer to the requested storage
   */
  void *allocate(std::size_t __bytes, std::size_t __alignment = default_alignment) {
    return do_allocate(__bytes, __alignment);
  }

  /*!
   * \brief Deallocates the storage pointed to by `__p`.
   *
   * `__p` must have been returned by a prior call to `allocate(__bytes,
   * __alignment)` on a `memory_resource` that compares equal to `*this`, and
   * the storage it points to must not yet have been deallocated, otherwise
   * behavior is undefined.
   *
   * \throws Nothing.
   *
   * \param __p Pointer to storage to be deallocated
   * \param __bytes The size in bytes of the allocation. This must be equal to
   * the value of `__bytes` that was specified to the `allocate` call that
   * returned `__p`.
   * \param __alignment The alignment of the allocation. This must be equal to
   * the value of `__alignment` that was specified to the `allocate` call that
   * returned `__p`.
   */
  void deallocate(void *__p, std::size_t __bytes, std::size_t __alignment = default_alignment) {
    do_deallocate(__p, __bytes, __alignment);
  }

  /*!
   * \brief Compare this resource to another.
   *
   * Two resources compare equal if and only if memory allocated from one
   * resource can be deallocated from the other and vice versa.
   *
   * \param __other The other resource to compare against
   */
  bool is_equal(memory_resource const &__other) const noexcept {
    return do_is_equal(__other);
  }

 private:
  template <typename, typename...>
  friend class basic_resource_view;

  void *do_allocate(std::size_t __bytes, std::size_t __alignment) override = 0;

  void do_deallocate(void *__p, std::size_t __bytes, std::size_t __alignment) override = 0;

  // Default to identity comparison
  virtual bool do_is_equal(memory_resource const &__other) const noexcept {
    return this == &__other;
  }

  void *__do_as_kind(const ::std::type_info *__tag_type_id,
                     const void *__tag_type_fallback_id) const noexcept final {
    using __tag = detail::__type_pack<memory_kind>;
    return detail::__fallback_typeid::__is_type<__tag>(__tag_type_id, __tag_type_fallback_id) ?
               const_cast<memory_resource *>(this) :
               nullptr;
  }

  bool is_equal_base(const detail::memory_resource_base &__other) const noexcept final {
    if (auto *__other_res = __other.as_kind<memory_kind>()) {
      return do_is_equal(*__other_res);
    } else {
      return false;
    }
  }
};

template <typename _Kind>
inline _DALI_INLINE_VISIBILITY bool operator==(const memory_resource<_Kind> &__a,
                                               const memory_resource<_Kind> &__b) {
  return __a.is_equal(__b);
}

/*!
 * \brief Abstract interface for CUDA stream-ordered memory allocation.
 *
 * "Stream-ordered memory allocation" extends the CUDA programming model to
 * include memory allocation as stream-ordered operations.
 *
 * All asynchronous accesses of the allocation must happen between the stream
 * execution of the allocation and the free. If storage is accessed outside of
 * the promised stream order, a use before allocation / use after free error
 * will cause undefined behavior.
 *
 * Allocating on stream `s0` returns memory that is valid to access immediately
 * only on `s0`. Accessing it on any other stream (or the host) first requires
 * synchronization with `s0`, otherwise behavior is undefined.
 *
 * Deallocating memory on stream `s1` indicates that it is valid to reuse the
 * deallocated memory immediately for another allocation on `s1`.
 *
 * Asynchronous, stream-ordered operations ordered before deallocation on `s1`
 * may still access the storage after deallocation completes.
 *
 * Memory may be allocated and deallocated on different streams, `s0` and `s1`
 * respectively, but requires synchronization between `s0` and `s1` before the
 * deallocation occurs.
 *
 * \tparam _MemoryKind The kind of the allocated memory.
 */
template <typename _MemoryKind>
class stream_ordered_memory_resource : public virtual memory_resource<_MemoryKind>,
                                       private virtual detail::stream_ordered_memory_resource_base {
 public:
  using memory_kind = _MemoryKind;
  static constexpr std::size_t default_alignment = memory_resource<_MemoryKind>::default_alignment;

  /*!
   * \brief Allocates storage of size at least `__bytes` bytes in stream order
   * on `__stream`.
   *
   * The returned storage is aligned to `default_alignment`.
   *
   * The returned storage may be used immediately only on `__stream`. Accessing
   * it on any other stream (or the host) requires first synchronizing with
   * `__stream`.
   *
   * \throws If the storage of the requested size and `default_alignment` cannot
   * be obtained.
   *
   * \param __bytes The size in bytes of the allocation.
   * \param __stream The stream on which to perform the allocation.
   * \return Pointer to the requested storage.
   */
  void *allocate_async(std::size_t __bytes, stream_view __stream) {
    return do_allocate_async(__bytes, default_alignment, __stream);
  }

  /*!
   * \brief Allocates storage of size at least `__bytes` bytes in stream order
   * on `__stream`.
   *
   * The returned storage is aligned to the specified `__alignment` if such
   * alignment is supported.
   *
   * The returned storage may be used immediately only on `__stream`. Using it
   * on any other stream (or the host) requires first synchronizing with
   * `__stream`.
   *
   * \throws If the storage of the requested size and alignment cannot be
   * obtained.
   *
   * \param __bytes The size in bytes of the allocation.
   * \param __alignment The alignment of the allocation
   * \param __stream The stream on which to perform the allocation.
   * \return Pointer to the requested storage.
   */
  void *allocate_async(std::size_t __bytes, std::size_t __alignment, stream_view __stream) {
    return do_allocate_async(__bytes, __alignment, __stream);
  }

  /*!
   * \brief Deallocates the storage pointed to by `__p` in stream order on
   * `__stream`.
   *
   * `__p` must have been returned by a prior call to
   * `allocate_async(__bytes, default_alignment)` or `allocate(__bytes,
   * default_alignment)` on a `stream_ordered_memory_resource` that compares
   * equal to `*this`, and the storage it points to must not yet have been
   * deallocated, otherwise behavior is undefined.
   *
   * Asynchronous, stream-ordered operations on `__stream` initiated before
   * `deallocate_async(__p, __bytes, __stream)` may still access the storage
   * pointed to by `__p` after `deallocate_async` returns.
   *
   * Storage deallocated on `__stream` may be reused by a future
   * call to `allocate_async` on the same stream without synchronizing
   * `__stream`. Therefore,  `__stream` is typically the last stream on which
   * `__p` was last used. It is the caller's responsibility to ensure the
   * storage pointed to by `__p` is not in use on any other stream (or the
   * host), or behavior is undefined.
   *
   * \param __p Pointer to storage to be deallocated.
   * \param __bytes The size in bytes of the allocation. This must be equal to
   * the value of `__bytes` that was specified to the `allocate` or
   * `allocate_async` call that returned `__p`.
   * \param __stream The stream on which to perform the deallocation.
   */
  void deallocate_async(void *__p, std::size_t __bytes, stream_view __stream) {
    do_deallocate_async(__p, __bytes, default_alignment, __stream);
  }

  /*!
   * \brief Deallocates the storage pointed to by `__p` in stream order on
   * `__stream`.
   *
   * `__p` must have been returned by a prior call to
   * `allocate_async(__bytes, __alignment)` or `allocate(__bytes,
   * __alignment)` on a `stream_ordered_memory_resource` that compares
   * equal to `*this`, and the storage it points to must not yet have been
   * deallocated, otherwise behavior is undefined.
   *
   * Asynchronous, stream-ordered operations on `__stream` initiated before
   * `deallocate_async(__p, __bytes, __stream)` may still access the storage
   * pointed to by `__p` after `deallocate_async` returns.
   *
   * Storage deallocated on `__stream` may be reused by a future
   * call to `allocate_async` on the same stream without synchronizing
   * `__stream`. Therefore,  `__stream` is typically the last stream on which
   * `__p` was last used. It is the caller's responsibility to ensure the
   * storage pointed to by `__p` is not in use on any other stream (or the
   * host), or behavior is undefined.
   *
   * \param __p Pointer to storage to be deallocated.
   * \param __bytes The size in bytes of the allocation. This must be equal to
   * the value of `__bytes` that was specified to the `allocate` or
   * `allocate_async` call that returned `__p`.
   * \param __alignment The alignment of the allocation. This must be equal to
   * the value of `__alignment` that was specified to the `allocate` or
   * `allocate_async` call that returned `__p`.
   * \param __stream The stream on which to perform the deallocation.
   */
  void deallocate_async(void *__p, std::size_t __bytes, std::size_t __alignment,
                        stream_view __stream) {
    do_deallocate_async(__p, __bytes, __alignment, __stream);
  }

 private:
  template <typename, typename...>
  friend class basic_resource_view;

  /// Default synchronous implementation of `memory_resource::do_allocate`
  void *do_allocate(std::size_t __bytes, std::size_t __alignment) override {
    auto const __default_stream = stream_view{};
    auto __p = do_allocate_async(__bytes, __alignment, __default_stream);
    __default_stream.wait();
    return __p;
  }

  /// Default synchronous implementation of `memory_resource::do_deallocate`
  void do_deallocate(void *__p, std::size_t __bytes, std::size_t __alignment) override {
    auto const __default_stream = stream_view{};
    __default_stream.wait();
    do_deallocate_async(__p, __bytes, __alignment, __default_stream);
  }

  void *do_allocate_async(std::size_t __bytes, std::size_t __alignment,
                          stream_view __stream) override = 0;

  void do_deallocate_async(void *__p, std::size_t __bytes, std::size_t __alignment,
                           stream_view __stream) override = 0;
};


/*!
 * \brief Indicates whether a memory kind `_MemoryKind` has a property `__property`.
 */
template <typename _MemoryKind, typename __property>
struct kind_has_property : std::false_type {};

/*!
 * \brief A special property telling that given resource/resource view allocates
 *        memory of specific kind.
 *
 * When a view defines this property, it implicitly has all properties of this
 * memory kind.
 * This property is also a property in itself and views defining properties of
 * the underlying memory kind cannot be converted to a view defining this property.
 * This allows for future extension of the set of properties.
 */
template <typename _MemoryKind>
struct is_kind;

template <typename _MemoryKind>
struct kind_has_property<_MemoryKind, is_kind<_MemoryKind>> : std::true_type {};

#define _DALI_MEMORY_KIND_PROPERTY(__kind, __property) \
  template <>                                          \
  struct kind_has_property<__kind, __property> : std::true_type {};

_DALI_MEMORY_KIND_PROPERTY(memory_kind::host, memory_access::host);
_DALI_MEMORY_KIND_PROPERTY(memory_kind::host, oversubscribable);
_DALI_MEMORY_KIND_PROPERTY(memory_kind::host, memory_location::host);

_DALI_MEMORY_KIND_PROPERTY(memory_kind::pinned, memory_access::host);
_DALI_MEMORY_KIND_PROPERTY(memory_kind::pinned, memory_access::device);
_DALI_MEMORY_KIND_PROPERTY(memory_kind::pinned, resident);
_DALI_MEMORY_KIND_PROPERTY(memory_kind::pinned, memory_location::host);

_DALI_MEMORY_KIND_PROPERTY(memory_kind::device, memory_access::device);
_DALI_MEMORY_KIND_PROPERTY(memory_kind::device, resident);
_DALI_MEMORY_KIND_PROPERTY(memory_kind::device, memory_location::device);

_DALI_MEMORY_KIND_PROPERTY(memory_kind::managed, memory_access::host);
_DALI_MEMORY_KIND_PROPERTY(memory_kind::managed, memory_access::device);
_DALI_MEMORY_KIND_PROPERTY(memory_kind::managed, oversubscribable);
_DALI_MEMORY_KIND_PROPERTY(memory_kind::managed, memory_location::host);
_DALI_MEMORY_KIND_PROPERTY(memory_kind::managed, memory_location::device);

namespace detail {
template <typename _Property>
std::false_type Has_Property(...);
template <typename _Property, typename _MemoryKind>
kind_has_property<_MemoryKind, _Property> Has_Property(const memory_resource<_MemoryKind> *);
template <typename _Property, typename _MemoryKind>
kind_has_property<_MemoryKind, _Property> Has_Property(
    const stream_ordered_memory_resource<_MemoryKind> *);
}  // namespace detail

template <typename _Target, typename _Property>
struct has_property : decltype(detail::Has_Property<_Property>(
                          std::declval<std::remove_reference_t<_Target> *>())) {};

namespace detail {
template <typename _Property, typename... _Properties>
struct is_property_in : std::false_type {};

template <typename _Property, typename _Mismatch, typename... _Tail>
struct is_property_in<_Property, _Mismatch, _Tail...> : is_property_in<_Property, _Tail...> {};

template <typename _Property, typename... _Tail>
struct is_property_in<_Property, _Property, _Tail...> : std::true_type {};

template <typename _Property, typename _MemoryKind, typename... Tail>
struct is_property_in<_Property, is_kind<_MemoryKind>, Tail...>
    : kind_has_property<_MemoryKind, _Property> {};

template <typename _MemoryKind, typename... Tail>
struct is_property_in<is_kind<_MemoryKind>, is_kind<_MemoryKind>, Tail...> : std::true_type {};

template <typename _FromPointer, typename _ToPointer>
struct is_resource_pointer_convertible : std::is_convertible<_FromPointer, _ToPointer> {};
// Private inheritance from (stream_ordered_)memory_resource_base* requires explicit partial
// specializations as `is_convertible` will return false

template <typename _FromPointer>
struct is_resource_pointer_convertible<_FromPointer, detail::memory_resource_base *>
    : std::conjunction<
          std::is_pointer<_FromPointer>,
          std::is_base_of<detail::memory_resource_base,
                          typename ::std::pointer_traits<_FromPointer>::element_type>> {};

template <typename _FromPointer>
struct is_resource_pointer_convertible<_FromPointer, detail::stream_ordered_memory_resource_base *>
    : std::conjunction<
          std::is_pointer<_FromPointer>,
          std::is_base_of<detail::stream_ordered_memory_resource_base,
                          typename ::std::pointer_traits<_FromPointer>::element_type>> {};

}  // namespace detail

template <typename _Pointer, typename... _Properties, typename _Property>
struct has_property<basic_resource_view<_Pointer, _Properties...>, _Property>
    : detail::is_property_in<_Property, _Properties...> {};

template <typename __from, typename __to>
struct is_view_convertible;

template <typename _FromPointer, typename... _FromProperties, typename _ToPointer,
          typename... _ToProperties>
struct is_view_convertible<basic_resource_view<_FromPointer, _FromProperties...>,
                           basic_resource_view<_ToPointer, _ToProperties...>>
    : std::conjunction<
          detail::is_resource_pointer_convertible<_FromPointer, _ToPointer>,
          has_property<basic_resource_view<_FromPointer, _FromProperties...>, _ToProperties>...> {};

/*!
 * \brief A pointer-like object to a memory resource based on resource.
 *
 * Resource view is an object that acts as a memory resource pointer, but provides
 * enhanced implicit conversions. The idea behind this type is that a user of
 * a memory resource may be interested in many kinds of resources as long as they
 * have certain properties. For example, a function may work with any resource
 * that can provide host-accessible memory, regardless of whether it is plain host
 * memory, pinned memory, managed memory, or some,yet-to-be-defined future kind
 * of memory.
 *
 * A resource view can be created from a memory resource pointer or from another
 * resource view that defines a superset of the target properties.
 *
 * The resource view exposes the underlying resource's interface via `operator->`.
 *
 * The `basic_resource_view` class can be parameterized with the resource pointer type,
 * which can be either one of the base resource classes or a concrete resource type.
 *
 * \tparam _ResourcePointer a pointer-like object to the underlying memory resource
 * \tparam _Properies properties of a memory resource required by resource view
 */
template <typename _ResourcePointer, typename... _Properties>
class basic_resource_view {
 public:
  static_assert(
      std::is_base_of<detail::memory_resource_base,
                      typename ::std::pointer_traits<_ResourcePointer>::element_type>::value ||
          std::is_base_of<detail::stream_ordered_memory_resource_base,
                          typename ::std::pointer_traits<_ResourcePointer>::element_type>::value,
      "ResourcePointer must be a pointer to a memory_resource_base, "
      "stream_ordered_memory_resource_base or a derived class");

  basic_resource_view() = default;

  basic_resource_view(int) = delete;  // NOLINT(runtime/explicit)

  basic_resource_view(std::nullptr_t) {}  // NOLINT(runtime/explicit)

  /*!
   * \brief Constructs a resource view from a compatible memory resource pointer.
   *
   * The memory resource is considered compatible if a pointer to it can be converted to
   * `_ResourcePointer` and the resource type has the required properties listed
   * in `_Properties`.
   *
   * \tparam _Resource Type of a mmeory resource object.
   * \param __p pointer to a memory resource object.
   */
  template <typename _Resource,
            typename = std::enable_if_t<
                detail::is_resource_pointer_convertible<_Resource *, _ResourcePointer>::value &&
                std::conjunction<has_property<_Resource, _Properties>...>::value>>
  basic_resource_view(_Resource *__p) : __pointer(__p) {}  // NOLINT(runtime/explicit)

  /*!
   * \brief Constructs a resource view by copying the resource pointer from a compatible resource
   * view.
   *
   * A resource view is considered compatible if it defines all properties required by this
   * view in `_Properties`.
   *
   * \tparam _OtherPointer The resource pointer type of the source resource view
   * \tparam _OtherProperties The properties defined byt the source resource view
   */
  template <
      typename _OtherPointer, typename... _OtherProperties,
      typename = std::enable_if_t<is_view_convertible<
          basic_resource_view<_OtherPointer, _OtherProperties...>, basic_resource_view>::value>>
  basic_resource_view(basic_resource_view<_OtherPointer, _OtherProperties...> v)  // NOLINT
      : __pointer(v.__pointer) {}

  /*!
   * \brief Exposes the interface of the underlying memory resource.
   *
   * \note This method should not be used to obtain the pointer to the memory resource.
   */
  _ResourcePointer operator->() const noexcept {
    return __pointer;
  }

  template <typename _Ptr2, typename... _Props2>
  bool operator==(
      const cuda_for_dali::basic_resource_view<_Ptr2, _Props2...> &__v2) const noexcept {
    using __view1_t = basic_resource_view;
    using __view2_t = basic_resource_view<_Ptr2, _Props2...>;
    if (__pointer == nullptr || __v2.__pointer == nullptr)
      return __pointer == nullptr && __v2.__pointer == nullptr;
    return static_cast<const detail::memory_resource_base *>(__pointer)->is_equal(*__v2.__pointer);
  }

  template <typename _Ptr2, typename... _Props2>
  bool operator!=(
      const cuda_for_dali::basic_resource_view<_Ptr2, _Props2...> &__v2) const noexcept {
    return !(*this == __v2);
  }

  /*!
   * \brief Returns true if the underlying pointer is not null.
   */
  constexpr explicit operator bool() const noexcept {
    return !!__pointer;
  }

 private:
  template <typename, typename...>
  friend class basic_resource_view;

  _ResourcePointer __pointer{};
};

template <typename _FirstProperty, typename... _Properties, typename _ResourcePointer>
basic_resource_view<_ResourcePointer, _FirstProperty, _Properties...> view_resource(
    _ResourcePointer __rsrc_ptr) {
  return __rsrc_ptr;
}


template <typename _ResourcePointer>
basic_resource_view<_ResourcePointer,
                    is_kind<typename std::remove_pointer_t<_ResourcePointer>::memory_kind>>
view_resource(_ResourcePointer __rsrc_ptr) {
  return __rsrc_ptr;
}


template <typename _ResourcePointer, typename... _Properties, typename _Kind>
bool operator==(const basic_resource_view<_ResourcePointer, _Properties...> &__view,
                const memory_resource<_Kind> *__mr) {
  return __view == view_resource(__mr);
}

template <typename _ResourcePointer, typename... _Properties, typename _Kind>
bool operator!=(const basic_resource_view<_ResourcePointer, _Properties...> &__view,
                const memory_resource<_Kind> *__mr) {
  return __view != view_resource(__mr);
}

template <typename _ResourcePointer, typename... _Properties, typename _Kind>
bool operator==(const memory_resource<_Kind> *__mr,
                const basic_resource_view<_ResourcePointer, _Properties...> &__view) {
  return view_resource(__mr) == __view;
}

template <typename _ResourcePointer, typename... _Properties, typename _Kind>
bool operator!=(const memory_resource<_Kind> *__mr,
                const basic_resource_view<_ResourcePointer, _Properties...> &__view) {
  return view_resource(__mr) != __view;
}

template <typename... _Properties>
using resource_view = basic_resource_view<detail::memory_resource_base *, _Properties...>;

template <typename... _Properties>
using stream_ordered_resource_view =
    basic_resource_view<detail::stream_ordered_memory_resource_base *, _Properties...>;

#if _DALI_STD_VER > 14

#if defined(_DALI_STD_PMR_NS)

namespace detail {
class __pmr_adaptor_base : public _DALI_STD_PMR_NS::memory_resource {
 public:
  virtual cuda_for_dali::memory_resource<cuda_for_dali::memory_kind::host> *resource()
      const noexcept = 0;
};
}  // namespace detail

template <typename _Pointer>
class pmr_adaptor final : public detail::__pmr_adaptor_base {
  using resource_type = std::remove_reference_t<decltype(*std::declval<_Pointer>())>;

  static constexpr bool __is_host_accessible_resource =
      has_property<resource_type, memory_access::host>::value;
  static_assert(
      __is_host_accessible_resource,
      "Pointer must be a pointer-like type to a type that allocates host-accessible memory.");

 public:
  pmr_adaptor(_Pointer __mr) : __mr_{std::move(__mr)} {}  // NOLINT(runtime/explicit)

  using raw_pointer = std::remove_reference_t<decltype(&*std::declval<_Pointer>())>;

  raw_pointer resource() const noexcept override {
    return &*__mr_;
  }

 private:
  void *do_allocate(std::size_t __bytes, std::size_t __alignment) override {
    return __mr_->allocate(__bytes, __alignment);
  }

  void do_deallocate(void *__p, std::size_t __bytes, std::size_t __alignment) override {
    return __mr_->deallocate(__p, __bytes, __alignment);
  }

  bool do_is_equal(_DALI_STD_PMR_NS::memory_resource const &__other) const noexcept override {
    auto __other_p = dynamic_cast<detail::__pmr_adaptor_base const *>(&__other);
    return __other_p &&
           (__other_p->resource() == resource() || __other_p->resource()->is_equal(*resource()));
  }

  _Pointer __mr_;
};
#endif  // defined(_DALI_STD_PMR_NS)
#endif  // _DALI_STD_VER > 14

}  // namespace cuda_for_dali
}  // namespace dali

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif  // DALI_CORE_MM_CUDA_MEMORY_RESOURCE_H_

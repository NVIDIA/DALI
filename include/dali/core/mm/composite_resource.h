// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_MM_COMPOSITE_RESOURCE_H_
#define DALI_CORE_MM_COMPOSITE_RESOURCE_H_

#include <type_traits>
#include <memory>
#include <tuple>
#include <utility>
#include "dali/core/mm/memory_resource.h"

namespace dali {
namespace mm {

namespace detail {

template <typename Kind, typename Context>
memory_resource<Kind, Context> &GetResourceInterface(const memory_resource<Kind, Context> &);

template <typename Kind>
async_memory_resource<Kind> &GetResourceInterface(const async_memory_resource<Kind> &);

template <typename Resource>
using resource_interface_t = std::remove_reference_t<
    decltype(GetResourceInterface(std::declval<Resource>()))>;


template <typename Interface, typename Resource, typename...Extra>
class CompositeResourceBase : public Interface {
 public:
  using memory_kind = typename Resource::memory_kind;
  CompositeResourceBase() = default;

  template <typename... ExtraArgs>
  CompositeResourceBase(std::shared_ptr<Resource> resource,
                    ExtraArgs... extra)
  : extra{std::forward<ExtraArgs>(extra)...}
  , resource(std::move(resource)) {
    static_assert(sizeof...(ExtraArgs) == sizeof...(Extra), "Incorrect number of extra values");
  }

 protected:
  std::tuple<Extra... > extra;
  std::shared_ptr<Resource> resource;

 private:
  bool do_is_equal(const memory_resource<memory_kind> &other) const noexcept override {
    if (auto *other_composite = dynamic_cast<const CompositeResourceBase *>(&other)) {
      if ((resource != nullptr) != (other_composite->resource != nullptr))
        return false;  // one is null, the other is not

      // both null or really equal
      return !resource || resource->is_equal(*other_composite ->resource);
    } else {
      return resource->is_equal(other);
    }
  }
  void *do_allocate(size_t bytes, size_t alignment) override {
    return resource->allocate(bytes, alignment);
  }
  void do_deallocate(void *mem, size_t bytes, size_t alignment) override {
    resource->deallocate(mem, bytes, alignment);
  }
};

template <typename Interface, typename Resource, typename... Extra>
class CompositeResourceImpl;

template <typename Kind, typename Context, typename Resource, typename... Extra>
class CompositeResourceImpl<memory_resource<Kind, Context>, Resource, Extra...>
: public CompositeResourceBase<memory_resource<Kind, Context>, Resource, Extra...> {
 public:
  using Base = CompositeResourceBase<memory_resource<Kind, Context>, Resource, Extra...>;
  using Base::Base;
};

template <typename Kind, typename Resource, typename... Extra>
class CompositeResourceImpl<async_memory_resource<Kind>, Resource, Extra...>
: public CompositeResourceBase<async_memory_resource<Kind>, Resource, Extra...> {
 public:
  using Base = CompositeResourceBase<async_memory_resource<Kind>, Resource, Extra...>;
  using Base::Base;

 private:
  void *do_allocate_async(size_t bytes, size_t alignment, stream_view stream) override {
    return this->resource->allocate_async(bytes, alignment, stream);
  }
  void do_deallocate_async(void *mem, size_t bytes, size_t alignment, stream_view stream) override {
    this->resource->deallocate_async(mem, bytes, alignment, stream);
  }
};

}  // namespace detail

/**
 * @brief Aggregates a memory resource and some additional data.
 *
 * Typical examples of extra data would be upstream resources. The object is constructed
 * in a way that guarantees that the extra data is not destroyed before the resource is,
 * enabling proper cleanup.
 */
template <typename Resource, typename... Extra>
class CompositeResource
: public detail::CompositeResourceImpl<detail::resource_interface_t<Resource>, Resource, Extra...> {
 public:
  using interface_type = detail::resource_interface_t<Resource>;
  using Base = detail::CompositeResourceImpl<interface_type, Resource, Extra...>;
  using Base::Base;
};


template <typename Resource, typename... Extra>
auto make_composite_resource(std::shared_ptr<Resource> resource, Extra... args) {
  return CompositeResource<Resource, Extra...>(std::move(resource), std::move(args)...);
}

template <typename Resource, typename... Extra>
auto make_shared_composite_resource(std::shared_ptr<Resource> resource, Extra... args) {
  return std::make_shared<CompositeResource<Resource, Extra...>>(
    std::move(resource), std::move(args)...);
}

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_COMPOSITE_RESOURCE_H_

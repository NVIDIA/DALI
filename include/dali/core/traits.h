// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_TRAITS_H_
#define DALI_CORE_TRAITS_H_

#include <type_traits>
#include <array>
#include <vector>

namespace dali {

template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A> > : std::true_type {};


template <typename T>
struct is_std_array : std::false_type {};

template <typename T, size_t A>
struct is_std_array<std::array<T, A> > : std::true_type {};


template <typename T>
using remove_const_t = std::remove_const_t<T>;

template <typename T>
using remove_cv_t = std::remove_cv_t<T>;

template <bool Value, typename Type = void>
using enable_if_t = std::enable_if_t<Value, Type>;


template <typename T, typename _ = void>
struct is_container : std::false_type {};

template <typename... Ts>
struct is_container_helper {};

template <typename T>
struct is_container<T,
        std::conditional_t<
                false,
                is_container_helper<
                        typename T::value_type,
                        typename T::size_type,
                        typename T::allocator_type,
                        typename T::iterator,
                        typename T::const_iterator,
                        decltype(std::declval<T>().size()),
                        decltype(std::declval<T>().begin()),
                        decltype(std::declval<T>().end()),
                        decltype(std::declval<T>().cbegin()),
                        decltype(std::declval<T>().cend())
                >, void
        >
> : public std::true_type {};

}  // namespace dali

#endif  // DALI_CORE_TRAITS_H_

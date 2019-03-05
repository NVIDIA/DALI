// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_KERNELS_TUPLE_HELPERS_H_
#define DALI_KERNELS_TUPLE_HELPERS_H_

#include <tuple>
#include <utility>

namespace dali {
namespace detail {

template <typename... tuples>
struct tuple_cat_helper;

template <typename t0>
struct tuple_cat_helper<t0> { using type = t0; };

template <typename t0, typename... tuples>
struct tuple_cat_helper<t0, tuples...>
 : tuple_cat_helper<t0, typename tuple_cat_helper<tuples...>::type> {};

template <typename... T1, typename... T2>
struct tuple_cat_helper<std::tuple<T1...>, std::tuple<T2...>> {
  using type = std::tuple<T1..., T2...>;
};

template <typename... tuples>
using tuple_cat_t = typename tuple_cat_helper<tuples...>::type;

/// @brief Compile-time integer sequence, see C++14 stl
template <int... indices>
struct seq {};

template <typename sequence1, typename sequence2>
struct seq_cat_impl;

template <int... s1, int... s2>
struct seq_cat_impl<seq<s1...>, seq<s2...>> {
  using type = seq<s1..., s2...>;
};

/// @brief Concatenates sequence types
template <typename sequence1, typename sequence2>
using seq_cat_t = typename seq_cat_impl<sequence1, sequence2>::type;

template <int offset, typename sequence>
struct seq_offset_impl;

template <int offset, int... s>
struct seq_offset_impl<offset, seq<s...>> {
  using type = seq<(offset+s)...>;
};

/// @brief Applies constant offset to all elements in the sequence
template <int offset, typename sequence>
using seq_offset_t = typename seq_offset_impl<offset, sequence>::type;

template <int start, int count>
struct build_seq_impl;

/// @brief Builds an integer sequence (start, start+1, ..., start+count-1)
template <int start, int count>
using build_seq_t = typename build_seq_impl<start, count>::type;

template <int start, int count>
struct build_seq_impl {
  // first, apply offset "start" and generate 0-based sequence (for efficiency)
  using type = seq_offset_t<start, build_seq_t<0, count>>;
};

template <int count>
struct build_seq_impl<0, count> {
  // concatenate two sub-sequences; since we only generate 0-based sequences,
  // it should run in relatively short time
  using type = seq_cat_t<build_seq_t<0, count/2>, build_seq_t<count/2, count-count/2>>;
};

template <>
struct build_seq_impl<0, 0> { using type = seq<>; };

template <>
struct build_seq_impl<0, 1> { using type = seq<0>; };

// for compilation speed
template <>
struct build_seq_impl<0, 2> { using type = seq<0, 1>; };

// for compilation speed
template <>
struct build_seq_impl<0, 3> { using type = seq<0, 1, 2>; };

// for compilation speed
template <>
struct build_seq_impl<0, 4> { using type = seq<0, 1, 2, 3>; };

// for compilation speed
template <>
struct build_seq_impl<0, 5> { using type = seq<0, 1, 2, 3, 4>; };

template <typename Tuple>
struct tuple_indices_helper;

template <typename... T>
struct tuple_indices_helper<std::tuple<T...>> {
  using type = build_seq_t<0, sizeof...(T)>;
};

template <typename Tuple>
using tuple_indices_t = typename tuple_indices_helper<Tuple>::type;

template <typename... T>
using pack_indices_t = build_seq_t<0, sizeof...(T)>;

template <typename... T>
constexpr pack_indices_t<T...> tuple_indices(const std::tuple<T...> &) { return {}; }

template <typename T>
constexpr std::tuple<T&&> as_tuple(T &&t) {
  return std::tuple<T&&>(std::forward<T>(t));
}

template <typename... T>
constexpr std::tuple<T...> &&as_tuple(std::tuple<T...> &&t) { return t; }

template <typename... T>
constexpr std::tuple<T...> &as_tuple(std::tuple<T...> &t) { return t; }

template <typename... T>
constexpr const std::tuple<T...> &as_tuple(const std::tuple<T...> &t) { return t; }

template <typename F, typename... T, int... indices>
constexpr auto apply_indexed(F &&f, const std::tuple<T...> &args, seq<indices...>)
  ->decltype(f(std::get<indices>(args)...)) {
  return f(std::get<indices>(args)...);
}

template <typename F, typename... T>
constexpr auto apply(F &&f, std::tuple<T...> &&args)
  ->decltype(apply_indexed(f, args, tuple_indices(args))) {
  return apply_indexed(f, args, tuple_indices(args));
}


template <typename F, typename... Args>
constexpr auto apply_all(F &&f, Args&&... args)
  ->decltype(apply(f, std::tuple_cat(as_tuple(args)...))) {
  return apply(f, std::tuple_cat(as_tuple(args)...));
}

template <size_t total, typename Type, typename Tuple>
struct tuple_index_impl;

template <size_t total, typename Type, typename... Tail>
struct tuple_index_impl<total, Type, std::tuple<Type, Tail...>>
    : std::integral_constant<size_t, total> {};

template <size_t total, typename Type, typename Head, typename... Tail>
struct tuple_index_impl<total, Type, std::tuple<Head, Tail...>>
    : tuple_index_impl<total + 1, Type, std::tuple<Tail...>> {};

template <size_t total, typename Type>
struct tuple_index_impl<total, Type, std::tuple<>> : std::integral_constant<size_t, total> {
  static_assert(total < 0, "Type not found");
};

/**
 * @brief Find index of first occurence of Type in std::tuple<Ts...>
 *
 * If Type is not present among Ts..., static_assert is invoked
 */
template <typename Type, typename Tuple>
struct tuple_index : tuple_index_impl<0, Type, Tuple> {};

/**
 * @brief c++14's std::get<Type>(std::tuple<Types...> )
 */
template <class T, class... Types>
T &get(std::tuple<Types...> &t) noexcept {
  return std::get<tuple_index<T, std::tuple<Types...>>::value>(t);
}

template <class T, class... Types>
T &&get(std::tuple<Types...> &&t) noexcept {
  return std::get<tuple_index<T, std::tuple<Types...>>::value>(t);
}

template <class T, class... Types>
const T &get(const std::tuple<Types...> &t) noexcept {
  return std::get<tuple_index<T, std::tuple<Types...>>::value>(t);
}

template <class T, class... Types>
const T &&get(const std::tuple<Types...> &&t) noexcept {
  return std::get<tuple_index<T, std::tuple<Types...>>::value>(t);
}


template <template <int> class type_generator, typename Sequence>
struct tuple_generator_type;

template <template <int> class type_generator, int... sequence>
struct tuple_generator_type<type_generator, detail::seq<sequence...>> {
  using type = std::tuple<type_generator<sequence>...>;
};

/**
 * @brief Generate tuple type using `type_generator` and `Sequence`.
 *
 * Used to generate tuple containing types that match values of some
 * enumeration.
 *
 * @tparam type_generator Template taking int to produce a type
 * @tparam Sequence
 */
template <template <int> class type_generator, typename Sequence>
using tuple_generator_t = typename tuple_generator_type<type_generator, Sequence>::type;


}  // namespace detail

using detail::apply;
using detail::apply_all;

}  // namespace dali

#endif  // DALI_KERNELS_TUPLE_HELPERS_H_

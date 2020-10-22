// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_CORE_STATIC_MAP_H_
#define DALI_CORE_STATIC_MAP_H_

#ifdef __CUDACC__
#ifdef BOOST_PP_VARIADICS
#undef BOOST_PP_VARIADICS
#endif
#define BOOST_PP_VARIADICS 1
#endif

/** @file
 * 
 * Parts of this file are based on uSHET Library - CPP Magic
 *
 * This file defines compile-time map that are suitable for
 * specializing over different type pairs.
 *
 * The #TYPE_MAP macro switches over types and provides a typedef
 * with given name for that type in each case block.
 *
 * Types containing commas should be enclosed in parenthesis.
 *
 * Code blocks (case_code, default_type1_code, default_type2_code) 
 * must be enclosed in parenthesis if they contain commas.
 *
*/

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <boost/preprocessor/punctuation/remove_parens.hpp>

#define DALI_FIRST_ARG(a, ...) a
#define DALI_SECOND_ARG(a, b, ...) b

#define DALI_IS_PROBE(...) DALI_SECOND_ARG(__VA_ARGS__, 0)
#define DALI_PROBE() ~, 1

#define DALI_NOT(x) DALI_IS_PROBE(BOOST_PP_CAT(_NOT_, x))
#define _NOT_0 DALI_PROBE()

#define DALI_BOOL(x) DALI_NOT(DALI_NOT(x))

#define DALI_HAS_ARGS(...) DALI_BOOL(DALI_FIRST_ARG(_END_OF_ARGUMENTS_ __VA_ARGS__)())
#define _END_OF_ARGUMENTS_() 0

#define DALI_EVAL_IMPL(...) __VA_ARGS__
#define DALI_EVAL(...) DALI_EVAL_IMPL(__VA_ARGS__)

#define DALI_DEFER(m) m BOOST_PP_EMPTY BOOST_PP_EMPTY()()

#define DALI_IF(condition) BOOST_PP_CAT(DALI_IF_, condition)
#define DALI_IF_1(...) __VA_ARGS__
#define DALI_IF_0(...)

#define DALI_MAP(m, args, first, ...) \
  m(args, first) \
  DALI_IF(DALI_HAS_ARGS(__VA_ARGS__))(DALI_DEFER(_DALI_MAP)()(m, args, __VA_ARGS__)) \

#define _DALI_MAP() DALI_MAP

#define TYPE_MAP_GET_KEY(type_map) BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(0, type_map))
#define TYPE_MAP_GET_VALUES(type_map) BOOST_PP_TUPLE_ELEM(1, type_map)

#define TYPE_MAP_IMPL3(args, type) \
    case BOOST_PP_TUPLE_ELEM(1, args)<type>::value: \
    { \
        using BOOST_PP_TUPLE_ELEM(3, args) = type; { \
        BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(4, args)); \
        } \
    } \
    break; \

#define TYPE_MAP_IMPL2(args, ...) DALI_EVAL(DALI_MAP(TYPE_MAP_IMPL3, args, __VA_ARGS__))

#define TYPE_MAP_IMPL(r, args, type_map) \
case BOOST_PP_TUPLE_ELEM(1, args)<TYPE_MAP_GET_KEY(type_map)>::value: \
{ \
    using BOOST_PP_TUPLE_ELEM(2, args) = TYPE_MAP_GET_KEY(type_map); \
    switch (BOOST_PP_TUPLE_ELEM(0, args)) { \
            TYPE_MAP_IMPL2(args, BOOST_PP_REMOVE_PARENS(TYPE_MAP_GET_VALUES(type_map))) \
        default: { BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(5, args)); } \
    } \
} \
break; \

/// Pastes the case_code specialized for each type pair defined by type map as a switch of switches.
/// The specialization is performed by aliasing a particular type with a typedef named type1_name
/// for outer switch and type2_name for inner switch.
/// @param type1_id           - numerical id of the type 1
/// @param type2_id           - numerical id of the type 2
/// @param type_tag           - a class template usable as type_tag<type>::value
///                             the value should be a type id for type
/// @param type1_name         - a name given for selected type in the outer switch
/// @param type2_name         - a name given for selected type in the inner switch
/// @param type_map           - parenthesised, comma-separated list of pairs. First element is
///                             parenthesised key type. Second element is parenthesised,
///                             comma-separated list of value types
///                             e.g. (int, float, (std::conditional<val, bool, char>::type))
/// @param case_code          - code to execute for matching cases
/// @param default_type1_code - code to execute when id doesn't match any type in outer switch
/// @param default_type2_code - code to execute when id doesn't match any type in inner switch
///
/// Usage:
/// ```
/// #define TEST_TYPES_MAP (
///     ((uint8_t), (uint8_t, uint64_t, float)),
///     ((int8_t), (int64_t))
/// TYPE_MAP(
///     input_type,
///     output_type,
///     TypeTag,
///     InputType,
///     OutputType,
///     TEST_TYPES_MAP,
///     (TypedFunc<InputType, OutputType>();),
///     (cout << "Outer default"),
///     (cout << "Innder default"))
///
/// ```
#define TYPE_MAP(type1_id, type2_id, type_tag, type1_name, type2_name, \
  type_map, case_code, default_type1_code, default_type2_code) \
switch(type1_id) { \
    BOOST_PP_SEQ_FOR_EACH( \
    TYPE_MAP_IMPL, \
    (type2_id, type_tag, type1_name, type2_name, case_code, default_type2_code), \
    BOOST_PP_TUPLE_TO_SEQ(type_map)) \
    default: { BOOST_PP_REMOVE_PARENS(default_type1_code); } \
} \

#endif  // DALI_CORE_STATIC_MAP_H_

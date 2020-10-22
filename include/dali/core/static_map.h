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

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <boost/preprocessor/punctuation/remove_parens.hpp>

#define FIRST(a, ...) a
#define SECOND(a, b, ...) b

#define IS_PROBE(...) SECOND(__VA_ARGS__, 0)
#define PROBE() ~, 1

#define NOT(x) IS_PROBE(BOOST_PP_CAT(_NOT_, x))
#define _NOT_0 PROBE()

#define BOOL(x) NOT(NOT(x))

#define HAS_ARGS(...) BOOL(FIRST(_END_OF_ARGUMENTS_ __VA_ARGS__)())
#define _END_OF_ARGUMENTS_() 0

#define DALI_EVAL_IMPL(...) __VA_ARGS__
#define DALI_EVAL(...) DALI_EVAL_IMPL(__VA_ARGS__)

#define DALI_DEFER(m) m BOOST_PP_EMPTY BOOST_PP_EMPTY()()

#define DALI_IF(condition) BOOST_PP_CAT(DALI_IF_, condition)
#define DALI_IF_1(...) __VA_ARGS__ 
#define DALI_IF_0(...) 

#define DALI_MAP(m, args, first, ...) \
  m(args, first) \
  DALI_IF(HAS_ARGS(__VA_ARGS__))(DALI_DEFER(_DALI_MAP)()(m, args, __VA_ARGS__)) \

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

#define TYPE_MAP( \
  type1_id, type2_id, type_tag, type1_name, type2_name, type_map, case_code, default_type1_code, default_type2_code) \
switch(type1_id) { \
    BOOST_PP_SEQ_FOR_EACH( \
    TYPE_MAP_IMPL, \
    (type2_id, type_tag, type1_name, type2_name, case_code, default_type2_code), \
    BOOST_PP_TUPLE_TO_SEQ(type_map)) \
    default: { BOOST_PP_REMOVE_PARENS(default_type1_code); } \
} \

#endif  // DALI_CORE_STATIC_MAP_H_

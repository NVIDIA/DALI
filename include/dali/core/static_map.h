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

#define EMPTY()

#define EVAL(...) EVAL1024(__VA_ARGS__)
#define EVAL1024(...) EVAL512(EVAL512(__VA_ARGS__))
#define EVAL512(...) EVAL256(EVAL256(__VA_ARGS__))
#define EVAL256(...) EVAL128(EVAL128(__VA_ARGS__))
#define EVAL128(...) EVAL64(EVAL64(__VA_ARGS__))
#define EVAL64(...) EVAL32(EVAL32(__VA_ARGS__))
#define EVAL32(...) EVAL16(EVAL16(__VA_ARGS__))
#define EVAL16(...) EVAL8(EVAL8(__VA_ARGS__))
#define EVAL8(...) EVAL4(EVAL4(__VA_ARGS__))
#define EVAL4(...) EVAL2(EVAL2(__VA_ARGS__))
#define EVAL2(...) EVAL1(EVAL1(__VA_ARGS__))
#define EVAL1(...) __VA_ARGS__

#define DEFER1(m) m EMPTY()
#define DEFER2(m) m EMPTY EMPTY()()
#define DEFER3(m) m EMPTY EMPTY EMPTY()()()
#define DEFER4(m) m EMPTY EMPTY EMPTY EMPTY()()()()

#define IS_PROBE(...) SECOND(__VA_ARGS__, 0)
#define PROBE() ~, 1

#define CAT(a, b) a ## b

#define NOT(x) IS_PROBE(CAT(_NOT_, x))
#define _NOT_0 PROBE()

#define BOOL(x) NOT(NOT(x))

#define IF_ELSE(condition) _IF_ELSE(BOOL(condition))
#define _IF_ELSE(condition) CAT(_IF_, condition)

#define _IF_1(...) __VA_ARGS__ _IF_1_ELSE
#define _IF_0(...)             _IF_0_ELSE

#define _IF_1_ELSE(...)
#define _IF_0_ELSE(...) __VA_ARGS__

#define HAS_ARGS(...) BOOL(FIRST(_END_OF_ARGUMENTS_ __VA_ARGS__)())
#define _END_OF_ARGUMENTS_() 0

#define MAP(m, args, first, ...)           \
  m(args, first)                           \
  IF_ELSE(HAS_ARGS(__VA_ARGS__))(          \
    DEFER2(_MAP)()(m, args, __VA_ARGS__))  \
  (                                        \
    /* Do nothing, just terminate */       \
  )
#define _MAP() MAP

#define GET_KEY(type_map) BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(0, type_map))
#define GET_VALUES(type_map) BOOST_PP_TUPLE_ELEM(1, type_map)

#define TYPE_MAP_IMPL3(args, type) \
    case type2id<type>::value: \
    { \
        using BOOST_PP_TUPLE_ELEM(3, args) = type; { \
        BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(4, args)); \
        } \
    } \
    break; \

#define TYPE_MAP_INNER_IMPL(args, ...) EVAL(MAP(TYPE_MAP_IMPL3, args, __VA_ARGS__))

#define TYPE_MAP_IMPL(r, args, type_map) \
case type2id<GET_KEY(type_map)>::value: \
{ \
    using BOOST_PP_TUPLE_ELEM(2, args) = GET_KEY(type_map); \
    switch (BOOST_PP_TUPLE_ELEM(1, args)) { \
            TYPE_MAP_INNER_IMPL(args, BOOST_PP_REMOVE_PARENS(GET_VALUES(type_map))) \
        default: { BOOST_PP_REMOVE_PARENS(BOOST_PP_TUPLE_ELEM(5, args)); } \
    } \
} \
break;

#define TYPE_MAP( \
  type_map, in_id, out_id, in_tag, out_tag, case_code, default_in_code, default_out_code) \
switch(in_id) { \
    BOOST_PP_SEQ_FOR_EACH( \
    TYPE_MAP_IMPL, \
    (in_id, out_id, in_tag, out_tag, case_code, default_out_code), \
    BOOST_PP_TUPLE_TO_SEQ(type_map)) \
    default: { BOOST_PP_REMOVE_PARENS(default_in_code); } \
}


#endif  // DALI_CORE_STATIC_MAP_H_

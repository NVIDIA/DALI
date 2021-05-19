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

#ifndef DALI_CORE_STATIC_SWITCH_H_
#define DALI_CORE_STATIC_SWITCH_H_

#ifdef __CUDACC__
#ifdef BOOST_PP_VARIADICS
#undef BOOST_PP_VARIADICS
#endif
#define BOOST_PP_VARIADICS 1
#endif

/** @file
 *
 * This file defines two compile-time "switches" that are suitable for
 * specializing over different values or types.
 *
 * The #TYPE_SWITCH macro switches over types and provides a typedef
 * with given name for that type in each case block.
 *
 * The #VALUE_SWITCH macro does the same with values and defines a costant
 * with given name within each case block.
 *
 * The macros can be safely (?) nested within each other.
 *
 * Types and values containing commas should be enclosed in parenthesis.
 *
 * Code blocks (case, default) must be enclosed in parenthesis if they contain commas.
 *
 * Proposed usage in DALI pipeline:
 * ```
 * #define DALI_TYPE_SWITCH(id, type, types, ...) \
 *    TYPE_SWITCH(id, type, types, DALITypeTag, (__VA_ARGS__), \
 *                (DALI_FAIL("Type id does not match any of " #types);)
 * ```
 *
 * DALITypeTag is a proposed name for mapping types to DALITypeId
*/

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <boost/preprocessor/punctuation/remove_parens.hpp>

// HC SVNT DRACONES

#define DALI_REMOVE_PAREN_IMPL(...) __VA_ARGS__
#define DALI_REMOVE_PAREN(args) DALI_REMOVE_PAREN_IMPL args

#define DALI_TYPE_SWITCH_IMPL3(type_, type_tag_, type_name_, code_) \
  case type_tag_<BOOST_PP_REMOVE_PARENS(type_)>::value: { \
  using type_name_ = BOOST_PP_REMOVE_PARENS(type_); \
    BOOST_PP_REMOVE_PARENS(code_); \
  } break; \

#define DALI_TYPE_SWITCH_IMPL2(...) DALI_TYPE_SWITCH_IMPL3(__VA_ARGS__)

#define DALI_TYPE_SWITCH_IMPL(r, args, type) DALI_TYPE_SWITCH_IMPL2(type, DALI_REMOVE_PAREN(args))

/// Pastes the case_ code specialized for each type in types_.
/// The specialization is performed by aliasing a particular type with a typedef named type_name_.
/// @param id_         - numerical id of the type
/// @param type_tag_   - a class template usable as type_tag<type>::value
///                      the value should be a type id for type
/// @param type_name_  - a name given for selected type in the switch
/// @param types_      - parenthesised, comma-separated list of types
///                      types containing commas should be enclosed with parenthesis
///                      e.g. (int, float, (std::conditional<val, bool, char>::type))
/// @param case_       - code to execute for matching cases
/// @param default_    - code to execute when id doesn't match any type in types
///
/// Usage:
/// ```
/// TYPE_SWITCH(input_type, TypeTag, IType, (int, float, (some_type<args>::type), int64_t), (
///    TYPE_SWITCH(output_type, TypeTag, OType, (int, double, int64_t), (
///        VALUE_SWITCH(channels, num_channels, (1, 2, 3, 4), (
///            SomeFunctor<IType, OType, num_channels>(
///            inputs[0].data<IType>(), outputs[0].mutable_data<OType>());
///          ), assert(!"Unsupported number of channels");
///        )
///      ), assert(!"Unsupported output type");
///    )
///  ), assert(!"Unsupported input type");
/// )
/// ```
#define TYPE_SWITCH(id_, type_tag_, type_name_, types, case_, default_) switch (id_) { \
    BOOST_PP_SEQ_FOR_EACH(DALI_TYPE_SWITCH_IMPL, (type_tag_, type_name_, case_), \
                          BOOST_PP_VARIADIC_TO_SEQ(DALI_REMOVE_PAREN(types))) \
  default: \
  { BOOST_PP_REMOVE_PARENS(default_); } \
  }

#define DALI_VALUE_SWITCH_IMPL3(value_, value_name_, code_) case value_: { \
  const auto value_name_ = value_; \
    BOOST_PP_REMOVE_PARENS(code_); \
  } break; \

#define DALI_VALUE_SWITCH_IMPL2(...) DALI_VALUE_SWITCH_IMPL3(__VA_ARGS__)

#define DALI_VALUE_SWITCH_IMPL(r, args, value) \
  DALI_VALUE_SWITCH_IMPL2(value, DALI_REMOVE_PAREN(args))

/// Pastes the case_ code specialized for each value in values.
/// The specialization is performed by aliasing a value with a constant named constant_name_
/// @param value_         - a value to switch by
/// @param constant_name_ - a name given for selected type in the switch
/// @param values_        - parenthesised, comma-separated list of case labels;
///                         expressions containing commas should be enclosed with parenthesis
/// @param case_          - code to execute for matching cases
/// @param default_       - code to execute when value doesn't match any in values
///
/// Usage:
/// ```
/// VALUE_SWITCH(channels, num_channels, (1, 2, 3, 4), (
///     SomeFunctor<IType, OType, num_channels>(
///     inputs[0].data<IType>(), outputs[0].mutable_data<OType>());
///   ), assert(!"Unsupported number of channels");
/// )
/// ```
#define VALUE_SWITCH(value_, value_name_, values, case_, default_) switch (value_) { \
    BOOST_PP_SEQ_FOR_EACH(DALI_VALUE_SWITCH_IMPL, (value_name_, case_), \
                          BOOST_PP_VARIADIC_TO_SEQ(DALI_REMOVE_PAREN(values))) \
  default: \
  { BOOST_PP_REMOVE_PARENS(default_); } \
  }

/// Pastes the case_ code specialized for true and false values.
/// The specialization is performed by aliasing a value with a constant named constant_name_
/// @param expr_       - a boolean expression to switch by
/// @param const_name_ - a name given for the constexpr bool variable.
/// @param code_       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, (
///     std::integral_constant<bool, BoolConst> constant;
///     some_function<BoolConst>(...);
///   )
/// )
/// ```
#define BOOL_SWITCH(expr_, const_name_, code_)   \
  if (expr_) {                                   \
    constexpr bool const_name_ = true;           \
    BOOST_PP_REMOVE_PARENS(code_);               \
  } else {                                       \
    constexpr bool const_name_ = false;          \
    BOOST_PP_REMOVE_PARENS(code_);               \
  }

#endif  // DALI_CORE_STATIC_SWITCH_H_

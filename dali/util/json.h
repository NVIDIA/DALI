/*
    __ _____ _____ _____
 __|  |   __|     |   | |  JSON for Modern C++
|  |  |__   |  |  | | | |  version 3.1.2
|_____|_____|_____|_|___|  https://github.com/nlohmann/json

Licensed under the MIT License <http://opensource.org/licenses/MIT>.
SPDX-License-Identifier: MIT
Copyright (c) 2013-2018 Niels Lohmann <http://nlohmann.me>.

Permission is hereby  granted, free of charge, to any  person obtaining a copy
of this software and associated  documentation files (the "Software"), to deal
in the Software  without restriction, including without  limitation the rights
to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef NLOHMANN_JSON_HPP
#define NLOHMANN_JSON_HPP

#define NLOHMANN_JSON_VERSION_MAJOR 3
#define NLOHMANN_JSON_VERSION_MINOR 1
#define NLOHMANN_JSON_VERSION_PATCH 2

#include <algorithm> // all_of, find, for_each
#include <cassert> // assert
#include <ciso646> // and, not, or
#include <cstddef> // nullptr_t, ptrdiff_t, size_t
#include <functional> // hash, less
#include <initializer_list> // initializer_list
#include <iosfwd> // istream, ostream
#include <iterator> // iterator_traits, random_access_iterator_tag
#include <numeric> // accumulate
#include <string> // string, stoi, to_string
#include <utility> // declval, forward, move, pair, swap

// #include <nlohmann/json_fwd.hpp>
#ifndef NLOHMANN_JSON_FWD_HPP
#define NLOHMANN_JSON_FWD_HPP

#include <cstdint> // int64_t, uint64_t
#include <map> // map
#include <memory> // allocator
#include <string> // string
#include <vector> // vector

/*!
@brief namespace for Niels Lohmann
@see https://github.com/nlohmann
@since version 1.0.0
*/
namespace nlohmann {
/*!
@brief default JSONSerializer template argument

This serializer ignores the template arguments and uses ADL
([argument-dependent lookup](https://en.cppreference.com/w/cpp/language/adl))
for serialization.
*/
template<typename T = void, typename SFINAE = void>
struct adl_serializer;

template<template<typename U, typename V, typename... Args> class ObjectType =
         std::map,
         template<typename U, typename... Args> class ArrayType = std::vector,
         class StringType = std::string, class BooleanType = bool,
         class NumberIntegerType = std::int64_t,
         class NumberUnsignedType = std::uint64_t,
         class NumberFloatType = double,
         template<typename U> class AllocatorType = std::allocator,
         template<typename T, typename SFINAE = void> class JSONSerializer =
         adl_serializer>
class basic_json;

/*!
@brief JSON Pointer

A JSON pointer defines a string syntax for identifying a specific value
within a JSON document. It can be used with functions `at` and
`operator[]`. Furthermore, JSON pointers are the base for JSON patches.

@sa [RFC 6901](https://tools.ietf.org/html/rfc6901)

@since version 2.0.0
*/
template<typename BasicJsonType>
class json_pointer;

/*!
@brief default JSON class

This type is the default specialization of the @ref basic_json class which
uses the standard template types.

@since version 1.0.0
*/
using json = basic_json<>;
}

#endif

// #include <nlohmann/detail/macro_scope.hpp>


// This file contains all internal macro definitions
// You MUST include macro_unscope.hpp at the end of json.hpp to undef all of them

// exclude unsupported compilers
#if !defined(JSON_SKIP_UNSUPPORTED_COMPILER_CHECK)
    #if defined(__clang__)
        #if (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__) < 30400
            #error "unsupported Clang version - see https://github.com/nlohmann/json#supported-compilers"
        #endif
    #elif defined(__GNUC__) && !(defined(__ICC) || defined(__INTEL_COMPILER))
        #if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 40900
            #error "unsupported GCC version - see https://github.com/nlohmann/json#supported-compilers"
        #endif
    #endif
#endif

// disable float-equal warnings on GCC/clang
#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

// disable documentation warnings on clang
#if defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdocumentation"
#endif

// allow for portable deprecation warnings
#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
    #define JSON_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
    #define JSON_DEPRECATED __declspec(deprecated)
#else
    #define JSON_DEPRECATED
#endif

// allow to disable exceptions
#if (defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)) && !defined(JSON_NOEXCEPTION)
    #define JSON_THROW(exception) throw exception
    #define JSON_TRY try
    #define JSON_CATCH(exception) catch(exception)
#else
    #define JSON_THROW(exception) std::abort()
    #define JSON_TRY if(true)
    #define JSON_CATCH(exception) if(false)
#endif

// override exception macros
#if defined(JSON_THROW_USER)
    #undef JSON_THROW
    #define JSON_THROW JSON_THROW_USER
#endif
#if defined(JSON_TRY_USER)
    #undef JSON_TRY
    #define JSON_TRY JSON_TRY_USER
#endif
#if defined(JSON_CATCH_USER)
    #undef JSON_CATCH
    #define JSON_CATCH JSON_CATCH_USER
#endif

// manual branch prediction
#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
    #define JSON_LIKELY(x)      __builtin_expect(!!(x), 1)
    #define JSON_UNLIKELY(x)    __builtin_expect(!!(x), 0)
#else
    #define JSON_LIKELY(x)      x
    #define JSON_UNLIKELY(x)    x
#endif

// C++ language standard detection
#if (defined(__cplusplus) && __cplusplus >= 201703L) || (defined(_HAS_CXX17) && _HAS_CXX17 == 1) // fix for issue #464
    #define JSON_HAS_CPP_17
    #define JSON_HAS_CPP_14
#elif (defined(__cplusplus) && __cplusplus >= 201402L) || (defined(_HAS_CXX14) && _HAS_CXX14 == 1)
    #define JSON_HAS_CPP_14
#endif

// Ugly macros to avoid uglier copy-paste when specializing basic_json. They
// may be removed in the future once the class is split.

#define NLOHMANN_BASIC_JSON_TPL_DECLARATION                                \
    template<template<typename, typename, typename...> class ObjectType,   \
             template<typename, typename...> class ArrayType,              \
             class StringType, class BooleanType, class NumberIntegerType, \
             class NumberUnsignedType, class NumberFloatType,              \
             template<typename> class AllocatorType,                       \
             template<typename, typename = void> class JSONSerializer>

#define NLOHMANN_BASIC_JSON_TPL                                            \
    basic_json<ObjectType, ArrayType, StringType, BooleanType,             \
    NumberIntegerType, NumberUnsignedType, NumberFloatType,                \
    AllocatorType, JSONSerializer>

/*!
@brief Helper to determine whether there's a key_type for T.

This helper is used to tell associative containers apart from other containers
such as sequence containers. For instance, `std::map` passes the test as it
contains a `mapped_type`, whereas `std::vector` fails the test.

@sa http://stackoverflow.com/a/7728728/266378
@since version 1.0.0, overworked in version 2.0.6
*/
#define NLOHMANN_JSON_HAS_HELPER(type)                                        \
    template<typename T> struct has_##type {                                  \
    private:                                                                  \
        template<typename U, typename = typename U::type>                     \
        static int detect(U &&);                                              \
        static void detect(...);                                              \
    public:                                                                   \
        static constexpr bool value =                                         \
                std::is_integral<decltype(detect(std::declval<T>()))>::value; \
    }

// #include <nlohmann/detail/meta.hpp>


#include <ciso646> // not
#include <cstddef> // size_t
#include <limits> // numeric_limits
#include <type_traits> // conditional, enable_if, false_type, integral_constant, is_constructible, is_integral, is_same, remove_cv, remove_reference, true_type
#include <utility> // declval

// #include <nlohmann/json_fwd.hpp>

// #include <nlohmann/detail/macro_scope.hpp>


namespace nlohmann
{
/*!
@brief detail namespace with internal helper functions

This namespace collects functions that should not be exposed,
implementations of some @ref basic_json methods, and meta-programming helpers.

@since version 2.1.0
*/
namespace detail
{
/////////////
// helpers //
/////////////

template<typename> struct is_basic_json : std::false_type {};

NLOHMANN_BASIC_JSON_TPL_DECLARATION
struct is_basic_json<NLOHMANN_BASIC_JSON_TPL> : std::true_type {};

// alias templates to reduce boilerplate
template<bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template<typename T>
using uncvref_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

// implementation of C++14 index_sequence and affiliates
// source: https://stackoverflow.com/a/32223343
template<std::size_t... Ints>
struct index_sequence
{
    using type = index_sequence;
    using value_type = std::size_t;
    static constexpr std::size_t size() noexcept
    {
        return sizeof...(Ints);
    }
};

template<class Sequence1, class Sequence2>
struct merge_and_renumber;

template<std::size_t... I1, std::size_t... I2>
struct merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
        : index_sequence < I1..., (sizeof...(I1) + I2)... > {};

template<std::size_t N>
struct make_index_sequence
    : merge_and_renumber < typename make_index_sequence < N / 2 >::type,
      typename make_index_sequence < N - N / 2 >::type > {};

template<> struct make_index_sequence<0> : index_sequence<> {};
template<> struct make_index_sequence<1> : index_sequence<0> {};

template<typename... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

/*
Implementation of two C++17 constructs: conjunction, negation. This is needed
to avoid evaluating all the traits in a condition

For example: not std::is_same<void, T>::value and has_value_type<T>::value
will not compile when T = void (on MSVC at least). Whereas
conjunction<negation<std::is_same<void, T>>, has_value_type<T>>::value will
stop evaluating if negation<...>::value == false

Please note that those constructs must be used with caution, since symbols can
become very long quickly (which can slow down compilation and cause MSVC
internal compiler errors). Only use it when you have to (see example ahead).
*/
template<class...> struct conjunction : std::true_type {};
template<class B1> struct conjunction<B1> : B1 {};
template<class B1, class... Bn>
struct conjunction<B1, Bn...> : std::conditional<bool(B1::value), conjunction<Bn...>, B1>::type {};

template<class B> struct negation : std::integral_constant<bool, not B::value> {};

// dispatch utility (taken from ranges-v3)
template<unsigned N> struct priority_tag : priority_tag < N - 1 > {};
template<> struct priority_tag<0> {};

////////////////////////
// has_/is_ functions //
////////////////////////

// source: https://stackoverflow.com/a/37193089/4116453

template <typename T, typename = void>
struct is_complete_type : std::false_type {};

template <typename T>
struct is_complete_type<T, decltype(void(sizeof(T)))> : std::true_type {};

NLOHMANN_JSON_HAS_HELPER(mapped_type);
NLOHMANN_JSON_HAS_HELPER(key_type);
NLOHMANN_JSON_HAS_HELPER(value_type);
NLOHMANN_JSON_HAS_HELPER(iterator);

template<bool B, class RealType, class CompatibleObjectType>
struct is_compatible_object_type_impl : std::false_type {};

template<class RealType, class CompatibleObjectType>
struct is_compatible_object_type_impl<true, RealType, CompatibleObjectType>
{
    static constexpr auto value =
        std::is_constructible<typename RealType::key_type, typename CompatibleObjectType::key_type>::value and
        std::is_constructible<typename RealType::mapped_type, typename CompatibleObjectType::mapped_type>::value;
};

template<bool B, class RealType, class CompatibleStringType>
struct is_compatible_string_type_impl : std::false_type {};

template<class RealType, class CompatibleStringType>
struct is_compatible_string_type_impl<true, RealType, CompatibleStringType>
{
    static constexpr auto value =
        std::is_same<typename RealType::value_type, typename CompatibleStringType::value_type>::value and
        std::is_constructible<RealType, CompatibleStringType>::value;
};

template<class BasicJsonType, class CompatibleObjectType>
struct is_compatible_object_type
{
    static auto constexpr value = is_compatible_object_type_impl <
                                  conjunction<negation<std::is_same<void, CompatibleObjectType>>,
                                  has_mapped_type<CompatibleObjectType>,
                                  has_key_type<CompatibleObjectType>>::value,
                                  typename BasicJsonType::object_t, CompatibleObjectType >::value;
};

template<class BasicJsonType, class CompatibleStringType>
struct is_compatible_string_type
{
    static auto constexpr value = is_compatible_string_type_impl <
                                  conjunction<negation<std::is_same<void, CompatibleStringType>>,
                                  has_value_type<CompatibleStringType>>::value,
                                  typename BasicJsonType::string_t, CompatibleStringType >::value;
};

template<typename BasicJsonType, typename T>
struct is_basic_json_nested_type
{
    static auto constexpr value = std::is_same<T, typename BasicJsonType::iterator>::value or
                                  std::is_same<T, typename BasicJsonType::const_iterator>::value or
                                  std::is_same<T, typename BasicJsonType::reverse_iterator>::value or
                                  std::is_same<T, typename BasicJsonType::const_reverse_iterator>::value;
};

template<class BasicJsonType, class CompatibleArrayType>
struct is_compatible_array_type
{
    static auto constexpr value =
        conjunction<negation<std::is_same<void, CompatibleArrayType>>,
        negation<is_compatible_object_type<
        BasicJsonType, CompatibleArrayType>>,
        negation<std::is_constructible<typename BasicJsonType::string_t,
        CompatibleArrayType>>,
        negation<is_basic_json_nested_type<BasicJsonType, CompatibleArrayType>>,
        has_value_type<CompatibleArrayType>,
        has_iterator<CompatibleArrayType>>::value;
};

template<bool, typename, typename>
struct is_compatible_integer_type_impl : std::false_type {};

template<typename RealIntegerType, typename CompatibleNumberIntegerType>
struct is_compatible_integer_type_impl<true, RealIntegerType, CompatibleNumberIntegerType>
{
    // is there an assert somewhere on overflows?
    using RealLimits = std::numeric_limits<RealIntegerType>;
    using CompatibleLimits = std::numeric_limits<CompatibleNumberIntegerType>;

    static constexpr auto value =
        std::is_constructible<RealIntegerType, CompatibleNumberIntegerType>::value and
        CompatibleLimits::is_integer and
        RealLimits::is_signed == CompatibleLimits::is_signed;
};

template<typename RealIntegerType, typename CompatibleNumberIntegerType>
struct is_compatible_integer_type
{
    static constexpr auto value =
        is_compatible_integer_type_impl <
        std::is_integral<CompatibleNumberIntegerType>::value and
        not std::is_same<bool, CompatibleNumberIntegerType>::value,
        RealIntegerType, CompatibleNumberIntegerType > ::value;
};

// trait checking if JSONSerializer<T>::from_json(json const&, udt&) exists
template<typename BasicJsonType, typename T>
struct has_from_json
{
  private:
    // also check the return type of from_json
    template<typename U, typename = enable_if_t<std::is_same<void, decltype(uncvref_t<U>::from_json(
                 std::declval<BasicJsonType>(), std::declval<T&>()))>::value>>
    static int detect(U&&);
    static void detect(...);

  public:
    static constexpr bool value = std::is_integral<decltype(
                                      detect(std::declval<typename BasicJsonType::template json_serializer<T, void>>()))>::value;
};

// This trait checks if JSONSerializer<T>::from_json(json const&) exists
// this overload is used for non-default-constructible user-defined-types
template<typename BasicJsonType, typename T>
struct has_non_default_from_json
{
  private:
    template <
        typename U,
        typename = enable_if_t<std::is_same<
                                   T, decltype(uncvref_t<U>::from_json(std::declval<BasicJsonType>()))>::value >>
    static int detect(U&&);
    static void detect(...);

  public:
    static constexpr bool value = std::is_integral<decltype(detect(
                                      std::declval<typename BasicJsonType::template json_serializer<T, void>>()))>::value;
};

// This trait checks if BasicJsonType::json_serializer<T>::to_json exists
template<typename BasicJsonType, typename T>
struct has_to_json
{
  private:
    template<typename U, typename = decltype(uncvref_t<U>::to_json(
                 std::declval<BasicJsonType&>(), std::declval<T>()))>
    static int detect(U&&);
    static void detect(...);

  public:
    static constexpr bool value = std::is_integral<decltype(detect(
                                      std::declval<typename BasicJsonType::template json_serializer<T, void>>()))>::value;
};

template <typename BasicJsonType, typename CompatibleCompleteType>
struct is_compatible_complete_type
{
    static constexpr bool value =
        not std::is_base_of<std::istream, CompatibleCompleteType>::value and
        not is_basic_json<CompatibleCompleteType>::value and
        not is_basic_json_nested_type<BasicJsonType, CompatibleCompleteType>::value and
        has_to_json<BasicJsonType, CompatibleCompleteType>::value;
};

template <typename BasicJsonType, typename CompatibleType>
struct is_compatible_type
    : conjunction<is_complete_type<CompatibleType>,
      is_compatible_complete_type<BasicJsonType, CompatibleType>>
{
};

// taken from ranges-v3
template<typename T>
struct static_const
{
    static constexpr T value{};
};

template<typename T>
constexpr T static_const<T>::value;
}
}

// #include <nlohmann/detail/exceptions.hpp>


#include <exception> // exception
#include <stdexcept> // runtime_error
#include <string> // to_string

namespace nlohmann
{
namespace detail
{
////////////////
// exceptions //
////////////////

/*!
@brief general exception of the @ref basic_json class

This class is an extension of `std::exception` objects with a member @a id for
exception ids. It is used as the base class for all exceptions thrown by the
@ref basic_json class. This class can hence be used as "wildcard" to catch
exceptions.

Subclasses:
- @ref parse_error for exceptions indicating a parse error
- @ref invalid_iterator for exceptions indicating errors with iterators
- @ref type_error for exceptions indicating executing a member function with
                  a wrong type
- @ref out_of_range for exceptions indicating access out of the defined range
- @ref other_error for exceptions indicating other library errors

@internal
@note To have nothrow-copy-constructible exceptions, we internally use
      `std::runtime_error` which can cope with arbitrary-length error messages.
      Intermediate strings are built with static functions and then passed to
      the actual constructor.
@endinternal

@liveexample{The following code shows how arbitrary library exceptions can be
caught.,exception}

@since version 3.0.0
*/
class exception : public std::exception
{
  public:
    /// returns the explanatory string
    const char* what() const noexcept override
    {
        return m.what();
    }

    /// the id of the exception
    const int id;

  protected:
    exception(int id_, const char* what_arg) : id(id_), m(what_arg) {}

    static std::string name(const std::string& ename, int id_)
    {
        return "[json.exception." + ename + "." + std::to_string(id_) + "] ";
    }

  private:
    /// an exception object as storage for error messages
    std::runtime_error m;
};

/*!
@brief exception indicating a parse error

This exception is thrown by the library when a parse error occurs. Parse errors
can occur during the deserialization of JSON text, CBOR, MessagePack, as well
as when using JSON Patch.

Member @a byte holds the byte index of the last read character in the input
file.

Exceptions have ids 1xx.

name / id                      | example message | description
------------------------------ | --------------- | -------------------------
json.exception.parse_error.101 | parse error at 2: unexpected end of input; expected string literal | This error indicates a syntax error while deserializing a JSON text. The error message describes that an unexpected token (character) was encountered, and the member @a byte indicates the error position.
json.exception.parse_error.102 | parse error at 14: missing or wrong low surrogate | JSON uses the `\uxxxx` format to describe Unicode characters. Code points above above 0xFFFF are split into two `\uxxxx` entries ("surrogate pairs"). This error indicates that the surrogate pair is incomplete or contains an invalid code point.
json.exception.parse_error.103 | parse error: code points above 0x10FFFF are invalid | Unicode supports code points up to 0x10FFFF. Code points above 0x10FFFF are invalid.
json.exception.parse_error.104 | parse error: JSON patch must be an array of objects | [RFC 6902](https://tools.ietf.org/html/rfc6902) requires a JSON Patch document to be a JSON document that represents an array of objects.
json.exception.parse_error.105 | parse error: operation must have string member 'op' | An operation of a JSON Patch document must contain exactly one "op" member, whose value indicates the operation to perform. Its value must be one of "add", "remove", "replace", "move", "copy", or "test"; other values are errors.
json.exception.parse_error.106 | parse error: array index '01' must not begin with '0' | An array index in a JSON Pointer ([RFC 6901](https://tools.ietf.org/html/rfc6901)) may be `0` or any number without a leading `0`.
json.exception.parse_error.107 | parse error: JSON pointer must be empty or begin with '/' - was: 'foo' | A JSON Pointer must be a Unicode string containing a sequence of zero or more reference tokens, each prefixed by a `/` character.
json.exception.parse_error.108 | parse error: escape character '~' must be followed with '0' or '1' | In a JSON Pointer, only `~0` and `~1` are valid escape sequences.
json.exception.parse_error.109 | parse error: array index 'one' is not a number | A JSON Pointer array index must be a number.
json.exception.parse_error.110 | parse error at 1: cannot read 2 bytes from vector | When parsing CBOR or MessagePack, the byte vector ends before the complete value has been read.
json.exception.parse_error.112 | parse error at 1: error reading CBOR; last byte: 0xF8 | Not all types of CBOR or MessagePack are supported. This exception occurs if an unsupported byte was read.
json.exception.parse_error.113 | parse error at 2: expected a CBOR string; last byte: 0x98 | While parsing a map key, a value that is not a string has been read.

@note For an input with n bytes, 1 is the index of the first character and n+1
      is the index of the terminating null byte or the end of file. This also
      holds true when reading a byte vector (CBOR or MessagePack).

@liveexample{The following code shows how a `parse_error` exception can be
caught.,parse_error}

@sa @ref exception for the base class of the library exceptions
@sa @ref invalid_iterator for exceptions indicating errors with iterators
@sa @ref type_error for exceptions indicating executing a member function with
                    a wrong type
@sa @ref out_of_range for exceptions indicating access out of the defined range
@sa @ref other_error for exceptions indicating other library errors

@since version 3.0.0
*/
class parse_error : public exception
{
  public:
    /*!
    @brief create a parse error exception
    @param[in] id_       the id of the exception
    @param[in] byte_     the byte index where the error occurred (or 0 if the
                         position cannot be determined)
    @param[in] what_arg  the explanatory string
    @return parse_error object
    */
    static parse_error create(int id_, std::size_t byte_, const std::string& what_arg)
    {
        std::string w = exception::name("parse_error", id_) + "parse error" +
                        (byte_ != 0 ? (" at " + std::to_string(byte_)) : "") +
                        ": " + what_arg;
        return parse_error(id_, byte_, w.c_str());
    }

    /*!
    @brief byte index of the parse error

    The byte index of the last read character in the input file.

    @note For an input with n bytes, 1 is the index of the first character and
          n+1 is the index of the terminating null byte or the end of file.
          This also holds true when reading a byte vector (CBOR or MessagePack).
    */
    const std::size_t byte;

  private:
    parse_error(int id_, std::size_t byte_, const char* what_arg)
        : exception(id_, what_arg), byte(byte_) {}
};

/*!
@brief exception indicating errors with iterators

This exception is thrown if iterators passed to a library function do not match
the expected semantics.

Exceptions have ids 2xx.

name / id                           | example message | description
----------------------------------- | --------------- | -------------------------
json.exception.invalid_iterator.201 | iterators are not compatible | The iterators passed to constructor @ref basic_json(InputIT first, InputIT last) are not compatible, meaning they do not belong to the same container. Therefore, the range (@a first, @a last) is invalid.
json.exception.invalid_iterator.202 | iterator does not fit current value | In an erase or insert function, the passed iterator @a pos does not belong to the JSON value for which the function was called. It hence does not define a valid position for the deletion/insertion.
json.exception.invalid_iterator.203 | iterators do not fit current value | Either iterator passed to function @ref erase(IteratorType first, IteratorType last) does not belong to the JSON value from which values shall be erased. It hence does not define a valid range to delete values from.
json.exception.invalid_iterator.204 | iterators out of range | When an iterator range for a primitive type (number, boolean, or string) is passed to a constructor or an erase function, this range has to be exactly (@ref begin(), @ref end()), because this is the only way the single stored value is expressed. All other ranges are invalid.
json.exception.invalid_iterator.205 | iterator out of range | When an iterator for a primitive type (number, boolean, or string) is passed to an erase function, the iterator has to be the @ref begin() iterator, because it is the only way to address the stored value. All other iterators are invalid.
json.exception.invalid_iterator.206 | cannot construct with iterators from null | The iterators passed to constructor @ref basic_json(InputIT first, InputIT last) belong to a JSON null value and hence to not define a valid range.
json.exception.invalid_iterator.207 | cannot use key() for non-object iterators | The key() member function can only be used on iterators belonging to a JSON object, because other types do not have a concept of a key.
json.exception.invalid_iterator.208 | cannot use operator[] for object iterators | The operator[] to specify a concrete offset cannot be used on iterators belonging to a JSON object, because JSON objects are unordered.
json.exception.invalid_iterator.209 | cannot use offsets with object iterators | The offset operators (+, -, +=, -=) cannot be used on iterators belonging to a JSON object, because JSON objects are unordered.
json.exception.invalid_iterator.210 | iterators do not fit | The iterator range passed to the insert function are not compatible, meaning they do not belong to the same container. Therefore, the range (@a first, @a last) is invalid.
json.exception.invalid_iterator.211 | passed iterators may not belong to container | The iterator range passed to the insert function must not be a subrange of the container to insert to.
json.exception.invalid_iterator.212 | cannot compare iterators of different containers | When two iterators are compared, they must belong to the same container.
json.exception.invalid_iterator.213 | cannot compare order of object iterators | The order of object iterators cannot be compared, because JSON objects are unordered.
json.exception.invalid_iterator.214 | cannot get value | Cannot get value for iterator: Either the iterator belongs to a null value or it is an iterator to a primitive type (number, boolean, or string), but the iterator is different to @ref begin().

@liveexample{The following code shows how an `invalid_iterator` exception can be
caught.,invalid_iterator}

@sa @ref exception for the base class of the library exceptions
@sa @ref parse_error for exceptions indicating a parse error
@sa @ref type_error for exceptions indicating executing a member function with
                    a wrong type
@sa @ref out_of_range for exceptions indicating access out of the defined range
@sa @ref other_error for exceptions indicating other library errors

@since version 3.0.0
*/
class invalid_iterator : public exception
{
  public:
    static invalid_iterator create(int id_, const std::string& what_arg)
    {
        std::string w = exception::name("invalid_iterator", id_) + what_arg;
        return invalid_iterator(id_, w.c_str());
    }

  private:
    invalid_iterator(int id_, const char* what_arg)
        : exception(id_, what_arg) {}
};

/*!
@brief exception indicating executing a member function with a wrong type

This exception is thrown in case of a type error; that is, a library function is
executed on a JSON value whose type does not match the expected semantics.

Exceptions have ids 3xx.

name / id                     | example message | description
----------------------------- | --------------- | -------------------------
json.exception.type_error.301 | cannot create object from initializer list | To create an object from an initializer list, the initializer list must consist only of a list of pairs whose first element is a string. When this constraint is violated, an array is created instead.
json.exception.type_error.302 | type must be object, but is array | During implicit or explicit value conversion, the JSON type must be compatible to the target type. For instance, a JSON string can only be converted into string types, but not into numbers or boolean types.
json.exception.type_error.303 | incompatible ReferenceType for get_ref, actual type is object | To retrieve a reference to a value stored in a @ref basic_json object with @ref get_ref, the type of the reference must match the value type. For instance, for a JSON array, the @a ReferenceType must be @ref array_t&.
json.exception.type_error.304 | cannot use at() with string | The @ref at() member functions can only be executed for certain JSON types.
json.exception.type_error.305 | cannot use operator[] with string | The @ref operator[] member functions can only be executed for certain JSON types.
json.exception.type_error.306 | cannot use value() with string | The @ref value() member functions can only be executed for certain JSON types.
json.exception.type_error.307 | cannot use erase() with string | The @ref erase() member functions can only be executed for certain JSON types.
json.exception.type_error.308 | cannot use push_back() with string | The @ref push_back() and @ref operator+= member functions can only be executed for certain JSON types.
json.exception.type_error.309 | cannot use insert() with | The @ref insert() member functions can only be executed for certain JSON types.
json.exception.type_error.310 | cannot use swap() with number | The @ref swap() member functions can only be executed for certain JSON types.
json.exception.type_error.311 | cannot use emplace_back() with string | The @ref emplace_back() member function can only be executed for certain JSON types.
json.exception.type_error.312 | cannot use update() with string | The @ref update() member functions can only be executed for certain JSON types.
json.exception.type_error.313 | invalid value to unflatten | The @ref unflatten function converts an object whose keys are JSON Pointers back into an arbitrary nested JSON value. The JSON Pointers must not overlap, because then the resulting value would not be well defined.
json.exception.type_error.314 | only objects can be unflattened | The @ref unflatten function only works for an object whose keys are JSON Pointers.
json.exception.type_error.315 | values in object must be primitive | The @ref unflatten function only works for an object whose keys are JSON Pointers and whose values are primitive.
json.exception.type_error.316 | invalid UTF-8 byte at index 10: 0x7E | The @ref dump function only works with UTF-8 encoded strings; that is, if you assign a `std::string` to a JSON value, make sure it is UTF-8 encoded. |

@liveexample{The following code shows how a `type_error` exception can be
caught.,type_error}

@sa @ref exception for the base class of the library exceptions
@sa @ref parse_error for exceptions indicating a parse error
@sa @ref invalid_iterator for exceptions indicating errors with iterators
@sa @ref out_of_range for exceptions indicating access out of the defined range
@sa @ref other_error for exceptions indicating other library errors

@since version 3.0.0
*/
class type_error : public exception
{
  public:
    static type_error create(int id_, const std::string& what_arg)
    {
        std::string w = exception::name("type_error", id_) + what_arg;
        return type_error(id_, w.c_str());
    }

  private:
    type_error(int id_, const char* what_arg) : exception(id_, what_arg) {}
};

/*!
@brief exception indicating access out of the defined range

This exception is thrown in case a library function is called on an input
parameter that exceeds the expected range, for instance in case of array
indices or nonexisting object keys.

Exceptions have ids 4xx.

name / id                       | example message | description
------------------------------- | --------------- | -------------------------
json.exception.out_of_range.401 | array index 3 is out of range | The provided array index @a i is larger than @a size-1.
json.exception.out_of_range.402 | array index '-' (3) is out of range | The special array index `-` in a JSON Pointer never describes a valid element of the array, but the index past the end. That is, it can only be used to add elements at this position, but not to read it.
json.exception.out_of_range.403 | key 'foo' not found | The provided key was not found in the JSON object.
json.exception.out_of_range.404 | unresolved reference token 'foo' | A reference token in a JSON Pointer could not be resolved.
json.exception.out_of_range.405 | JSON pointer has no parent | The JSON Patch operations 'remove' and 'add' can not be applied to the root element of the JSON value.
json.exception.out_of_range.406 | number overflow parsing '10E1000' | A parsed number could not be stored as without changing it to NaN or INF.
json.exception.out_of_range.407 | number overflow serializing '9223372036854775808' | UBJSON only supports integers numbers up to 9223372036854775807. |
json.exception.out_of_range.408 | excessive array size: 8658170730974374167 | The size (following `#`) of an UBJSON array or object exceeds the maximal capacity. |

@liveexample{The following code shows how an `out_of_range` exception can be
caught.,out_of_range}

@sa @ref exception for the base class of the library exceptions
@sa @ref parse_error for exceptions indicating a parse error
@sa @ref invalid_iterator for exceptions indicating errors with iterators
@sa @ref type_error for exceptions indicating executing a member function with
                    a wrong type
@sa @ref other_error for exceptions indicating other library errors

@since version 3.0.0
*/
class out_of_range : public exception
{
  public:
    static out_of_range create(int id_, const std::string& what_arg)
    {
        std::string w = exception::name("out_of_range", id_) + what_arg;
        return out_of_range(id_, w.c_str());
    }

  private:
    out_of_range(int id_, const char* what_arg) : exception(id_, what_arg) {}
};

/*!
@brief exception indicating other library errors

This exception is thrown in case of errors that cannot be classified with the
other exception types.

Exceptions have ids 5xx.

name / id                      | example message | description
------------------------------ | --------------- | -------------------------
json.exception.other_error.501 | unsuccessful: {"op":"test","path":"/baz", "value":"bar"} | A JSON Patch operation 'test' failed. The unsuccessful operation is also printed.

@sa @ref exception for the base class of the library exceptions
@sa @ref parse_error for exceptions indicating a parse error
@sa @ref invalid_iterator for exceptions indicating errors with iterators
@sa @ref type_error for exceptions indicating executing a member function with
                    a wrong type
@sa @ref out_of_range for exceptions indicating access out of the defined range

@liveexample{The following code shows how an `other_error` exception can be
caught.,other_error}

@since version 3.0.0
*/
class other_error : public exception
{
  public:
    static other_error create(int id_, const std::string& what_arg)
    {
        std::string w = exception::name("other_error", id_) + what_arg;
        return other_error(id_, w.c_str());
    }

  private:
    other_error(int id_, const char* what_arg) : exception(id_, what_arg) {}
};
}
}

// #include <nlohmann/detail/value_t.hpp>


#include <array> // array
#include <ciso646> // and
#include <cstddef> // size_t
#include <cstdint> // uint8_t

namespace nlohmann
{
namespace detail
{
///////////////////////////
// JSON type enumeration //
///////////////////////////

/*!
@brief the JSON type enumeration

This enumeration collects the different JSON types. It is internally used to
distinguish the stored values, and the functions @ref basic_json::is_null(),
@ref basic_json::is_object(), @ref basic_json::is_array(),
@ref basic_json::is_string(), @ref basic_json::is_boolean(),
@ref basic_json::is_number() (with @ref basic_json::is_number_integer(),
@ref basic_json::is_number_unsigned(), and @ref basic_json::is_number_float()),
@ref basic_json::is_discarded(), @ref basic_json::is_primitive(), and
@ref basic_json::is_structured() rely on it.

@note There are three enumeration entries (number_integer, number_unsigned, and
number_float), because the library distinguishes these three types for numbers:
@ref basic_json::number_unsigned_t is used for unsigned integers,
@ref basic_json::number_integer_t is used for signed integers, and
@ref basic_json::number_float_t is used for floating-point numbers or to
approximate integers which do not fit in the limits of their respective type.

@sa @ref basic_json::basic_json(const value_t value_type) -- create a JSON
value with the default value for a given type

@since version 1.0.0
*/
enum class value_t : std::uint8_t
{
    null,             ///< null value
    object,           ///< object (unordered set of name/value pairs)
    array,            ///< array (ordered collection of values)
    string,           ///< string value
    boolean,          ///< boolean value
    number_integer,   ///< number value (signed integer)
    number_unsigned,  ///< number value (unsigned integer)
    number_float,     ///< number value (floating-point)
    discarded         ///< discarded by the the parser callback function
};

/*!
@brief comparison operator for JSON types

Returns an ordering that is similar to Python:
- order: null < boolean < number < object < array < string
- furthermore, each type is not smaller than itself
- discarded values are not comparable

@since version 1.0.0
*/
inline bool operator<(const value_t lhs, const value_t rhs) noexcept
{
    static constexpr std::array<std::uint8_t, 8> order = {{
            0 /* null */, 3 /* object */, 4 /* array */, 5 /* string */,
            1 /* boolean */, 2 /* integer */, 2 /* unsigned */, 2 /* float */
        }
    };

    const auto l_index = static_cast<std::size_t>(lhs);
    const auto r_index = static_cast<std::size_t>(rhs);
    return l_index < order.size() and r_index < order.size() and order[l_index] < order[r_index];
}
}
}

// #include <nlohmann/detail/conversions/from_json.hpp>


#include <algorithm> // transform
#include <array> // array
#include <ciso646> // and, not
#include <forward_list> // forward_list
#include <iterator> // inserter, front_inserter, end
#include <map> // map
#include <string> // string
#include <tuple> // tuple, make_tuple
#include <type_traits> // is_arithmetic, is_same, is_enum, underlying_type, is_convertible
#include <unordered_map> // unordered_map
#include <utility> // pair, declval
#include <valarray> // valarray

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta.hpp>

// #include <nlohmann/detail/value_t.hpp>


namespace nlohmann
{
namespace detail
{
template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename std::nullptr_t& n)
{
    if (JSON_UNLIKELY(not j.is_null()))
    {
        JSON_THROW(type_error::create(302, "type must be null, but is " + std::string(j.type_name())));
    }
    n = nullptr;
}

// overloads for basic_json template parameters
template<typename BasicJsonType, typename ArithmeticType,
         enable_if_t<std::is_arithmetic<ArithmeticType>::value and
                     not std::is_same<ArithmeticType, typename BasicJsonType::boolean_t>::value,
                     int> = 0>
void get_arithmetic_value(const BasicJsonType& j, ArithmeticType& val)
{
    switch (static_cast<value_t>(j))
    {
        case value_t::number_unsigned:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_unsigned_t*>());
            break;
        }
        case value_t::number_integer:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_integer_t*>());
            break;
        }
        case value_t::number_float:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_float_t*>());
            break;
        }

        default:
            JSON_THROW(type_error::create(302, "type must be number, but is " + std::string(j.type_name())));
    }
}

template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::boolean_t& b)
{
    if (JSON_UNLIKELY(not j.is_boolean()))
    {
        JSON_THROW(type_error::create(302, "type must be boolean, but is " + std::string(j.type_name())));
    }
    b = *j.template get_ptr<const typename BasicJsonType::boolean_t*>();
}

template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::string_t& s)
{
    if (JSON_UNLIKELY(not j.is_string()))
    {
        JSON_THROW(type_error::create(302, "type must be string, but is " + std::string(j.type_name())));
    }
    s = *j.template get_ptr<const typename BasicJsonType::string_t*>();
}

template <
    typename BasicJsonType, typename CompatibleStringType,
    enable_if_t <
        is_compatible_string_type<BasicJsonType, CompatibleStringType>::value and
        not std::is_same<typename BasicJsonType::string_t,
                         CompatibleStringType>::value,
        int > = 0 >
void from_json(const BasicJsonType& j, CompatibleStringType& s)
{
    if (JSON_UNLIKELY(not j.is_string()))
    {
        JSON_THROW(type_error::create(302, "type must be string, but is " + std::string(j.type_name())));
    }

    s = *j.template get_ptr<const typename BasicJsonType::string_t*>();
}

template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::number_float_t& val)
{
    get_arithmetic_value(j, val);
}

template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::number_unsigned_t& val)
{
    get_arithmetic_value(j, val);
}

template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::number_integer_t& val)
{
    get_arithmetic_value(j, val);
}

template<typename BasicJsonType, typename EnumType,
         enable_if_t<std::is_enum<EnumType>::value, int> = 0>
void from_json(const BasicJsonType& j, EnumType& e)
{
    typename std::underlying_type<EnumType>::type val;
    get_arithmetic_value(j, val);
    e = static_cast<EnumType>(val);
}

template<typename BasicJsonType>
void from_json(const BasicJsonType& j, typename BasicJsonType::array_t& arr)
{
    if (JSON_UNLIKELY(not j.is_array()))
    {
        JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(j.type_name())));
    }
    arr = *j.template get_ptr<const typename BasicJsonType::array_t*>();
}

// forward_list doesn't have an insert method
template<typename BasicJsonType, typename T, typename Allocator,
         enable_if_t<std::is_convertible<BasicJsonType, T>::value, int> = 0>
void from_json(const BasicJsonType& j, std::forward_list<T, Allocator>& l)
{
    if (JSON_UNLIKELY(not j.is_array()))
    {
        JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(j.type_name())));
    }
    std::transform(j.rbegin(), j.rend(),
                   std::front_inserter(l), [](const BasicJsonType & i)
    {
        return i.template get<T>();
    });
}

// valarray doesn't have an insert method
template<typename BasicJsonType, typename T,
         enable_if_t<std::is_convertible<BasicJsonType, T>::value, int> = 0>
void from_json(const BasicJsonType& j, std::valarray<T>& l)
{
    if (JSON_UNLIKELY(not j.is_array()))
    {
        JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(j.type_name())));
    }
    l.resize(j.size());
    std::copy(j.m_value.array->begin(), j.m_value.array->end(), std::begin(l));
}

template<typename BasicJsonType, typename CompatibleArrayType>
void from_json_array_impl(const BasicJsonType& j, CompatibleArrayType& arr, priority_tag<0> /*unused*/)
{
    using std::end;

    std::transform(j.begin(), j.end(),
                   std::inserter(arr, end(arr)), [](const BasicJsonType & i)
    {
        // get<BasicJsonType>() returns *this, this won't call a from_json
        // method when value_type is BasicJsonType
        return i.template get<typename CompatibleArrayType::value_type>();
    });
}

template<typename BasicJsonType, typename CompatibleArrayType>
auto from_json_array_impl(const BasicJsonType& j, CompatibleArrayType& arr, priority_tag<1> /*unused*/)
-> decltype(
    arr.reserve(std::declval<typename CompatibleArrayType::size_type>()),
    void())
{
    using std::end;

    arr.reserve(j.size());
    std::transform(j.begin(), j.end(),
                   std::inserter(arr, end(arr)), [](const BasicJsonType & i)
    {
        // get<BasicJsonType>() returns *this, this won't call a from_json
        // method when value_type is BasicJsonType
        return i.template get<typename CompatibleArrayType::value_type>();
    });
}

template<typename BasicJsonType, typename T, std::size_t N>
void from_json_array_impl(const BasicJsonType& j, std::array<T, N>& arr, priority_tag<2> /*unused*/)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        arr[i] = j.at(i).template get<T>();
    }
}

template <
    typename BasicJsonType, typename CompatibleArrayType,
    enable_if_t <
        is_compatible_array_type<BasicJsonType, CompatibleArrayType>::value and
        not std::is_same<typename BasicJsonType::array_t,
                         CompatibleArrayType>::value and
        std::is_constructible <
            BasicJsonType, typename CompatibleArrayType::value_type >::value,
        int > = 0 >
void from_json(const BasicJsonType& j, CompatibleArrayType& arr)
{
    if (JSON_UNLIKELY(not j.is_array()))
    {
        JSON_THROW(type_error::create(302, "type must be array, but is " +
                                      std::string(j.type_name())));
    }

    from_json_array_impl(j, arr, priority_tag<2> {});
}

template<typename BasicJsonType, typename CompatibleObjectType,
         enable_if_t<is_compatible_object_type<BasicJsonType, CompatibleObjectType>::value, int> = 0>
void from_json(const BasicJsonType& j, CompatibleObjectType& obj)
{
    if (JSON_UNLIKELY(not j.is_object()))
    {
        JSON_THROW(type_error::create(302, "type must be object, but is " + std::string(j.type_name())));
    }

    auto inner_object = j.template get_ptr<const typename BasicJsonType::object_t*>();
    using value_type = typename CompatibleObjectType::value_type;
    std::transform(
        inner_object->begin(), inner_object->end(),
        std::inserter(obj, obj.begin()),
        [](typename BasicJsonType::object_t::value_type const & p)
    {
        return value_type(p.first, p.second.template get<typename CompatibleObjectType::mapped_type>());
    });
}

// overload for arithmetic types, not chosen for basic_json template arguments
// (BooleanType, etc..); note: Is it really necessary to provide explicit
// overloads for boolean_t etc. in case of a custom BooleanType which is not
// an arithmetic type?
template<typename BasicJsonType, typename ArithmeticType,
         enable_if_t <
             std::is_arithmetic<ArithmeticType>::value and
             not std::is_same<ArithmeticType, typename BasicJsonType::number_unsigned_t>::value and
             not std::is_same<ArithmeticType, typename BasicJsonType::number_integer_t>::value and
             not std::is_same<ArithmeticType, typename BasicJsonType::number_float_t>::value and
             not std::is_same<ArithmeticType, typename BasicJsonType::boolean_t>::value,
             int> = 0>
void from_json(const BasicJsonType& j, ArithmeticType& val)
{
    switch (static_cast<value_t>(j))
    {
        case value_t::number_unsigned:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_unsigned_t*>());
            break;
        }
        case value_t::number_integer:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_integer_t*>());
            break;
        }
        case value_t::number_float:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::number_float_t*>());
            break;
        }
        case value_t::boolean:
        {
            val = static_cast<ArithmeticType>(*j.template get_ptr<const typename BasicJsonType::boolean_t*>());
            break;
        }

        default:
            JSON_THROW(type_error::create(302, "type must be number, but is " + std::string(j.type_name())));
    }
}

template<typename BasicJsonType, typename A1, typename A2>
void from_json(const BasicJsonType& j, std::pair<A1, A2>& p)
{
    p = {j.at(0).template get<A1>(), j.at(1).template get<A2>()};
}

template<typename BasicJsonType, typename Tuple, std::size_t... Idx>
void from_json_tuple_impl(const BasicJsonType& j, Tuple& t, index_sequence<Idx...>)
{
    t = std::make_tuple(j.at(Idx).template get<typename std::tuple_element<Idx, Tuple>::type>()...);
}

template<typename BasicJsonType, typename... Args>
void from_json(const BasicJsonType& j, std::tuple<Args...>& t)
{
    from_json_tuple_impl(j, t, index_sequence_for<Args...> {});
}

template <typename BasicJsonType, typename Key, typename Value, typename Compare, typename Allocator,
          typename = enable_if_t<not std::is_constructible<
                                     typename BasicJsonType::string_t, Key>::value>>
void from_json(const BasicJsonType& j, std::map<Key, Value, Compare, Allocator>& m)
{
    if (JSON_UNLIKELY(not j.is_array()))
    {
        JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(j.type_name())));
    }
    for (const auto& p : j)
    {
        if (JSON_UNLIKELY(not p.is_array()))
        {
            JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(p.type_name())));
        }
        m.emplace(p.at(0).template get<Key>(), p.at(1).template get<Value>());
    }
}

template <typename BasicJsonType, typename Key, typename Value, typename Hash, typename KeyEqual, typename Allocator,
          typename = enable_if_t<not std::is_constructible<
                                     typename BasicJsonType::string_t, Key>::value>>
void from_json(const BasicJsonType& j, std::unordered_map<Key, Value, Hash, KeyEqual, Allocator>& m)
{
    if (JSON_UNLIKELY(not j.is_array()))
    {
        JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(j.type_name())));
    }
    for (const auto& p : j)
    {
        if (JSON_UNLIKELY(not p.is_array()))
        {
            JSON_THROW(type_error::create(302, "type must be array, but is " + std::string(p.type_name())));
        }
        m.emplace(p.at(0).template get<Key>(), p.at(1).template get<Value>());
    }
}

struct from_json_fn
{
  private:
    template<typename BasicJsonType, typename T>
    auto call(const BasicJsonType& j, T& val, priority_tag<1> /*unused*/) const
    noexcept(noexcept(from_json(j, val)))
    -> decltype(from_json(j, val), void())
    {
        return from_json(j, val);
    }

    template<typename BasicJsonType, typename T>
    void call(const BasicJsonType& /*unused*/, T& /*unused*/, priority_tag<0> /*unused*/) const noexcept
    {
        static_assert(sizeof(BasicJsonType) == 0,
                      "could not find from_json() method in T's namespace");
#ifdef _MSC_VER
        // MSVC does not show a stacktrace for the above assert
        using decayed = uncvref_t<T>;
        static_assert(sizeof(typename decayed::force_msvc_stacktrace) == 0,
                      "forcing MSVC stacktrace to show which T we're talking about.");
#endif
    }

  public:
    template<typename BasicJsonType, typename T>
    void operator()(const BasicJsonType& j, T& val) const
    noexcept(noexcept(std::declval<from_json_fn>().call(j, val, priority_tag<1> {})))
    {
        return call(j, val, priority_tag<1> {});
    }
};
}

/// namespace to hold default `from_json` function
/// to see why this is required:
/// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4381.html
namespace
{
constexpr const auto& from_json = detail::static_const<detail::from_json_fn>::value;
}
}

// #include <nlohmann/detail/conversions/to_json.hpp>


#include <ciso646> // or, and, not
#include <iterator> // begin, end
#include <tuple> // tuple, get
#include <type_traits> // is_same, is_constructible, is_floating_point, is_enum, underlying_type
#include <utility> // move, forward, declval, pair
#include <valarray> // valarray
#include <vector> // vector

// #include <nlohmann/detail/meta.hpp>

// #include <nlohmann/detail/value_t.hpp>

// #include <nlohmann/detail/iterators/iteration_proxy.hpp>


#include <cstddef> // size_t
#include <string> // string, to_string
#include <iterator> // input_iterator_tag

// #include <nlohmann/detail/value_t.hpp>


namespace nlohmann
{
namespace detail
{
/// proxy class for the items() function
template<typename IteratorType> class iteration_proxy
{
  private:
    /// helper class for iteration
    class iteration_proxy_internal
    {
      public:
        using difference_type = std::ptrdiff_t;
        using value_type = iteration_proxy_internal;
        using pointer = iteration_proxy_internal*;
        using reference = iteration_proxy_internal&;
        using iterator_category = std::input_iterator_tag;

      private:
        /// the iterator
        IteratorType anchor;
        /// an index for arrays (used to create key names)
        std::size_t array_index = 0;
        /// last stringified array index
        mutable std::size_t array_index_last = 0;
        /// a string representation of the array index
        mutable std::string array_index_str = "0";
        /// an empty string (to return a reference for primitive values)
        const std::string empty_str = "";

      public:
        explicit iteration_proxy_internal(IteratorType it) noexcept : anchor(it) {}

        iteration_proxy_internal(const iteration_proxy_internal&) = default;
        iteration_proxy_internal& operator=(const iteration_proxy_internal&) = default;

        /// dereference operator (needed for range-based for)
        iteration_proxy_internal& operator*()
        {
            return *this;
        }

        /// increment operator (needed for range-based for)
        iteration_proxy_internal& operator++()
        {
            ++anchor;
            ++array_index;

            return *this;
        }

        /// equality operator (needed for InputIterator)
        bool operator==(const iteration_proxy_internal& o) const noexcept
        {
            return anchor == o.anchor;
        }

        /// inequality operator (needed for range-based for)
        bool operator!=(const iteration_proxy_internal& o) const noexcept
        {
            return anchor != o.anchor;
        }

        /// return key of the iterator
        const std::string& key() const
        {
            assert(anchor.m_object != nullptr);

            switch (anchor.m_object->type())
            {
                // use integer array index as key
                case value_t::array:
                {
                    if (array_index != array_index_last)
                    {
                        array_index_str = std::to_string(array_index);
                        array_index_last = array_index;
                    }
                    return array_index_str;
                }

                // use key from the object
                case value_t::object:
                    return anchor.key();

                // use an empty key for all primitive types
                default:
                    return empty_str;
            }
        }

        /// return value of the iterator
        typename IteratorType::reference value() const
        {
            return anchor.value();
        }
    };

    /// the container to iterate
    typename IteratorType::reference container;

  public:
    /// construct iteration proxy from a container
    explicit iteration_proxy(typename IteratorType::reference cont) noexcept
        : container(cont) {}

    /// return iterator begin (needed for range-based for)
    iteration_proxy_internal begin() noexcept
    {
        return iteration_proxy_internal(container.begin());
    }

    /// return iterator end (needed for range-based for)
    iteration_proxy_internal end() noexcept
    {
        return iteration_proxy_internal(container.end());
    }
};
}
}


namespace nlohmann
{
namespace detail
{
//////////////////
// constructors //
//////////////////

template<value_t> struct external_constructor;

template<>
struct external_constructor<value_t::boolean>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::boolean_t b) noexcept
    {
        j.m_type = value_t::boolean;
        j.m_value = b;
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::string>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, const typename BasicJsonType::string_t& s)
    {
        j.m_type = value_t::string;
        j.m_value = s;
        j.assert_invariant();
    }

    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::string_t&& s)
    {
        j.m_type = value_t::string;
        j.m_value = std::move(s);
        j.assert_invariant();
    }

    template<typename BasicJsonType, typename CompatibleStringType,
             enable_if_t<not std::is_same<CompatibleStringType, typename BasicJsonType::string_t>::value,
                         int> = 0>
    static void construct(BasicJsonType& j, const CompatibleStringType& str)
    {
        j.m_type = value_t::string;
        j.m_value.string = j.template create<typename BasicJsonType::string_t>(str);
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::number_float>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::number_float_t val) noexcept
    {
        j.m_type = value_t::number_float;
        j.m_value = val;
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::number_unsigned>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::number_unsigned_t val) noexcept
    {
        j.m_type = value_t::number_unsigned;
        j.m_value = val;
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::number_integer>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::number_integer_t val) noexcept
    {
        j.m_type = value_t::number_integer;
        j.m_value = val;
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::array>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, const typename BasicJsonType::array_t& arr)
    {
        j.m_type = value_t::array;
        j.m_value = arr;
        j.assert_invariant();
    }

    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::array_t&& arr)
    {
        j.m_type = value_t::array;
        j.m_value = std::move(arr);
        j.assert_invariant();
    }

    template<typename BasicJsonType, typename CompatibleArrayType,
             enable_if_t<not std::is_same<CompatibleArrayType, typename BasicJsonType::array_t>::value,
                         int> = 0>
    static void construct(BasicJsonType& j, const CompatibleArrayType& arr)
    {
        using std::begin;
        using std::end;
        j.m_type = value_t::array;
        j.m_value.array = j.template create<typename BasicJsonType::array_t>(begin(arr), end(arr));
        j.assert_invariant();
    }

    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, const std::vector<bool>& arr)
    {
        j.m_type = value_t::array;
        j.m_value = value_t::array;
        j.m_value.array->reserve(arr.size());
        for (const bool x : arr)
        {
            j.m_value.array->push_back(x);
        }
        j.assert_invariant();
    }

    template<typename BasicJsonType, typename T,
             enable_if_t<std::is_convertible<T, BasicJsonType>::value, int> = 0>
    static void construct(BasicJsonType& j, const std::valarray<T>& arr)
    {
        j.m_type = value_t::array;
        j.m_value = value_t::array;
        j.m_value.array->resize(arr.size());
        std::copy(std::begin(arr), std::end(arr), j.m_value.array->begin());
        j.assert_invariant();
    }
};

template<>
struct external_constructor<value_t::object>
{
    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, const typename BasicJsonType::object_t& obj)
    {
        j.m_type = value_t::object;
        j.m_value = obj;
        j.assert_invariant();
    }

    template<typename BasicJsonType>
    static void construct(BasicJsonType& j, typename BasicJsonType::object_t&& obj)
    {
        j.m_type = value_t::object;
        j.m_value = std::move(obj);
        j.assert_invariant();
    }

    template<typename BasicJsonType, typename CompatibleObjectType,
             enable_if_t<not std::is_same<CompatibleObjectType, typename BasicJsonType::object_t>::value, int> = 0>
    static void construct(BasicJsonType& j, const CompatibleObjectType& obj)
    {
        using std::begin;
        using std::end;

        j.m_type = value_t::object;
        j.m_value.object = j.template create<typename BasicJsonType::object_t>(begin(obj), end(obj));
        j.assert_invariant();
    }
};

/////////////
// to_json //
/////////////

template<typename BasicJsonType, typename T,
         enable_if_t<std::is_same<T, typename BasicJsonType::boolean_t>::value, int> = 0>
void to_json(BasicJsonType& j, T b) noexcept
{
    external_constructor<value_t::boolean>::construct(j, b);
}

template<typename BasicJsonType, typename CompatibleString,
         enable_if_t<std::is_constructible<typename BasicJsonType::string_t, CompatibleString>::value, int> = 0>
void to_json(BasicJsonType& j, const CompatibleString& s)
{
    external_constructor<value_t::string>::construct(j, s);
}

template<typename BasicJsonType>
void to_json(BasicJsonType& j, typename BasicJsonType::string_t&& s)
{
    external_constructor<value_t::string>::construct(j, std::move(s));
}

template<typename BasicJsonType, typename FloatType,
         enable_if_t<std::is_floating_point<FloatType>::value, int> = 0>
void to_json(BasicJsonType& j, FloatType val) noexcept
{
    external_constructor<value_t::number_float>::construct(j, static_cast<typename BasicJsonType::number_float_t>(val));
}

template<typename BasicJsonType, typename CompatibleNumberUnsignedType,
         enable_if_t<is_compatible_integer_type<typename BasicJsonType::number_unsigned_t, CompatibleNumberUnsignedType>::value, int> = 0>
void to_json(BasicJsonType& j, CompatibleNumberUnsignedType val) noexcept
{
    external_constructor<value_t::number_unsigned>::construct(j, static_cast<typename BasicJsonType::number_unsigned_t>(val));
}

template<typename BasicJsonType, typename CompatibleNumberIntegerType,
         enable_if_t<is_compatible_integer_type<typename BasicJsonType::number_integer_t, CompatibleNumberIntegerType>::value, int> = 0>
void to_json(BasicJsonType& j, CompatibleNumberIntegerType val) noexcept
{
    external_constructor<value_t::number_integer>::construct(j, static_cast<typename BasicJsonType::number_integer_t>(val));
}

template<typename BasicJsonType, typename EnumType,
         enable_if_t<std::is_enum<EnumType>::value, int> = 0>
void to_json(BasicJsonType& j, EnumType e) noexcept
{
    using underlying_type = typename std::underlying_type<EnumType>::type;
    external_constructor<value_t::number_integer>::construct(j, static_cast<underlying_type>(e));
}

template<typename BasicJsonType>
void to_json(BasicJsonType& j, const std::vector<bool>& e)
{
    external_constructor<value_t::array>::construct(j, e);
}

template<typename BasicJsonType, typename CompatibleArrayType,
         enable_if_t<is_compatible_array_type<BasicJsonType, CompatibleArrayType>::value or
                     std::is_same<typename BasicJsonType::array_t, CompatibleArrayType>::value,
                     int> = 0>
void to_json(BasicJsonType& j, const CompatibleArrayType& arr)
{
    external_constructor<value_t::array>::construct(j, arr);
}

template<typename BasicJsonType, typename T,
         enable_if_t<std::is_convertible<T, BasicJsonType>::value, int> = 0>
void to_json(BasicJsonType& j, const std::valarray<T>& arr)
{
    external_constructor<value_t::array>::construct(j, std::move(arr));
}

template<typename BasicJsonType>
void to_json(BasicJsonType& j, typename BasicJsonType::array_t&& arr)
{
    external_constructor<value_t::array>::construct(j, std::move(arr));
}

template<typename BasicJsonType, typename CompatibleObjectType,
         enable_if_t<is_compatible_object_type<BasicJsonType, CompatibleObjectType>::value, int> = 0>
void to_json(BasicJsonType& j, const CompatibleObjectType& obj)
{
    external_constructor<value_t::object>::construct(j, obj);
}

template<typename BasicJsonType>
void to_json(BasicJsonType& j, typename BasicJsonType::object_t&& obj)
{
    external_constructor<value_t::object>::construct(j, std::move(obj));
}

template<typename BasicJsonType, typename T, std::size_t N,
         enable_if_t<not std::is_constructible<typename BasicJsonType::string_t, T (&)[N]>::value, int> = 0>
void to_json(BasicJsonType& j, T (&arr)[N])
{
    external_constructor<value_t::array>::construct(j, arr);
}

template<typename BasicJsonType, typename... Args>
void to_json(BasicJsonType& j, const std::pair<Args...>& p)
{
    j = {p.first, p.second};
}

// for https://github.com/nlohmann/json/pull/1134
template<typename BasicJsonType, typename T,
         enable_if_t<std::is_same<T, typename iteration_proxy<typename BasicJsonType::iterator>::iteration_proxy_internal>::value, int> = 0>
void to_json(BasicJsonType& j, T b) noexcept
{
    j = {{b.key(), b.value()}};
}

template<typename BasicJsonType, typename Tuple, std::size_t... Idx>
void to_json_tuple_impl(BasicJsonType& j, const Tuple& t, index_sequence<Idx...>)
{
    j = {std::get<Idx>(t)...};
}

template<typename BasicJsonType, typename... Args>
void to_json(BasicJsonType& j, const std::tuple<Args...>& t)
{
    to_json_tuple_impl(j, t, index_sequence_for<Args...> {});
}

struct to_json_fn
{
  private:
    template<typename BasicJsonType, typename T>
    auto call(BasicJsonType& j, T&& val, priority_tag<1> /*unused*/) const noexcept(noexcept(to_json(j, std::forward<T>(val))))
    -> decltype(to_json(j, std::forward<T>(val)), void())
    {
        return to_json(j, std::forward<T>(val));
    }

    template<typename BasicJsonType, typename T>
    void call(BasicJsonType& /*unused*/, T&& /*unused*/, priority_tag<0> /*unused*/) const noexcept
    {
        static_assert(sizeof(BasicJsonType) == 0,
                      "could not find to_json() method in T's namespace");

#ifdef _MSC_VER
        // MSVC does not show a stacktrace for the above assert
        using decayed = uncvref_t<T>;
        static_assert(sizeof(typename decayed::force_msvc_stacktrace) == 0,
                      "forcing MSVC stacktrace to show which T we're talking about.");
#endif
    }

  public:
    template<typename BasicJsonType, typename T>
    void operator()(BasicJsonType& j, T&& val) const
    noexcept(noexcept(std::declval<to_json_fn>().call(j, std::forward<T>(val), priority_tag<1> {})))
    {
        return call(j, std::forward<T>(val), priority_tag<1> {});
    }
};
}

/// namespace to hold default `to_json` function
namespace
{
constexpr const auto& to_json = detail::static_const<detail::to_json_fn>::value;
}
}

// #include <nlohmann/detail/input/input_adapters.hpp>


#include <cassert> // assert
#include <cstddef> // size_t
#include <cstring> // strlen
#include <istream> // istream
#include <iterator> // begin, end, iterator_traits, random_access_iterator_tag, distance, next
#include <memory> // shared_ptr, make_shared, addressof
#include <numeric> // accumulate
#include <string> // string, char_traits
#include <type_traits> // enable_if, is_base_of, is_pointer, is_integral, remove_pointer
#include <utility> // pair, declval

// #include <nlohmann/detail/macro_scope.hpp>


namespace nlohmann
{
namespace detail
{
/// the supported input formats
enum class input_format_t { json, cbor, msgpack, ubjson };

////////////////////
// input adapters //
////////////////////

/*!
@brief abstract input adapter interface

Produces a stream of std::char_traits<char>::int_type characters from a
std::istream, a buffer, or some other input type. Accepts the return of
exactly one non-EOF character for future input. The int_type characters
returned consist of all valid char values as positive values (typically
unsigned char), plus an EOF value outside that range, specified by the value
of the function std::char_traits<char>::eof(). This value is typically -1, but
could be any arbitrary value which is not a valid char value.
*/
struct input_adapter_protocol
{
    /// get a character [0,255] or std::char_traits<char>::eof().
    virtual std::char_traits<char>::int_type get_character() = 0;
    virtual ~input_adapter_protocol() = default;
};

/// a type to simplify interfaces
using input_adapter_t = std::shared_ptr<input_adapter_protocol>;

/*!
Input adapter for a (caching) istream. Ignores a UFT Byte Order Mark at
beginning of input. Does not support changing the underlying std::streambuf
in mid-input. Maintains underlying std::istream and std::streambuf to support
subsequent use of standard std::istream operations to process any input
characters following those used in parsing the JSON input.  Clears the
std::istream flags; any input errors (e.g., EOF) will be detected by the first
subsequent call for input from the std::istream.
*/
class input_stream_adapter : public input_adapter_protocol
{
  public:
    ~input_stream_adapter() override
    {
        // clear stream flags; we use underlying streambuf I/O, do not
        // maintain ifstream flags
        is.clear();
    }

    explicit input_stream_adapter(std::istream& i)
        : is(i), sb(*i.rdbuf())
    {}

    // delete because of pointer members
    input_stream_adapter(const input_stream_adapter&) = delete;
    input_stream_adapter& operator=(input_stream_adapter&) = delete;

    // std::istream/std::streambuf use std::char_traits<char>::to_int_type, to
    // ensure that std::char_traits<char>::eof() and the character 0xFF do not
    // end up as the same value, eg. 0xFFFFFFFF.
    std::char_traits<char>::int_type get_character() override
    {
        return sb.sbumpc();
    }

  private:
    /// the associated input stream
    std::istream& is;
    std::streambuf& sb;
};

/// input adapter for buffer input
class input_buffer_adapter : public input_adapter_protocol
{
  public:
    input_buffer_adapter(const char* b, const std::size_t l)
        : cursor(b), limit(b + l)
    {}

    // delete because of pointer members
    input_buffer_adapter(const input_buffer_adapter&) = delete;
    input_buffer_adapter& operator=(input_buffer_adapter&) = delete;

    std::char_traits<char>::int_type get_character() noexcept override
    {
        if (JSON_LIKELY(cursor < limit))
        {
            return std::char_traits<char>::to_int_type(*(cursor++));
        }

        return std::char_traits<char>::eof();
    }

  private:
    /// pointer to the current character
    const char* cursor;
    /// pointer past the last character
    const char* const limit;
};

template<typename WideStringType>
class wide_string_input_adapter : public input_adapter_protocol
{
  public:
    explicit wide_string_input_adapter(const WideStringType& w) : str(w) {}

    std::char_traits<char>::int_type get_character() noexcept override
    {
        // check if buffer needs to be filled
        if (utf8_bytes_index == utf8_bytes_filled)
        {
            if (sizeof(typename WideStringType::value_type) == 2)
            {
                fill_buffer_utf16();
            }
            else
            {
                fill_buffer_utf32();
            }

            assert(utf8_bytes_filled > 0);
            assert(utf8_bytes_index == 0);
        }

        // use buffer
        assert(utf8_bytes_filled > 0);
        assert(utf8_bytes_index < utf8_bytes_filled);
        return utf8_bytes[utf8_bytes_index++];
    }

  private:
    void fill_buffer_utf16()
    {
        utf8_bytes_index = 0;

        if (current_wchar == str.size())
        {
            utf8_bytes[0] = std::char_traits<char>::eof();
            utf8_bytes_filled = 1;
        }
        else
        {
            // get the current character
            const int wc = static_cast<int>(str[current_wchar++]);

            // UTF-16 to UTF-8 encoding
            if (wc < 0x80)
            {
                utf8_bytes[0] = wc;
                utf8_bytes_filled = 1;
            }
            else if (wc <= 0x7FF)
            {
                utf8_bytes[0] = 0xC0 | ((wc >> 6));
                utf8_bytes[1] = 0x80 | (wc & 0x3F);
                utf8_bytes_filled = 2;
            }
            else if (0xD800 > wc or wc >= 0xE000)
            {
                utf8_bytes[0] = 0xE0 | ((wc >> 12));
                utf8_bytes[1] = 0x80 | ((wc >> 6) & 0x3F);
                utf8_bytes[2] = 0x80 | (wc & 0x3F);
                utf8_bytes_filled = 3;
            }
            else
            {
                if (current_wchar < str.size())
                {
                    const int wc2 = static_cast<int>(str[current_wchar++]);
                    const int charcode = 0x10000 + (((wc & 0x3FF) << 10) | (wc2 & 0x3FF));
                    utf8_bytes[0] = 0xf0 | (charcode >> 18);
                    utf8_bytes[1] = 0x80 | ((charcode >> 12) & 0x3F);
                    utf8_bytes[2] = 0x80 | ((charcode >> 6) & 0x3F);
                    utf8_bytes[3] = 0x80 | (charcode & 0x3F);
                    utf8_bytes_filled = 4;
                }
                else
                {
                    // unknown character
                    ++current_wchar;
                    utf8_bytes[0] = wc;
                    utf8_bytes_filled = 1;
                }
            }
        }
    }

    void fill_buffer_utf32()
    {
        utf8_bytes_index = 0;

        if (current_wchar == str.size())
        {
            utf8_bytes[0] = std::char_traits<char>::eof();
            utf8_bytes_filled = 1;
        }
        else
        {
            // get the current character
            const int wc = static_cast<int>(str[current_wchar++]);

            // UTF-32 to UTF-8 encoding
            if (wc < 0x80)
            {
                utf8_bytes[0] = wc;
                utf8_bytes_filled = 1;
            }
            else if (wc <= 0x7FF)
            {
                utf8_bytes[0] = 0xC0 | ((wc >> 6) & 0x1F);
                utf8_bytes[1] = 0x80 | (wc & 0x3F);
                utf8_bytes_filled = 2;
            }
            else if (wc <= 0xFFFF)
            {
                utf8_bytes[0] = 0xE0 | ((wc >> 12) & 0x0F);
                utf8_bytes[1] = 0x80 | ((wc >> 6) & 0x3F);
                utf8_bytes[2] = 0x80 | (wc & 0x3F);
                utf8_bytes_filled = 3;
            }
            else if (wc <= 0x10FFFF)
            {
                utf8_bytes[0] = 0xF0 | ((wc >> 18 ) & 0x07);
                utf8_bytes[1] = 0x80 | ((wc >> 12) & 0x3F);
                utf8_bytes[2] = 0x80 | ((wc >> 6) & 0x3F);
                utf8_bytes[3] = 0x80 | (wc & 0x3F);
                utf8_bytes_filled = 4;
            }
            else
            {
                // unknown character
                utf8_bytes[0] = wc;
                utf8_bytes_filled = 1;
            }
        }
    }

  private:
    /// the wstring to process
    const WideStringType& str;

    /// index of the current wchar in str
    std::size_t current_wchar = 0;

    /// a buffer for UTF-8 bytes
    std::array<std::char_traits<char>::int_type, 4> utf8_bytes = {{0, 0, 0, 0}};

    /// index to the utf8_codes array for the next valid byte
    std::size_t utf8_bytes_index = 0;
    /// number of valid bytes in the utf8_codes array
    std::size_t utf8_bytes_filled = 0;
};

class input_adapter
{
  public:
    // native support

    /// input adapter for input stream
    input_adapter(std::istream& i)
        : ia(std::make_shared<input_stream_adapter>(i)) {}

    /// input adapter for input stream
    input_adapter(std::istream&& i)
        : ia(std::make_shared<input_stream_adapter>(i)) {}

    input_adapter(const std::wstring& ws)
        : ia(std::make_shared<wide_string_input_adapter<std::wstring>>(ws)) {}

    input_adapter(const std::u16string& ws)
        : ia(std::make_shared<wide_string_input_adapter<std::u16string>>(ws)) {}

    input_adapter(const std::u32string& ws)
        : ia(std::make_shared<wide_string_input_adapter<std::u32string>>(ws)) {}

    /// input adapter for buffer
    template<typename CharT,
             typename std::enable_if<
                 std::is_pointer<CharT>::value and
                 std::is_integral<typename std::remove_pointer<CharT>::type>::value and
                 sizeof(typename std::remove_pointer<CharT>::type) == 1,
                 int>::type = 0>
    input_adapter(CharT b, std::size_t l)
        : ia(std::make_shared<input_buffer_adapter>(reinterpret_cast<const char*>(b), l)) {}

    // derived support

    /// input adapter for string literal
    template<typename CharT,
             typename std::enable_if<
                 std::is_pointer<CharT>::value and
                 std::is_integral<typename std::remove_pointer<CharT>::type>::value and
                 sizeof(typename std::remove_pointer<CharT>::type) == 1,
                 int>::type = 0>
    input_adapter(CharT b)
        : input_adapter(reinterpret_cast<const char*>(b),
                        std::strlen(reinterpret_cast<const char*>(b))) {}

    /// input adapter for iterator range with contiguous storage
    template<class IteratorType,
             typename std::enable_if<
                 std::is_same<typename std::iterator_traits<IteratorType>::iterator_category, std::random_access_iterator_tag>::value,
                 int>::type = 0>
    input_adapter(IteratorType first, IteratorType last)
    {
        // assertion to check that the iterator range is indeed contiguous,
        // see http://stackoverflow.com/a/35008842/266378 for more discussion
        assert(std::accumulate(
                   first, last, std::pair<bool, int>(true, 0),
                   [&first](std::pair<bool, int> res, decltype(*first) val)
        {
            res.first &= (val == *(std::next(std::addressof(*first), res.second++)));
            return res;
        }).first);

        // assertion to check that each element is 1 byte long
        static_assert(
            sizeof(typename std::iterator_traits<IteratorType>::value_type) == 1,
            "each element in the iterator range must have the size of 1 byte");

        const auto len = static_cast<size_t>(std::distance(first, last));
        if (JSON_LIKELY(len > 0))
        {
            // there is at least one element: use the address of first
            ia = std::make_shared<input_buffer_adapter>(reinterpret_cast<const char*>(&(*first)), len);
        }
        else
        {
            // the address of first cannot be used: use nullptr
            ia = std::make_shared<input_buffer_adapter>(nullptr, len);
        }
    }

    /// input adapter for array
    template<class T, std::size_t N>
    input_adapter(T (&array)[N])
        : input_adapter(std::begin(array), std::end(array)) {}

    /// input adapter for contiguous container
    template<class ContiguousContainer, typename
             std::enable_if<not std::is_pointer<ContiguousContainer>::value and
                            std::is_base_of<std::random_access_iterator_tag, typename std::iterator_traits<decltype(std::begin(std::declval<ContiguousContainer const>()))>::iterator_category>::value,
                            int>::type = 0>
    input_adapter(const ContiguousContainer& c)
        : input_adapter(std::begin(c), std::end(c)) {}

    operator input_adapter_t()
    {
        return ia;
    }

  private:
    /// the actual adapter
    input_adapter_t ia = nullptr;
};
}
}

// #include <nlohmann/detail/input/lexer.hpp>


#include <clocale> // localeconv
#include <cstddef> // size_t
#include <cstdlib> // strtof, strtod, strtold, strtoll, strtoull
#include <cstdio> // snprintf
#include <initializer_list> // initializer_list
#include <string> // char_traits, string
#include <vector> // vector

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/input/input_adapters.hpp>


namespace nlohmann
{
namespace detail
{
///////////
// lexer //
///////////

/*!
@brief lexical analysis

This class organizes the lexical analysis during JSON deserialization.
*/
template<typename BasicJsonType>
class lexer
{
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;

  public:
    /// token types for the parser
    enum class token_type
    {
        uninitialized,    ///< indicating the scanner is uninitialized
        literal_true,     ///< the `true` literal
        literal_false,    ///< the `false` literal
        literal_null,     ///< the `null` literal
        value_string,     ///< a string -- use get_string() for actual value
        value_unsigned,   ///< an unsigned integer -- use get_number_unsigned() for actual value
        value_integer,    ///< a signed integer -- use get_number_integer() for actual value
        value_float,      ///< an floating point number -- use get_number_float() for actual value
        begin_array,      ///< the character for array begin `[`
        begin_object,     ///< the character for object begin `{`
        end_array,        ///< the character for array end `]`
        end_object,       ///< the character for object end `}`
        name_separator,   ///< the name separator `:`
        value_separator,  ///< the value separator `,`
        parse_error,      ///< indicating a parse error
        end_of_input,     ///< indicating the end of the input buffer
        literal_or_value  ///< a literal or the begin of a value (only for diagnostics)
    };

    /// return name of values of type token_type (only used for errors)
    static const char* token_type_name(const token_type t) noexcept
    {
        switch (t)
        {
            case token_type::uninitialized:
                return "<uninitialized>";
            case token_type::literal_true:
                return "true literal";
            case token_type::literal_false:
                return "false literal";
            case token_type::literal_null:
                return "null literal";
            case token_type::value_string:
                return "string literal";
            case lexer::token_type::value_unsigned:
            case lexer::token_type::value_integer:
            case lexer::token_type::value_float:
                return "number literal";
            case token_type::begin_array:
                return "'['";
            case token_type::begin_object:
                return "'{'";
            case token_type::end_array:
                return "']'";
            case token_type::end_object:
                return "'}'";
            case token_type::name_separator:
                return "':'";
            case token_type::value_separator:
                return "','";
            case token_type::parse_error:
                return "<parse error>";
            case token_type::end_of_input:
                return "end of input";
            case token_type::literal_or_value:
                return "'[', '{', or a literal";
            default: // catch non-enum values
                return "unknown token"; // LCOV_EXCL_LINE
        }
    }

    explicit lexer(detail::input_adapter_t&& adapter)
        : ia(std::move(adapter)), decimal_point_char(get_decimal_point()) {}

    // delete because of pointer members
    lexer(const lexer&) = delete;
    lexer& operator=(lexer&) = delete;

  private:
    /////////////////////
    // locales
    /////////////////////

    /// return the locale-dependent decimal point
    static char get_decimal_point() noexcept
    {
        const auto loc = localeconv();
        assert(loc != nullptr);
        return (loc->decimal_point == nullptr) ? '.' : *(loc->decimal_point);
    }

    /////////////////////
    // scan functions
    /////////////////////

    /*!
    @brief get codepoint from 4 hex characters following `\u`

    For input "\u c1 c2 c3 c4" the codepoint is:
      (c1 * 0x1000) + (c2 * 0x0100) + (c3 * 0x0010) + c4
    = (c1 << 12) + (c2 << 8) + (c3 << 4) + (c4 << 0)

    Furthermore, the possible characters '0'..'9', 'A'..'F', and 'a'..'f'
    must be converted to the integers 0x0..0x9, 0xA..0xF, 0xA..0xF, resp. The
    conversion is done by subtracting the offset (0x30, 0x37, and 0x57)
    between the ASCII value of the character and the desired integer value.

    @return codepoint (0x0000..0xFFFF) or -1 in case of an error (e.g. EOF or
            non-hex character)
    */
    int get_codepoint()
    {
        // this function only makes sense after reading `\u`
        assert(current == 'u');
        int codepoint = 0;

        const auto factors = { 12, 8, 4, 0 };
        for (const auto factor : factors)
        {
            get();

            if (current >= '0' and current <= '9')
            {
                codepoint += ((current - 0x30) << factor);
            }
            else if (current >= 'A' and current <= 'F')
            {
                codepoint += ((current - 0x37) << factor);
            }
            else if (current >= 'a' and current <= 'f')
            {
                codepoint += ((current - 0x57) << factor);
            }
            else
            {
                return -1;
            }
        }

        assert(0x0000 <= codepoint and codepoint <= 0xFFFF);
        return codepoint;
    }

    /*!
    @brief check if the next byte(s) are inside a given range

    Adds the current byte and, for each passed range, reads a new byte and
    checks if it is inside the range. If a violation was detected, set up an
    error message and return false. Otherwise, return true.

    @param[in] ranges  list of integers; interpreted as list of pairs of
                       inclusive lower and upper bound, respectively

    @pre The passed list @a ranges must have 2, 4, or 6 elements; that is,
         1, 2, or 3 pairs. This precondition is enforced by an assertion.

    @return true if and only if no range violation was detected
    */
    bool next_byte_in_range(std::initializer_list<int> ranges)
    {
        assert(ranges.size() == 2 or ranges.size() == 4 or ranges.size() == 6);
        add(current);

        for (auto range = ranges.begin(); range != ranges.end(); ++range)
        {
            get();
            if (JSON_LIKELY(*range <= current and current <= *(++range)))
            {
                add(current);
            }
            else
            {
                error_message = "invalid string: ill-formed UTF-8 byte";
                return false;
            }
        }

        return true;
    }

    /*!
    @brief scan a string literal

    This function scans a string according to Sect. 7 of RFC 7159. While
    scanning, bytes are escaped and copied into buffer token_buffer. Then the
    function returns successfully, token_buffer is *not* null-terminated (as it
    may contain \0 bytes), and token_buffer.size() is the number of bytes in the
    string.

    @return token_type::value_string if string could be successfully scanned,
            token_type::parse_error otherwise

    @note In case of errors, variable error_message contains a textual
          description.
    */
    token_type scan_string()
    {
        // reset token_buffer (ignore opening quote)
        reset();

        // we entered the function by reading an open quote
        assert(current == '\"');

        while (true)
        {
            // get next character
            switch (get())
            {
                // end of file while parsing string
                case std::char_traits<char>::eof():
                {
                    error_message = "invalid string: missing closing quote";
                    return token_type::parse_error;
                }

                // closing quote
                case '\"':
                {
                    return token_type::value_string;
                }

                // escapes
                case '\\':
                {
                    switch (get())
                    {
                        // quotation mark
                        case '\"':
                            add('\"');
                            break;
                        // reverse solidus
                        case '\\':
                            add('\\');
                            break;
                        // solidus
                        case '/':
                            add('/');
                            break;
                        // backspace
                        case 'b':
                            add('\b');
                            break;
                        // form feed
                        case 'f':
                            add('\f');
                            break;
                        // line feed
                        case 'n':
                            add('\n');
                            break;
                        // carriage return
                        case 'r':
                            add('\r');
                            break;
                        // tab
                        case 't':
                            add('\t');
                            break;

                        // unicode escapes
                        case 'u':
                        {
                            const int codepoint1 = get_codepoint();
                            int codepoint = codepoint1; // start with codepoint1

                            if (JSON_UNLIKELY(codepoint1 == -1))
                            {
                                error_message = "invalid string: '\\u' must be followed by 4 hex digits";
                                return token_type::parse_error;
                            }

                            // check if code point is a high surrogate
                            if (0xD800 <= codepoint1 and codepoint1 <= 0xDBFF)
                            {
                                // expect next \uxxxx entry
                                if (JSON_LIKELY(get() == '\\' and get() == 'u'))
                                {
                                    const int codepoint2 = get_codepoint();

                                    if (JSON_UNLIKELY(codepoint2 == -1))
                                    {
                                        error_message = "invalid string: '\\u' must be followed by 4 hex digits";
                                        return token_type::parse_error;
                                    }

                                    // check if codepoint2 is a low surrogate
                                    if (JSON_LIKELY(0xDC00 <= codepoint2 and codepoint2 <= 0xDFFF))
                                    {
                                        // overwrite codepoint
                                        codepoint =
                                            // high surrogate occupies the most significant 22 bits
                                            (codepoint1 << 10)
                                            // low surrogate occupies the least significant 15 bits
                                            + codepoint2
                                            // there is still the 0xD800, 0xDC00 and 0x10000 noise
                                            // in the result so we have to subtract with:
                                            // (0xD800 << 10) + DC00 - 0x10000 = 0x35FDC00
                                            - 0x35FDC00;
                                    }
                                    else
                                    {
                                        error_message = "invalid string: surrogate U+DC00..U+DFFF must be followed by U+DC00..U+DFFF";
                                        return token_type::parse_error;
                                    }
                                }
                                else
                                {
                                    error_message = "invalid string: surrogate U+DC00..U+DFFF must be followed by U+DC00..U+DFFF";
                                    return token_type::parse_error;
                                }
                            }
                            else
                            {
                                if (JSON_UNLIKELY(0xDC00 <= codepoint1 and codepoint1 <= 0xDFFF))
                                {
                                    error_message = "invalid string: surrogate U+DC00..U+DFFF must follow U+D800..U+DBFF";
                                    return token_type::parse_error;
                                }
                            }

                            // result of the above calculation yields a proper codepoint
                            assert(0x00 <= codepoint and codepoint <= 0x10FFFF);

                            // translate codepoint into bytes
                            if (codepoint < 0x80)
                            {
                                // 1-byte characters: 0xxxxxxx (ASCII)
                                add(codepoint);
                            }
                            else if (codepoint <= 0x7FF)
                            {
                                // 2-byte characters: 110xxxxx 10xxxxxx
                                add(0xC0 | (codepoint >> 6));
                                add(0x80 | (codepoint & 0x3F));
                            }
                            else if (codepoint <= 0xFFFF)
                            {
                                // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
                                add(0xE0 | (codepoint >> 12));
                                add(0x80 | ((codepoint >> 6) & 0x3F));
                                add(0x80 | (codepoint & 0x3F));
                            }
                            else
                            {
                                // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                                add(0xF0 | (codepoint >> 18));
                                add(0x80 | ((codepoint >> 12) & 0x3F));
                                add(0x80 | ((codepoint >> 6) & 0x3F));
                                add(0x80 | (codepoint & 0x3F));
                            }

                            break;
                        }

                        // other characters after escape
                        default:
                            error_message = "invalid string: forbidden character after backslash";
                            return token_type::parse_error;
                    }

                    break;
                }

                // invalid control characters
                case 0x00:
                case 0x01:
                case 0x02:
                case 0x03:
                case 0x04:
                case 0x05:
                case 0x06:
                case 0x07:
                case 0x08:
                case 0x09:
                case 0x0A:
                case 0x0B:
                case 0x0C:
                case 0x0D:
                case 0x0E:
                case 0x0F:
                case 0x10:
                case 0x11:
                case 0x12:
                case 0x13:
                case 0x14:
                case 0x15:
                case 0x16:
                case 0x17:
                case 0x18:
                case 0x19:
                case 0x1A:
                case 0x1B:
                case 0x1C:
                case 0x1D:
                case 0x1E:
                case 0x1F:
                {
                    error_message = "invalid string: control character must be escaped";
                    return token_type::parse_error;
                }

                // U+0020..U+007F (except U+0022 (quote) and U+005C (backspace))
                case 0x20:
                case 0x21:
                case 0x23:
                case 0x24:
                case 0x25:
                case 0x26:
                case 0x27:
                case 0x28:
                case 0x29:
                case 0x2A:
                case 0x2B:
                case 0x2C:
                case 0x2D:
                case 0x2E:
                case 0x2F:
                case 0x30:
                case 0x31:
                case 0x32:
                case 0x33:
                case 0x34:
                case 0x35:
                case 0x36:
                case 0x37:
                case 0x38:
                case 0x39:
                case 0x3A:
                case 0x3B:
                case 0x3C:
                case 0x3D:
                case 0x3E:
                case 0x3F:
                case 0x40:
                case 0x41:
                case 0x42:
                case 0x43:
                case 0x44:
                case 0x45:
                case 0x46:
                case 0x47:
                case 0x48:
                case 0x49:
                case 0x4A:
                case 0x4B:
                case 0x4C:
                case 0x4D:
                case 0x4E:
                case 0x4F:
                case 0x50:
                case 0x51:
                case 0x52:
                case 0x53:
                case 0x54:
                case 0x55:
                case 0x56:
                case 0x57:
                case 0x58:
                case 0x59:
                case 0x5A:
                case 0x5B:
                case 0x5D:
                case 0x5E:
                case 0x5F:
                case 0x60:
                case 0x61:
                case 0x62:
                case 0x63:
                case 0x64:
                case 0x65:
                case 0x66:
                case 0x67:
                case 0x68:
                case 0x69:
                case 0x6A:
                case 0x6B:
                case 0x6C:
                case 0x6D:
                case 0x6E:
                case 0x6F:
                case 0x70:
                case 0x71:
                case 0x72:
                case 0x73:
                case 0x74:
                case 0x75:
                case 0x76:
                case 0x77:
                case 0x78:
                case 0x79:
                case 0x7A:
                case 0x7B:
                case 0x7C:
                case 0x7D:
                case 0x7E:
                case 0x7F:
                {
                    add(current);
                    break;
                }

                // U+0080..U+07FF: bytes C2..DF 80..BF
                case 0xC2:
                case 0xC3:
                case 0xC4:
                case 0xC5:
                case 0xC6:
                case 0xC7:
                case 0xC8:
                case 0xC9:
                case 0xCA:
                case 0xCB:
                case 0xCC:
                case 0xCD:
                case 0xCE:
                case 0xCF:
                case 0xD0:
                case 0xD1:
                case 0xD2:
                case 0xD3:
                case 0xD4:
                case 0xD5:
                case 0xD6:
                case 0xD7:
                case 0xD8:
                case 0xD9:
                case 0xDA:
                case 0xDB:
                case 0xDC:
                case 0xDD:
                case 0xDE:
                case 0xDF:
                {
                    if (JSON_UNLIKELY(not next_byte_in_range({0x80, 0xBF})))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+0800..U+0FFF: bytes E0 A0..BF 80..BF
                case 0xE0:
                {
                    if (JSON_UNLIKELY(not (next_byte_in_range({0xA0, 0xBF, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+1000..U+CFFF: bytes E1..EC 80..BF 80..BF
                // U+E000..U+FFFF: bytes EE..EF 80..BF 80..BF
                case 0xE1:
                case 0xE2:
                case 0xE3:
                case 0xE4:
                case 0xE5:
                case 0xE6:
                case 0xE7:
                case 0xE8:
                case 0xE9:
                case 0xEA:
                case 0xEB:
                case 0xEC:
                case 0xEE:
                case 0xEF:
                {
                    if (JSON_UNLIKELY(not (next_byte_in_range({0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+D000..U+D7FF: bytes ED 80..9F 80..BF
                case 0xED:
                {
                    if (JSON_UNLIKELY(not (next_byte_in_range({0x80, 0x9F, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+10000..U+3FFFF F0 90..BF 80..BF 80..BF
                case 0xF0:
                {
                    if (JSON_UNLIKELY(not (next_byte_in_range({0x90, 0xBF, 0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+40000..U+FFFFF F1..F3 80..BF 80..BF 80..BF
                case 0xF1:
                case 0xF2:
                case 0xF3:
                {
                    if (JSON_UNLIKELY(not (next_byte_in_range({0x80, 0xBF, 0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // U+100000..U+10FFFF F4 80..8F 80..BF 80..BF
                case 0xF4:
                {
                    if (JSON_UNLIKELY(not (next_byte_in_range({0x80, 0x8F, 0x80, 0xBF, 0x80, 0xBF}))))
                    {
                        return token_type::parse_error;
                    }
                    break;
                }

                // remaining bytes (80..C1 and F5..FF) are ill-formed
                default:
                {
                    error_message = "invalid string: ill-formed UTF-8 byte";
                    return token_type::parse_error;
                }
            }
        }
    }

    static void strtof(float& f, const char* str, char** endptr) noexcept
    {
        f = std::strtof(str, endptr);
    }

    static void strtof(double& f, const char* str, char** endptr) noexcept
    {
        f = std::strtod(str, endptr);
    }

    static void strtof(long double& f, const char* str, char** endptr) noexcept
    {
        f = std::strtold(str, endptr);
    }

    /*!
    @brief scan a number literal

    This function scans a string according to Sect. 6 of RFC 7159.

    The function is realized with a deterministic finite state machine derived
    from the grammar described in RFC 7159. Starting in state "init", the
    input is read and used to determined the next state. Only state "done"
    accepts the number. State "error" is a trap state to model errors. In the
    table below, "anything" means any character but the ones listed before.

    state    | 0        | 1-9      | e E      | +       | -       | .        | anything
    ---------|----------|----------|----------|---------|---------|----------|-----------
    init     | zero     | any1     | [error]  | [error] | minus   | [error]  | [error]
    minus    | zero     | any1     | [error]  | [error] | [error] | [error]  | [error]
    zero     | done     | done     | exponent | done    | done    | decimal1 | done
    any1     | any1     | any1     | exponent | done    | done    | decimal1 | done
    decimal1 | decimal2 | [error]  | [error]  | [error] | [error] | [error]  | [error]
    decimal2 | decimal2 | decimal2 | exponent | done    | done    | done     | done
    exponent | any2     | any2     | [error]  | sign    | sign    | [error]  | [error]
    sign     | any2     | any2     | [error]  | [error] | [error] | [error]  | [error]
    any2     | any2     | any2     | done     | done    | done    | done     | done

    The state machine is realized with one label per state (prefixed with
    "scan_number_") and `goto` statements between them. The state machine
    contains cycles, but any cycle can be left when EOF is read. Therefore,
    the function is guaranteed to terminate.

    During scanning, the read bytes are stored in token_buffer. This string is
    then converted to a signed integer, an unsigned integer, or a
    floating-point number.

    @return token_type::value_unsigned, token_type::value_integer, or
            token_type::value_float if number could be successfully scanned,
            token_type::parse_error otherwise

    @note The scanner is independent of the current locale. Internally, the
          locale's decimal point is used instead of `.` to work with the
          locale-dependent converters.
    */
    token_type scan_number()
    {
        // reset token_buffer to store the number's bytes
        reset();

        // the type of the parsed number; initially set to unsigned; will be
        // changed if minus sign, decimal point or exponent is read
        token_type number_type = token_type::value_unsigned;

        // state (init): we just found out we need to scan a number
        switch (current)
        {
            case '-':
            {
                add(current);
                goto scan_number_minus;
            }

            case '0':
            {
                add(current);
                goto scan_number_zero;
            }

            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any1;
            }

            default:
            {
                // all other characters are rejected outside scan_number()
                assert(false); // LCOV_EXCL_LINE
            }
        }

scan_number_minus:
        // state: we just parsed a leading minus sign
        number_type = token_type::value_integer;
        switch (get())
        {
            case '0':
            {
                add(current);
                goto scan_number_zero;
            }

            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any1;
            }

            default:
            {
                error_message = "invalid number; expected digit after '-'";
                return token_type::parse_error;
            }
        }

scan_number_zero:
        // state: we just parse a zero (maybe with a leading minus sign)
        switch (get())
        {
            case '.':
            {
                add(decimal_point_char);
                goto scan_number_decimal1;
            }

            case 'e':
            case 'E':
            {
                add(current);
                goto scan_number_exponent;
            }

            default:
                goto scan_number_done;
        }

scan_number_any1:
        // state: we just parsed a number 0-9 (maybe with a leading minus sign)
        switch (get())
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any1;
            }

            case '.':
            {
                add(decimal_point_char);
                goto scan_number_decimal1;
            }

            case 'e':
            case 'E':
            {
                add(current);
                goto scan_number_exponent;
            }

            default:
                goto scan_number_done;
        }

scan_number_decimal1:
        // state: we just parsed a decimal point
        number_type = token_type::value_float;
        switch (get())
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_decimal2;
            }

            default:
            {
                error_message = "invalid number; expected digit after '.'";
                return token_type::parse_error;
            }
        }

scan_number_decimal2:
        // we just parsed at least one number after a decimal point
        switch (get())
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_decimal2;
            }

            case 'e':
            case 'E':
            {
                add(current);
                goto scan_number_exponent;
            }

            default:
                goto scan_number_done;
        }

scan_number_exponent:
        // we just parsed an exponent
        number_type = token_type::value_float;
        switch (get())
        {
            case '+':
            case '-':
            {
                add(current);
                goto scan_number_sign;
            }

            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any2;
            }

            default:
            {
                error_message =
                    "invalid number; expected '+', '-', or digit after exponent";
                return token_type::parse_error;
            }
        }

scan_number_sign:
        // we just parsed an exponent sign
        switch (get())
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any2;
            }

            default:
            {
                error_message = "invalid number; expected digit after exponent sign";
                return token_type::parse_error;
            }
        }

scan_number_any2:
        // we just parsed a number after the exponent or exponent sign
        switch (get())
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            {
                add(current);
                goto scan_number_any2;
            }

            default:
                goto scan_number_done;
        }

scan_number_done:
        // unget the character after the number (we only read it to know that
        // we are done scanning a number)
        unget();

        char* endptr = nullptr;
        errno = 0;

        // try to parse integers first and fall back to floats
        if (number_type == token_type::value_unsigned)
        {
            const auto x = std::strtoull(token_buffer.data(), &endptr, 10);

            // we checked the number format before
            assert(endptr == token_buffer.data() + token_buffer.size());

            if (errno == 0)
            {
                value_unsigned = static_cast<number_unsigned_t>(x);
                if (value_unsigned == x)
                {
                    return token_type::value_unsigned;
                }
            }
        }
        else if (number_type == token_type::value_integer)
        {
            const auto x = std::strtoll(token_buffer.data(), &endptr, 10);

            // we checked the number format before
            assert(endptr == token_buffer.data() + token_buffer.size());

            if (errno == 0)
            {
                value_integer = static_cast<number_integer_t>(x);
                if (value_integer == x)
                {
                    return token_type::value_integer;
                }
            }
        }

        // this code is reached if we parse a floating-point number or if an
        // integer conversion above failed
        strtof(value_float, token_buffer.data(), &endptr);

        // we checked the number format before
        assert(endptr == token_buffer.data() + token_buffer.size());

        return token_type::value_float;
    }

    /*!
    @param[in] literal_text  the literal text to expect
    @param[in] length        the length of the passed literal text
    @param[in] return_type   the token type to return on success
    */
    token_type scan_literal(const char* literal_text, const std::size_t length,
                            token_type return_type)
    {
        assert(current == literal_text[0]);
        for (std::size_t i = 1; i < length; ++i)
        {
            if (JSON_UNLIKELY(get() != literal_text[i]))
            {
                error_message = "invalid literal";
                return token_type::parse_error;
            }
        }
        return return_type;
    }

    /////////////////////
    // input management
    /////////////////////

    /// reset token_buffer; current character is beginning of token
    void reset() noexcept
    {
        token_buffer.clear();
        token_string.clear();
        token_string.push_back(std::char_traits<char>::to_char_type(current));
    }

    /*
    @brief get next character from the input

    This function provides the interface to the used input adapter. It does
    not throw in case the input reached EOF, but returns a
    `std::char_traits<char>::eof()` in that case.  Stores the scanned characters
    for use in error messages.

    @return character read from the input
    */
    std::char_traits<char>::int_type get()
    {
        ++chars_read;
        if (next_unget)
        {
            // just reset the next_unget variable and work with current
            next_unget = false;
        }
        else
        {
            current = ia->get_character();
        }

        if (JSON_LIKELY(current != std::char_traits<char>::eof()))
        {
            token_string.push_back(std::char_traits<char>::to_char_type(current));
        }
        return current;
    }

    /*!
    @brief unget current character (read it again on next get)

    We implement unget by setting variable next_unget to true. The input is not
    changed - we just simulate ungetting by modifying chars_read and
    token_string. The next call to get() will behave as if the unget character
    is read again.
    */
    void unget()
    {
        next_unget = true;
        --chars_read;
        if (JSON_LIKELY(current != std::char_traits<char>::eof()))
        {
            assert(token_string.size() != 0);
            token_string.pop_back();
        }
    }

    /// add a character to token_buffer
    void add(int c)
    {
        token_buffer.push_back(std::char_traits<char>::to_char_type(c));
    }

  public:
    /////////////////////
    // value getters
    /////////////////////

    /// return integer value
    constexpr number_integer_t get_number_integer() const noexcept
    {
        return value_integer;
    }

    /// return unsigned integer value
    constexpr number_unsigned_t get_number_unsigned() const noexcept
    {
        return value_unsigned;
    }

    /// return floating-point value
    constexpr number_float_t get_number_float() const noexcept
    {
        return value_float;
    }

    /// return current string value (implicitly resets the token; useful only once)
    string_t& get_string()
    {
        return token_buffer;
    }

    /////////////////////
    // diagnostics
    /////////////////////

    /// return position of last read token
    constexpr std::size_t get_position() const noexcept
    {
        return chars_read;
    }

    /// return the last read token (for errors only).  Will never contain EOF
    /// (an arbitrary value that is not a valid char value, often -1), because
    /// 255 may legitimately occur.  May contain NUL, which should be escaped.
    std::string get_token_string() const
    {
        // escape control characters
        std::string result;
        for (const auto c : token_string)
        {
            if ('\x00' <= c and c <= '\x1F')
            {
                // escape control characters
                char cs[9];
                snprintf(cs, 9, "<U+%.4hhX>", static_cast<unsigned char>(c));
                result += cs;
            }
            else
            {
                // add character as is
                result.push_back(c);
            }
        }

        return result;
    }

    /// return syntax error message
    constexpr const char* get_error_message() const noexcept
    {
        return error_message;
    }

    /////////////////////
    // actual scanner
    /////////////////////

    /*!
    @brief skip the UTF-8 byte order mark
    @return true iff there is no BOM or the correct BOM has been skipped
    */
    bool skip_bom()
    {
        if (get() == 0xEF)
        {
            if (get() == 0xBB and get() == 0xBF)
            {
                // we completely parsed the BOM
                return true;
            }
            else
            {
                // after reading 0xEF, an unexpected character followed
                return false;
            }
        }
        else
        {
            // the first character is not the beginning of the BOM; unget it to
            // process is later
            unget();
            return true;
        }
    }

    token_type scan()
    {
        // initially, skip the BOM
        if (chars_read == 0 and not skip_bom())
        {
            error_message = "invalid BOM; must be 0xEF 0xBB 0xBF if given";
            return token_type::parse_error;
        }

        // read next character and ignore whitespace
        do
        {
            get();
        }
        while (current == ' ' or current == '\t' or current == '\n' or current == '\r');

        switch (current)
        {
            // structural characters
            case '[':
                return token_type::begin_array;
            case ']':
                return token_type::end_array;
            case '{':
                return token_type::begin_object;
            case '}':
                return token_type::end_object;
            case ':':
                return token_type::name_separator;
            case ',':
                return token_type::value_separator;

            // literals
            case 't':
                return scan_literal("true", 4, token_type::literal_true);
            case 'f':
                return scan_literal("false", 5, token_type::literal_false);
            case 'n':
                return scan_literal("null", 4, token_type::literal_null);

            // string
            case '\"':
                return scan_string();

            // number
            case '-':
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                return scan_number();

            // end of input (the null byte is needed when parsing from
            // string literals)
            case '\0':
            case std::char_traits<char>::eof():
                return token_type::end_of_input;

            // error
            default:
                error_message = "invalid literal";
                return token_type::parse_error;
        }
    }

  private:
    /// input adapter
    detail::input_adapter_t ia = nullptr;

    /// the current character
    std::char_traits<char>::int_type current = std::char_traits<char>::eof();

    /// whether the next get() call should just return current
    bool next_unget = false;

    /// the number of characters read
    std::size_t chars_read = 0;

    /// raw input token string (for error messages)
    std::vector<char> token_string {};

    /// buffer for variable-length tokens (numbers, strings)
    string_t token_buffer {};

    /// a description of occurred lexer errors
    const char* error_message = "";

    // number values
    number_integer_t value_integer = 0;
    number_unsigned_t value_unsigned = 0;
    number_float_t value_float = 0;

    /// the decimal point
    const char decimal_point_char = '.';
};
}
}

// #include <nlohmann/detail/input/parser.hpp>


#include <cassert> // assert
#include <cmath> // isfinite
#include <cstdint> // uint8_t
#include <functional> // function
#include <string> // string
#include <utility> // move

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/input/input_adapters.hpp>

// #include <nlohmann/detail/input/json_sax.hpp>


#include <cstddef>
#include <string>
#include <vector>

// #include <nlohmann/detail/input/parser.hpp>

// #include <nlohmann/detail/exceptions.hpp>


namespace nlohmann
{

/*!
@brief SAX interface
*/
template<typename BasicJsonType>
struct json_sax
{
    /// type for (signed) integers
    using number_integer_t = typename BasicJsonType::number_integer_t;
    /// type for unsigned integers
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    /// type for floating-point numbers
    using number_float_t = typename BasicJsonType::number_float_t;
    /// type for strings
    using string_t = typename BasicJsonType::string_t;

    /// constant to indicate that no size limit is given for array or object
    static constexpr auto no_limit = std::size_t(-1);

    /*!
    @brief a null value was read
    @return whether parsing should proceed
    */
    virtual bool null() = 0;

    /*!
    @brief a boolean value was read
    @param[in] val  boolean value
    @return whether parsing should proceed
    */
    virtual bool boolean(bool val) = 0;

    /*!
    @brief an integer number was read
    @param[in] val  integer value
    @return whether parsing should proceed
    */
    virtual bool number_integer(number_integer_t val) = 0;

    /*!
    @brief an unsigned integer number was read
    @param[in] val  unsigned integer value
    @return whether parsing should proceed
    */
    virtual bool number_unsigned(number_unsigned_t val) = 0;

    /*!
    @brief an floating-point number was read
    @param[in] val  floating-point value
    @param[in] s    raw token value
    @return whether parsing should proceed
    */
    virtual bool number_float(number_float_t val, const string_t& s) = 0;

    /*!
    @brief a string was read
    @param[in] val  string value
    @return whether parsing should proceed
    */
    virtual bool string(string_t& val) = 0;

    /*!
    @brief the beginning of an object was read
    @param[in] elements  number of object elements or no_limit if unknown
    @return whether parsing should proceed
    @note binary formats may report the number of elements
    */
    virtual bool start_object(std::size_t elements = no_limit) = 0;

    /*!
    @brief an object key was read
    @param[in] val  object key
    @return whether parsing should proceed
    */
    virtual bool key(string_t& val) = 0;

    /*!
    @brief the end of an object was read
    @return whether parsing should proceed
    */
    virtual bool end_object() = 0;

    /*!
    @brief the beginning of an array was read
    @param[in] elements  number of array elements or no_limit if unknown
    @return whether parsing should proceed
    @note binary formats may report the number of elements
    */
    virtual bool start_array(std::size_t elements = no_limit) = 0;

    /*!
    @brief the end of an array was read
    @return whether parsing should proceed
    */
    virtual bool end_array() = 0;

    /*!
    @brief a parse error occurred
    @param[in] position    the position in the input where the error occurs
    @param[in] last_token  the last read token
    @param[in] error_msg   a detailed error message
    @return whether parsing should proceed (must return false)
    */
    virtual bool parse_error(std::size_t position,
                             const std::string& last_token,
                             const detail::exception& ex) = 0;

    virtual ~json_sax() = default;
};


namespace detail
{
/*!
@brief SAX implementation to create a JSON value from SAX events

This class implements the @ref json_sax interface and processes the SAX events
to create a JSON value which makes it basically a DOM parser. The structure or
hierarchy of the JSON value is managed by the stack `ref_stack` which contains
a pointer to the respective array or object for each recursion depth.

After successful parsing, the value that is passed by reference to the
constructor contains the parsed value.

@tparam BasicJsonType  the JSON type
*/
template<typename BasicJsonType>
class json_sax_dom_parser : public json_sax<BasicJsonType>
{
  public:
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;

    /*!
    @param[in, out] r  reference to a JSON value that is manipulated while
                       parsing
    @param[in] allow_exceptions_  whether parse errors yield exceptions
    */
    json_sax_dom_parser(BasicJsonType& r, const bool allow_exceptions_ = true)
        : root(r), allow_exceptions(allow_exceptions_)
    {}

    bool null() override
    {
        handle_value(nullptr);
        return true;
    }

    bool boolean(bool val) override
    {
        handle_value(val);
        return true;
    }

    bool number_integer(number_integer_t val) override
    {
        handle_value(val);
        return true;
    }

    bool number_unsigned(number_unsigned_t val) override
    {
        handle_value(val);
        return true;
    }

    bool number_float(number_float_t val, const string_t&) override
    {
        handle_value(val);
        return true;
    }

    bool string(string_t& val) override
    {
        handle_value(val);
        return true;
    }

    bool start_object(std::size_t len) override
    {
        ref_stack.push_back(handle_value(BasicJsonType::value_t::object));

        if (JSON_UNLIKELY(len != json_sax<BasicJsonType>::no_limit and len > ref_stack.back()->max_size()))
        {
            JSON_THROW(out_of_range::create(408,
                                            "excessive object size: " + std::to_string(len)));
        }

        return true;
    }

    bool key(string_t& val) override
    {
        // add null at given key and store the reference for later
        object_element = &(ref_stack.back()->m_value.object->operator[](val));
        return true;
    }

    bool end_object() override
    {
        ref_stack.pop_back();
        return true;
    }

    bool start_array(std::size_t len) override
    {
        ref_stack.push_back(handle_value(BasicJsonType::value_t::array));

        if (JSON_UNLIKELY(len != json_sax<BasicJsonType>::no_limit and len > ref_stack.back()->max_size()))
        {
            JSON_THROW(out_of_range::create(408,
                                            "excessive array size: " + std::to_string(len)));
        }

        return true;
    }

    bool end_array() override
    {
        ref_stack.pop_back();
        return true;
    }

    bool parse_error(std::size_t, const std::string&,
                     const detail::exception& ex) override
    {
        errored = true;
        if (allow_exceptions)
        {
            // determine the proper exception type from the id
            switch ((ex.id / 100) % 100)
            {
                case 1:
                    JSON_THROW(*reinterpret_cast<const detail::parse_error*>(&ex));
                case 2:
                    JSON_THROW(*reinterpret_cast<const detail::invalid_iterator*>(&ex));  // LCOV_EXCL_LINE
                case 3:
                    JSON_THROW(*reinterpret_cast<const detail::type_error*>(&ex));  // LCOV_EXCL_LINE
                case 4:
                    JSON_THROW(*reinterpret_cast<const detail::out_of_range*>(&ex));
                case 5:
                    JSON_THROW(*reinterpret_cast<const detail::other_error*>(&ex));  // LCOV_EXCL_LINE
                default:
                    assert(false);  // LCOV_EXCL_LINE
            }
        }
        return false;
    }

    constexpr bool is_errored() const
    {
        return errored;
    }

  private:
    /*!
    @invariant If the ref stack is empty, then the passed value will be the new
               root.
    @invariant If the ref stack contains a value, then it is an array or an
               object to which we can add elements
    */
    template<typename Value>
    BasicJsonType* handle_value(Value&& v)
    {
        if (ref_stack.empty())
        {
            root = BasicJsonType(std::forward<Value>(v));
            return &root;
        }
        else
        {
            assert(ref_stack.back()->is_array() or ref_stack.back()->is_object());
            if (ref_stack.back()->is_array())
            {
                ref_stack.back()->m_value.array->emplace_back(std::forward<Value>(v));
                return &(ref_stack.back()->m_value.array->back());
            }
            else
            {
                assert(object_element);
                *object_element = BasicJsonType(std::forward<Value>(v));
                return object_element;
            }
        }
    }

    /// the parsed JSON value
    BasicJsonType& root;
    /// stack to model hierarchy of values
    std::vector<BasicJsonType*> ref_stack;
    /// helper to hold the reference for the next object element
    BasicJsonType* object_element = nullptr;
    /// whether a syntax error occurred
    bool errored = false;
    /// whether to throw exceptions in case of errors
    const bool allow_exceptions = true;
};

template<typename BasicJsonType>
class json_sax_dom_callback_parser : public json_sax<BasicJsonType>
{
  public:
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using parser_callback_t = typename BasicJsonType::parser_callback_t;
    using parse_event_t = typename BasicJsonType::parse_event_t;

    json_sax_dom_callback_parser(BasicJsonType& r,
                                 const parser_callback_t cb,
                                 const bool allow_exceptions_ = true)
        : root(r), callback(cb), allow_exceptions(allow_exceptions_)
    {
        keep_stack.push_back(true);
    }

    bool null() override
    {
        handle_value(nullptr);
        return true;
    }

    bool boolean(bool val) override
    {
        handle_value(val);
        return true;
    }

    bool number_integer(number_integer_t val) override
    {
        handle_value(val);
        return true;
    }

    bool number_unsigned(number_unsigned_t val) override
    {
        handle_value(val);
        return true;
    }

    bool number_float(number_float_t val, const string_t&) override
    {
        handle_value(val);
        return true;
    }

    bool string(string_t& val) override
    {
        handle_value(val);
        return true;
    }

    bool start_object(std::size_t len) override
    {
        // check callback for object start
        const bool keep = callback(static_cast<int>(ref_stack.size()), parse_event_t::object_start, discarded);
        keep_stack.push_back(keep);

        auto val = handle_value(BasicJsonType::value_t::object, true);
        ref_stack.push_back(val.second);

        // check object limit
        if (ref_stack.back())
        {
            if (JSON_UNLIKELY(len != json_sax<BasicJsonType>::no_limit and len > ref_stack.back()->max_size()))
            {
                JSON_THROW(out_of_range::create(408,
                                                "excessive object size: " + std::to_string(len)));
            }
        }

        return true;
    }

    bool key(string_t& val) override
    {
        BasicJsonType k = BasicJsonType(val);

        // check callback for key
        const bool keep = callback(static_cast<int>(ref_stack.size()), parse_event_t::key, k);
        key_keep_stack.push_back(keep);

        // add discarded value at given key and store the reference for later
        if (keep and ref_stack.back())
        {
            object_element = &(ref_stack.back()->m_value.object->operator[](val) = discarded);
        }

        return true;
    }

    bool end_object() override
    {
        if (ref_stack.back())
        {
            if (not callback(static_cast<int>(ref_stack.size()) - 1, parse_event_t::object_end, *ref_stack.back()))
            {
                // discard object
                *ref_stack.back() = discarded;
            }
        }

        assert(not ref_stack.empty());
        assert(not keep_stack.empty());
        ref_stack.pop_back();
        keep_stack.pop_back();

        if (not ref_stack.empty() and ref_stack.back())
        {
            // remove discarded value
            if (ref_stack.back()->is_object())
            {
                for (auto it = ref_stack.back()->begin(); it != ref_stack.back()->end(); ++it)
                {
                    if (it->is_discarded())
                    {
                        ref_stack.back()->erase(it);
                        break;
                    }
                }
            }
        }

        return true;
    }

    bool start_array(std::size_t len) override
    {
        const bool keep = callback(static_cast<int>(ref_stack.size()), parse_event_t::array_start, discarded);
        keep_stack.push_back(keep);

        auto val = handle_value(BasicJsonType::value_t::array, true);
        ref_stack.push_back(val.second);

        // check array limit
        if (ref_stack.back())
        {
            if (JSON_UNLIKELY(len != json_sax<BasicJsonType>::no_limit and len > ref_stack.back()->max_size()))
            {
                JSON_THROW(out_of_range::create(408,
                                                "excessive array size: " + std::to_string(len)));
            }
        }

        return true;
    }

    bool end_array() override
    {
        bool keep = true;

        if (ref_stack.back())
        {
            keep = callback(static_cast<int>(ref_stack.size()) - 1, parse_event_t::array_end, *ref_stack.back());
            if (not keep)
            {
                // discard array
                *ref_stack.back() = discarded;
            }
        }

        assert(not ref_stack.empty());
        assert(not keep_stack.empty());
        ref_stack.pop_back();
        keep_stack.pop_back();

        // remove discarded value
        if (not keep and not ref_stack.empty())
        {
            if (ref_stack.back()->is_array())
            {
                ref_stack.back()->m_value.array->pop_back();
            }
        }

        return true;
    }

    bool parse_error(std::size_t, const std::string&,
                     const detail::exception& ex) override
    {
        errored = true;
        if (allow_exceptions)
        {
            // determine the proper exception type from the id
            switch ((ex.id / 100) % 100)
            {
                case 1:
                    JSON_THROW(*reinterpret_cast<const detail::parse_error*>(&ex));
                case 2:
                    JSON_THROW(*reinterpret_cast<const detail::invalid_iterator*>(&ex));  // LCOV_EXCL_LINE
                case 3:
                    JSON_THROW(*reinterpret_cast<const detail::type_error*>(&ex));  // LCOV_EXCL_LINE
                case 4:
                    JSON_THROW(*reinterpret_cast<const detail::out_of_range*>(&ex));
                case 5:
                    JSON_THROW(*reinterpret_cast<const detail::other_error*>(&ex));  // LCOV_EXCL_LINE
                default:
                    assert(false);  // LCOV_EXCL_LINE
            }
        }
        return false;
    }

    constexpr bool is_errored() const
    {
        return errored;
    }

  private:
    /*!
    @param[in] v  value to add to the JSON value we build during parsing
    @param[in] skip_callback  whether we should skip calling the callback
               function; this is required after start_array() and
               start_object() SAX events, because otherwise we would call the
               callback function with an empty array or object, respectively.

    @invariant If the ref stack is empty, then the passed value will be the new
               root.
    @invariant If the ref stack contains a value, then it is an array or an
               object to which we can add elements

    @return pair of boolean (whether value should be kept) and pointer (to the
            passed value in the ref_stack hierarchy; nullptr if not kept)
    */
    template<typename Value>
    std::pair<bool, BasicJsonType*> handle_value(Value&& v, const bool skip_callback = false)
    {
        assert(not keep_stack.empty());

        // do not handle this value if we know it would be added to a discarded
        // container
        if (not keep_stack.back())
        {
            return {false, nullptr};
        }

        // create value
        auto value = BasicJsonType(std::forward<Value>(v));

        // check callback
        const bool keep = skip_callback or callback(static_cast<int>(ref_stack.size()), parse_event_t::value, value);

        // do not handle this value if we just learnt it shall be discarded
        if (not keep)
        {
            return {false, nullptr};
        }

        if (ref_stack.empty())
        {
            root = std::move(value);
            return {true, &root};
        }
        else
        {
            assert(ref_stack.back()->is_array() or ref_stack.back()->is_object());
            if (ref_stack.back()->is_array())
            {
                ref_stack.back()->m_value.array->push_back(std::move(value));
                return {true, &(ref_stack.back()->m_value.array->back())};
            }
            else
            {
                // check if we should store an element for the current key
                assert(not key_keep_stack.empty());
                const bool store_element = key_keep_stack.back();
                key_keep_stack.pop_back();

                if (not store_element)
                {
                    return {false, nullptr};
                }

                assert(object_element);
                *object_element = std::move(value);
                return {true, object_element};
            }
        }
    }

    /// the parsed JSON value
    BasicJsonType& root;
    /// stack to model hierarchy of values
    std::vector<BasicJsonType*> ref_stack;
    /// stack to manage which values to keep
    std::vector<bool> keep_stack;
    /// stack to manage which object keys to keep
    std::vector<bool> key_keep_stack;
    /// helper to hold the reference for the next object element
    BasicJsonType* object_element = nullptr;
    /// whether a syntax error occurred
    bool errored = false;
    /// callback function
    const parser_callback_t callback = nullptr;
    /// whether to throw exceptions in case of errors
    const bool allow_exceptions = true;
    /// a discarded value for the callback
    BasicJsonType discarded = BasicJsonType::value_t::discarded;
};

template<typename BasicJsonType>
class json_sax_acceptor : public json_sax<BasicJsonType>
{
  public:
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;

    bool null() override
    {
        return true;
    }

    bool boolean(bool) override
    {
        return true;
    }

    bool number_integer(number_integer_t) override
    {
        return true;
    }

    bool number_unsigned(number_unsigned_t) override
    {
        return true;
    }

    bool number_float(number_float_t, const string_t&) override
    {
        return true;
    }

    bool string(string_t&) override
    {
        return true;
    }

    bool start_object(std::size_t) override
    {
        return true;
    }

    bool key(string_t&) override
    {
        return true;
    }

    bool end_object() override
    {
        return true;
    }

    bool start_array(std::size_t) override
    {
        return true;
    }

    bool end_array() override
    {
        return true;
    }

    bool parse_error(std::size_t, const std::string&, const detail::exception&) override
    {
        return false;
    }
};
}

}

// #include <nlohmann/detail/input/lexer.hpp>

// #include <nlohmann/detail/value_t.hpp>


namespace nlohmann
{
namespace detail
{
////////////
// parser //
////////////

/*!
@brief syntax analysis

This class implements a recursive decent parser.
*/
template<typename BasicJsonType>
class parser
{
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using lexer_t = lexer<BasicJsonType>;
    using token_type = typename lexer_t::token_type;

  public:
    enum class parse_event_t : uint8_t
    {
        /// the parser read `{` and started to process a JSON object
        object_start,
        /// the parser read `}` and finished processing a JSON object
        object_end,
        /// the parser read `[` and started to process a JSON array
        array_start,
        /// the parser read `]` and finished processing a JSON array
        array_end,
        /// the parser read a key of a value in an object
        key,
        /// the parser finished reading a JSON value
        value
    };

    using json_sax_t = json_sax<BasicJsonType>;

    using parser_callback_t =
        std::function<bool(int depth, parse_event_t event, BasicJsonType& parsed)>;

    /// a parser reading from an input adapter
    explicit parser(detail::input_adapter_t&& adapter,
                    const parser_callback_t cb = nullptr,
                    const bool allow_exceptions_ = true)
        : callback(cb), m_lexer(std::move(adapter)), allow_exceptions(allow_exceptions_)
    {
        // read first token
        get_token();
    }

    /*!
    @brief public parser interface

    @param[in] strict      whether to expect the last token to be EOF
    @param[in,out] result  parsed JSON value

    @throw parse_error.101 in case of an unexpected token
    @throw parse_error.102 if to_unicode fails or surrogate error
    @throw parse_error.103 if to_unicode fails
    */
    void parse(const bool strict, BasicJsonType& result)
    {
        if (callback)
        {
            json_sax_dom_callback_parser<BasicJsonType> sdp(result, callback, allow_exceptions);
            sax_parse_internal(&sdp);
            result.assert_invariant();

            // in strict mode, input must be completely read
            if (strict and (get_token() != token_type::end_of_input))
            {
                sdp.parse_error(m_lexer.get_position(),
                                m_lexer.get_token_string(),
                                parse_error::create(101, m_lexer.get_position(), exception_message(token_type::end_of_input)));
            }

            // in case of an error, return discarded value
            if (sdp.is_errored())
            {
                result = value_t::discarded;
                return;
            }

            // set top-level value to null if it was discarded by the callback
            // function
            if (result.is_discarded())
            {
                result = nullptr;
            }
        }
        else
        {
            json_sax_dom_parser<BasicJsonType> sdp(result, allow_exceptions);
            sax_parse_internal(&sdp);
            result.assert_invariant();

            // in strict mode, input must be completely read
            if (strict and (get_token() != token_type::end_of_input))
            {
                sdp.parse_error(m_lexer.get_position(),
                                m_lexer.get_token_string(),
                                parse_error::create(101, m_lexer.get_position(), exception_message(token_type::end_of_input)));
            }

            // in case of an error, return discarded value
            if (sdp.is_errored())
            {
                result = value_t::discarded;
                return;
            }
        }
    }

    /*!
    @brief public accept interface

    @param[in] strict  whether to expect the last token to be EOF
    @return whether the input is a proper JSON text
    */
    bool accept(const bool strict = true)
    {
        json_sax_acceptor<BasicJsonType> sax_acceptor;
        return sax_parse(&sax_acceptor, strict);
    }

    bool sax_parse(json_sax_t* sax, const bool strict = true)
    {
        const bool result = sax_parse_internal(sax);

        // strict mode: next byte must be EOF
        if (result and strict and (get_token() != token_type::end_of_input))
        {
            return sax->parse_error(m_lexer.get_position(),
                                    m_lexer.get_token_string(),
                                    parse_error::create(101, m_lexer.get_position(), exception_message(token_type::end_of_input)));
        }

        return result;
    }

  private:
    bool sax_parse_internal(json_sax_t* sax)
    {
        // stack to remember the hieararchy of structured values we are parsing
        // true = array; false = object
        std::vector<bool> states;
        // value to avoid a goto (see comment where set to true)
        bool skip_to_state_evaluation = false;

        while (true)
        {
            if (not skip_to_state_evaluation)
            {
                // invariant: get_token() was called before each iteration
                switch (last_token)
                {
                    case token_type::begin_object:
                    {
                        if (JSON_UNLIKELY(not sax->start_object()))
                        {
                            return false;
                        }

                        // closing } -> we are done
                        if (get_token() == token_type::end_object)
                        {
                            if (JSON_UNLIKELY(not sax->end_object()))
                            {
                                return false;
                            }
                            break;
                        }

                        // parse key
                        if (JSON_UNLIKELY(last_token != token_type::value_string))
                        {
                            return sax->parse_error(m_lexer.get_position(),
                                                    m_lexer.get_token_string(),
                                                    parse_error::create(101, m_lexer.get_position(), exception_message(token_type::value_string)));
                        }
                        else
                        {
                            if (JSON_UNLIKELY(not sax->key(m_lexer.get_string())))
                            {
                                return false;
                            }
                        }

                        // parse separator (:)
                        if (JSON_UNLIKELY(get_token() != token_type::name_separator))
                        {
                            return sax->parse_error(m_lexer.get_position(),
                                                    m_lexer.get_token_string(),
                                                    parse_error::create(101, m_lexer.get_position(), exception_message(token_type::name_separator)));
                        }

                        // remember we are now inside an object
                        states.push_back(false);

                        // parse values
                        get_token();
                        continue;
                    }

                    case token_type::begin_array:
                    {
                        if (JSON_UNLIKELY(not sax->start_array()))
                        {
                            return false;
                        }

                        // closing ] -> we are done
                        if (get_token() == token_type::end_array)
                        {
                            if (JSON_UNLIKELY(not sax->end_array()))
                            {
                                return false;
                            }
                            break;
                        }

                        // remember we are now inside an array
                        states.push_back(true);

                        // parse values (no need to call get_token)
                        continue;
                    }

                    case token_type::value_float:
                    {
                        const auto res = m_lexer.get_number_float();

                        if (JSON_UNLIKELY(not std::isfinite(res)))
                        {
                            return sax->parse_error(m_lexer.get_position(),
                                                    m_lexer.get_token_string(),
                                                    out_of_range::create(406, "number overflow parsing '" + m_lexer.get_token_string() + "'"));
                        }
                        else
                        {
                            if (JSON_UNLIKELY(not sax->number_float(res, m_lexer.get_string())))
                            {
                                return false;
                            }
                            break;
                        }
                    }

                    case token_type::literal_false:
                    {
                        if (JSON_UNLIKELY(not sax->boolean(false)))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::literal_null:
                    {
                        if (JSON_UNLIKELY(not sax->null()))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::literal_true:
                    {
                        if (JSON_UNLIKELY(not sax->boolean(true)))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::value_integer:
                    {
                        if (JSON_UNLIKELY(not sax->number_integer(m_lexer.get_number_integer())))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::value_string:
                    {
                        if (JSON_UNLIKELY(not sax->string(m_lexer.get_string())))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::value_unsigned:
                    {
                        if (JSON_UNLIKELY(not sax->number_unsigned(m_lexer.get_number_unsigned())))
                        {
                            return false;
                        }
                        break;
                    }

                    case token_type::parse_error:
                    {
                        // using "uninitialized" to avoid "expected" message
                        return sax->parse_error(m_lexer.get_position(),
                                                m_lexer.get_token_string(),
                                                parse_error::create(101, m_lexer.get_position(), exception_message(token_type::uninitialized)));
                    }

                    default: // the last token was unexpected
                    {
                        return sax->parse_error(m_lexer.get_position(),
                                                m_lexer.get_token_string(),
                                                parse_error::create(101, m_lexer.get_position(), exception_message(token_type::literal_or_value)));
                    }
                }
            }
            else
            {
                skip_to_state_evaluation = false;
            }

            // we reached this line after we successfully parsed a value
            if (states.empty())
            {
                // empty stack: we reached the end of the hieararchy: done
                return true;
            }
            else
            {
                if (states.back())  // array
                {
                    // comma -> next value
                    if (get_token() == token_type::value_separator)
                    {
                        // parse a new value
                        get_token();
                        continue;
                    }

                    // closing ]
                    if (JSON_LIKELY(last_token == token_type::end_array))
                    {
                        if (JSON_UNLIKELY(not sax->end_array()))
                        {
                            return false;
                        }

                        // We are done with this array. Before we can parse a
                        // new value, we need to evaluate the new state first.
                        // By setting skip_to_state_evaluation to false, we
                        // are effectively jumping to the beginning of this if.
                        assert(not states.empty());
                        states.pop_back();
                        skip_to_state_evaluation = true;
                        continue;
                    }
                    else
                    {
                        return sax->parse_error(m_lexer.get_position(),
                                                m_lexer.get_token_string(),
                                                parse_error::create(101, m_lexer.get_position(), exception_message(token_type::end_array)));
                    }
                }
                else  // object
                {
                    // comma -> next value
                    if (get_token() == token_type::value_separator)
                    {
                        // parse key
                        if (JSON_UNLIKELY(get_token() != token_type::value_string))
                        {
                            return sax->parse_error(m_lexer.get_position(),
                                                    m_lexer.get_token_string(),
                                                    parse_error::create(101, m_lexer.get_position(), exception_message(token_type::value_string)));
                        }
                        else
                        {
                            if (JSON_UNLIKELY(not sax->key(m_lexer.get_string())))
                            {
                                return false;
                            }
                        }

                        // parse separator (:)
                        if (JSON_UNLIKELY(get_token() != token_type::name_separator))
                        {
                            return sax->parse_error(m_lexer.get_position(),
                                                    m_lexer.get_token_string(),
                                                    parse_error::create(101, m_lexer.get_position(), exception_message(token_type::name_separator)));
                        }

                        // parse values
                        get_token();
                        continue;
                    }

                    // closing }
                    if (JSON_LIKELY(last_token == token_type::end_object))
                    {
                        if (JSON_UNLIKELY(not sax->end_object()))
                        {
                            return false;
                        }

                        // We are done with this object. Before we can parse a
                        // new value, we need to evaluate the new state first.
                        // By setting skip_to_state_evaluation to false, we
                        // are effectively jumping to the beginning of this if.
                        assert(not states.empty());
                        states.pop_back();
                        skip_to_state_evaluation = true;
                        continue;
                    }
                    else
                    {
                        return sax->parse_error(m_lexer.get_position(),
                                                m_lexer.get_token_string(),
                                                parse_error::create(101, m_lexer.get_position(), exception_message(token_type::end_object)));
                    }
                }
            }
        }
    }

    /// get next token from lexer
    token_type get_token()
    {
        return (last_token = m_lexer.scan());
    }

    std::string exception_message(const token_type expected)
    {
        std::string error_msg = "syntax error - ";
        if (last_token == token_type::parse_error)
        {
            error_msg += std::string(m_lexer.get_error_message()) + "; last read: '" +
                         m_lexer.get_token_string() + "'";
        }
        else
        {
            error_msg += "unexpected " + std::string(lexer_t::token_type_name(last_token));
        }

        if (expected != token_type::uninitialized)
        {
            error_msg += "; expected " + std::string(lexer_t::token_type_name(expected));
        }

        return error_msg;
    }

  private:
    /// callback function
    const parser_callback_t callback = nullptr;
    /// the type of the last read token
    token_type last_token = token_type::uninitialized;
    /// the lexer
    lexer_t m_lexer;
    /// whether to throw exceptions in case of errors
    const bool allow_exceptions = true;
};
}
}

// #include <nlohmann/detail/iterators/primitive_iterator.hpp>


#include <cstddef> // ptrdiff_t
#include <limits>  // numeric_limits

namespace nlohmann
{
namespace detail
{
/*
@brief an iterator for primitive JSON types

This class models an iterator for primitive JSON types (boolean, number,
string). It's only purpose is to allow the iterator/const_iterator classes
to "iterate" over primitive values. Internally, the iterator is modeled by
a `difference_type` variable. Value begin_value (`0`) models the begin,
end_value (`1`) models past the end.
*/
class primitive_iterator_t
{
  private:
    using difference_type = std::ptrdiff_t;
    static constexpr difference_type begin_value = 0;
    static constexpr difference_type end_value = begin_value + 1;

    /// iterator as signed integer type
    difference_type m_it = (std::numeric_limits<std::ptrdiff_t>::min)();

  public:
    constexpr difference_type get_value() const noexcept
    {
        return m_it;
    }

    /// set iterator to a defined beginning
    void set_begin() noexcept
    {
        m_it = begin_value;
    }

    /// set iterator to a defined past the end
    void set_end() noexcept
    {
        m_it = end_value;
    }

    /// return whether the iterator can be dereferenced
    constexpr bool is_begin() const noexcept
    {
        return m_it == begin_value;
    }

    /// return whether the iterator is at end
    constexpr bool is_end() const noexcept
    {
        return m_it == end_value;
    }

    friend constexpr bool operator==(primitive_iterator_t lhs, primitive_iterator_t rhs) noexcept
    {
        return lhs.m_it == rhs.m_it;
    }

    friend constexpr bool operator<(primitive_iterator_t lhs, primitive_iterator_t rhs) noexcept
    {
        return lhs.m_it < rhs.m_it;
    }

    primitive_iterator_t operator+(difference_type n) noexcept
    {
        auto result = *this;
        result += n;
        return result;
    }

    friend constexpr difference_type operator-(primitive_iterator_t lhs, primitive_iterator_t rhs) noexcept
    {
        return lhs.m_it - rhs.m_it;
    }

    primitive_iterator_t& operator++() noexcept
    {
        ++m_it;
        return *this;
    }

    primitive_iterator_t const operator++(int) noexcept
    {
        auto result = *this;
        ++m_it;
        return result;
    }

    primitive_iterator_t& operator--() noexcept
    {
        --m_it;
        return *this;
    }

    primitive_iterator_t const operator--(int) noexcept
    {
        auto result = *this;
        --m_it;
        return result;
    }

    primitive_iterator_t& operator+=(difference_type n) noexcept
    {
        m_it += n;
        return *this;
    }

    primitive_iterator_t& operator-=(difference_type n) noexcept
    {
        m_it -= n;
        return *this;
    }
};
}
}

// #include <nlohmann/detail/iterators/internal_iterator.hpp>


// #include <nlohmann/detail/iterators/primitive_iterator.hpp>


namespace nlohmann
{
namespace detail
{
/*!
@brief an iterator value

@note This structure could easily be a union, but MSVC currently does not allow
unions members with complex constructors, see https://github.com/nlohmann/json/pull/105.
*/
template<typename BasicJsonType> struct internal_iterator
{
    /// iterator for JSON objects
    typename BasicJsonType::object_t::iterator object_iterator {};
    /// iterator for JSON arrays
    typename BasicJsonType::array_t::iterator array_iterator {};
    /// generic iterator for all other types
    primitive_iterator_t primitive_iterator {};
};
}
}

// #include <nlohmann/detail/iterators/iter_impl.hpp>


#include <ciso646> // not
#include <iterator> // iterator, random_access_iterator_tag, bidirectional_iterator_tag, advance, next
#include <type_traits> // conditional, is_const, remove_const

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/iterators/internal_iterator.hpp>

// #include <nlohmann/detail/iterators/primitive_iterator.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/meta.hpp>

// #include <nlohmann/detail/value_t.hpp>


namespace nlohmann
{
namespace detail
{
// forward declare, to be able to friend it later on
template<typename IteratorType> class iteration_proxy;

/*!
@brief a template for a bidirectional iterator for the @ref basic_json class

This class implements a both iterators (iterator and const_iterator) for the
@ref basic_json class.

@note An iterator is called *initialized* when a pointer to a JSON value has
      been set (e.g., by a constructor or a copy assignment). If the iterator is
      default-constructed, it is *uninitialized* and most methods are undefined.
      **The library uses assertions to detect calls on uninitialized iterators.**

@requirement The class satisfies the following concept requirements:
-
[BidirectionalIterator](https://en.cppreference.com/w/cpp/named_req/BidirectionalIterator):
  The iterator that can be moved can be moved in both directions (i.e.
  incremented and decremented).

@since version 1.0.0, simplified in version 2.0.9, change to bidirectional
       iterators in version 3.0.0 (see https://github.com/nlohmann/json/issues/593)
*/
template<typename BasicJsonType>
class iter_impl
{
    /// allow basic_json to access private members
    friend iter_impl<typename std::conditional<std::is_const<BasicJsonType>::value, typename std::remove_const<BasicJsonType>::type, const BasicJsonType>::type>;
    friend BasicJsonType;
    friend iteration_proxy<iter_impl>;

    using object_t = typename BasicJsonType::object_t;
    using array_t = typename BasicJsonType::array_t;
    // make sure BasicJsonType is basic_json or const basic_json
    static_assert(is_basic_json<typename std::remove_const<BasicJsonType>::type>::value,
                  "iter_impl only accepts (const) basic_json");

  public:

    /// The std::iterator class template (used as a base class to provide typedefs) is deprecated in C++17.
    /// The C++ Standard has never required user-defined iterators to derive from std::iterator.
    /// A user-defined iterator should provide publicly accessible typedefs named
    /// iterator_category, value_type, difference_type, pointer, and reference.
    /// Note that value_type is required to be non-const, even for constant iterators.
    using iterator_category = std::bidirectional_iterator_tag;

    /// the type of the values when the iterator is dereferenced
    using value_type = typename BasicJsonType::value_type;
    /// a type to represent differences between iterators
    using difference_type = typename BasicJsonType::difference_type;
    /// defines a pointer to the type iterated over (value_type)
    using pointer = typename std::conditional<std::is_const<BasicJsonType>::value,
          typename BasicJsonType::const_pointer,
          typename BasicJsonType::pointer>::type;
    /// defines a reference to the type iterated over (value_type)
    using reference =
        typename std::conditional<std::is_const<BasicJsonType>::value,
        typename BasicJsonType::const_reference,
        typename BasicJsonType::reference>::type;

    /// default constructor
    iter_impl() = default;

    /*!
    @brief constructor for a given JSON instance
    @param[in] object  pointer to a JSON object for this iterator
    @pre object != nullptr
    @post The iterator is initialized; i.e. `m_object != nullptr`.
    */
    explicit iter_impl(pointer object) noexcept : m_object(object)
    {
        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
            {
                m_it.object_iterator = typename object_t::iterator();
                break;
            }

            case value_t::array:
            {
                m_it.array_iterator = typename array_t::iterator();
                break;
            }

            default:
            {
                m_it.primitive_iterator = primitive_iterator_t();
                break;
            }
        }
    }

    /*!
    @note The conventional copy constructor and copy assignment are implicitly
          defined. Combined with the following converting constructor and
          assignment, they support: (1) copy from iterator to iterator, (2)
          copy from const iterator to const iterator, and (3) conversion from
          iterator to const iterator. However conversion from const iterator
          to iterator is not defined.
    */

    /*!
    @brief converting constructor
    @param[in] other  non-const iterator to copy from
    @note It is not checked whether @a other is initialized.
    */
    iter_impl(const iter_impl<typename std::remove_const<BasicJsonType>::type>& other) noexcept
        : m_object(other.m_object), m_it(other.m_it) {}

    /*!
    @brief converting assignment
    @param[in,out] other  non-const iterator to copy from
    @return const/non-const iterator
    @note It is not checked whether @a other is initialized.
    */
    iter_impl& operator=(const iter_impl<typename std::remove_const<BasicJsonType>::type>& other) noexcept
    {
        m_object = other.m_object;
        m_it = other.m_it;
        return *this;
    }

  private:
    /*!
    @brief set the iterator to the first value
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    void set_begin() noexcept
    {
        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
            {
                m_it.object_iterator = m_object->m_value.object->begin();
                break;
            }

            case value_t::array:
            {
                m_it.array_iterator = m_object->m_value.array->begin();
                break;
            }

            case value_t::null:
            {
                // set to end so begin()==end() is true: null is empty
                m_it.primitive_iterator.set_end();
                break;
            }

            default:
            {
                m_it.primitive_iterator.set_begin();
                break;
            }
        }
    }

    /*!
    @brief set the iterator past the last value
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    void set_end() noexcept
    {
        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
            {
                m_it.object_iterator = m_object->m_value.object->end();
                break;
            }

            case value_t::array:
            {
                m_it.array_iterator = m_object->m_value.array->end();
                break;
            }

            default:
            {
                m_it.primitive_iterator.set_end();
                break;
            }
        }
    }

  public:
    /*!
    @brief return a reference to the value pointed to by the iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    reference operator*() const
    {
        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
            {
                assert(m_it.object_iterator != m_object->m_value.object->end());
                return m_it.object_iterator->second;
            }

            case value_t::array:
            {
                assert(m_it.array_iterator != m_object->m_value.array->end());
                return *m_it.array_iterator;
            }

            case value_t::null:
                JSON_THROW(invalid_iterator::create(214, "cannot get value"));

            default:
            {
                if (JSON_LIKELY(m_it.primitive_iterator.is_begin()))
                {
                    return *m_object;
                }

                JSON_THROW(invalid_iterator::create(214, "cannot get value"));
            }
        }
    }

    /*!
    @brief dereference the iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    pointer operator->() const
    {
        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
            {
                assert(m_it.object_iterator != m_object->m_value.object->end());
                return &(m_it.object_iterator->second);
            }

            case value_t::array:
            {
                assert(m_it.array_iterator != m_object->m_value.array->end());
                return &*m_it.array_iterator;
            }

            default:
            {
                if (JSON_LIKELY(m_it.primitive_iterator.is_begin()))
                {
                    return m_object;
                }

                JSON_THROW(invalid_iterator::create(214, "cannot get value"));
            }
        }
    }

    /*!
    @brief post-increment (it++)
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl const operator++(int)
    {
        auto result = *this;
        ++(*this);
        return result;
    }

    /*!
    @brief pre-increment (++it)
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl& operator++()
    {
        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
            {
                std::advance(m_it.object_iterator, 1);
                break;
            }

            case value_t::array:
            {
                std::advance(m_it.array_iterator, 1);
                break;
            }

            default:
            {
                ++m_it.primitive_iterator;
                break;
            }
        }

        return *this;
    }

    /*!
    @brief post-decrement (it--)
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl const operator--(int)
    {
        auto result = *this;
        --(*this);
        return result;
    }

    /*!
    @brief pre-decrement (--it)
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl& operator--()
    {
        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
            {
                std::advance(m_it.object_iterator, -1);
                break;
            }

            case value_t::array:
            {
                std::advance(m_it.array_iterator, -1);
                break;
            }

            default:
            {
                --m_it.primitive_iterator;
                break;
            }
        }

        return *this;
    }

    /*!
    @brief  comparison: equal
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    bool operator==(const iter_impl& other) const
    {
        // if objects are not the same, the comparison is undefined
        if (JSON_UNLIKELY(m_object != other.m_object))
        {
            JSON_THROW(invalid_iterator::create(212, "cannot compare iterators of different containers"));
        }

        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
                return (m_it.object_iterator == other.m_it.object_iterator);

            case value_t::array:
                return (m_it.array_iterator == other.m_it.array_iterator);

            default:
                return (m_it.primitive_iterator == other.m_it.primitive_iterator);
        }
    }

    /*!
    @brief  comparison: not equal
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    bool operator!=(const iter_impl& other) const
    {
        return not operator==(other);
    }

    /*!
    @brief  comparison: smaller
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    bool operator<(const iter_impl& other) const
    {
        // if objects are not the same, the comparison is undefined
        if (JSON_UNLIKELY(m_object != other.m_object))
        {
            JSON_THROW(invalid_iterator::create(212, "cannot compare iterators of different containers"));
        }

        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
                JSON_THROW(invalid_iterator::create(213, "cannot compare order of object iterators"));

            case value_t::array:
                return (m_it.array_iterator < other.m_it.array_iterator);

            default:
                return (m_it.primitive_iterator < other.m_it.primitive_iterator);
        }
    }

    /*!
    @brief  comparison: less than or equal
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    bool operator<=(const iter_impl& other) const
    {
        return not other.operator < (*this);
    }

    /*!
    @brief  comparison: greater than
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    bool operator>(const iter_impl& other) const
    {
        return not operator<=(other);
    }

    /*!
    @brief  comparison: greater than or equal
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    bool operator>=(const iter_impl& other) const
    {
        return not operator<(other);
    }

    /*!
    @brief  add to iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl& operator+=(difference_type i)
    {
        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
                JSON_THROW(invalid_iterator::create(209, "cannot use offsets with object iterators"));

            case value_t::array:
            {
                std::advance(m_it.array_iterator, i);
                break;
            }

            default:
            {
                m_it.primitive_iterator += i;
                break;
            }
        }

        return *this;
    }

    /*!
    @brief  subtract from iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl& operator-=(difference_type i)
    {
        return operator+=(-i);
    }

    /*!
    @brief  add to iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl operator+(difference_type i) const
    {
        auto result = *this;
        result += i;
        return result;
    }

    /*!
    @brief  addition of distance and iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    friend iter_impl operator+(difference_type i, const iter_impl& it)
    {
        auto result = it;
        result += i;
        return result;
    }

    /*!
    @brief  subtract from iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    iter_impl operator-(difference_type i) const
    {
        auto result = *this;
        result -= i;
        return result;
    }

    /*!
    @brief  return difference
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    difference_type operator-(const iter_impl& other) const
    {
        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
                JSON_THROW(invalid_iterator::create(209, "cannot use offsets with object iterators"));

            case value_t::array:
                return m_it.array_iterator - other.m_it.array_iterator;

            default:
                return m_it.primitive_iterator - other.m_it.primitive_iterator;
        }
    }

    /*!
    @brief  access to successor
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    reference operator[](difference_type n) const
    {
        assert(m_object != nullptr);

        switch (m_object->m_type)
        {
            case value_t::object:
                JSON_THROW(invalid_iterator::create(208, "cannot use operator[] for object iterators"));

            case value_t::array:
                return *std::next(m_it.array_iterator, n);

            case value_t::null:
                JSON_THROW(invalid_iterator::create(214, "cannot get value"));

            default:
            {
                if (JSON_LIKELY(m_it.primitive_iterator.get_value() == -n))
                {
                    return *m_object;
                }

                JSON_THROW(invalid_iterator::create(214, "cannot get value"));
            }
        }
    }

    /*!
    @brief  return the key of an object iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    const typename object_t::key_type& key() const
    {
        assert(m_object != nullptr);

        if (JSON_LIKELY(m_object->is_object()))
        {
            return m_it.object_iterator->first;
        }

        JSON_THROW(invalid_iterator::create(207, "cannot use key() for non-object iterators"));
    }

    /*!
    @brief  return the value of an iterator
    @pre The iterator is initialized; i.e. `m_object != nullptr`.
    */
    reference value() const
    {
        return operator*();
    }

  private:
    /// associated JSON instance
    pointer m_object = nullptr;
    /// the actual iterator of the associated instance
    internal_iterator<typename std::remove_const<BasicJsonType>::type> m_it;
};
}
}

// #include <nlohmann/detail/iterators/iteration_proxy.hpp>

// #include <nlohmann/detail/iterators/json_reverse_iterator.hpp>


#include <cstddef> // ptrdiff_t
#include <iterator> // reverse_iterator
#include <utility> // declval

namespace nlohmann
{
namespace detail
{
//////////////////////
// reverse_iterator //
//////////////////////

/*!
@brief a template for a reverse iterator class

@tparam Base the base iterator type to reverse. Valid types are @ref
iterator (to create @ref reverse_iterator) and @ref const_iterator (to
create @ref const_reverse_iterator).

@requirement The class satisfies the following concept requirements:
-
[BidirectionalIterator](https://en.cppreference.com/w/cpp/named_req/BidirectionalIterator):
  The iterator that can be moved can be moved in both directions (i.e.
  incremented and decremented).
- [OutputIterator](https://en.cppreference.com/w/cpp/named_req/OutputIterator):
  It is possible to write to the pointed-to element (only if @a Base is
  @ref iterator).

@since version 1.0.0
*/
template<typename Base>
class json_reverse_iterator : public std::reverse_iterator<Base>
{
  public:
    using difference_type = std::ptrdiff_t;
    /// shortcut to the reverse iterator adapter
    using base_iterator = std::reverse_iterator<Base>;
    /// the reference type for the pointed-to element
    using reference = typename Base::reference;

    /// create reverse iterator from iterator
    explicit json_reverse_iterator(const typename base_iterator::iterator_type& it) noexcept
        : base_iterator(it) {}

    /// create reverse iterator from base class
    explicit json_reverse_iterator(const base_iterator& it) noexcept : base_iterator(it) {}

    /// post-increment (it++)
    json_reverse_iterator const operator++(int)
    {
        return static_cast<json_reverse_iterator>(base_iterator::operator++(1));
    }

    /// pre-increment (++it)
    json_reverse_iterator& operator++()
    {
        return static_cast<json_reverse_iterator&>(base_iterator::operator++());
    }

    /// post-decrement (it--)
    json_reverse_iterator const operator--(int)
    {
        return static_cast<json_reverse_iterator>(base_iterator::operator--(1));
    }

    /// pre-decrement (--it)
    json_reverse_iterator& operator--()
    {
        return static_cast<json_reverse_iterator&>(base_iterator::operator--());
    }

    /// add to iterator
    json_reverse_iterator& operator+=(difference_type i)
    {
        return static_cast<json_reverse_iterator&>(base_iterator::operator+=(i));
    }

    /// add to iterator
    json_reverse_iterator operator+(difference_type i) const
    {
        return static_cast<json_reverse_iterator>(base_iterator::operator+(i));
    }

    /// subtract from iterator
    json_reverse_iterator operator-(difference_type i) const
    {
        return static_cast<json_reverse_iterator>(base_iterator::operator-(i));
    }

    /// return difference
    difference_type operator-(const json_reverse_iterator& other) const
    {
        return base_iterator(*this) - base_iterator(other);
    }

    /// access to successor
    reference operator[](difference_type n) const
    {
        return *(this->operator+(n));
    }

    /// return the key of an object iterator
    auto key() const -> decltype(std::declval<Base>().key())
    {
        auto it = --this->base();
        return it.key();
    }

    /// return the value of an iterator
    reference value() const
    {
        auto it = --this->base();
        return it.operator * ();
    }
};
}
}

// #include <nlohmann/detail/output/output_adapters.hpp>


#include <algorithm> // copy
#include <cstddef> // size_t
#include <ios> // streamsize
#include <iterator> // back_inserter
#include <memory> // shared_ptr, make_shared
#include <ostream> // basic_ostream
#include <string> // basic_string
#include <vector> // vector

namespace nlohmann
{
namespace detail
{
/// abstract output adapter interface
template<typename CharType> struct output_adapter_protocol
{
    virtual void write_character(CharType c) = 0;
    virtual void write_characters(const CharType* s, std::size_t length) = 0;
    virtual ~output_adapter_protocol() = default;
};

/// a type to simplify interfaces
template<typename CharType>
using output_adapter_t = std::shared_ptr<output_adapter_protocol<CharType>>;

/// output adapter for byte vectors
template<typename CharType>
class output_vector_adapter : public output_adapter_protocol<CharType>
{
  public:
    explicit output_vector_adapter(std::vector<CharType>& vec) : v(vec) {}

    void write_character(CharType c) override
    {
        v.push_back(c);
    }

    void write_characters(const CharType* s, std::size_t length) override
    {
        std::copy(s, s + length, std::back_inserter(v));
    }

  private:
    std::vector<CharType>& v;
};

/// output adapter for output streams
template<typename CharType>
class output_stream_adapter : public output_adapter_protocol<CharType>
{
  public:
    explicit output_stream_adapter(std::basic_ostream<CharType>& s) : stream(s) {}

    void write_character(CharType c) override
    {
        stream.put(c);
    }

    void write_characters(const CharType* s, std::size_t length) override
    {
        stream.write(s, static_cast<std::streamsize>(length));
    }

  private:
    std::basic_ostream<CharType>& stream;
};

/// output adapter for basic_string
template<typename CharType, typename StringType = std::basic_string<CharType>>
class output_string_adapter : public output_adapter_protocol<CharType>
{
  public:
    explicit output_string_adapter(StringType& s) : str(s) {}

    void write_character(CharType c) override
    {
        str.push_back(c);
    }

    void write_characters(const CharType* s, std::size_t length) override
    {
        str.append(s, length);
    }

  private:
    StringType& str;
};

template<typename CharType, typename StringType = std::basic_string<CharType>>
class output_adapter
{
  public:
    output_adapter(std::vector<CharType>& vec)
        : oa(std::make_shared<output_vector_adapter<CharType>>(vec)) {}

    output_adapter(std::basic_ostream<CharType>& s)
        : oa(std::make_shared<output_stream_adapter<CharType>>(s)) {}

    output_adapter(StringType& s)
        : oa(std::make_shared<output_string_adapter<CharType, StringType>>(s)) {}

    operator output_adapter_t<CharType>()
    {
        return oa;
    }

  private:
    output_adapter_t<CharType> oa = nullptr;
};
}
}

// #include <nlohmann/detail/input/binary_reader.hpp>


#include <algorithm> // generate_n
#include <array> // array
#include <cassert> // assert
#include <cmath> // ldexp
#include <cstddef> // size_t
#include <cstdint> // uint8_t, uint16_t, uint32_t, uint64_t
#include <cstdio> // snprintf
#include <cstring> // memcpy
#include <iterator> // back_inserter
#include <limits> // numeric_limits
#include <string> // char_traits, string
#include <utility> // make_pair, move

// #include <nlohmann/detail/input/input_adapters.hpp>

// #include <nlohmann/detail/input/json_sax.hpp>

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/value_t.hpp>


namespace nlohmann
{
namespace detail
{
///////////////////
// binary reader //
///////////////////

/*!
@brief deserialization of CBOR, MessagePack, and UBJSON values
*/
template<typename BasicJsonType>
class binary_reader
{
    using number_integer_t = typename BasicJsonType::number_integer_t;
    using number_unsigned_t = typename BasicJsonType::number_unsigned_t;
    using number_float_t = typename BasicJsonType::number_float_t;
    using string_t = typename BasicJsonType::string_t;
    using json_sax_t = json_sax<BasicJsonType>;

  public:
    /*!
    @brief create a binary reader

    @param[in] adapter  input adapter to read from
    */
    explicit binary_reader(input_adapter_t adapter) : ia(std::move(adapter))
    {
        assert(ia);
    }

    /*!
    @param[in] format  the binary format to parse
    @param[in] sax_    a SAX event processor
    @param[in] strict  whether to expect the input to be consumed completed

    @return
    */
    bool sax_parse(const input_format_t format,
                   json_sax_t* sax_,
                   const bool strict = true)
    {
        sax = sax_;
        bool result;

        switch (format)
        {
            case input_format_t::cbor:
                result = parse_cbor_internal();
                break;

            case input_format_t::msgpack:
                result = parse_msgpack_internal();
                break;

            case input_format_t::ubjson:
                result = parse_ubjson_internal();
                break;

            default:
                assert(false);  // LCOV_EXCL_LINE
        }

        // strict mode: next byte must be EOF
        if (result and strict)
        {
            if (format == input_format_t::ubjson)
            {
                get_ignore_noop();
            }
            else
            {
                get();
            }

            if (JSON_UNLIKELY(current != std::char_traits<char>::eof()))
            {
                return sax->parse_error(chars_read, get_token_string(), parse_error::create(110, chars_read, "expected end of input"));
            }
        }

        return result;
    }

    /*!
    @brief determine system byte order

    @return true if and only if system's byte order is little endian

    @note from http://stackoverflow.com/a/1001328/266378
    */
    static constexpr bool little_endianess(int num = 1) noexcept
    {
        return (*reinterpret_cast<char*>(&num) == 1);
    }

  private:
    /*!
    @param[in] get_char  whether a new character should be retrieved from the
                         input (true, default) or whether the last read
                         character should be considered instead

    @return whether a valid CBOR value was passed to the SAX parser
    */
    bool parse_cbor_internal(const bool get_char = true)
    {
        switch (get_char ? get() : current)
        {
            // EOF
            case std::char_traits<char>::eof():
                return unexpect_eof();

            // Integer 0x00..0x17 (0..23)
            case 0x00:
            case 0x01:
            case 0x02:
            case 0x03:
            case 0x04:
            case 0x05:
            case 0x06:
            case 0x07:
            case 0x08:
            case 0x09:
            case 0x0A:
            case 0x0B:
            case 0x0C:
            case 0x0D:
            case 0x0E:
            case 0x0F:
            case 0x10:
            case 0x11:
            case 0x12:
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
            case 0x17:
                return sax->number_unsigned(static_cast<number_unsigned_t>(current));

            case 0x18: // Unsigned integer (one-byte uint8_t follows)
            {
                uint8_t number;
                return get_number(number) and sax->number_unsigned(number);
            }

            case 0x19: // Unsigned integer (two-byte uint16_t follows)
            {
                uint16_t number;
                return get_number(number) and sax->number_unsigned(number);
            }

            case 0x1A: // Unsigned integer (four-byte uint32_t follows)
            {
                uint32_t number;
                return get_number(number) and sax->number_unsigned(number);
            }

            case 0x1B: // Unsigned integer (eight-byte uint64_t follows)
            {
                uint64_t number;
                return get_number(number) and sax->number_unsigned(number);
            }

            // Negative integer -1-0x00..-1-0x17 (-1..-24)
            case 0x20:
            case 0x21:
            case 0x22:
            case 0x23:
            case 0x24:
            case 0x25:
            case 0x26:
            case 0x27:
            case 0x28:
            case 0x29:
            case 0x2A:
            case 0x2B:
            case 0x2C:
            case 0x2D:
            case 0x2E:
            case 0x2F:
            case 0x30:
            case 0x31:
            case 0x32:
            case 0x33:
            case 0x34:
            case 0x35:
            case 0x36:
            case 0x37:
                return sax->number_integer(static_cast<int8_t>(0x20 - 1 - current));

            case 0x38: // Negative integer (one-byte uint8_t follows)
            {
                uint8_t number;
                return get_number(number) and sax->number_integer(static_cast<number_integer_t>(-1) - number);
            }

            case 0x39: // Negative integer -1-n (two-byte uint16_t follows)
            {
                uint16_t number;
                return get_number(number) and sax->number_integer(static_cast<number_integer_t>(-1) - number);
            }

            case 0x3A: // Negative integer -1-n (four-byte uint32_t follows)
            {
                uint32_t number;
                return get_number(number) and sax->number_integer(static_cast<number_integer_t>(-1) - number);
            }

            case 0x3B: // Negative integer -1-n (eight-byte uint64_t follows)
            {
                uint64_t number;
                return get_number(number) and sax->number_integer(static_cast<number_integer_t>(-1)
                        - static_cast<number_integer_t>(number));
            }

            // UTF-8 string (0x00..0x17 bytes follow)
            case 0x60:
            case 0x61:
            case 0x62:
            case 0x63:
            case 0x64:
            case 0x65:
            case 0x66:
            case 0x67:
            case 0x68:
            case 0x69:
            case 0x6A:
            case 0x6B:
            case 0x6C:
            case 0x6D:
            case 0x6E:
            case 0x6F:
            case 0x70:
            case 0x71:
            case 0x72:
            case 0x73:
            case 0x74:
            case 0x75:
            case 0x76:
            case 0x77:
            case 0x78: // UTF-8 string (one-byte uint8_t for n follows)
            case 0x79: // UTF-8 string (two-byte uint16_t for n follow)
            case 0x7A: // UTF-8 string (four-byte uint32_t for n follow)
            case 0x7B: // UTF-8 string (eight-byte uint64_t for n follow)
            case 0x7F: // UTF-8 string (indefinite length)
            {
                string_t s;
                return get_cbor_string(s) and sax->string(s);
            }

            // array (0x00..0x17 data items follow)
            case 0x80:
            case 0x81:
            case 0x82:
            case 0x83:
            case 0x84:
            case 0x85:
            case 0x86:
            case 0x87:
            case 0x88:
            case 0x89:
            case 0x8A:
            case 0x8B:
            case 0x8C:
            case 0x8D:
            case 0x8E:
            case 0x8F:
            case 0x90:
            case 0x91:
            case 0x92:
            case 0x93:
            case 0x94:
            case 0x95:
            case 0x96:
            case 0x97:
                return get_cbor_array(static_cast<std::size_t>(current & 0x1F));

            case 0x98: // array (one-byte uint8_t for n follows)
            {
                uint8_t len;
                return get_number(len) and get_cbor_array(static_cast<std::size_t>(len));
            }

            case 0x99: // array (two-byte uint16_t for n follow)
            {
                uint16_t len;
                return get_number(len) and get_cbor_array(static_cast<std::size_t>(len));
            }

            case 0x9A: // array (four-byte uint32_t for n follow)
            {
                uint32_t len;
                return get_number(len) and get_cbor_array(static_cast<std::size_t>(len));
            }

            case 0x9B: // array (eight-byte uint64_t for n follow)
            {
                uint64_t len;
                return get_number(len) and get_cbor_array(static_cast<std::size_t>(len));
            }

            case 0x9F: // array (indefinite length)
                return get_cbor_array(json_sax_t::no_limit);

            // map (0x00..0x17 pairs of data items follow)
            case 0xA0:
            case 0xA1:
            case 0xA2:
            case 0xA3:
            case 0xA4:
            case 0xA5:
            case 0xA6:
            case 0xA7:
            case 0xA8:
            case 0xA9:
            case 0xAA:
            case 0xAB:
            case 0xAC:
            case 0xAD:
            case 0xAE:
            case 0xAF:
            case 0xB0:
            case 0xB1:
            case 0xB2:
            case 0xB3:
            case 0xB4:
            case 0xB5:
            case 0xB6:
            case 0xB7:
                return get_cbor_object(static_cast<std::size_t>(current & 0x1F));

            case 0xB8: // map (one-byte uint8_t for n follows)
            {
                uint8_t len;
                return get_number(len) and get_cbor_object(static_cast<std::size_t>(len));
            }

            case 0xB9: // map (two-byte uint16_t for n follow)
            {
                uint16_t len;
                return get_number(len) and get_cbor_object(static_cast<std::size_t>(len));
            }

            case 0xBA: // map (four-byte uint32_t for n follow)
            {
                uint32_t len;
                return get_number(len) and get_cbor_object(static_cast<std::size_t>(len));
            }

            case 0xBB: // map (eight-byte uint64_t for n follow)
            {
                uint64_t len;
                return get_number(len) and get_cbor_object(static_cast<std::size_t>(len));
            }

            case 0xBF: // map (indefinite length)
                return get_cbor_object(json_sax_t::no_limit);

            case 0xF4: // false
                return sax->boolean(false);

            case 0xF5: // true
                return sax->boolean(true);

            case 0xF6: // null
                return sax->null();

            case 0xF9: // Half-Precision Float (two-byte IEEE 754)
            {
                const int byte1 = get();
                if (JSON_UNLIKELY(not unexpect_eof()))
                {
                    return false;
                }
                const int byte2 = get();
                if (JSON_UNLIKELY(not unexpect_eof()))
                {
                    return false;
                }

                // code from RFC 7049, Appendix D, Figure 3:
                // As half-precision floating-point numbers were only added
                // to IEEE 754 in 2008, today's programming platforms often
                // still only have limited support for them. It is very
                // easy to include at least decoding support for them even
                // without such support. An example of a small decoder for
                // half-precision floating-point numbers in the C language
                // is shown in Fig. 3.
                const int half = (byte1 << 8) + byte2;
                const double val = [&half]
                {
                    const int exp = (half >> 10) & 0x1F;
                    const int mant = half & 0x3FF;
                    assert(0 <= exp and exp <= 32);
                    assert(0 <= mant and mant <= 1024);
                    switch (exp)
                    {
                        case 0:
                            return std::ldexp(mant, -24);
                        case 31:
                            return (mant == 0)
                            ? std::numeric_limits<double>::infinity()
                            : std::numeric_limits<double>::quiet_NaN();
                        default:
                            return std::ldexp(mant + 1024, exp - 25);
                    }
                }();
                return sax->number_float((half & 0x8000) != 0
                                         ? static_cast<number_float_t>(-val)
                                         : static_cast<number_float_t>(val), "");
            }

            case 0xFA: // Single-Precision Float (four-byte IEEE 754)
            {
                float number;
                return get_number(number) and sax->number_float(static_cast<number_float_t>(number), "");
            }

            case 0xFB: // Double-Precision Float (eight-byte IEEE 754)
            {
                double number;
                return get_number(number) and sax->number_float(static_cast<number_float_t>(number), "");
            }

            default: // anything else (0xFF is handled inside the other types)
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read, "error reading CBOR; last byte: 0x" + last_token));
            }
        }
    }

    /*!
    @return whether a valid MessagePack value was passed to the SAX parser
    */
    bool parse_msgpack_internal()
    {
        switch (get())
        {
            // EOF
            case std::char_traits<char>::eof():
                return unexpect_eof();

            // positive fixint
            case 0x00:
            case 0x01:
            case 0x02:
            case 0x03:
            case 0x04:
            case 0x05:
            case 0x06:
            case 0x07:
            case 0x08:
            case 0x09:
            case 0x0A:
            case 0x0B:
            case 0x0C:
            case 0x0D:
            case 0x0E:
            case 0x0F:
            case 0x10:
            case 0x11:
            case 0x12:
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
            case 0x17:
            case 0x18:
            case 0x19:
            case 0x1A:
            case 0x1B:
            case 0x1C:
            case 0x1D:
            case 0x1E:
            case 0x1F:
            case 0x20:
            case 0x21:
            case 0x22:
            case 0x23:
            case 0x24:
            case 0x25:
            case 0x26:
            case 0x27:
            case 0x28:
            case 0x29:
            case 0x2A:
            case 0x2B:
            case 0x2C:
            case 0x2D:
            case 0x2E:
            case 0x2F:
            case 0x30:
            case 0x31:
            case 0x32:
            case 0x33:
            case 0x34:
            case 0x35:
            case 0x36:
            case 0x37:
            case 0x38:
            case 0x39:
            case 0x3A:
            case 0x3B:
            case 0x3C:
            case 0x3D:
            case 0x3E:
            case 0x3F:
            case 0x40:
            case 0x41:
            case 0x42:
            case 0x43:
            case 0x44:
            case 0x45:
            case 0x46:
            case 0x47:
            case 0x48:
            case 0x49:
            case 0x4A:
            case 0x4B:
            case 0x4C:
            case 0x4D:
            case 0x4E:
            case 0x4F:
            case 0x50:
            case 0x51:
            case 0x52:
            case 0x53:
            case 0x54:
            case 0x55:
            case 0x56:
            case 0x57:
            case 0x58:
            case 0x59:
            case 0x5A:
            case 0x5B:
            case 0x5C:
            case 0x5D:
            case 0x5E:
            case 0x5F:
            case 0x60:
            case 0x61:
            case 0x62:
            case 0x63:
            case 0x64:
            case 0x65:
            case 0x66:
            case 0x67:
            case 0x68:
            case 0x69:
            case 0x6A:
            case 0x6B:
            case 0x6C:
            case 0x6D:
            case 0x6E:
            case 0x6F:
            case 0x70:
            case 0x71:
            case 0x72:
            case 0x73:
            case 0x74:
            case 0x75:
            case 0x76:
            case 0x77:
            case 0x78:
            case 0x79:
            case 0x7A:
            case 0x7B:
            case 0x7C:
            case 0x7D:
            case 0x7E:
            case 0x7F:
                return sax->number_unsigned(static_cast<number_unsigned_t>(current));

            // fixmap
            case 0x80:
            case 0x81:
            case 0x82:
            case 0x83:
            case 0x84:
            case 0x85:
            case 0x86:
            case 0x87:
            case 0x88:
            case 0x89:
            case 0x8A:
            case 0x8B:
            case 0x8C:
            case 0x8D:
            case 0x8E:
            case 0x8F:
                return get_msgpack_object(static_cast<std::size_t>(current & 0x0F));

            // fixarray
            case 0x90:
            case 0x91:
            case 0x92:
            case 0x93:
            case 0x94:
            case 0x95:
            case 0x96:
            case 0x97:
            case 0x98:
            case 0x99:
            case 0x9A:
            case 0x9B:
            case 0x9C:
            case 0x9D:
            case 0x9E:
            case 0x9F:
                return get_msgpack_array(static_cast<std::size_t>(current & 0x0F));

            // fixstr
            case 0xA0:
            case 0xA1:
            case 0xA2:
            case 0xA3:
            case 0xA4:
            case 0xA5:
            case 0xA6:
            case 0xA7:
            case 0xA8:
            case 0xA9:
            case 0xAA:
            case 0xAB:
            case 0xAC:
            case 0xAD:
            case 0xAE:
            case 0xAF:
            case 0xB0:
            case 0xB1:
            case 0xB2:
            case 0xB3:
            case 0xB4:
            case 0xB5:
            case 0xB6:
            case 0xB7:
            case 0xB8:
            case 0xB9:
            case 0xBA:
            case 0xBB:
            case 0xBC:
            case 0xBD:
            case 0xBE:
            case 0xBF:
            {
                string_t s;
                return get_msgpack_string(s) and sax->string(s);
            }

            case 0xC0: // nil
                return sax->null();

            case 0xC2: // false
                return sax->boolean(false);

            case 0xC3: // true
                return sax->boolean(true);

            case 0xCA: // float 32
            {
                float number;
                return get_number(number) and sax->number_float(static_cast<number_float_t>(number), "");
            }

            case 0xCB: // float 64
            {
                double number;
                return get_number(number) and sax->number_float(static_cast<number_float_t>(number), "");
            }

            case 0xCC: // uint 8
            {
                uint8_t number;
                return get_number(number) and sax->number_unsigned(number);
            }

            case 0xCD: // uint 16
            {
                uint16_t number;
                return get_number(number) and sax->number_unsigned(number);
            }

            case 0xCE: // uint 32
            {
                uint32_t number;
                return get_number(number) and sax->number_unsigned(number);
            }

            case 0xCF: // uint 64
            {
                uint64_t number;
                return get_number(number) and sax->number_unsigned(number);
            }

            case 0xD0: // int 8
            {
                int8_t number;
                return get_number(number) and sax->number_integer(number);
            }

            case 0xD1: // int 16
            {
                int16_t number;
                return get_number(number) and sax->number_integer(number);
            }

            case 0xD2: // int 32
            {
                int32_t number;
                return get_number(number) and sax->number_integer(number);
            }

            case 0xD3: // int 64
            {
                int64_t number;
                return get_number(number) and sax->number_integer(number);
            }

            case 0xD9: // str 8
            case 0xDA: // str 16
            case 0xDB: // str 32
            {
                string_t s;
                return get_msgpack_string(s) and sax->string(s);
            }

            case 0xDC: // array 16
            {
                uint16_t len;
                return get_number(len) and get_msgpack_array(static_cast<std::size_t>(len));
            }

            case 0xDD: // array 32
            {
                uint32_t len;
                return get_number(len) and get_msgpack_array(static_cast<std::size_t>(len));
            }

            case 0xDE: // map 16
            {
                uint16_t len;
                return get_number(len) and get_msgpack_object(static_cast<std::size_t>(len));
            }

            case 0xDF: // map 32
            {
                uint32_t len;
                return get_number(len) and get_msgpack_object(static_cast<std::size_t>(len));
            }

            // negative fixint
            case 0xE0:
            case 0xE1:
            case 0xE2:
            case 0xE3:
            case 0xE4:
            case 0xE5:
            case 0xE6:
            case 0xE7:
            case 0xE8:
            case 0xE9:
            case 0xEA:
            case 0xEB:
            case 0xEC:
            case 0xED:
            case 0xEE:
            case 0xEF:
            case 0xF0:
            case 0xF1:
            case 0xF2:
            case 0xF3:
            case 0xF4:
            case 0xF5:
            case 0xF6:
            case 0xF7:
            case 0xF8:
            case 0xF9:
            case 0xFA:
            case 0xFB:
            case 0xFC:
            case 0xFD:
            case 0xFE:
            case 0xFF:
                return sax->number_integer(static_cast<int8_t>(current));

            default: // anything else
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read, "error reading MessagePack; last byte: 0x" + last_token));
            }
        }
    }

    /*!
    @param[in] get_char  whether a new character should be retrieved from the
                         input (true, default) or whether the last read
                         character should be considered instead

    @return whether a valid UBJSON value was passed to the SAX parser
    */
    bool parse_ubjson_internal(const bool get_char = true)
    {
        return get_ubjson_value(get_char ? get_ignore_noop() : current);
    }

    /*!
    @brief get next character from the input

    This function provides the interface to the used input adapter. It does
    not throw in case the input reached EOF, but returns a -'ve valued
    `std::char_traits<char>::eof()` in that case.

    @return character read from the input
    */
    int get()
    {
        ++chars_read;
        return (current = ia->get_character());
    }

    /*!
    @return character read from the input after ignoring all 'N' entries
    */
    int get_ignore_noop()
    {
        do
        {
            get();
        }
        while (current == 'N');

        return current;
    }

    /*
    @brief read a number from the input

    @tparam NumberType the type of the number
    @param[out] result  number of type @a NumberType

    @return whether conversion completed

    @note This function needs to respect the system's endianess, because
          bytes in CBOR, MessagePack, and UBJSON are stored in network order
          (big endian) and therefore need reordering on little endian systems.
    */
    template<typename NumberType>
    bool get_number(NumberType& result)
    {
        // step 1: read input into array with system's byte order
        std::array<uint8_t, sizeof(NumberType)> vec;
        for (std::size_t i = 0; i < sizeof(NumberType); ++i)
        {
            get();
            if (JSON_UNLIKELY(not unexpect_eof()))
            {
                return false;
            }

            // reverse byte order prior to conversion if necessary
            if (is_little_endian)
            {
                vec[sizeof(NumberType) - i - 1] = static_cast<uint8_t>(current);
            }
            else
            {
                vec[i] = static_cast<uint8_t>(current); // LCOV_EXCL_LINE
            }
        }

        // step 2: convert array into number of type T and return
        std::memcpy(&result, vec.data(), sizeof(NumberType));
        return true;
    }

    /*!
    @brief create a string by reading characters from the input

    @tparam NumberType the type of the number
    @param[in] len number of characters to read
    @param[out] string created by reading @a len bytes

    @return whether string creation completed

    @note We can not reserve @a len bytes for the result, because @a len
          may be too large. Usually, @ref unexpect_eof() detects the end of
          the input before we run out of string memory.
    */
    template<typename NumberType>
    bool get_string(const NumberType len, string_t& result)
    {
        bool success = true;
        std::generate_n(std::back_inserter(result), len, [this, &success]()
        {
            get();
            if (JSON_UNLIKELY(not unexpect_eof()))
            {
                success = false;
            }
            return static_cast<char>(current);
        });
        return success;
    }

    /*!
    @brief reads a CBOR string

    This function first reads starting bytes to determine the expected
    string length and then copies this number of bytes into a string.
    Additionally, CBOR's strings with indefinite lengths are supported.

    @param[out] result  created string

    @return whether string creation completed
    */
    bool get_cbor_string(string_t& result)
    {
        if (JSON_UNLIKELY(not unexpect_eof()))
        {
            return false;
        }

        switch (current)
        {
            // UTF-8 string (0x00..0x17 bytes follow)
            case 0x60:
            case 0x61:
            case 0x62:
            case 0x63:
            case 0x64:
            case 0x65:
            case 0x66:
            case 0x67:
            case 0x68:
            case 0x69:
            case 0x6A:
            case 0x6B:
            case 0x6C:
            case 0x6D:
            case 0x6E:
            case 0x6F:
            case 0x70:
            case 0x71:
            case 0x72:
            case 0x73:
            case 0x74:
            case 0x75:
            case 0x76:
            case 0x77:
            {
                return get_string(current & 0x1F, result);
            }

            case 0x78: // UTF-8 string (one-byte uint8_t for n follows)
            {
                uint8_t len;
                return get_number(len) and get_string(len, result);
            }

            case 0x79: // UTF-8 string (two-byte uint16_t for n follow)
            {
                uint16_t len;
                return get_number(len) and get_string(len, result);
            }

            case 0x7A: // UTF-8 string (four-byte uint32_t for n follow)
            {
                uint32_t len;
                return get_number(len) and get_string(len, result);
            }

            case 0x7B: // UTF-8 string (eight-byte uint64_t for n follow)
            {
                uint64_t len;
                return get_number(len) and get_string(len, result);
            }

            case 0x7F: // UTF-8 string (indefinite length)
            {
                while (get() != 0xFF)
                {
                    string_t chunk;
                    if (not get_cbor_string(chunk))
                    {
                        return false;
                    }
                    result.append(chunk);
                }
                return true;
            }

            default:
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, "expected a CBOR string; last byte: 0x" + last_token));
            }
        }
    }

    /*!
    @param[in] len  the length of the array or json_sax_t::no_limit for an
                    array of indefinite size
    @return whether array creation completed
    */
    bool get_cbor_array(const std::size_t len)
    {
        if (JSON_UNLIKELY(not sax->start_array(len)))
        {
            return false;
        }

        if (len != json_sax_t::no_limit)
            for (std::size_t i = 0; i < len; ++i)
            {
                if (JSON_UNLIKELY(not parse_cbor_internal()))
                {
                    return false;
                }
            }
        else
        {
            while (get() != 0xFF)
            {
                if (JSON_UNLIKELY(not parse_cbor_internal(false)))
                {
                    return false;
                }
            }
        }

        return sax->end_array();
    }

    /*!
    @param[in] len  the length of the object or json_sax_t::no_limit for an
                    object of indefinite size
    @return whether object creation completed
    */
    bool get_cbor_object(const std::size_t len)
    {
        if (not JSON_UNLIKELY(sax->start_object(len)))
        {
            return false;
        }

        string_t key;
        if (len != json_sax_t::no_limit)
        {
            for (std::size_t i = 0; i < len; ++i)
            {
                get();
                if (JSON_UNLIKELY(not get_cbor_string(key) or not sax->key(key)))
                {
                    return false;
                }

                if (JSON_UNLIKELY(not parse_cbor_internal()))
                {
                    return false;
                }
                key.clear();
            }
        }
        else
        {
            while (get() != 0xFF)
            {
                if (JSON_UNLIKELY(not get_cbor_string(key) or not sax->key(key)))
                {
                    return false;
                }

                if (JSON_UNLIKELY(not parse_cbor_internal()))
                {
                    return false;
                }
                key.clear();
            }
        }

        return sax->end_object();
    }

    /*!
    @brief reads a MessagePack string

    This function first reads starting bytes to determine the expected
    string length and then copies this number of bytes into a string.

    @param[out] result  created string

    @return whether string creation completed
    */
    bool get_msgpack_string(string_t& result)
    {
        if (JSON_UNLIKELY(not unexpect_eof()))
        {
            return false;
        }

        switch (current)
        {
            // fixstr
            case 0xA0:
            case 0xA1:
            case 0xA2:
            case 0xA3:
            case 0xA4:
            case 0xA5:
            case 0xA6:
            case 0xA7:
            case 0xA8:
            case 0xA9:
            case 0xAA:
            case 0xAB:
            case 0xAC:
            case 0xAD:
            case 0xAE:
            case 0xAF:
            case 0xB0:
            case 0xB1:
            case 0xB2:
            case 0xB3:
            case 0xB4:
            case 0xB5:
            case 0xB6:
            case 0xB7:
            case 0xB8:
            case 0xB9:
            case 0xBA:
            case 0xBB:
            case 0xBC:
            case 0xBD:
            case 0xBE:
            case 0xBF:
            {
                return get_string(current & 0x1F, result);
            }

            case 0xD9: // str 8
            {
                uint8_t len;
                return get_number(len) and get_string(len, result);
            }

            case 0xDA: // str 16
            {
                uint16_t len;
                return get_number(len) and get_string(len, result);
            }

            case 0xDB: // str 32
            {
                uint32_t len;
                return get_number(len) and get_string(len, result);
            }

            default:
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, "expected a MessagePack string; last byte: 0x" + last_token));
            }
        }
    }

    /*!
    @param[in] len  the length of the array
    @return whether array creation completed
    */
    bool get_msgpack_array(const std::size_t len)
    {
        if (JSON_UNLIKELY(not sax->start_array(len)))
        {
            return false;
        }

        for (std::size_t i = 0; i < len; ++i)
        {
            if (JSON_UNLIKELY(not parse_msgpack_internal()))
            {
                return false;
            }
        }

        return sax->end_array();
    }

    /*!
    @param[in] len  the length of the object
    @return whether object creation completed
    */
    bool get_msgpack_object(const std::size_t len)
    {
        if (JSON_UNLIKELY(not sax->start_object(len)))
        {
            return false;
        }

        string_t key;
        for (std::size_t i = 0; i < len; ++i)
        {
            get();
            if (JSON_UNLIKELY(not get_msgpack_string(key) or not sax->key(key)))
            {
                return false;
            }

            if (JSON_UNLIKELY(not parse_msgpack_internal()))
            {
                return false;
            }
            key.clear();
        }

        return sax->end_object();
    }

    /*!
    @brief reads a UBJSON string

    This function is either called after reading the 'S' byte explicitly
    indicating a string, or in case of an object key where the 'S' byte can be
    left out.

    @param[out] result   created string
    @param[in] get_char  whether a new character should be retrieved from the
                         input (true, default) or whether the last read
                         character should be considered instead

    @return whether string creation completed
    */
    bool get_ubjson_string(string_t& result, const bool get_char = true)
    {
        if (get_char)
        {
            get();  // TODO: may we ignore N here?
        }

        if (JSON_UNLIKELY(not unexpect_eof()))
        {
            return false;
        }

        switch (current)
        {
            case 'U':
            {
                uint8_t len;
                return get_number(len) and get_string(len, result);
            }

            case 'i':
            {
                int8_t len;
                return get_number(len) and get_string(len, result);
            }

            case 'I':
            {
                int16_t len;
                return get_number(len) and get_string(len, result);
            }

            case 'l':
            {
                int32_t len;
                return get_number(len) and get_string(len, result);
            }

            case 'L':
            {
                int64_t len;
                return get_number(len) and get_string(len, result);
            }

            default:
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, "expected a UBJSON string; last byte: 0x" + last_token));
        }
    }

    /*!
    @param[out] result  determined size
    @return whether size determination completed
    */
    bool get_ubjson_size_value(std::size_t& result)
    {
        switch (get_ignore_noop())
        {
            case 'U':
            {
                uint8_t number;
                if (JSON_UNLIKELY(not get_number(number)))
                {
                    return false;
                }
                result = static_cast<std::size_t>(number);
                return true;
            }

            case 'i':
            {
                int8_t number;
                if (JSON_UNLIKELY(not get_number(number)))
                {
                    return false;
                }
                result = static_cast<std::size_t>(number);
                return true;
            }

            case 'I':
            {
                int16_t number;
                if (JSON_UNLIKELY(not get_number(number)))
                {
                    return false;
                }
                result = static_cast<std::size_t>(number);
                return true;
            }

            case 'l':
            {
                int32_t number;
                if (JSON_UNLIKELY(not get_number(number)))
                {
                    return false;
                }
                result = static_cast<std::size_t>(number);
                return true;
            }

            case 'L':
            {
                int64_t number;
                if (JSON_UNLIKELY(not get_number(number)))
                {
                    return false;
                }
                result = static_cast<std::size_t>(number);
                return true;
            }

            default:
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, "byte after '#' must denote a number type; last byte: 0x" + last_token));
            }
        }
    }

    /*!
    @brief determine the type and size for a container

    In the optimized UBJSON format, a type and a size can be provided to allow
    for a more compact representation.

    @param[out] result  pair of the size and the type

    @return whether pair creation completed
    */
    bool get_ubjson_size_type(std::pair<std::size_t, int>& result)
    {
        result.first = string_t::npos; // size
        result.second = 0; // type

        get_ignore_noop();

        if (current == '$')
        {
            result.second = get();  // must not ignore 'N', because 'N' maybe the type
            if (JSON_UNLIKELY(not unexpect_eof()))
            {
                return false;
            }

            get_ignore_noop();
            if (JSON_UNLIKELY(current != '#'))
            {
                if (JSON_UNLIKELY(not unexpect_eof()))
                {
                    return false;
                }
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read, "expected '#' after UBJSON type information; last byte: 0x" + last_token));
            }

            return get_ubjson_size_value(result.first);
        }
        else if (current == '#')
        {
            return get_ubjson_size_value(result.first);
        }
        return true;
    }

    /*!
    @param prefix  the previously read or set type prefix
    @return whether value creation completed
    */
    bool get_ubjson_value(const int prefix)
    {
        switch (prefix)
        {
            case std::char_traits<char>::eof():  // EOF
                return unexpect_eof();

            case 'T':  // true
                return sax->boolean(true);
            case 'F':  // false
                return sax->boolean(false);

            case 'Z':  // null
                return sax->null();

            case 'U':
            {
                uint8_t number;
                return get_number(number) and sax->number_unsigned(number);
            }

            case 'i':
            {
                int8_t number;
                return get_number(number) and sax->number_integer(number);
            }

            case 'I':
            {
                int16_t number;
                return get_number(number) and sax->number_integer(number);
            }

            case 'l':
            {
                int32_t number;
                return get_number(number) and sax->number_integer(number);
            }

            case 'L':
            {
                int64_t number;
                return get_number(number) and sax->number_integer(number);
            }

            case 'd':
            {
                float number;
                return get_number(number) and sax->number_float(static_cast<number_float_t>(number), "");
            }

            case 'D':
            {
                double number;
                return get_number(number) and sax->number_float(static_cast<number_float_t>(number), "");
            }

            case 'C':  // char
            {
                get();
                if (JSON_UNLIKELY(not unexpect_eof()))
                {
                    return false;
                }
                if (JSON_UNLIKELY(current > 127))
                {
                    auto last_token = get_token_string();
                    return sax->parse_error(chars_read, last_token, parse_error::create(113, chars_read, "byte after 'C' must be in range 0x00..0x7F; last byte: 0x" + last_token));
                }
                string_t s(1, static_cast<char>(current));
                return sax->string(s);
            }

            case 'S':  // string
            {
                string_t s;
                return get_ubjson_string(s) and sax->string(s);
            }

            case '[':  // array
                return get_ubjson_array();

            case '{':  // object
                return get_ubjson_object();

            default: // anything else
            {
                auto last_token = get_token_string();
                return sax->parse_error(chars_read, last_token, parse_error::create(112, chars_read, "error reading UBJSON; last byte: 0x" + last_token));
            }
        }
    }

    /*!
    @return whether array creation completed
    */
    bool get_ubjson_array()
    {
        std::pair<std::size_t, int> size_and_type;
        if (JSON_UNLIKELY(not get_ubjson_size_type(size_and_type)))
        {
            return false;
        }

        if (size_and_type.first != string_t::npos)
        {
            if (JSON_UNLIKELY(not sax->start_array(size_and_type.first)))
            {
                return false;
            }

            if (size_and_type.second != 0)
            {
                if (size_and_type.second != 'N')
                {
                    for (std::size_t i = 0; i < size_and_type.first; ++i)
                    {
                        if (JSON_UNLIKELY(not get_ubjson_value(size_and_type.second)))
                        {
                            return false;
                        }
                    }
                }
            }
            else
            {
                for (std::size_t i = 0; i < size_and_type.first; ++i)
                {
                    if (JSON_UNLIKELY(not parse_ubjson_internal()))
                    {
                        return false;
                    }
                }
            }
        }
        else
        {
            if (JSON_UNLIKELY(not sax->start_array()))
            {
                return false;
            }

            while (current != ']')
            {
                if (JSON_UNLIKELY(not parse_ubjson_internal(false)))
                {
                    return false;
                }
                get_ignore_noop();
            }
        }

        return sax->end_array();
    }

    /*!
    @return whether object creation completed
    */
    bool get_ubjson_object()
    {
        std::pair<std::size_t, int> size_and_type;
        if (JSON_UNLIKELY(not get_ubjson_size_type(size_and_type)))
        {
            return false;
        }

        string_t key;
        if (size_and_type.first != string_t::npos)
        {
            if (JSON_UNLIKELY(not sax->start_object(size_and_type.first)))
            {
                return false;
            }

            if (size_and_type.second != 0)
            {
                for (std::size_t i = 0; i < size_and_type.first; ++i)
                {
                    if (JSON_UNLIKELY(not get_ubjson_string(key) or not sax->key(key)))
                    {
                        return false;
                    }
                    if (JSON_UNLIKELY(not get_ubjson_value(size_and_type.second)))
                    {
                        return false;
                    }
                    key.clear();
                }
            }
            else
            {
                for (std::size_t i = 0; i < size_and_type.first; ++i)
                {
                    if (JSON_UNLIKELY(not get_ubjson_string(key) or not sax->key(key)))
                    {
                        return false;
                    }
                    if (JSON_UNLIKELY(not parse_ubjson_internal()))
                    {
                        return false;
                    }
                    key.clear();
                }
            }
        }
        else
        {
            if (JSON_UNLIKELY(not sax->start_object()))
            {
                return false;
            }

            while (current != '}')
            {
                if (JSON_UNLIKELY(not get_ubjson_string(key, false) or not sax->key(key)))
                {
                    return false;
                }
                if (JSON_UNLIKELY(not parse_ubjson_internal()))
                {
                    return false;
                }
                get_ignore_noop();
                key.clear();
            }
        }

        return sax->end_object();
    }

    /*!
    @return whether the last read character is not EOF
    */
    bool unexpect_eof() const
    {
        if (JSON_UNLIKELY(current == std::char_traits<char>::eof()))
        {
            return sax->parse_error(chars_read, "<end of file>", parse_error::create(110, chars_read, "unexpected end of input"));
        }
        return true;
    }

    /*!
    @return a string representation of the last read byte
    */
    std::string get_token_string() const
    {
        char cr[3];
        snprintf(cr, 3, "%.2hhX", static_cast<unsigned char>(current));
        return std::string{cr};
    }

  private:
    /// input adapter
    input_adapter_t ia = nullptr;

    /// the current character
    int current = std::char_traits<char>::eof();

    /// the number of characters read
    std::size_t chars_read = 0;

    /// whether we can assume little endianess
    const bool is_little_endian = little_endianess();

    /// the SAX parser
    json_sax_t* sax = nullptr;
};
}
}

// #include <nlohmann/detail/json_ref.hpp>


#include <initializer_list>
#include <utility>

namespace nlohmann
{
namespace detail
{
template<typename BasicJsonType>
class json_ref
{
  public:
    using value_type = BasicJsonType;

    json_ref(value_type&& value)
        : owned_value(std::move(value)), value_ref(&owned_value), is_rvalue(true)
    {}

    json_ref(const value_type& value)
        : value_ref(const_cast<value_type*>(&value)), is_rvalue(false)
    {}

    json_ref(std::initializer_list<json_ref> init)
        : owned_value(init), value_ref(&owned_value), is_rvalue(true)
    {}

    template<class... Args>
    json_ref(Args&& ... args)
        : owned_value(std::forward<Args>(args)...), value_ref(&owned_value), is_rvalue(true)
    {}

    // class should be movable only
    json_ref(json_ref&&) = default;
    json_ref(const json_ref&) = delete;
    json_ref& operator=(const json_ref&) = delete;

    value_type moved_or_copied() const
    {
        if (is_rvalue)
        {
            return std::move(*value_ref);
        }
        return *value_ref;
    }

    value_type const& operator*() const
    {
        return *static_cast<value_type const*>(value_ref);
    }

    value_type const* operator->() const
    {
        return static_cast<value_type const*>(value_ref);
    }

  private:
    mutable value_type owned_value = nullptr;
    value_type* value_ref = nullptr;
    const bool is_rvalue;
};
}
}

// #include <nlohmann/detail/json_pointer.hpp>


#include <cassert> // assert
#include <numeric> // accumulate
#include <string> // string
#include <vector> // vector

// #include <nlohmann/detail/macro_scope.hpp>

// #include <nlohmann/detail/exceptions.hpp>

// #include <nlohmann/detail/value_t.hpp>


namespace nlohmann
{
template<typename BasicJsonType>
class json_pointer
{
    // allow basic_json to access private members
    NLOHMANN_BASIC_JSON_TPL_DECLARATION
    friend class basic_json;

  public:
    /*!
    @brief create JSON pointer

    Create a JSON pointer according to the syntax described in
    [Section 3 of RFC6901](https://tools.ietf.org/html/rfc6901#section-3).

    @param[in] s  string representing the JSON pointer; if omitted, the empty
                  string is assumed which references the whole JSON value

    @throw parse_error.107 if the given JSON pointer @a s is nonempty and does
                           not begin with a slash (`/`); see example below

    @throw parse_error.108 if a tilde (`~`) in the given JSON pointer @a s is
    not followed by `0` (representing `~`) or `1` (representing `/`); see
    example below

    @liveexample{The example shows the construction several valid JSON pointers
    as well as the exceptional behavior.,json_pointer}

    @since version 2.0.0
    */
    explicit json_pointer(const std::string& s = "")
        : reference_tokens(split(s))
    {}

    /*!
    @brief return a string representation of the JSON pointer

    @invariant For each JSON pointer `ptr`, it holds:
    @code {.cpp}
    ptr == json_pointer(ptr.to_string());
    @endcode

    @return a string representation of the JSON pointer

    @liveexample{The example shows the result of `to_string`.,
    json_pointer__to_string}

    @since version 2.0.0
    */
    std::string to_string() const noexcept
    {
        return std::accumulate(reference_tokens.begin(), reference_tokens.end(),
                               std::string{},
                               [](const std::string & a, const std::string & b)
        {
            return a + "/" + escape(b);
        });
    }

    /// @copydoc to_string()
    operator std::string() const
    {
        return to_string();
    }

    /*!
    @param[in] s  reference token to be converted into an array index

    @return integer representation of @a s

    @throw out_of_range.404 if string @a s could not be converted to an integer
    */
    static int array_index(const std::string& s)
    {
        std::size_t processed_chars = 0;
        const int res = std::stoi(s, &processed_chars);

        // check if the string was completely read
        if (JSON_UNLIKELY(processed_chars != s.size()))
        {
            JSON_THROW(detail::out_of_range::create(404, "unresolved reference token '" + s + "'"));
        }

        return res;
    }

  private:
    /*!
    @brief remove and return last reference pointer
    @throw out_of_range.405 if JSON pointer has no parent
    */
    std::string pop_back()
    {
        if (JSON_UNLIKELY(is_root()))
        {
            JSON_THROW(detail::out_of_range::create(405, "JSON pointer has no parent"));
        }

        auto last = reference_tokens.back();
        reference_tokens.pop_back();
        return last;
    }

    /// return whether pointer points to the root document
    bool is_root() const
    {
        return reference_tokens.empty();
    }

    json_pointer top() const
    {
        if (JSON_UNLIKELY(is_root()))
        {
            JSON_THROW(detail::out_of_range::create(405, "JSON pointer has no parent"));
        }

        json_pointer result = *this;
        result.reference_tokens = {reference_tokens[0]};
        return result;
    }

    /*!
    @brief create and return a reference to the pointed to value

    @complexity Linear in the number of reference tokens.

    @throw parse_error.109 if array index is not a number
    @throw type_error.313 if value cannot be unflattened
    */
    BasicJsonType& get_and_create(BasicJsonType& j) const
    {
        using size_type = typename BasicJsonType::size_type;
        auto result = &j;

        // in case no reference tokens exist, return a reference to the JSON value
        // j which will be overwritten by a primitive value
        for (const auto& reference_token : reference_tokens)
        {
            switch (result->m_type)
            {
                case detail::value_t::null:
                {
                    if (reference_token == "0")
                    {
                        // start a new array if reference token is 0
                        result = &result->operator[](0);
                    }
                    else
                    {
                        // start a new object otherwise
                        result = &result->operator[](reference_token);
                    }
                    break;
                }

                case detail::value_t::object:
                {
                    // create an entry in the object
                    result = &result->operator[](reference_token);
                    break;
                }

                case detail::value_t::array:
                {
                    // create an entry in the array
                    JSON_TRY
                    {
                        result = &result->operator[](static_cast<size_type>(array_index(reference_token)));
                    }
                    JSON_CATCH(std::invalid_argument&)
                    {
                        JSON_THROW(detail::parse_error::create(109, 0, "array index '" + reference_token + "' is not a number"));
                    }
                    break;
                }

                /*
                The following code is only reached if there exists a reference
                token _and_ the current value is primitive. In this case, we have
                an error situation, because primitive values may only occur as
                single value; that is, with an empty list of reference tokens.
                */
                default:
                    JSON_THROW(detail::type_error::create(313, "invalid value to unflatten"));
            }
        }

        return *result;
    }

    /*!
    @brief return a reference to the pointed to value

    @note This version does not throw if a value is not present, but tries to
          create nested values instead. For instance, calling this function
          with pointer `"/this/that"` on a null value is equivalent to calling
          `operator[]("this").operator[]("that")` on that value, effectively
          changing the null value to an object.

    @param[in] ptr  a JSON value

    @return reference to the JSON value pointed to by the JSON pointer

    @complexity Linear in the length of the JSON pointer.

    @throw parse_error.106   if an array index begins with '0'
    @throw parse_error.109   if an array index was not a number
    @throw out_of_range.404  if the JSON pointer can not be resolved
    */
    BasicJsonType& get_unchecked(BasicJsonType* ptr) const
    {
        using size_type = typename BasicJsonType::size_type;
        for (const auto& reference_token : reference_tokens)
        {
            // convert null values to arrays or objects before continuing
            if (ptr->m_type == detail::value_t::null)
            {
                // check if reference token is a number
                const bool nums =
                    std::all_of(reference_token.begin(), reference_token.end(),
                                [](const char x)
                {
                    return (x >= '0' and x <= '9');
                });

                // change value to array for numbers or "-" or to object otherwise
                *ptr = (nums or reference_token == "-")
                       ? detail::value_t::array
                       : detail::value_t::object;
            }

            switch (ptr->m_type)
            {
                case detail::value_t::object:
                {
                    // use unchecked object access
                    ptr = &ptr->operator[](reference_token);
                    break;
                }

                case detail::value_t::array:
                {
                    // error condition (cf. RFC 6901, Sect. 4)
                    if (JSON_UNLIKELY(reference_token.size() > 1 and reference_token[0] == '0'))
                    {
                        JSON_THROW(detail::parse_error::create(106, 0,
                                                               "array index '" + reference_token +
                                                               "' must not begin with '0'"));
                    }

                    if (reference_token == "-")
                    {
                        // explicitly treat "-" as index beyond the end
                        ptr = &ptr->operator[](ptr->m_value.array->size());
                    }
                    else
                    {
                        // convert array index to number; unchecked access
                        JSON_TRY
                        {
                            ptr = &ptr->operator[](
                                static_cast<size_type>(array_index(reference_token)));
                        }
                        JSON_CATCH(std::invalid_argument&)
                        {
                            JSON_THROW(detail::parse_error::create(109, 0, "array index '" + reference_token + "' is not a number"));
                        }
                    }
                    break;
                }

                default:
                    JSON_THROW(detail::out_of_range::create(404, "unresolved reference token '" + reference_token + "'"));
            }
        }

        return *ptr;
    }

    /*!
    @throw parse_error.106   if an array index begins with '0'
    @throw parse_error.109   if an array index was not a number
    @throw out_of_range.402  if the array index '-' is used
    @throw out_of_range.404  if the JSON pointer can not be resolved
    */
    BasicJsonType& get_checked(BasicJsonType* ptr) const
    {
        using size_type = typename BasicJsonType::size_type;
        for (const auto& reference_token : reference_tokens)
        {
            switch (ptr->m_type)
            {
                case detail::value_t::object:
                {
                    // note: at performs range check
                    ptr = &ptr->at(reference_token);
                    break;
                }

                case detail::value_t::array:
                {
                    if (JSON_UNLIKELY(reference_token == "-"))
                    {
                        // "-" always fails the range check
                        JSON_THROW(detail::out_of_range::create(402,
                                                                "array index '-' (" + std::to_string(ptr->m_value.array->size()) +
                                                                ") is out of range"));
                    }

                    // error condition (cf. RFC 6901, Sect. 4)
                    if (JSON_UNLIKELY(reference_token.size() > 1 and reference_token[0] == '0'))
                    {
                        JSON_THROW(detail::parse_error::create(106, 0,
                                                               "array index '" + reference_token +
                                                               "' must not begin with '0'"));
                    }

                    // note: at performs range check
                    JSON_TRY
                    {
                        ptr = &ptr->at(static_cast<size_type>(array_index(reference_token)));
                    }
                    JSON_CATCH(std::invalid_argument&)
                    {
                        JSON_THROW(detail::parse_error::create(109, 0, "array index '" + reference_token + "' is not a number"));
                    }
                    break;
                }

                default:
                    JSON_THROW(detail::out_of_range::create(404, "unresolved reference token '" + reference_token + "'"));
            }
        }

        return *ptr;
    }

    /*!
    @brief return a const reference to the pointed to value

    @param[in] ptr  a JSON value

    @return const reference to the JSON value pointed to by the JSON
    pointer

    @throw parse_error.106   if an array index begins with '0'
    @throw parse_error.109   if an array index was not a number
    @throw out_of_range.402  if the array index '-' is used
    @throw out_of_range.404  if the JSON pointer can not be resolved
    */
    const BasicJsonType& get_unchecked(const BasicJsonType* ptr) const
    {
        using size_type = typename BasicJsonType::size_type;
        for (const auto& reference_token : reference_tokens)
        {
            switch (ptr->m_type)
            {
                case detail::value_t::object:
                {
                    // use unchecked object access
                    ptr = &ptr->operator[](reference_token);
                    break;
                }

                case detail::value_t::array:
                {
                    if (JSON_UNLIKELY(reference_token == "-"))
                    {
                        // "-" cannot be used for const access
                        JSON_THROW(detail::out_of_range::create(402,
                                                                "array index '-' (" + std::to_string(ptr->m_value.array->size()) +
                                                                ") is out of range"));
                    }

                    // error condition (cf. RFC 6901, Sect. 4)
                    if (JSON_UNLIKELY(reference_token.size() > 1 and reference_token[0] == '0'))
                    {
                        JSON_THROW(detail::parse_error::create(106, 0,
                                                               "array index '" + reference_token +
                                                               "' must not begin with '0'"));
                    }

                    // use unchecked array access
                    JSON_TRY
                    {
                        ptr = &ptr->operator[](
                            static_cast<size_type>(array_index(reference_token)));
                    }
                    JSON_CATCH(std::invalid_argument&)
                    {
                        JSON_THROW(detail::parse_error::create(109, 0, "array index '" + reference_token + "' is not a number"));
                    }
                    break;
                }

                default:
                    JSON_THROW(detail::out_of_range::create(404, "unresolved reference token '" + reference_token + "'"));
            }
        }

        return *ptr;
    }

    /*!
    @throw parse_error.106   if an array index begins with '0'
    @throw parse_error.109   if an array index was not a number
    @throw out_of_range.402  if the array index '-' is used
    @throw out_of_range.404  if the JSON pointer can not be resolved
    */
    const BasicJsonType& get_checked(const BasicJsonType* ptr) const
    {
        using size_type = typename BasicJsonType::size_type;
        for (const auto& reference_token : reference_tokens)
        {
            switch (ptr->m_type)
            {
                case detail::value_t::object:
                {
                    // note: at performs range check
                    ptr = &ptr->at(reference_token);
                    break;
                }

                case detail::value_t::array:
                {
                    if (JSON_UNLIKELY(reference_token == "-"))
                    {
                        // "-" always fails the range check
                        JSON_THROW(detail::out_of_range::create(402,
                                                                "array index '-' (" + std::to_string(ptr->m_value.array->size()) +
                                                                ") is out of range"));
                    }

                    // error condition (cf. RFC 6901, Sect. 4)
                    if (JSON_UNLIKELY(reference_token.size() > 1 and reference_token[0] == '0'))
                    {
                        JSON_THROW(detail::parse_error::create(106, 0,
                                                               "array index '" + reference_token +
                                                               "' must not begin with '0'"));
                    }

                    // note: at performs range check
                    JSON_TRY
                    {
                        ptr = &ptr->at(static_cast<size_type>(array_index(reference_token)));
                    }
                    JSON_CATCH(std::invalid_argument&)
                    {
                        JSON_THROW(detail::parse_error::create(109, 0, "array index '" + reference_token + "' is not a number"));
                    }
                    break;
                }

                default:
                    JSON_THROW(detail::out_of_range::create(404, "unresolved reference token '" + reference_token + "'"));
            }
        }

        return *ptr;
    }

    /*!
    @brief split the string input to reference tokens

    @note This function is only called by the json_pointer constructor.
          All exceptions below are documented there.

    @throw parse_error.107  if the pointer is not empty or begins with '/'
    @throw parse_error.108  if character '~' is not followed by '0' or '1'
    */
    static std::vector<std::string> split(const std::string& reference_string)
    {
        std::vector<std::string> result;

        // special case: empty reference string -> no reference tokens
        if (reference_string.empty())
        {
            return result;
        }

        // check if nonempty reference string begins with slash
        if (JSON_UNLIKELY(reference_string[0] != '/'))
        {
            JSON_THROW(detail::parse_error::create(107, 1,
                                                   "JSON pointer must be empty or begin with '/' - was: '" +
                                                   reference_string + "'"));
        }

        // extract the reference tokens:
        // - slash: position of the last read slash (or end of string)
        // - start: position after the previous slash
        for (
            // search for the first slash after the first character
            std::size_t slash = reference_string.find_first_of('/', 1),
            // set the beginning of the first reference token
            start = 1;
            // we can stop if start == string::npos+1 = 0
            start != 0;
            // set the beginning of the next reference token
            // (will eventually be 0 if slash == std::string::npos)
            start = slash + 1,
            // find next slash
            slash = reference_string.find_first_of('/', start))
        {
            // use the text between the beginning of the reference token
            // (start) and the last slash (slash).
            auto reference_token = reference_string.substr(start, slash - start);

            // check reference tokens are properly escaped
            for (std::size_t pos = reference_token.find_first_of('~');
                    pos != std::string::npos;
                    pos = reference_token.find_first_of('~', pos + 1))
            {
                assert(reference_token[pos] == '~');

                // ~ must be followed by 0 or 1
                if (JSON_UNLIKELY(pos == reference_token.size() - 1 or
                                  (reference_token[pos + 1] != '0' and
                                   reference_token[pos + 1] != '1')))
                {
                    JSON_THROW(detail::parse_error::create(108, 0, "escape character '~' must be followed with '0' or '1'"));
                }
            }

            // finally, store the reference token
            unescape(reference_token);
            result.push_back(reference_token);
        }

        return result;
    }

    /*!
    @brief replace all occurrences of a substring by another string

    @param[in,out] s  the string to manipulate; changed so that all
                   occurrences of @a f are replaced with @a t
    @param[in]     f  the substring to replace with @a t
    @param[in]     t  the string to replace @a f

    @pre The search string @a f must not be empty. **This precondition is
    enforced with an assertion.**

    @since version 2.0.0
    */
    static void replace_substring(std::string& s, const std::string& f,
                                  const std::string& t)
    {
        assert(not f.empty());
        for (auto pos = s.find(f);                // find first occurrence of f
                pos != std::string::npos;         // make sure f was found
                s.replace(pos, f.size(), t),      // replace with t, and
                pos = s.find(f, pos + t.size()))  // find next occurrence of f
        {}
    }

    /// escape "~"" to "~0" and "/" to "~1"
    static std::string escape(std::string s)
    {
        replace_substring(s, "~", "~0");
        replace_substring(s, "/", "~1");
        return s;
    }

    /// unescape "~1" to tilde and "~0" to slash (order is important!)
    static void unescape(std::string& s)
    {
        replace_substring(s, "~1", "/");
        replace_substring(s, "~0", "~");
    }

    /*!
    @param[in] reference_string  the reference string to the current value
    @param[in] value             the value to consider
    @param[in,out] result        the result object to insert values to

    @note Empty objects or arrays are flattened to `null`.
    */
    static void flatten(const std::string& reference_string,
                        const BasicJsonType& value,
                        BasicJsonType& result)
    {
        switch (value.m_type)
        {
            case detail::value_t::array:
            {
                if (value.m_value.array->empty())
                {
                    // flatten empty array as null
                    result[reference_string] = nullptr;
                }
                else
                {
                    // iterate array and use index as reference string
                    for (std::size_t i = 0; i < value.m_value.array->size(); ++i)
                    {
                        flatten(reference_string + "/" + std::to_string(i),
                                value.m_value.array->operator[](i), result);
                    }
                }
                break;
            }

            case detail::value_t::object:
            {
                if (value.m_value.object->empty())
                {
                    // flatten empty object as null
                    result[reference_string] = nullptr;
                }
                else
                {
                    // iterate object and use keys as reference string
                    for (const auto& element : *value.m_value.object)
                    {
                        flatten(reference_string + "/" + escape(element.first), element.second, result);
                    }
                }
                break;
            }

            default:
            {
                // add primitive value with its reference string
                result[reference_string] = value;
                break;
            }
        }
    }

    /*!
    @param[in] value  flattened JSON

    @return unflattened JSON

    @throw parse_error.109 if array index is not a number
    @throw type_error.314  if value is not an object
    @throw type_error.315  if object values are not primitive
    @throw type_error.313  if value cannot be unflattened
    */
    static BasicJsonType
    unflatten(const BasicJsonType& value)
    {
        if (JSON_UNLIKELY(not value.is_object()))
        {
            JSON_THROW(detail::type_error::create(314, "only objects can be unflattened"));
        }

        BasicJsonType result;

        // iterate the JSON object values
        for (const auto& element : *value.m_value.object)
        {
            if (JSON_UNLIKELY(not element.second.is_primitive()))
            {
                JSON_THROW(detail::type_error::create(315, "values in object must be primitive"));
            }

            // assign value to reference pointed to by JSON pointer; Note that if
            // the JSON pointer is "" (i.e., points to the whole value), function
            // get_and_create returns a reference to result itself. An assignment
            // will then create a primitive value.
            json_pointer(element.first).get_and_create(result) = element.second;
        }

        return result;
    }

    friend bool operator==(json_pointer const& lhs,
                           json_pointer const& rhs) noexcept
    {
        return (lhs.reference_tokens == rhs.reference_tokens);
    }

    friend bool operator!=(json_pointer const& lhs,
                           json_pointer const& rhs) noexcept
    {
        return not (lhs == rhs);
    }

    /// the reference tokens
    std::vector<std::string> reference_tokens;
};
}

// #include <nlohmann/adl_serializer.hpp>


#include <utility>

// #include <nlohmann/detail/conversions/from_json.hpp>

// #include <nlohmann/detail/conversions/to_json.hpp>


namespace nlohmann
{
template<typename, typename>
struct adl_serializer
{
    /*!
    @brief convert a JSON value to any value type

    This function is usually called by the `get()` function of the
    @ref basic_json class (either explicit or via conversion operators).

    @param[in] j         JSON value to read from
    @param[in,out] val  value to write to
    */
    template<typename BasicJsonType, typename ValueType>
    static void from_json(BasicJsonType&& j, ValueType& val) noexcept(
        noexcept(::nlohmann::from_json(std::forward<BasicJsonType>(j), val)))
    {
        ::nlohmann::from_json(std::forward<BasicJsonType>(j), val);
    }

    /*!
    @brief convert any value type to a JSON value

    This function is usually called by the constructors of the @ref basic_json
    class.

    @param[in,out] j  JSON value to write to
    @param[in] val     value to read from
    */
    template<typename BasicJsonType, typename ValueType>
    static void to_json(BasicJsonType& j, ValueType&& val) noexcept(
        noexcept(::nlohmann::to_json(j, std::forward<ValueType>(val))))
    {
        ::nlohmann::to_json(j, std::forward<ValueType>(val));
    }
};
}


/*!
@brief namespace for Niels Lohmann
@see https://github.com/nlohmann
@since version 1.0.0
*/
namespace nlohmann
{

/*!
@brief a class to store JSON values

@tparam ObjectType type for JSON objects (`std::map` by default; will be used
in @ref object_t)
@tparam ArrayType type for JSON arrays (`std::vector` by default; will be used
in @ref array_t)
@tparam StringType type for JSON strings and object keys (`std::string` by
default; will be used in @ref string_t)
@tparam BooleanType type for JSON booleans (`bool` by default; will be used
in @ref boolean_t)
@tparam NumberIntegerType type for JSON integer numbers (`int64_t` by
default; will be used in @ref number_integer_t)
@tparam NumberUnsignedType type for JSON unsigned integer numbers (@c
`uint64_t` by default; will be used in @ref number_unsigned_t)
@tparam NumberFloatType type for JSON floating-point numbers (`double` by
default; will be used in @ref number_float_t)
@tparam AllocatorType type of the allocator to use (`std::allocator` by
default)
@tparam JSONSerializer the serializer to resolve internal calls to `to_json()`
and `from_json()` (@ref adl_serializer by default)

@requirement The class satisfies the following concept requirements:
- Basic
 - [DefaultConstructible](https://en.cppreference.com/w/cpp/named_req/DefaultConstructible):
   JSON values can be default constructed. The result will be a JSON null
   value.
 - [MoveConstructible](https://en.cppreference.com/w/cpp/named_req/MoveConstructible):
   A JSON value can be constructed from an rvalue argument.
 - [CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible):
   A JSON value can be copy-constructed from an lvalue expression.
 - [MoveAssignable](https://en.cppreference.com/w/cpp/named_req/MoveAssignable):
   A JSON value van be assigned from an rvalue argument.
 - [CopyAssignable](https://en.cppreference.com/w/cpp/named_req/CopyAssignable):
   A JSON value can be copy-assigned from an lvalue expression.
 - [Destructible](https://en.cppreference.com/w/cpp/named_req/Destructible):
   JSON values can be destructed.
- Layout
 - [StandardLayoutType](https://en.cppreference.com/w/cpp/named_req/StandardLayoutType):
   JSON values have
   [standard layout](https://en.cppreference.com/w/cpp/language/data_members#Standard_layout):
   All non-static data members are private and standard layout types, the
   class has no virtual functions or (virtual) base classes.
- Library-wide
 - [EqualityComparable](https://en.cppreference.com/w/cpp/named_req/EqualityComparable):
   JSON values can be compared with `==`, see @ref
   operator==(const_reference,const_reference).
 - [LessThanComparable](https://en.cppreference.com/w/cpp/named_req/LessThanComparable):
   JSON values can be compared with `<`, see @ref
   operator<(const_reference,const_reference).
 - [Swappable](https://en.cppreference.com/w/cpp/named_req/Swappable):
   Any JSON lvalue or rvalue of can be swapped with any lvalue or rvalue of
   other compatible types, using unqualified function call @ref swap().
 - [NullablePointer](https://en.cppreference.com/w/cpp/named_req/NullablePointer):
   JSON values can be compared against `std::nullptr_t` objects which are used
   to model the `null` value.
- Container
 - [Container](https://en.cppreference.com/w/cpp/named_req/Container):
   JSON values can be used like STL containers and provide iterator access.
 - [ReversibleContainer](https://en.cppreference.com/w/cpp/named_req/ReversibleContainer);
   JSON values can be used like STL containers and provide reverse iterator
   access.

@invariant The member variables @a m_value and @a m_type have the following
relationship:
- If `m_type == value_t::object`, then `m_value.object != nullptr`.
- If `m_type == value_t::array`, then `m_value.array != nullptr`.
- If `m_type == value_t::string`, then `m_value.string != nullptr`.
The invariants are checked by member function assert_invariant().

@internal
@note ObjectType trick from http://stackoverflow.com/a/9860911
@endinternal

@see [RFC 7159: The JavaScript Object Notation (JSON) Data Interchange
Format](http://rfc7159.net/rfc7159)

@since version 1.0.0

@nosubgrouping
*/
NLOHMANN_BASIC_JSON_TPL_DECLARATION
class basic_json
{
  private:
    template<detail::value_t> friend struct detail::external_constructor;
    friend ::nlohmann::json_pointer<basic_json>;
    friend ::nlohmann::detail::parser<basic_json>;
    template<typename BasicJsonType>
    friend class ::nlohmann::detail::iter_impl;
    template<typename BasicJsonType>
    friend class ::nlohmann::detail::binary_reader;
    template<typename BasicJsonType>
    friend class ::nlohmann::detail::json_sax_dom_parser;
    template<typename BasicJsonType>
    friend class ::nlohmann::detail::json_sax_dom_callback_parser;

    /// workaround type for MSVC
    using basic_json_t = NLOHMANN_BASIC_JSON_TPL;

    // convenience aliases for types residing in namespace detail;
    using lexer = ::nlohmann::detail::lexer<basic_json>;
    using parser = ::nlohmann::detail::parser<basic_json>;

    using primitive_iterator_t = ::nlohmann::detail::primitive_iterator_t;
    template<typename BasicJsonType>
    using internal_iterator = ::nlohmann::detail::internal_iterator<BasicJsonType>;
    template<typename BasicJsonType>
    using iter_impl = ::nlohmann::detail::iter_impl<BasicJsonType>;
    template<typename Iterator>
    using iteration_proxy = ::nlohmann::detail::iteration_proxy<Iterator>;
    template<typename Base> using json_reverse_iterator = ::nlohmann::detail::json_reverse_iterator<Base>;

    template<typename CharType>
    using output_adapter_t = ::nlohmann::detail::output_adapter_t<CharType>;

    using binary_reader = ::nlohmann::detail::binary_reader<basic_json>;

  public:
    using value_t = detail::value_t;
    /// @copydoc nlohmann::json_pointer
    using json_pointer = ::nlohmann::json_pointer<basic_json>;
    template<typename T, typename SFINAE>
    using json_serializer = JSONSerializer<T, SFINAE>;
    /// helper type for initializer lists of basic_json values
    using initializer_list_t = std::initializer_list<detail::json_ref<basic_json>>;

    using input_format_t = detail::input_format_t;

    ////////////////
    // exceptions //
    ////////////////

    /// @name exceptions
    /// Classes to implement user-defined exceptions.
    /// @{

    /// @copydoc detail::exception
    using exception = detail::exception;
    /// @copydoc detail::parse_error
    using parse_error = detail::parse_error;
    /// @copydoc detail::invalid_iterator
    using invalid_iterator = detail::invalid_iterator;
    /// @copydoc detail::type_error
    using type_error = detail::type_error;
    /// @copydoc detail::out_of_range
    using out_of_range = detail::out_of_range;
    /// @copydoc detail::other_error
    using other_error = detail::other_error;

    /// @}


    /////////////////////
    // container types //
    /////////////////////

    /// @name container types
    /// The canonic container types to use @ref basic_json like any other STL
    /// container.
    /// @{

    /// the type of elements in a basic_json container
    using value_type = basic_json;

    /// the type of an element reference
    using reference = value_type&;
    /// the type of an element const reference
    using const_reference = const value_type&;

    /// a type to represent differences between iterators
    using difference_type = std::ptrdiff_t;
    /// a type to represent container sizes
    using size_type = std::size_t;

    /// the allocator type
    using allocator_type = AllocatorType<basic_json>;

    /// the type of an element pointer
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    /// the type of an element const pointer
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;

    /// an iterator for a basic_json container
    using iterator = iter_impl<basic_json>;
    /// a const iterator for a basic_json container
    using const_iterator = iter_impl<const basic_json>;
    /// a reverse iterator for a basic_json container
    using reverse_iterator = json_reverse_iterator<typename basic_json::iterator>;
    /// a const reverse iterator for a basic_json container
    using const_reverse_iterator = json_reverse_iterator<typename basic_json::const_iterator>;

    /// @}


    /*!
    @brief returns the allocator associated with the container
    */
    static allocator_type get_allocator()
    {
        return allocator_type();
    }

    /*!
    @brief returns version information on the library

    This function returns a JSON object with information about the library,
    including the version number and information on the platform and compiler.

    @return JSON object holding version information
    key         | description
    ----------- | ---------------
    `compiler`  | Information on the used compiler. It is an object with the following keys: `c++` (the used C++ standard), `family` (the compiler family; possible values are `clang`, `icc`, `gcc`, `ilecpp`, `msvc`, `pgcpp`, `sunpro`, and `unknown`), and `version` (the compiler version).
    `copyright` | The copyright line for the library as string.
    `name`      | The name of the library as string.
    `platform`  | The used platform as string. Possible values are `win32`, `linux`, `apple`, `unix`, and `unknown`.
    `url`       | The URL of the project as string.
    `version`   | The version of the library. It is an object with the following keys: `major`, `minor`, and `patch` as defined by [Semantic Versioning](http://semver.org), and `string` (the version string).

    @liveexample{The following code shows an example output of the `meta()`
    function.,meta}

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes to any JSON value.

    @complexity Constant.

    @since 2.1.0
    */
    static basic_json meta()
    {
        basic_json result;

        result["copyright"] = "(C) 2013-2017 Niels Lohmann";
        result["name"] = "JSON for Modern C++";
        result["url"] = "https://github.com/nlohmann/json";
        result["version"]["string"] =
            std::to_string(NLOHMANN_JSON_VERSION_MAJOR) + "." +
            std::to_string(NLOHMANN_JSON_VERSION_MINOR) + "." +
            std::to_string(NLOHMANN_JSON_VERSION_PATCH);
        result["version"]["major"] = NLOHMANN_JSON_VERSION_MAJOR;
        result["version"]["minor"] = NLOHMANN_JSON_VERSION_MINOR;
        result["version"]["patch"] = NLOHMANN_JSON_VERSION_PATCH;

#ifdef _WIN32
        result["platform"] = "win32";
#elif defined __linux__
        result["platform"] = "linux";
#elif defined __APPLE__
        result["platform"] = "apple";
#elif defined __unix__
        result["platform"] = "unix";
#else
        result["platform"] = "unknown";
#endif

#if defined(__ICC) || defined(__INTEL_COMPILER)
        result["compiler"] = {{"family", "icc"}, {"version", __INTEL_COMPILER}};
#elif defined(__clang__)
        result["compiler"] = {{"family", "clang"}, {"version", __clang_version__}};
#elif defined(__GNUC__) || defined(__GNUG__)
        result["compiler"] = {{"family", "gcc"}, {"version", std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__)}};
#elif defined(__HP_cc) || defined(__HP_aCC)
        result["compiler"] = "hp"
#elif defined(__IBMCPP__)
        result["compiler"] = {{"family", "ilecpp"}, {"version", __IBMCPP__}};
#elif defined(_MSC_VER)
        result["compiler"] = {{"family", "msvc"}, {"version", _MSC_VER}};
#elif defined(__PGI)
        result["compiler"] = {{"family", "pgcpp"}, {"version", __PGI}};
#elif defined(__SUNPRO_CC)
        result["compiler"] = {{"family", "sunpro"}, {"version", __SUNPRO_CC}};
#else
        result["compiler"] = {{"family", "unknown"}, {"version", "unknown"}};
#endif

#ifdef __cplusplus
        result["compiler"]["c++"] = std::to_string(__cplusplus);
#else
        result["compiler"]["c++"] = "unknown";
#endif
        return result;
    }


    ///////////////////////////
    // JSON value data types //
    ///////////////////////////

    /// @name JSON value data types
    /// The data types to store a JSON value. These types are derived from
    /// the template arguments passed to class @ref basic_json.
    /// @{

#if defined(JSON_HAS_CPP_14)
    // Use transparent comparator if possible, combined with perfect forwarding
    // on find() and count() calls prevents unnecessary string construction.
    using object_comparator_t = std::less<>;
#else
    using object_comparator_t = std::less<StringType>;
#endif

    /*!
    @brief a type for an object

    [RFC 7159](http://rfc7159.net/rfc7159) describes JSON objects as follows:
    > An object is an unordered collection of zero or more name/value pairs,
    > where a name is a string and a value is a string, number, boolean, null,
    > object, or array.

    To store objects in C++, a type is defined by the template parameters
    described below.

    @tparam ObjectType  the container to store objects (e.g., `std::map` or
    `std::unordered_map`)
    @tparam StringType the type of the keys or names (e.g., `std::string`).
    The comparison function `std::less<StringType>` is used to order elements
    inside the container.
    @tparam AllocatorType the allocator to use for objects (e.g.,
    `std::allocator`)

    #### Default type

    With the default values for @a ObjectType (`std::map`), @a StringType
    (`std::string`), and @a AllocatorType (`std::allocator`), the default
    value for @a object_t is:

    @code {.cpp}
    std::map<
      std::string, // key_type
      basic_json, // value_type
      std::less<std::string>, // key_compare
      std::allocator<std::pair<const std::string, basic_json>> // allocator_type
    >
    @endcode

    #### Behavior

    The choice of @a object_t influences the behavior of the JSON class. With
    the default type, objects have the following behavior:

    - When all names are unique, objects will be interoperable in the sense
      that all software implementations receiving that object will agree on
      the name-value mappings.
    - When the names within an object are not unique, it is unspecified which
      one of the values for a given key will be chosen. For instance,
      `{"key": 2, "key": 1}` could be equal to either `{"key": 1}` or
      `{"key": 2}`.
    - Internally, name/value pairs are stored in lexicographical order of the
      names. Objects will also be serialized (see @ref dump) in this order.
      For instance, `{"b": 1, "a": 2}` and `{"a": 2, "b": 1}` will be stored
      and serialized as `{"a": 2, "b": 1}`.
    - When comparing objects, the order of the name/value pairs is irrelevant.
      This makes objects interoperable in the sense that they will not be
      affected by these differences. For instance, `{"b": 1, "a": 2}` and
      `{"a": 2, "b": 1}` will be treated as equal.

    #### Limits

    [RFC 7159](http://rfc7159.net/rfc7159) specifies:
    > An implementation may set limits on the maximum depth of nesting.

    In this class, the object's limit of nesting is not explicitly constrained.
    However, a maximum depth of nesting may be introduced by the compiler or
    runtime environment. A theoretical limit can be queried by calling the
    @ref max_size function of a JSON object.

    #### Storage

    Objects are stored as pointers in a @ref basic_json type. That is, for any
    access to object values, a pointer of type `object_t*` must be
    dereferenced.

    @sa @ref array_t -- type for an array value

    @since version 1.0.0

    @note The order name/value pairs are added to the object is *not*
    preserved by the library. Therefore, iterating an object may return
    name/value pairs in a different order than they were originally stored. In
    fact, keys will be traversed in alphabetical order as `std::map` with
    `std::less` is used by default. Please note this behavior conforms to [RFC
    7159](http://rfc7159.net/rfc7159), because any order implements the
    specified "unordered" nature of JSON objects.
    */
    using object_t = ObjectType<StringType,
          basic_json,
          object_comparator_t,
          AllocatorType<std::pair<const StringType,
          basic_json>>>;

    /*!
    @brief a type for an array

    [RFC 7159](http://rfc7159.net/rfc7159) describes JSON arrays as follows:
    > An array is an ordered sequence of zero or more values.

    To store objects in C++, a type is defined by the template parameters
    explained below.

    @tparam ArrayType  container type to store arrays (e.g., `std::vector` or
    `std::list`)
    @tparam AllocatorType allocator to use for arrays (e.g., `std::allocator`)

    #### Default type

    With the default values for @a ArrayType (`std::vector`) and @a
    AllocatorType (`std::allocator`), the default value for @a array_t is:

    @code {.cpp}
    std::vector<
      basic_json, // value_type
      std::allocator<basic_json> // allocator_type
    >
    @endcode

    #### Limits

    [RFC 7159](http://rfc7159.net/rfc7159) specifies:
    > An implementation may set limits on the maximum depth of nesting.

    In this class, the array's limit of nesting is not explicitly constrained.
    However, a maximum depth of nesting may be introduced by the compiler or
    runtime environment. A theoretical limit can be queried by calling the
    @ref max_size function of a JSON array.

    #### Storage

    Arrays are stored as pointers in a @ref basic_json type. That is, for any
    access to array values, a pointer of type `array_t*` must be dereferenced.

    @sa @ref object_t -- type for an object value

    @since version 1.0.0
    */
    using array_t = ArrayType<basic_json, AllocatorType<basic_json>>;

    /*!
    @brief a type for a string

    [RFC 7159](http://rfc7159.net/rfc7159) describes JSON strings as follows:
    > A string is a sequence of zero or more Unicode characters.

    To store objects in C++, a type is defined by the template parameter
    described below. Unicode values are split by the JSON class into
    byte-sized characters during deserialization.

    @tparam StringType  the container to store strings (e.g., `std::string`).
    Note this container is used for keys/names in objects, see @ref object_t.

    #### Default type

    With the default values for @a StringType (`std::string`), the default
    value for @a string_t is:

    @code {.cpp}
    std::string
    @endcode

    #### Encoding

    Strings are stored in UTF-8 encoding. Therefore, functions like
    `std::string::size()` or `std::string::length()` return the number of
    bytes in the string rather than the number of characters or glyphs.

    #### String comparison

    [RFC 7159](http://rfc7159.net/rfc7159) states:
    > Software implementations are typically required to test names of object
    > members for equality. Implementations that transform the textual
    > representation into sequences of Unicode code units and then perform the
    > comparison numerically, code unit by code unit, are interoperable in the
    > sense that implementations will agree in all cases on equality or
    > inequality of two strings. For example, implementations that compare
    > strings with escaped characters unconverted may incorrectly find that
    > `"a\\b"` and `"a\u005Cb"` are not equal.

    This implementation is interoperable as it does compare strings code unit
    by code unit.

    #### Storage

    String values are stored as pointers in a @ref basic_json type. That is,
    for any access to string values, a pointer of type `string_t*` must be
    dereferenced.

    @since version 1.0.0
    */
    using string_t = StringType;

    /*!
    @brief a type for a boolean

    [RFC 7159](http://rfc7159.net/rfc7159) implicitly describes a boolean as a
    type which differentiates the two literals `true` and `false`.

    To store objects in C++, a type is defined by the template parameter @a
    BooleanType which chooses the type to use.

    #### Default type

    With the default values for @a BooleanType (`bool`), the default value for
    @a boolean_t is:

    @code {.cpp}
    bool
    @endcode

    #### Storage

    Boolean values are stored directly inside a @ref basic_json type.

    @since version 1.0.0
    */
    using boolean_t = BooleanType;

    /*!
    @brief a type for a number (integer)

    [RFC 7159](http://rfc7159.net/rfc7159) describes numbers as follows:
    > The representation of numbers is similar to that used in most
    > programming languages. A number is represented in base 10 using decimal
    > digits. It contains an integer component that may be prefixed with an
    > optional minus sign, which may be followed by a fraction part and/or an
    > exponent part. Leading zeros are not allowed. (...) Numeric values that
    > cannot be represented in the grammar below (such as Infinity and NaN)
    > are not permitted.

    This description includes both integer and floating-point numbers.
    However, C++ allows more precise storage if it is known whether the number
    is a signed integer, an unsigned integer or a floating-point number.
    Therefore, three different types, @ref number_integer_t, @ref
    number_unsigned_t and @ref number_float_t are used.

    To store integer numbers in C++, a type is defined by the template
    parameter @a NumberIntegerType which chooses the type to use.

    #### Default type

    With the default values for @a NumberIntegerType (`int64_t`), the default
    value for @a number_integer_t is:

    @code {.cpp}
    int64_t
    @endcode

    #### Default behavior

    - The restrictions about leading zeros is not enforced in C++. Instead,
      leading zeros in integer literals lead to an interpretation as octal
      number. Internally, the value will be stored as decimal number. For
      instance, the C++ integer literal `010` will be serialized to `8`.
      During deserialization, leading zeros yield an error.
    - Not-a-number (NaN) values will be serialized to `null`.

    #### Limits

    [RFC 7159](http://rfc7159.net/rfc7159) specifies:
    > An implementation may set limits on the range and precision of numbers.

    When the default type is used, the maximal integer number that can be
    stored is `9223372036854775807` (INT64_MAX) and the minimal integer number
    that can be stored is `-9223372036854775808` (INT64_MIN). Integer numbers
    that are out of range will yield over/underflow when used in a
    constructor. During deserialization, too large or small integer numbers
    will be automatically be stored as @ref number_unsigned_t or @ref
    number_float_t.

    [RFC 7159](http://rfc7159.net/rfc7159) further states:
    > Note that when such software is used, numbers that are integers and are
    > in the range \f$[-2^{53}+1, 2^{53}-1]\f$ are interoperable in the sense
    > that implementations will agree exactly on their numeric values.

    As this range is a subrange of the exactly supported range [INT64_MIN,
    INT64_MAX], this class's integer type is interoperable.

    #### Storage

    Integer number values are stored directly inside a @ref basic_json type.

    @sa @ref number_float_t -- type for number values (floating-point)

    @sa @ref number_unsigned_t -- type for number values (unsigned integer)

    @since version 1.0.0
    */
    using number_integer_t = NumberIntegerType;

    /*!
    @brief a type for a number (unsigned)

    [RFC 7159](http://rfc7159.net/rfc7159) describes numbers as follows:
    > The representation of numbers is similar to that used in most
    > programming languages. A number is represented in base 10 using decimal
    > digits. It contains an integer component that may be prefixed with an
    > optional minus sign, which may be followed by a fraction part and/or an
    > exponent part. Leading zeros are not allowed. (...) Numeric values that
    > cannot be represented in the grammar below (such as Infinity and NaN)
    > are not permitted.

    This description includes both integer and floating-point numbers.
    However, C++ allows more precise storage if it is known whether the number
    is a signed integer, an unsigned integer or a floating-point number.
    Therefore, three different types, @ref number_integer_t, @ref
    number_unsigned_t and @ref number_float_t are used.

    To store unsigned integer numbers in C++, a type is defined by the
    template parameter @a NumberUnsignedType which chooses the type to use.

    #### Default type

    With the default values for @a NumberUnsignedType (`uint64_t`), the
    default value for @a number_unsigned_t is:

    @code {.cpp}
    uint64_t
    @endcode

    #### Default behavior

    - The restrictions about leading zeros is not enforced in C++. Instead,
      leading zeros in integer literals lead to an interpretation as octal
      number. Internally, the value will be stored as decimal number. For
      instance, the C++ integer literal `010` will be serialized to `8`.
      During deserialization, leading zeros yield an error.
    - Not-a-number (NaN) values will be serialized to `null`.

    #### Limits

    [RFC 7159](http://rfc7159.net/rfc7159) specifies:
    > An implementation may set limits on the range and precision of numbers.

    When the default type is used, the maximal integer number that can be
    stored is `18446744073709551615` (UINT64_MAX) and the minimal integer
    number that can be stored is `0`. Integer numbers that are out of range
    will yield over/underflow when used in a constructor. During
    deserialization, too large or small integer numbers will be automatically
    be stored as @ref number_integer_t or @ref number_float_t.

    [RFC 7159](http://rfc7159.net/rfc7159) further states:
    > Note that when such software is used, numbers that are integers and are
    > in the range \f$[-2^{53}+1, 2^{53}-1]\f$ are interoperable in the sense
    > that implementations will agree exactly on their numeric values.

    As this range is a subrange (when considered in conjunction with the
    number_integer_t type) of the exactly supported range [0, UINT64_MAX],
    this class's integer type is interoperable.

    #### Storage

    Integer number values are stored directly inside a @ref basic_json type.

    @sa @ref number_float_t -- type for number values (floating-point)
    @sa @ref number_integer_t -- type for number values (integer)

    @since version 2.0.0
    */
    using number_unsigned_t = NumberUnsignedType;

    /*!
    @brief a type for a number (floating-point)

    [RFC 7159](http://rfc7159.net/rfc7159) describes numbers as follows:
    > The representation of numbers is similar to that used in most
    > programming languages. A number is represented in base 10 using decimal
    > digits. It contains an integer component that may be prefixed with an
    > optional minus sign, which may be followed by a fraction part and/or an
    > exponent part. Leading zeros are not allowed. (...) Numeric values that
    > cannot be represented in the grammar below (such as Infinity and NaN)
    > are not permitted.

    This description includes both integer and floating-point numbers.
    However, C++ allows more precise storage if it is known whether the number
    is a signed integer, an unsigned integer or a floating-point number.
    Therefore, three different types, @ref number_integer_t, @ref
    number_unsigned_t and @ref number_float_t are used.

    To store floating-point numbers in C++, a type is defined by the template
    parameter @a NumberFloatType which chooses the type to use.

    #### Default type

    With the default values for @a NumberFloatType (`double`), the default
    value for @a number_float_t is:

    @code {.cpp}
    double
    @endcode

    #### Default behavior

    - The restrictions about leading zeros is not enforced in C++. Instead,
      leading zeros in floating-point literals will be ignored. Internally,
      the value will be stored as decimal number. For instance, the C++
      floating-point literal `01.2` will be serialized to `1.2`. During
      deserialization, leading zeros yield an error.
    - Not-a-number (NaN) values will be serialized to `null`.

    #### Limits

    [RFC 7159](http://rfc7159.net/rfc7159) states:
    > This specification allows implementations to set limits on the range and
    > precision of numbers accepted. Since software that implements IEEE
    > 754-2008 binary64 (double precision) numbers is generally available and
    > widely used, good interoperability can be achieved by implementations
    > that expect no more precision or range than these provide, in the sense
    > that implementations will approximate JSON numbers within the expected
    > precision.

    This implementation does exactly follow this approach, as it uses double
    precision floating-point numbers. Note values smaller than
    `-1.79769313486232e+308` and values greater than `1.79769313486232e+308`
    will be stored as NaN internally and be serialized to `null`.

    #### Storage

    Floating-point number values are stored directly inside a @ref basic_json
    type.

    @sa @ref number_integer_t -- type for number values (integer)

    @sa @ref number_unsigned_t -- type for number values (unsigned integer)

    @since version 1.0.0
    */
    using number_float_t = NumberFloatType;

    /// @}

  private:

    /// helper for exception-safe object creation
    template<typename T, typename... Args>
    static T* create(Args&& ... args)
    {
        AllocatorType<T> alloc;
        using AllocatorTraits = std::allocator_traits<AllocatorType<T>>;

        auto deleter = [&](T * object)
        {
            AllocatorTraits::deallocate(alloc, object, 1);
        };
        std::unique_ptr<T, decltype(deleter)> object(AllocatorTraits::allocate(alloc, 1), deleter);
        AllocatorTraits::construct(alloc, object.get(), std::forward<Args>(args)...);
        assert(object != nullptr);
        return object.release();
    }

    ////////////////////////
    // JSON value storage //
    ////////////////////////

    /*!
    @brief a JSON value

    The actual storage for a JSON value of the @ref basic_json class. This
    union combines the different storage types for the JSON value types
    defined in @ref value_t.

    JSON type | value_t type    | used type
    --------- | --------------- | ------------------------
    object    | object          | pointer to @ref object_t
    array     | array           | pointer to @ref array_t
    string    | string          | pointer to @ref string_t
    boolean   | boolean         | @ref boolean_t
    number    | number_integer  | @ref number_integer_t
    number    | number_unsigned | @ref number_unsigned_t
    number    | number_float    | @ref number_float_t
    null      | null            | *no value is stored*

    @note Variable-length types (objects, arrays, and strings) are stored as
    pointers. The size of the union should not exceed 64 bits if the default
    value types are used.

    @since version 1.0.0
    */
    union json_value
    {
        /// object (stored with pointer to save storage)
        object_t* object;
        /// array (stored with pointer to save storage)
        array_t* array;
        /// string (stored with pointer to save storage)
        string_t* string;
        /// boolean
        boolean_t boolean;
        /// number (integer)
        number_integer_t number_integer;
        /// number (unsigned integer)
        number_unsigned_t number_unsigned;
        /// number (floating-point)
        number_float_t number_float;

        /// default constructor (for null values)
        json_value() = default;
        /// constructor for booleans
        json_value(boolean_t v) noexcept : boolean(v) {}
        /// constructor for numbers (integer)
        json_value(number_integer_t v) noexcept : number_integer(v) {}
        /// constructor for numbers (unsigned)
        json_value(number_unsigned_t v) noexcept : number_unsigned(v) {}
        /// constructor for numbers (floating-point)
        json_value(number_float_t v) noexcept : number_float(v) {}
        /// constructor for empty values of a given type
        json_value(value_t t)
        {
            switch (t)
            {
                case value_t::object:
                {
                    object = create<object_t>();
                    break;
                }

                case value_t::array:
                {
                    array = create<array_t>();
                    break;
                }

                case value_t::string:
                {
                    string = create<string_t>("");
                    break;
                }

                case value_t::boolean:
                {
                    boolean = boolean_t(false);
                    break;
                }

                case value_t::number_integer:
                {
                    number_integer = number_integer_t(0);
                    break;
                }

                case value_t::number_unsigned:
                {
                    number_unsigned = number_unsigned_t(0);
                    break;
                }

                case value_t::number_float:
                {
                    number_float = number_float_t(0.0);
                    break;
                }

                case value_t::null:
                {
                    object = nullptr;  // silence warning, see #821
                    break;
                }

                default:
                {
                    object = nullptr;  // silence warning, see #821
                    if (JSON_UNLIKELY(t == value_t::null))
                    {
                        JSON_THROW(other_error::create(500, "961c151d2e87f2686a955a9be24d316f1362bf21 3.1.2")); // LCOV_EXCL_LINE
                    }
                    break;
                }
            }
        }

        /// constructor for strings
        json_value(const string_t& value)
        {
            string = create<string_t>(value);
        }

        /// constructor for rvalue strings
        json_value(string_t&& value)
        {
            string = create<string_t>(std::move(value));
        }

        /// constructor for objects
        json_value(const object_t& value)
        {
            object = create<object_t>(value);
        }

        /// constructor for rvalue objects
        json_value(object_t&& value)
        {
            object = create<object_t>(std::move(value));
        }

        /// constructor for arrays
        json_value(const array_t& value)
        {
            array = create<array_t>(value);
        }

        /// constructor for rvalue arrays
        json_value(array_t&& value)
        {
            array = create<array_t>(std::move(value));
        }

        void destroy(value_t t) noexcept
        {
            switch (t)
            {
                case value_t::object:
                {
                    AllocatorType<object_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, object);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, object, 1);
                    break;
                }

                case value_t::array:
                {
                    AllocatorType<array_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, array);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, array, 1);
                    break;
                }

                case value_t::string:
                {
                    AllocatorType<string_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, string);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, string, 1);
                    break;
                }

                default:
                {
                    break;
                }
            }
        }
    };

    /*!
    @brief checks the class invariants

    This function asserts the class invariants. It needs to be called at the
    end of every constructor to make sure that created objects respect the
    invariant. Furthermore, it has to be called each time the type of a JSON
    value is changed, because the invariant expresses a relationship between
    @a m_type and @a m_value.
    */
    void assert_invariant() const noexcept
    {
        assert(m_type != value_t::object or m_value.object != nullptr);
        assert(m_type != value_t::array or m_value.array != nullptr);
        assert(m_type != value_t::string or m_value.string != nullptr);
    }

  public:
    //////////////////////////
    // JSON parser callback //
    //////////////////////////

    /*!
    @brief parser event types

    The parser callback distinguishes the following events:
    - `object_start`: the parser read `{` and started to process a JSON object
    - `key`: the parser read a key of a value in an object
    - `object_end`: the parser read `}` and finished processing a JSON object
    - `array_start`: the parser read `[` and started to process a JSON array
    - `array_end`: the parser read `]` and finished processing a JSON array
    - `value`: the parser finished reading a JSON value

    @image html callback_events.png "Example when certain parse events are triggered"

    @sa @ref parser_callback_t for more information and examples
    */
    using parse_event_t = typename parser::parse_event_t;

    /*!
    @brief per-element parser callback type

    With a parser callback function, the result of parsing a JSON text can be
    influenced. When passed to @ref parse, it is called on certain events
    (passed as @ref parse_event_t via parameter @a event) with a set recursion
    depth @a depth and context JSON value @a parsed. The return value of the
    callback function is a boolean indicating whether the element that emitted
    the callback shall be kept or not.

    We distinguish six scenarios (determined by the event type) in which the
    callback function can be called. The following table describes the values
    of the parameters @a depth, @a event, and @a parsed.

    parameter @a event | description | parameter @a depth | parameter @a parsed
    ------------------ | ----------- | ------------------ | -------------------
    parse_event_t::object_start | the parser read `{` and started to process a JSON object | depth of the parent of the JSON object | a JSON value with type discarded
    parse_event_t::key | the parser read a key of a value in an object | depth of the currently parsed JSON object | a JSON string containing the key
    parse_event_t::object_end | the parser read `}` and finished processing a JSON object | depth of the parent of the JSON object | the parsed JSON object
    parse_event_t::array_start | the parser read `[` and started to process a JSON array | depth of the parent of the JSON array | a JSON value with type discarded
    parse_event_t::array_end | the parser read `]` and finished processing a JSON array | depth of the parent of the JSON array | the parsed JSON array
    parse_event_t::value | the parser finished reading a JSON value | depth of the value | the parsed JSON value

    @image html callback_events.png "Example when certain parse events are triggered"

    Discarding a value (i.e., returning `false`) has different effects
    depending on the context in which function was called:

    - Discarded values in structured types are skipped. That is, the parser
      will behave as if the discarded value was never read.
    - In case a value outside a structured type is skipped, it is replaced
      with `null`. This case happens if the top-level element is skipped.

    @param[in] depth  the depth of the recursion during parsing

    @param[in] event  an event of type parse_event_t indicating the context in
    the callback function has been called

    @param[in,out] parsed  the current intermediate parse result; note that
    writing to this value has no effect for parse_event_t::key events

    @return Whether the JSON value which called the function during parsing
    should be kept (`true`) or not (`false`). In the latter case, it is either
    skipped completely or replaced by an empty discarded object.

    @sa @ref parse for examples

    @since version 1.0.0
    */
    using parser_callback_t = typename parser::parser_callback_t;

    using json_sax_t = typename parser::json_sax_t;

    //////////////////
    // constructors //
    //////////////////

    /// @name constructors and destructors
    /// Constructors of class @ref basic_json, copy/move constructor, copy
    /// assignment, static functions creating objects, and the destructor.
    /// @{

    /*!
    @brief create an empty value with a given type

    Create an empty JSON value with a given type. The value will be default
    initialized with an empty value which depends on the type:

    Value type  | initial value
    ----------- | -------------
    null        | `null`
    boolean     | `false`
    string      | `""`
    number      | `0`
    object      | `{}`
    array       | `[]`

    @param[in] v  the type of the value to create

    @complexity Constant.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes to any JSON value.

    @liveexample{The following code shows the constructor for different @ref
    value_t values,basic_json__value_t}

    @sa @ref clear() -- restores the postcondition of this constructor

    @since version 1.0.0
    */
    basic_json(const value_t v)
        : m_type(v), m_value(v)
    {
        assert_invariant();
    }

    /*!
    @brief create a null object

    Create a `null` JSON value. It either takes a null pointer as parameter
    (explicitly creating `null`) or no parameter (implicitly creating `null`).
    The passed null pointer itself is not read -- it is only used to choose
    the right constructor.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this constructor never throws
    exceptions.

    @liveexample{The following code shows the constructor with and without a
    null pointer parameter.,basic_json__nullptr_t}

    @since version 1.0.0
    */
    basic_json(std::nullptr_t = nullptr) noexcept
        : basic_json(value_t::null)
    {
        assert_invariant();
    }

    /*!
    @brief create a JSON value

    This is a "catch all" constructor for all compatible JSON types; that is,
    types for which a `to_json()` method exists. The constructor forwards the
    parameter @a val to that method (to `json_serializer<U>::to_json` method
    with `U = uncvref_t<CompatibleType>`, to be exact).

    Template type @a CompatibleType includes, but is not limited to, the
    following types:
    - **arrays**: @ref array_t and all kinds of compatible containers such as
      `std::vector`, `std::deque`, `std::list`, `std::forward_list`,
      `std::array`, `std::valarray`, `std::set`, `std::unordered_set`,
      `std::multiset`, and `std::unordered_multiset` with a `value_type` from
      which a @ref basic_json value can be constructed.
    - **objects**: @ref object_t and all kinds of compatible associative
      containers such as `std::map`, `std::unordered_map`, `std::multimap`,
      and `std::unordered_multimap` with a `key_type` compatible to
      @ref string_t and a `value_type` from which a @ref basic_json value can
      be constructed.
    - **strings**: @ref string_t, string literals, and all compatible string
      containers can be used.
    - **numbers**: @ref number_integer_t, @ref number_unsigned_t,
      @ref number_float_t, and all convertible number types such as `int`,
      `size_t`, `int64_t`, `float` or `double` can be used.
    - **boolean**: @ref boolean_t / `bool` can be used.

    See the examples below.

    @tparam CompatibleType a type such that:
    - @a CompatibleType is not derived from `std::istream`,
    - @a CompatibleType is not @ref basic_json (to avoid hijacking copy/move
         constructors),
    - @a CompatibleType is not a different @ref basic_json type (i.e. with different template arguments)
    - @a CompatibleType is not a @ref basic_json nested type (e.g.,
         @ref json_pointer, @ref iterator, etc ...)
    - @ref @ref json_serializer<U> has a
         `to_json(basic_json_t&, CompatibleType&&)` method

    @tparam U = `uncvref_t<CompatibleType>`

    @param[in] val the value to be forwarded to the respective constructor

    @complexity Usually linear in the size of the passed @a val, also
                depending on the implementation of the called `to_json()`
                method.

    @exceptionsafety Depends on the called constructor. For types directly
    supported by the library (i.e., all types for which no `to_json()` function
    was provided), strong guarantee holds: if an exception is thrown, there are
    no changes to any JSON value.

    @liveexample{The following code shows the constructor with several
    compatible types.,basic_json__CompatibleType}

    @since version 2.1.0
    */
    template <typename CompatibleType,
              typename U = detail::uncvref_t<CompatibleType>,
              detail::enable_if_t<
                  detail::is_compatible_type<basic_json_t, U>::value, int> = 0>
    basic_json(CompatibleType && val) noexcept(noexcept(
                JSONSerializer<U>::to_json(std::declval<basic_json_t&>(),
                                           std::forward<CompatibleType>(val))))
    {
        JSONSerializer<U>::to_json(*this, std::forward<CompatibleType>(val));
        assert_invariant();
    }

    /*!
    @brief create a JSON value from an existing one

    This is a constructor for existing @ref basic_json types.
    It does not hijack copy/move constructors, since the parameter has different
    template arguments than the current ones.

    The constructor tries to convert the internal @ref m_value of the parameter.

    @tparam BasicJsonType a type such that:
    - @a BasicJsonType is a @ref basic_json type.
    - @a BasicJsonType has different template arguments than @ref basic_json_t.

    @param[in] val the @ref basic_json value to be converted.

    @complexity Usually linear in the size of the passed @a val, also
                depending on the implementation of the called `to_json()`
                method.

    @exceptionsafety Depends on the called constructor. For types directly
    supported by the library (i.e., all types for which no `to_json()` function
    was provided), strong guarantee holds: if an exception is thrown, there are
    no changes to any JSON value.

    @since version 3.1.2
    */
    template <typename BasicJsonType,
              detail::enable_if_t<
                  detail::is_basic_json<BasicJsonType>::value and not std::is_same<basic_json, BasicJsonType>::value, int> = 0>
    basic_json(const BasicJsonType& val)
    {
        using other_boolean_t = typename BasicJsonType::boolean_t;
        using other_number_float_t = typename BasicJsonType::number_float_t;
        using other_number_integer_t = typename BasicJsonType::number_integer_t;
        using other_number_unsigned_t = typename BasicJsonType::number_unsigned_t;
        using other_string_t = typename BasicJsonType::string_t;
        using other_object_t = typename BasicJsonType::object_t;
        using other_array_t = typename BasicJsonType::array_t;

        switch (val.type())
        {
            case value_t::boolean:
                JSONSerializer<other_boolean_t>::to_json(*this, val.template get<other_boolean_t>());
                break;
            case value_t::number_float:
                JSONSerializer<other_number_float_t>::to_json(*this, val.template get<other_number_float_t>());
                break;
            case value_t::number_integer:
                JSONSerializer<other_number_integer_t>::to_json(*this, val.template get<other_number_integer_t>());
                break;
            case value_t::number_unsigned:
                JSONSerializer<other_number_unsigned_t>::to_json(*this, val.template get<other_number_unsigned_t>());
                break;
            case value_t::string:
                JSONSerializer<other_string_t>::to_json(*this, val.template get_ref<const other_string_t&>());
                break;
            case value_t::object:
                JSONSerializer<other_object_t>::to_json(*this, val.template get_ref<const other_object_t&>());
                break;
            case value_t::array:
                JSONSerializer<other_array_t>::to_json(*this, val.template get_ref<const other_array_t&>());
                break;
            case value_t::null:
                *this = nullptr;
                break;
            case value_t::discarded:
                m_type = value_t::discarded;
                break;
        }
        assert_invariant();
    }

    /*!
    @brief create a container (array or object) from an initializer list

    Creates a JSON value of type array or object from the passed initializer
    list @a init. In case @a type_deduction is `true` (default), the type of
    the JSON value to be created is deducted from the initializer list @a init
    according to the following rules:

    1. If the list is empty, an empty JSON object value `{}` is created.
    2. If the list consists of pairs whose first element is a string, a JSON
       object value is created where the first elements of the pairs are
       treated as keys and the second elements are as values.
    3. In all other cases, an array is created.

    The rules aim to create the best fit between a C++ initializer list and
    JSON values. The rationale is as follows:

    1. The empty initializer list is written as `{}` which is exactly an empty
       JSON object.
    2. C++ has no way of describing mapped types other than to list a list of
       pairs. As JSON requires that keys must be of type string, rule 2 is the
       weakest constraint one can pose on initializer lists to interpret them
       as an object.
    3. In all other cases, the initializer list could not be interpreted as
       JSON object type, so interpreting it as JSON array type is safe.

    With the rules described above, the following JSON values cannot be
    expressed by an initializer list:

    - the empty array (`[]`): use @ref array(initializer_list_t)
      with an empty initializer list in this case
    - arrays whose elements satisfy rule 2: use @ref
      array(initializer_list_t) with the same initializer list
      in this case

    @note When used without parentheses around an empty initializer list, @ref
    basic_json() is called instead of this function, yielding the JSON null
    value.

    @param[in] init  initializer list with JSON values

    @param[in] type_deduction internal parameter; when set to `true`, the type
    of the JSON value is deducted from the initializer list @a init; when set
    to `false`, the type provided via @a manual_type is forced. This mode is
    used by the functions @ref array(initializer_list_t) and
    @ref object(initializer_list_t).

    @param[in] manual_type internal parameter; when @a type_deduction is set
    to `false`, the created JSON value will use the provided type (only @ref
    value_t::array and @ref value_t::object are valid); when @a type_deduction
    is set to `true`, this parameter has no effect

    @throw type_error.301 if @a type_deduction is `false`, @a manual_type is
    `value_t::object`, but @a init contains an element which is not a pair
    whose first element is a string. In this case, the constructor could not
    create an object. If @a type_deduction would have be `true`, an array
    would have been created. See @ref object(initializer_list_t)
    for an example.

    @complexity Linear in the size of the initializer list @a init.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes to any JSON value.

    @liveexample{The example below shows how JSON values are created from
    initializer lists.,basic_json__list_init_t}

    @sa @ref array(initializer_list_t) -- create a JSON array
    value from an initializer list
    @sa @ref object(initializer_list_t) -- create a JSON object
    value from an initializer list

    @since version 1.0.0
    */
    basic_json(initializer_list_t init,
               bool type_deduction = true,
               value_t manual_type = value_t::array)
    {
        // check if each element is an array with two elements whose first
        // element is a string
        bool is_an_object = std::all_of(init.begin(), init.end(),
                                        [](const detail::json_ref<basic_json>& element_ref)
        {
            return (element_ref->is_array() and element_ref->size() == 2 and (*element_ref)[0].is_string());
        });

        // adjust type if type deduction is not wanted
        if (not type_deduction)
        {
            // if array is wanted, do not create an object though possible
            if (manual_type == value_t::array)
            {
                is_an_object = false;
            }

            // if object is wanted but impossible, throw an exception
            if (JSON_UNLIKELY(manual_type == value_t::object and not is_an_object))
            {
                JSON_THROW(type_error::create(301, "cannot create object from initializer list"));
            }
        }

        if (is_an_object)
        {
            // the initializer list is a list of pairs -> create object
            m_type = value_t::object;
            m_value = value_t::object;

            std::for_each(init.begin(), init.end(), [this](const detail::json_ref<basic_json>& element_ref)
            {
                auto element = element_ref.moved_or_copied();
                m_value.object->emplace(
                    std::move(*((*element.m_value.array)[0].m_value.string)),
                    std::move((*element.m_value.array)[1]));
            });
        }
        else
        {
            // the initializer list describes an array -> create array
            m_type = value_t::array;
            m_value.array = create<array_t>(init.begin(), init.end());
        }

        assert_invariant();
    }

    /*!
    @brief explicitly create an array from an initializer list

    Creates a JSON array value from a given initializer list. That is, given a
    list of values `a, b, c`, creates the JSON value `[a, b, c]`. If the
    initializer list is empty, the empty array `[]` is created.

    @note This function is only needed to express two edge cases that cannot
    be realized with the initializer list constructor (@ref
    basic_json(initializer_list_t, bool, value_t)). These cases
    are:
    1. creating an array whose elements are all pairs whose first element is a
    string -- in this case, the initializer list constructor would create an
    object, taking the first elements as keys
    2. creating an empty array -- passing the empty initializer list to the
    initializer list constructor yields an empty object

    @param[in] init  initializer list with JSON values to create an array from
    (optional)

    @return JSON array value

    @complexity Linear in the size of @a init.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes to any JSON value.

    @liveexample{The following code shows an example for the `array`
    function.,array}

    @sa @ref basic_json(initializer_list_t, bool, value_t) --
    create a JSON value from an initializer list
    @sa @ref object(initializer_list_t) -- create a JSON object
    value from an initializer list

    @since version 1.0.0
    */
    static basic_json array(initializer_list_t init = {})
    {
        return basic_json(init, false, value_t::array);
    }

    /*!
    @brief explicitly create an object from an initializer list

    Creates a JSON object value from a given initializer list. The initializer
    lists elements must be pairs, and their first elements must be strings. If
    the initializer list is empty, the empty object `{}` is created.

    @note This function is only added for symmetry reasons. In contrast to the
    related function @ref array(initializer_list_t), there are
    no cases which can only be expressed by this function. That is, any
    initializer list @a init can also be passed to the initializer list
    constructor @ref basic_json(initializer_list_t, bool, value_t).

    @param[in] init  initializer list to create an object from (optional)

    @return JSON object value

    @throw type_error.301 if @a init is not a list of pairs whose first
    elements are strings. In this case, no object can be created. When such a
    value is passed to @ref basic_json(initializer_list_t, bool, value_t),
    an array would have been created from the passed initializer list @a init.
    See example below.

    @complexity Linear in the size of @a init.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes to any JSON value.

    @liveexample{The following code shows an example for the `object`
    function.,object}

    @sa @ref basic_json(initializer_list_t, bool, value_t) --
    create a JSON value from an initializer list
    @sa @ref array(initializer_list_t) -- create a JSON array
    value from an initializer list

    @since version 1.0.0
    */
    static basic_json object(initializer_list_t init = {})
    {
        return basic_json(init, false, value_t::object);
    }

    /*!
    @brief construct an array with count copies of given value

    Constructs a JSON array value by creating @a cnt copies of a passed value.
    In case @a cnt is `0`, an empty array is created.

    @param[in] cnt  the number of JSON copies of @a val to create
    @param[in] val  the JSON value to copy

    @post `std::distance(begin(),end()) == cnt` holds.

    @complexity Linear in @a cnt.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes to any JSON value.

    @liveexample{The following code shows examples for the @ref
    basic_json(size_type\, const basic_json&)
    constructor.,basic_json__size_type_basic_json}

    @since version 1.0.0
    */
    basic_json(size_type cnt, const basic_json& val)
        : m_type(value_t::array)
    {
        m_value.array = create<array_t>(cnt, val);
        assert_invariant();
    }

    /*!
    @brief construct a JSON container given an iterator range

    Constructs the JSON value with the contents of the range `[first, last)`.
    The semantics depends on the different types a JSON value can have:
    - In case of a null type, invalid_iterator.206 is thrown.
    - In case of other primitive types (number, boolean, or string), @a first
      must be `begin()` and @a last must be `end()`. In this case, the value is
      copied. Otherwise, invalid_iterator.204 is thrown.
    - In case of structured types (array, object), the constructor behaves as
      similar versions for `std::vector` or `std::map`; that is, a JSON array
      or object is constructed from the values in the range.

    @tparam InputIT an input iterator type (@ref iterator or @ref
    const_iterator)

    @param[in] first begin of the range to copy from (included)
    @param[in] last end of the range to copy from (excluded)

    @pre Iterators @a first and @a last must be initialized. **This
         precondition is enforced with an assertion (see warning).** If
         assertions are switched off, a violation of this precondition yields
         undefined behavior.

    @pre Range `[first, last)` is valid. Usually, this precondition cannot be
         checked efficiently. Only certain edge cases are detected; see the
         description of the exceptions below. A violation of this precondition
         yields undefined behavior.

    @warning A precondition is enforced with a runtime assertion that will
             result in calling `std::abort` if this precondition is not met.
             Assertions can be disabled by defining `NDEBUG` at compile time.
             See https://en.cppreference.com/w/cpp/error/assert for more
             information.

    @throw invalid_iterator.201 if iterators @a first and @a last are not
    compatible (i.e., do not belong to the same JSON value). In this case,
    the range `[first, last)` is undefined.
    @throw invalid_iterator.204 if iterators @a first and @a last belong to a
    primitive type (number, boolean, or string), but @a first does not point
    to the first element any more. In this case, the range `[first, last)` is
    undefined. See example code below.
    @throw invalid_iterator.206 if iterators @a first and @a last belong to a
    null value. In this case, the range `[first, last)` is undefined.

    @complexity Linear in distance between @a first and @a last.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes to any JSON value.

    @liveexample{The example below shows several ways to create JSON values by
    specifying a subrange with iterators.,basic_json__InputIt_InputIt}

    @since version 1.0.0
    */
    template<class InputIT, typename std::enable_if<
                 std::is_same<InputIT, typename basic_json_t::iterator>::value or
                 std::is_same<InputIT, typename basic_json_t::const_iterator>::value, int>::type = 0>
    basic_json(InputIT first, InputIT last)
    {
        assert(first.m_object != nullptr);
        assert(last.m_object != nullptr);

        // make sure iterator fits the current value
        if (JSON_UNLIKELY(first.m_object != last.m_object))
        {
            JSON_THROW(invalid_iterator::create(201, "iterators are not compatible"));
        }

        // copy type from first iterator
        m_type = first.m_object->m_type;

        // check if iterator range is complete for primitive values
        switch (m_type)
        {
            case value_t::boolean:
            case value_t::number_float:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::string:
            {
                if (JSON_UNLIKELY(not first.m_it.primitive_iterator.is_begin()
                                  or not last.m_it.primitive_iterator.is_end()))
                {
                    JSON_THROW(invalid_iterator::create(204, "iterators out of range"));
                }
                break;
            }

            default:
                break;
        }

        switch (m_type)
        {
            case value_t::number_integer:
            {
                m_value.number_integer = first.m_object->m_value.number_integer;
                break;
            }

            case value_t::number_unsigned:
            {
                m_value.number_unsigned = first.m_object->m_value.number_unsigned;
                break;
            }

            case value_t::number_float:
            {
                m_value.number_float = first.m_object->m_value.number_float;
                break;
            }

            case value_t::boolean:
            {
                m_value.boolean = first.m_object->m_value.boolean;
                break;
            }

            case value_t::string:
            {
                m_value = *first.m_object->m_value.string;
                break;
            }

            case value_t::object:
            {
                m_value.object = create<object_t>(first.m_it.object_iterator,
                                                  last.m_it.object_iterator);
                break;
            }

            case value_t::array:
            {
                m_value.array = create<array_t>(first.m_it.array_iterator,
                                                last.m_it.array_iterator);
                break;
            }

            default:
                JSON_THROW(invalid_iterator::create(206, "cannot construct with iterators from " +
                                                    std::string(first.m_object->type_name())));
        }

        assert_invariant();
    }


    ///////////////////////////////////////
    // other constructors and destructor //
    ///////////////////////////////////////

    /// @private
    basic_json(const detail::json_ref<basic_json>& ref)
        : basic_json(ref.moved_or_copied())
    {}

    /*!
    @brief copy constructor

    Creates a copy of a given JSON value.

    @param[in] other  the JSON value to copy

    @post `*this == other`

    @complexity Linear in the size of @a other.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes to any JSON value.

    @requirement This function helps `basic_json` satisfying the
    [Container](https://en.cppreference.com/w/cpp/named_req/Container)
    requirements:
    - The complexity is linear.
    - As postcondition, it holds: `other == basic_json(other)`.

    @liveexample{The following code shows an example for the copy
    constructor.,basic_json__basic_json}

    @since version 1.0.0
    */
    basic_json(const basic_json& other)
        : m_type(other.m_type)
    {
        // check of passed value is valid
        other.assert_invariant();

        switch (m_type)
        {
            case value_t::object:
            {
                m_value = *other.m_value.object;
                break;
            }

            case value_t::array:
            {
                m_value = *other.m_value.array;
                break;
            }

            case value_t::string:
            {
                m_value = *other.m_value.string;
                break;
            }

            case value_t::boolean:
            {
                m_value = other.m_value.boolean;
                break;
            }

            case value_t::number_integer:
            {
                m_value = other.m_value.number_integer;
                break;
            }

            case value_t::number_unsigned:
            {
                m_value = other.m_value.number_unsigned;
                break;
            }

            case value_t::number_float:
            {
                m_value = other.m_value.number_float;
                break;
            }

            default:
                break;
        }

        assert_invariant();
    }

    /*!
    @brief move constructor

    Move constructor. Constructs a JSON value with the contents of the given
    value @a other using move semantics. It "steals" the resources from @a
    other and leaves it as JSON null value.

    @param[in,out] other  value to move to this object

    @post `*this` has the same value as @a other before the call.
    @post @a other is a JSON null value.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this constructor never throws
    exceptions.

    @requirement This function helps `basic_json` satisfying the
    [MoveConstructible](https://en.cppreference.com/w/cpp/named_req/MoveConstructible)
    requirements.

    @liveexample{The code below shows the move constructor explicitly called
    via std::move.,basic_json__moveconstructor}

    @since version 1.0.0
    */
    basic_json(basic_json&& other) noexcept
        : m_type(std::move(other.m_type)),
          m_value(std::move(other.m_value))
    {
        // check that passed value is valid
        other.assert_invariant();

        // invalidate payload
        other.m_type = value_t::null;
        other.m_value = {};

        assert_invariant();
    }

    /*!
    @brief copy assignment

    Copy assignment operator. Copies a JSON value via the "copy and swap"
    strategy: It is expressed in terms of the copy constructor, destructor,
    and the `swap()` member function.

    @param[in] other  value to copy from

    @complexity Linear.

    @requirement This function helps `basic_json` satisfying the
    [Container](https://en.cppreference.com/w/cpp/named_req/Container)
    requirements:
    - The complexity is linear.

    @liveexample{The code below shows and example for the copy assignment. It
    creates a copy of value `a` which is then swapped with `b`. Finally\, the
    copy of `a` (which is the null value after the swap) is
    destroyed.,basic_json__copyassignment}

    @since version 1.0.0
    */
    reference& operator=(basic_json other) noexcept (
        std::is_nothrow_move_constructible<value_t>::value and
        std::is_nothrow_move_assignable<value_t>::value and
        std::is_nothrow_move_constructible<json_value>::value and
        std::is_nothrow_move_assignable<json_value>::value
    )
    {
        // check that passed value is valid
        other.assert_invariant();

        using std::swap;
        swap(m_type, other.m_type);
        swap(m_value, other.m_value);

        assert_invariant();
        return *this;
    }

    /*!
    @brief destructor

    Destroys the JSON value and frees all allocated memory.

    @complexity Linear.

    @requirement This function helps `basic_json` satisfying the
    [Container](https://en.cppreference.com/w/cpp/named_req/Container)
    requirements:
    - The complexity is linear.
    - All stored elements are destroyed and all memory is freed.

    @since version 1.0.0
    */
    ~basic_json() noexcept
    {
        assert_invariant();
        m_value.destroy(m_type);
    }

    /// @}

  public:
    ///////////////////////
    // object inspection //
    ///////////////////////

    /// @name object inspection
    /// Functions to inspect the type of a JSON value.
    /// @{

    /*!
    @brief return the type of the JSON value (explicit)

    Return the type of the JSON value as a value from the @ref value_t
    enumeration.

    @return the type of the JSON value
            Value type                | return value
            ------------------------- | -------------------------
            null                      | value_t::null
            boolean                   | value_t::boolean
            string                    | value_t::string
            number (integer)          | value_t::number_integer
            number (unsigned integer) | value_t::number_unsigned
            number (floating-point)   | value_t::number_float
            object                    | value_t::object
            array                     | value_t::array
            discarded                 | value_t::discarded

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `type()` for all JSON
    types.,type}

    @sa @ref operator value_t() -- return the type of the JSON value (implicit)
    @sa @ref type_name() -- return the type as string

    @since version 1.0.0
    */
    constexpr value_t type() const noexcept
    {
        return m_type;
    }

    /*!
    @brief return whether type is primitive

    This function returns true if and only if the JSON type is primitive
    (string, number, boolean, or null).

    @return `true` if type is primitive (string, number, boolean, or null),
    `false` otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_primitive()` for all JSON
    types.,is_primitive}

    @sa @ref is_structured() -- returns whether JSON value is structured
    @sa @ref is_null() -- returns whether JSON value is `null`
    @sa @ref is_string() -- returns whether JSON value is a string
    @sa @ref is_boolean() -- returns whether JSON value is a boolean
    @sa @ref is_number() -- returns whether JSON value is a number

    @since version 1.0.0
    */
    constexpr bool is_primitive() const noexcept
    {
        return is_null() or is_string() or is_boolean() or is_number();
    }

    /*!
    @brief return whether type is structured

    This function returns true if and only if the JSON type is structured
    (array or object).

    @return `true` if type is structured (array or object), `false` otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_structured()` for all JSON
    types.,is_structured}

    @sa @ref is_primitive() -- returns whether value is primitive
    @sa @ref is_array() -- returns whether value is an array
    @sa @ref is_object() -- returns whether value is an object

    @since version 1.0.0
    */
    constexpr bool is_structured() const noexcept
    {
        return is_array() or is_object();
    }

    /*!
    @brief return whether value is null

    This function returns true if and only if the JSON value is null.

    @return `true` if type is null, `false` otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_null()` for all JSON
    types.,is_null}

    @since version 1.0.0
    */
    constexpr bool is_null() const noexcept
    {
        return (m_type == value_t::null);
    }

    /*!
    @brief return whether value is a boolean

    This function returns true if and only if the JSON value is a boolean.

    @return `true` if type is boolean, `false` otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_boolean()` for all JSON
    types.,is_boolean}

    @since version 1.0.0
    */
    constexpr bool is_boolean() const noexcept
    {
        return (m_type == value_t::boolean);
    }

    /*!
    @brief return whether value is a number

    This function returns true if and only if the JSON value is a number. This
    includes both integer (signed and unsigned) and floating-point values.

    @return `true` if type is number (regardless whether integer, unsigned
    integer or floating-type), `false` otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_number()` for all JSON
    types.,is_number}

    @sa @ref is_number_integer() -- check if value is an integer or unsigned
    integer number
    @sa @ref is_number_unsigned() -- check if value is an unsigned integer
    number
    @sa @ref is_number_float() -- check if value is a floating-point number

    @since version 1.0.0
    */
    constexpr bool is_number() const noexcept
    {
        return is_number_integer() or is_number_float();
    }

    /*!
    @brief return whether value is an integer number

    This function returns true if and only if the JSON value is a signed or
    unsigned integer number. This excludes floating-point values.

    @return `true` if type is an integer or unsigned integer number, `false`
    otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_number_integer()` for all
    JSON types.,is_number_integer}

    @sa @ref is_number() -- check if value is a number
    @sa @ref is_number_unsigned() -- check if value is an unsigned integer
    number
    @sa @ref is_number_float() -- check if value is a floating-point number

    @since version 1.0.0
    */
    constexpr bool is_number_integer() const noexcept
    {
        return (m_type == value_t::number_integer or m_type == value_t::number_unsigned);
    }

    /*!
    @brief return whether value is an unsigned integer number

    This function returns true if and only if the JSON value is an unsigned
    integer number. This excludes floating-point and signed integer values.

    @return `true` if type is an unsigned integer number, `false` otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_number_unsigned()` for all
    JSON types.,is_number_unsigned}

    @sa @ref is_number() -- check if value is a number
    @sa @ref is_number_integer() -- check if value is an integer or unsigned
    integer number
    @sa @ref is_number_float() -- check if value is a floating-point number

    @since version 2.0.0
    */
    constexpr bool is_number_unsigned() const noexcept
    {
        return (m_type == value_t::number_unsigned);
    }

    /*!
    @brief return whether value is a floating-point number

    This function returns true if and only if the JSON value is a
    floating-point number. This excludes signed and unsigned integer values.

    @return `true` if type is a floating-point number, `false` otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_number_float()` for all
    JSON types.,is_number_float}

    @sa @ref is_number() -- check if value is number
    @sa @ref is_number_integer() -- check if value is an integer number
    @sa @ref is_number_unsigned() -- check if value is an unsigned integer
    number

    @since version 1.0.0
    */
    constexpr bool is_number_float() const noexcept
    {
        return (m_type == value_t::number_float);
    }

    /*!
    @brief return whether value is an object

    This function returns true if and only if the JSON value is an object.

    @return `true` if type is object, `false` otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_object()` for all JSON
    types.,is_object}

    @since version 1.0.0
    */
    constexpr bool is_object() const noexcept
    {
        return (m_type == value_t::object);
    }

    /*!
    @brief return whether value is an array

    This function returns true if and only if the JSON value is an array.

    @return `true` if type is array, `false` otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_array()` for all JSON
    types.,is_array}

    @since version 1.0.0
    */
    constexpr bool is_array() const noexcept
    {
        return (m_type == value_t::array);
    }

    /*!
    @brief return whether value is a string

    This function returns true if and only if the JSON value is a string.

    @return `true` if type is string, `false` otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_string()` for all JSON
    types.,is_string}

    @since version 1.0.0
    */
    constexpr bool is_string() const noexcept
    {
        return (m_type == value_t::string);
    }

    /*!
    @brief return whether value is discarded

    This function returns true if and only if the JSON value was discarded
    during parsing with a callback function (see @ref parser_callback_t).

    @note This function will always be `false` for JSON values after parsing.
    That is, discarded values can only occur during parsing, but will be
    removed when inside a structured value or replaced by null in other cases.

    @return `true` if type is discarded, `false` otherwise.

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies `is_discarded()` for all JSON
    types.,is_discarded}

    @since version 1.0.0
    */
    constexpr bool is_discarded() const noexcept
    {
        return (m_type == value_t::discarded);
    }

    /*!
    @brief return the type of the JSON value (implicit)

    Implicitly return the type of the JSON value as a value from the @ref
    value_t enumeration.

    @return the type of the JSON value

    @complexity Constant.

    @exceptionsafety No-throw guarantee: this member function never throws
    exceptions.

    @liveexample{The following code exemplifies the @ref value_t operator for
    all JSON types.,operator__value_t}

    @sa @ref type() -- return the type of the JSON value (explicit)
    @sa @ref type_name() -- return the type as string

    @since version 1.0.0
    */
    constexpr operator value_t() const noexcept
    {
        return m_type;
    }

    /// @}

  private:
    //////////////////
    // value access //
    //////////////////

    /// get a boolean (explicit)
    boolean_t get_impl(boolean_t* /*unused*/) const
    {
        if (JSON_LIKELY(is_boolean()))
        {
            return m_value.boolean;
        }

        JSON_THROW(type_error::create(302, "type must be boolean, but is " + std::string(type_name())));
    }

    /// get a pointer to the value (object)
    object_t* get_impl_ptr(object_t* /*unused*/) noexcept
    {
        return is_object() ? m_value.object : nullptr;
    }

    /// get a pointer to the value (object)
    constexpr const object_t* get_impl_ptr(const object_t* /*unused*/) const noexcept
    {
        return is_object() ? m_value.object : nullptr;
    }

    /// get a pointer to the value (array)
    array_t* get_impl_ptr(array_t* /*unused*/) noexcept
    {
        return is_array() ? m_value.array : nullptr;
    }

    /// get a pointer to the value (array)
    constexpr const array_t* get_impl_ptr(const array_t* /*unused*/) const noexcept
    {
        return is_array() ? m_value.array : nullptr;
    }

    /// get a pointer to the value (string)
    string_t* get_impl_ptr(string_t* /*unused*/) noexcept
    {
        return is_string() ? m_value.string : nullptr;
    }

    /// get a pointer to the value (string)
    constexpr const string_t* get_impl_ptr(const string_t* /*unused*/) const noexcept
    {
        return is_string() ? m_value.string : nullptr;
    }

    /// get a pointer to the value (boolean)
    boolean_t* get_impl_ptr(boolean_t* /*unused*/) noexcept
    {
        return is_boolean() ? &m_value.boolean : nullptr;
    }

    /// get a pointer to the value (boolean)
    constexpr const boolean_t* get_impl_ptr(const boolean_t* /*unused*/) const noexcept
    {
        return is_boolean() ? &m_value.boolean : nullptr;
    }

    /// get a pointer to the value (integer number)
    number_integer_t* get_impl_ptr(number_integer_t* /*unused*/) noexcept
    {
        return is_number_integer() ? &m_value.number_integer : nullptr;
    }

    /// get a pointer to the value (integer number)
    constexpr const number_integer_t* get_impl_ptr(const number_integer_t* /*unused*/) const noexcept
    {
        return is_number_integer() ? &m_value.number_integer : nullptr;
    }

    /// get a pointer to the value (unsigned number)
    number_unsigned_t* get_impl_ptr(number_unsigned_t* /*unused*/) noexcept
    {
        return is_number_unsigned() ? &m_value.number_unsigned : nullptr;
    }

    /// get a pointer to the value (unsigned number)
    constexpr const number_unsigned_t* get_impl_ptr(const number_unsigned_t* /*unused*/) const noexcept
    {
        return is_number_unsigned() ? &m_value.number_unsigned : nullptr;
    }

    /// get a pointer to the value (floating-point number)
    number_float_t* get_impl_ptr(number_float_t* /*unused*/) noexcept
    {
        return is_number_float() ? &m_value.number_float : nullptr;
    }

    /// get a pointer to the value (floating-point number)
    constexpr const number_float_t* get_impl_ptr(const number_float_t* /*unused*/) const noexcept
    {
        return is_number_float() ? &m_value.number_float : nullptr;
    }

    /*!
    @brief helper function to implement get_ref()

    This function helps to implement get_ref() without code duplication for
    const and non-const overloads

    @tparam ThisType will be deduced as `basic_json` or `const basic_json`

    @throw type_error.303 if ReferenceType does not match underlying value
    type of the current JSON
    */
    template<typename ReferenceType, typename ThisType>
    static ReferenceType get_ref_impl(ThisType& obj)
    {
        // delegate the call to get_ptr<>()
        auto ptr = obj.template get_ptr<typename std::add_pointer<ReferenceType>::type>();

        if (JSON_LIKELY(ptr != nullptr))
        {
            return *ptr;
        }

        JSON_THROW(type_error::create(303, "incompatible ReferenceType for get_ref, actual type is " + std::string(obj.type_name())));
    }

  public:
    /// @name value access
    /// Direct access to the stored value of a JSON value.
    /// @{

    /*!
    @brief get special-case overload

    This overloads avoids a lot of template boilerplate, it can be seen as the
    identity method

    @tparam BasicJsonType == @ref basic_json

    @return a copy of *this

    @complexity Constant.

    @since version 2.1.0
    */
    template<typename BasicJsonType, detail::enable_if_t<
                 std::is_same<typename std::remove_const<BasicJsonType>::type, basic_json_t>::value,
                 int> = 0>
    basic_json get() const
    {
        return *this;
    }

    /*!
    @brief get special-case overload

    This overloads converts the current @ref basic_json in a different
    @ref basic_json type

    @tparam BasicJsonType == @ref basic_json

    @return a copy of *this, converted into @tparam BasicJsonType

    @complexity Depending on the implementation of the called `from_json()`
                method.

    @since version 3.1.2
    */
    template<typename BasicJsonType, detail::enable_if_t<
                 not std::is_same<BasicJsonType, basic_json>::value and
                 detail::is_basic_json<BasicJsonType>::value, int> = 0>
    BasicJsonType get() const
    {
        return *this;
    }

    /*!
    @brief get a value (explicit)

    Explicit type conversion between the JSON value and a compatible value
    which is [CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible)
    and [DefaultConstructible](https://en.cppreference.com/w/cpp/named_req/DefaultConstructible).
    The value is converted by calling the @ref json_serializer<ValueType>
    `from_json()` method.

    The function is equivalent to executing
    @code {.cpp}
    ValueType ret;
    JSONSerializer<ValueType>::from_json(*this, ret);
    return ret;
    @endcode

    This overloads is chosen if:
    - @a ValueType is not @ref basic_json,
    - @ref json_serializer<ValueType> has a `from_json()` method of the form
      `void from_json(const basic_json&, ValueType&)`, and
    - @ref json_serializer<ValueType> does not have a `from_json()` method of
      the form `ValueType from_json(const basic_json&)`

    @tparam ValueTypeCV the provided value type
    @tparam ValueType the returned value type

    @return copy of the JSON value, converted to @a ValueType

    @throw what @ref json_serializer<ValueType> `from_json()` method throws

    @liveexample{The example below shows several conversions from JSON values
    to other types. There a few things to note: (1) Floating-point numbers can
    be converted to integers\, (2) A JSON array can be converted to a standard
    `std::vector<short>`\, (3) A JSON object can be converted to C++
    associative containers such as `std::unordered_map<std::string\,
    json>`.,get__ValueType_const}

    @since version 2.1.0
    */
    template<typename ValueTypeCV, typename ValueType = detail::uncvref_t<ValueTypeCV>,
             detail::enable_if_t <
                 not detail::is_basic_json<ValueType>::value and
                 detail::has_from_json<basic_json_t, ValueType>::value and
                 not detail::has_non_default_from_json<basic_json_t, ValueType>::value,
                 int> = 0>
    ValueType get() const noexcept(noexcept(
                                       JSONSerializer<ValueType>::from_json(std::declval<const basic_json_t&>(), std::declval<ValueType&>())))
    {
        // we cannot static_assert on ValueTypeCV being non-const, because
        // there is support for get<const basic_json_t>(), which is why we
        // still need the uncvref
        static_assert(not std::is_reference<ValueTypeCV>::value,
                      "get() cannot be used with reference types, you might want to use get_ref()");
        static_assert(std::is_default_constructible<ValueType>::value,
                      "types must be DefaultConstructible when used with get()");

        ValueType ret;
        JSONSerializer<ValueType>::from_json(*this, ret);
        return ret;
    }

    /*!
    @brief get a value (explicit); special case

    Explicit type conversion between the JSON value and a compatible value
    which is **not** [CopyConstructible](https://en.cppreference.com/w/cpp/named_req/CopyConstructible)
    and **not** [DefaultConstructible](https://en.cppreference.com/w/cpp/named_req/DefaultConstructible).
    The value is converted by calling the @ref json_serializer<ValueType>
    `from_json()` method.

    The function is equivalent to executing
    @code {.cpp}
    return JSONSerializer<ValueTypeCV>::from_json(*this);
    @endcode

    This overloads is chosen if:
    - @a ValueType is not @ref basic_json and
    - @ref json_serializer<ValueType> has a `from_json()` method of the form
      `ValueType from_json(const basic_json&)`

    @note If @ref json_serializer<ValueType> has both overloads of
    `from_json()`, this one is chosen.

    @tparam ValueTypeCV the provided value type
    @tparam ValueType the returned value type

    @return copy of the JSON value, converted to @a ValueType

    @throw what @ref json_serializer<ValueType> `from_json()` method throws

    @since version 2.1.0
    */
    template<typename ValueTypeCV, typename ValueType = detail::uncvref_t<ValueTypeCV>,
             detail::enable_if_t<not std::is_same<basic_json_t, ValueType>::value and
                                 detail::has_non_default_from_json<basic_json_t, ValueType>::value,
                                 int> = 0>
    ValueType get() const noexcept(noexcept(
                                       JSONSerializer<ValueTypeCV>::from_json(std::declval<const basic_json_t&>())))
    {
        static_assert(not std::is_reference<ValueTypeCV>::value,
                      "get() cannot be used with reference types, you might want to use get_ref()");
        return JSONSerializer<ValueTypeCV>::from_json(*this);
    }

    /*!
    @brief get a pointer value (explicit)

    Explicit pointer access to the internally stored JSON value. No copies are
    made.

    @warning The pointer becomes invalid if the underlying JSON object
    changes.

    @tparam PointerType pointer type; must be a pointer to @ref array_t, @ref
    object_t, @ref string_t, @ref boolean_t, @ref number_integer_t,
    @ref number_unsigned_t, or @ref number_float_t.

    @return pointer to the internally stored JSON value if the requested
    pointer type @a PointerType fits to the JSON value; `nullptr` otherwise

    @complexity Constant.

    @liveexample{The example below shows how pointers to internal values of a
    JSON value can be requested. Note that no type conversions are made and a
    `nullptr` is returned if the value and the requested pointer type does not
    match.,get__PointerType}

    @sa @ref get_ptr() for explicit pointer-member access

    @since version 1.0.0
    */
    template<typename PointerType, typename std::enable_if<
                 std::is_pointer<PointerType>::value, int>::type = 0>
    PointerType get() noexcept
    {
        // delegate the call to get_ptr
        return get_ptr<PointerType>();
    }

    /*!
    @brief get a pointer value (explicit)
    @copydoc get()
    */
    template<typename PointerType, typename std::enable_if<
                 std::is_pointer<PointerType>::value, int>::type = 0>
    constexpr const PointerType get() const noexcept
    {
        // delegate the call to get_ptr
        return get_ptr<PointerType>();
    }

    /*!
    @brief get a pointer value (implicit)

    Implicit pointer access to the internally stored JSON value. No copies are
    made.

    @warning Writing data to the pointee of the result yields an undefined
    state.

    @tparam PointerType pointer type; must be a pointer to @ref array_t, @ref
    object_t, @ref string_t, @ref boolean_t, @ref number_integer_t,
    @ref number_unsigned_t, or @ref number_float_t. Enforced by a static
    assertion.

    @return pointer to the internally stored JSON value if the requested
    pointer type @a PointerType fits to the JSON value; `nullptr` otherwise

    @complexity Constant.

    @liveexample{The example below shows how pointers to internal values of a
    JSON value can be requested. Note that no type conversions are made and a
    `nullptr` is returned if the value and the requested pointer type does not
    match.,get_ptr}

    @since version 1.0.0
    */
    template<typename PointerType, typename std::enable_if<
                 std::is_pointer<PointerType>::value, int>::type = 0>
    PointerType get_ptr() noexcept
    {
        // get the type of the PointerType (remove pointer and const)
        using pointee_t = typename std::remove_const<typename
                          std::remove_pointer<typename
                          std::remove_const<PointerType>::type>::type>::type;
        // make sure the type matches the allowed types
        static_assert(
            std::is_same<object_t, pointee_t>::value
            or std::is_same<array_t, pointee_t>::value
            or std::is_same<string_t, pointee_t>::value
            or std::is_same<boolean_t, pointee_t>::value
            or std::is_same<number_integer_t, pointee_t>::value
            or std::is_same<number_unsigned_t, pointee_t>::value
            or std::is_same<number_float_t, pointee_t>::value
            , "incompatible pointer type");

        // delegate the call to get_impl_ptr<>()
        return get_impl_ptr(static_cast<PointerType>(nullptr));
    }

    /*!
    @brief get a pointer value (implicit)
    @copydoc get_ptr()
    */
    template<typename PointerType, typename std::enable_if<
                 std::is_pointer<PointerType>::value and
                 std::is_const<typename std::remove_pointer<PointerType>::type>::value, int>::type = 0>
    constexpr const PointerType get_ptr() const noexcept
    {
        // get the type of the PointerType (remove pointer and const)
        using pointee_t = typename std::remove_const<typename
                          std::remove_pointer<typename
                          std::remove_const<PointerType>::type>::type>::type;
        // make sure the type matches the allowed types
        static_assert(
            std::is_same<object_t, pointee_t>::value
            or std::is_same<array_t, pointee_t>::value
            or std::is_same<string_t, pointee_t>::value
            or std::is_same<boolean_t, pointee_t>::value
            or std::is_same<number_integer_t, pointee_t>::value
            or std::is_same<number_unsigned_t, pointee_t>::value
            or std::is_same<number_float_t, pointee_t>::value
            , "incompatible pointer type");

        // delegate the call to get_impl_ptr<>() const
        return get_impl_ptr(static_cast<PointerType>(nullptr));
    }

    /*!
    @brief get a reference value (implicit)

    Implicit reference access to the internally stored JSON value. No copies
    are made.

    @warning Writing data to the referee of the result yields an undefined
    state.

    @tparam ReferenceType reference type; must be a reference to @ref array_t,
    @ref object_t, @ref string_t, @ref boolean_t, @ref number_integer_t, or
    @ref number_float_t. Enforced by static assertion.

    @return reference to the internally stored JSON value if the requested
    reference type @a ReferenceType fits to the JSON value; throws
    type_error.303 otherwise

    @throw type_error.303 in case passed type @a ReferenceType is incompatible
    with the stored JSON value; see example below

    @complexity Constant.

    @liveexample{The example shows several calls to `get_ref()`.,get_ref}

    @since version 1.1.0
    */
    template<typename ReferenceType, typename std::enable_if<
                 std::is_reference<ReferenceType>::value, int>::type = 0>
    ReferenceType get_ref()
    {
        // delegate call to get_ref_impl
        return get_ref_impl<ReferenceType>(*this);
    }

    /*!
    @brief get a reference value (implicit)
    @copydoc get_ref()
    */
    template<typename ReferenceType, typename std::enable_if<
                 std::is_reference<ReferenceType>::value and
                 std::is_const<typename std::remove_reference<ReferenceType>::type>::value, int>::type = 0>
    ReferenceType get_ref() const
    {
        // delegate call to get_ref_impl
        return get_ref_impl<ReferenceType>(*this);
    }

    /*!
    @brief get a value (implicit)

    Implicit type conversion between the JSON value and a compatible value.
    The call is realized by calling @ref get() const.

    @tparam ValueType non-pointer type compatible to the JSON value, for
    instance `int` for JSON integer numbers, `bool` for JSON booleans, or
    `std::vector` types for JSON arrays. The character type of @ref string_t
    as well as an initializer list of this type is excluded to avoid
    ambiguities as these types implicitly convert to `std::string`.

    @return copy of the JSON value, converted to type @a ValueType

    @throw type_error.302 in case passed type @a ValueType is incompatible
    to the JSON value type (e.g., the JSON value is of type boolean, but a
    string is requested); see example below

    @complexity Linear in the size of the JSON value.

    @liveexample{The example below shows several conversions from JSON values
    to other types. There a few things to note: (1) Floating-point numbers can
    be converted to integers\, (2) A JSON array can be converted to a standard
    `std::vector<short>`\, (3) A JSON object can be converted to C++
    associative containers such as `std::unordered_map<std::string\,
    json>`.,operator__ValueType}

    @since version 1.0.0
    */
    template < typename ValueType, typename std::enable_if <
                   not std::is_pointer<ValueType>::value and
                   not std::is_same<ValueType, detail::json_ref<basic_json>>::value and
                   not std::is_same<ValueType, typename string_t::value_type>::value and
                   not detail::is_basic_json<ValueType>::value
#ifndef _MSC_VER  // fix for issue #167 operator<< ambiguity under VS2015
                   and not std::is_same<ValueType, std::initializer_list<typename string_t::value_type>>::value
#if defined(JSON_HAS_CPP_17) && _MSC_VER <= 1914
                   and not std::is_same<ValueType, typename std::string_view>::value
#endif
#endif
                   , int >::type = 0 >
    operator ValueType() const
    {
        // delegate the call to get<>() const
        return get<ValueType>();
    }

    /// @}


    ////////////////////
    // element access //
    ////////////////////

    /// @name element access
    /// Access to the JSON value.
    /// @{

    /*!
    @brief access specified array element with bounds checking

    Returns a reference to the element at specified location @a idx, with
    bounds checking.

    @param[in] idx  index of the element to access

    @return reference to the element at index @a idx

    @throw type_error.304 if the JSON value is not an array; in this case,
    calling `at` with an index makes no sense. See example below.
    @throw out_of_range.401 if the index @a idx is out of range of the array;
    that is, `idx >= size()`. See example below.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes in the JSON value.

    @complexity Constant.

    @since version 1.0.0

    @liveexample{The example below shows how array elements can be read and
    written using `at()`. It also demonstrates the different exceptions that
    can be thrown.,at__size_type}
    */
    reference at(size_type idx)
    {
        // at only works for arrays
        if (JSON_LIKELY(is_array()))
        {
            JSON_TRY
            {
                return m_value.array->at(idx);
            }
            JSON_CATCH (std::out_of_range&)
            {
                // create better exception explanation
                JSON_THROW(out_of_range::create(401, "array index " + std::to_string(idx) + " is out of range"));
            }
        }
        else
        {
            JSON_THROW(type_error::create(304, "cannot use at() with " + std::string(type_name())));
        }
    }

    /*!
    @brief access specified array element with bounds checking

    Returns a const reference to the element at specified location @a idx,
    with bounds checking.

    @param[in] idx  index of the element to access

    @return const reference to the element at index @a idx

    @throw type_error.304 if the JSON value is not an array; in this case,
    calling `at` with an index makes no sense. See example below.
    @throw out_of_range.401 if the index @a idx is out of range of the array;
    that is, `idx >= size()`. See example below.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes in the JSON value.

    @complexity Constant.

    @since version 1.0.0

    @liveexample{The example below shows how array elements can be read using
    `at()`. It also demonstrates the different exceptions that can be thrown.,
    at__size_type_const}
    */
    const_reference at(size_type idx) const
    {
        // at only works for arrays
        if (JSON_LIKELY(is_array()))
        {
            JSON_TRY
            {
                return m_value.array->at(idx);
            }
            JSON_CATCH (std::out_of_range&)
            {
                // create better exception explanation
                JSON_THROW(out_of_range::create(401, "array index " + std::to_string(idx) + " is out of range"));
            }
        }
        else
        {
            JSON_THROW(type_error::create(304, "cannot use at() with " + std::string(type_name())));
        }
    }

    /*!
    @brief access specified object element with bounds checking

    Returns a reference to the element at with specified key @a key, with
    bounds checking.

    @param[in] key  key of the element to access

    @return reference to the element at key @a key

    @throw type_error.304 if the JSON value is not an object; in this case,
    calling `at` with a key makes no sense. See example below.
    @throw out_of_range.403 if the key @a key is is not stored in the object;
    that is, `find(key) == end()`. See example below.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes in the JSON value.

    @complexity Logarithmic in the size of the container.

    @sa @ref operator[](const typename object_t::key_type&) for unchecked
    access by reference
    @sa @ref value() for access by value with a default value

    @since version 1.0.0

    @liveexample{The example below shows how object elements can be read and
    written using `at()`. It also demonstrates the different exceptions that
    can be thrown.,at__object_t_key_type}
    */
    reference at(const typename object_t::key_type& key)
    {
        // at only works for objects
        if (JSON_LIKELY(is_object()))
        {
            JSON_TRY
            {
                return m_value.object->at(key);
            }
            JSON_CATCH (std::out_of_range&)
            {
                // create better exception explanation
                JSON_THROW(out_of_range::create(403, "key '" + key + "' not found"));
            }
        }
        else
        {
            JSON_THROW(type_error::create(304, "cannot use at() with " + std::string(type_name())));
        }
    }

    /*!
    @brief access specified object element with bounds checking

    Returns a const reference to the element at with specified key @a key,
    with bounds checking.

    @param[in] key  key of the element to access

    @return const reference to the element at key @a key

    @throw type_error.304 if the JSON value is not an object; in this case,
    calling `at` with a key makes no sense. See example below.
    @throw out_of_range.403 if the key @a key is is not stored in the object;
    that is, `find(key) == end()`. See example below.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes in the JSON value.

    @complexity Logarithmic in the size of the container.

    @sa @ref operator[](const typename object_t::key_type&) for unchecked
    access by reference
    @sa @ref value() for access by value with a default value

    @since version 1.0.0

    @liveexample{The example below shows how object elements can be read using
    `at()`. It also demonstrates the different exceptions that can be thrown.,
    at__object_t_key_type_const}
    */
    const_reference at(const typename object_t::key_type& key) const
    {
        // at only works for objects
        if (JSON_LIKELY(is_object()))
        {
            JSON_TRY
            {
                return m_value.object->at(key);
            }
            JSON_CATCH (std::out_of_range&)
            {
                // create better exception explanation
                JSON_THROW(out_of_range::create(403, "key '" + key + "' not found"));
            }
        }
        else
        {
            JSON_THROW(type_error::create(304, "cannot use at() with " + std::string(type_name())));
        }
    }

    /*!
    @brief access specified array element

    Returns a reference to the element at specified location @a idx.

    @note If @a idx is beyond the range of the array (i.e., `idx >= size()`),
    then the array is silently filled up with `null` values to make `idx` a
    valid reference to the last stored element.

    @param[in] idx  index of the element to access

    @return reference to the element at index @a idx

    @throw type_error.305 if the JSON value is not an array or null; in that
    cases, using the [] operator with an index makes no sense.

    @complexity Constant if @a idx is in the range of the array. Otherwise
    linear in `idx - size()`.

    @liveexample{The example below shows how array elements can be read and
    written using `[]` operator. Note the addition of `null`
    values.,operatorarray__size_type}

    @since version 1.0.0
    */
    reference operator[](size_type idx)
    {
        // implicitly convert null value to an empty array
        if (is_null())
        {
            m_type = value_t::array;
            m_value.array = create<array_t>();
            assert_invariant();
        }

        // operator[] only works for arrays
        if (JSON_LIKELY(is_array()))
        {
            // fill up array with null values if given idx is outside range
            if (idx >= m_value.array->size())
            {
                m_value.array->insert(m_value.array->end(),
                                      idx - m_value.array->size() + 1,
                                      basic_json());
            }

            return m_value.array->operator[](idx);
        }

        JSON_THROW(type_error::create(305, "cannot use operator[] with " + std::string(type_name())));
    }

    /*!
    @brief access specified array element

    Returns a const reference to the element at specified location @a idx.

    @param[in] idx  index of the element to access

    @return const reference to the element at index @a idx

    @throw type_error.305 if the JSON value is not an array; in that case,
    using the [] operator with an index makes no sense.

    @complexity Constant.

    @liveexample{The example below shows how array elements can be read using
    the `[]` operator.,operatorarray__size_type_const}

    @since version 1.0.0
    */
    const_reference operator[](size_type idx) const
    {
        // const operator[] only works for arrays
        if (JSON_LIKELY(is_array()))
        {
            return m_value.array->operator[](idx);
        }

        JSON_THROW(type_error::create(305, "cannot use operator[] with " + std::string(type_name())));
    }

    /*!
    @brief access specified object element

    Returns a reference to the element at with specified key @a key.

    @note If @a key is not found in the object, then it is silently added to
    the object and filled with a `null` value to make `key` a valid reference.
    In case the value was `null` before, it is converted to an object.

    @param[in] key  key of the element to access

    @return reference to the element at key @a key

    @throw type_error.305 if the JSON value is not an object or null; in that
    cases, using the [] operator with a key makes no sense.

    @complexity Logarithmic in the size of the container.

    @liveexample{The example below shows how object elements can be read and
    written using the `[]` operator.,operatorarray__key_type}

    @sa @ref at(const typename object_t::key_type&) for access by reference
    with range checking
    @sa @ref value() for access by value with a default value

    @since version 1.0.0
    */
    reference operator[](const typename object_t::key_type& key)
    {
        // implicitly convert null value to an empty object
        if (is_null())
        {
            m_type = value_t::object;
            m_value.object = create<object_t>();
            assert_invariant();
        }

        // operator[] only works for objects
        if (JSON_LIKELY(is_object()))
        {
            return m_value.object->operator[](key);
        }

        JSON_THROW(type_error::create(305, "cannot use operator[] with " + std::string(type_name())));
    }

    /*!
    @brief read-only access specified object element

    Returns a const reference to the element at with specified key @a key. No
    bounds checking is performed.

    @warning If the element with key @a key does not exist, the behavior is
    undefined.

    @param[in] key  key of the element to access

    @return const reference to the element at key @a key

    @pre The element with key @a key must exist. **This precondition is
         enforced with an assertion.**

    @throw type_error.305 if the JSON value is not an object; in that case,
    using the [] operator with a key makes no sense.

    @complexity Logarithmic in the size of the container.

    @liveexample{The example below shows how object elements can be read using
    the `[]` operator.,operatorarray__key_type_const}

    @sa @ref at(const typename object_t::key_type&) for access by reference
    with range checking
    @sa @ref value() for access by value with a default value

    @since version 1.0.0
    */
    const_reference operator[](const typename object_t::key_type& key) const
    {
        // const operator[] only works for objects
        if (JSON_LIKELY(is_object()))
        {
            assert(m_value.object->find(key) != m_value.object->end());
            return m_value.object->find(key)->second;
        }

        JSON_THROW(type_error::create(305, "cannot use operator[] with " + std::string(type_name())));
    }

    /*!
    @brief access specified object element

    Returns a reference to the element at with specified key @a key.

    @note If @a key is not found in the object, then it is silently added to
    the object and filled with a `null` value to make `key` a valid reference.
    In case the value was `null` before, it is converted to an object.

    @param[in] key  key of the element to access

    @return reference to the element at key @a key

    @throw type_error.305 if the JSON value is not an object or null; in that
    cases, using the [] operator with a key makes no sense.

    @complexity Logarithmic in the size of the container.

    @liveexample{The example below shows how object elements can be read and
    written using the `[]` operator.,operatorarray__key_type}

    @sa @ref at(const typename object_t::key_type&) for access by reference
    with range checking
    @sa @ref value() for access by value with a default value

    @since version 1.1.0
    */
    template<typename T>
    reference operator[](T* key)
    {
        // implicitly convert null to object
        if (is_null())
        {
            m_type = value_t::object;
            m_value = value_t::object;
            assert_invariant();
        }

        // at only works for objects
        if (JSON_LIKELY(is_object()))
        {
            return m_value.object->operator[](key);
        }

        JSON_THROW(type_error::create(305, "cannot use operator[] with " + std::string(type_name())));
    }

    /*!
    @brief read-only access specified object element

    Returns a const reference to the element at with specified key @a key. No
    bounds checking is performed.

    @warning If the element with key @a key does not exist, the behavior is
    undefined.

    @param[in] key  key of the element to access

    @return const reference to the element at key @a key

    @pre The element with key @a key must exist. **This precondition is
         enforced with an assertion.**

    @throw type_error.305 if the JSON value is not an object; in that case,
    using the [] operator with a key makes no sense.

    @complexity Logarithmic in the size of the container.

    @liveexample{The example below shows how object elements can be read using
    the `[]` operator.,operatorarray__key_type_const}

    @sa @ref at(const typename object_t::key_type&) for access by reference
    with range checking
    @sa @ref value() for access by value with a default value

    @since version 1.1.0
    */
    template<typename T>
    const_reference operator[](T* key) const
    {
        // at only works for objects
        if (JSON_LIKELY(is_object()))
        {
            assert(m_value.object->find(key) != m_value.object->end());
            return m_value.object->find(key)->second;
        }

        JSON_THROW(type_error::create(305, "cannot use operator[] with " + std::string(type_name())));
    }

    /*!
    @brief access specified object element with default value

    Returns either a copy of an object's element at the specified key @a key
    or a given default value if no element with key @a key exists.

    The function is basically equivalent to executing
    @code {.cpp}
    try {
        return at(key);
    } catch(out_of_range) {
        return default_value;
    }
    @endcode

    @note Unlike @ref at(const typename object_t::key_type&), this function
    does not throw if the given key @a key was not found.

    @note Unlike @ref operator[](const typename object_t::key_type& key), this
    function does not implicitly add an element to the position defined by @a
    key. This function is furthermore also applicable to const objects.

    @param[in] key  key of the element to access
    @param[in] default_value  the value to return if @a key is not found

    @tparam ValueType type compatible to JSON values, for instance `int` for
    JSON integer numbers, `bool` for JSON booleans, or `std::vector` types for
    JSON arrays. Note the type of the expected value at @a key and the default
    value @a default_value must be compatible.

    @return copy of the element at key @a key or @a default_value if @a key
    is not found

    @throw type_error.306 if the JSON value is not an object; in that case,
    using `value()` with a key makes no sense.

    @complexity Logarithmic in the size of the container.

    @liveexample{The example below shows how object elements can be queried
    with a default value.,basic_json__value}

    @sa @ref at(const typename object_t::key_type&) for access by reference
    with range checking
    @sa @ref operator[](const typename object_t::key_type&) for unchecked
    access by reference

    @since version 1.0.0
    */
    template<class ValueType, typename std::enable_if<
                 std::is_convertible<basic_json_t, ValueType>::value, int>::type = 0>
    ValueType value(const typename object_t::key_type& key, const ValueType& default_value) const
    {
        // at only works for objects
        if (JSON_LIKELY(is_object()))
        {
            // if key is found, return value and given default value otherwise
            const auto it = find(key);
            if (it != end())
            {
                return *it;
            }

            return default_value;
        }

        JSON_THROW(type_error::create(306, "cannot use value() with " + std::string(type_name())));
    }

    /*!
    @brief overload for a default value of type const char*
    @copydoc basic_json::value(const typename object_t::key_type&, ValueType) const
    */
    string_t value(const typename object_t::key_type& key, const char* default_value) const
    {
        return value(key, string_t(default_value));
    }

    /*!
    @brief access specified object element via JSON Pointer with default value

    Returns either a copy of an object's element at the specified key @a key
    or a given default value if no element with key @a key exists.

    The function is basically equivalent to executing
    @code {.cpp}
    try {
        return at(ptr);
    } catch(out_of_range) {
        return default_value;
    }
    @endcode

    @note Unlike @ref at(const json_pointer&), this function does not throw
    if the given key @a key was not found.

    @param[in] ptr  a JSON pointer to the element to access
    @param[in] default_value  the value to return if @a ptr found no value

    @tparam ValueType type compatible to JSON values, for instance `int` for
    JSON integer numbers, `bool` for JSON booleans, or `std::vector` types for
    JSON arrays. Note the type of the expected value at @a key and the default
    value @a default_value must be compatible.

    @return copy of the element at key @a key or @a default_value if @a key
    is not found

    @throw type_error.306 if the JSON value is not an object; in that case,
    using `value()` with a key makes no sense.

    @complexity Logarithmic in the size of the container.

    @liveexample{The example below shows how object elements can be queried
    with a default value.,basic_json__value_ptr}

    @sa @ref operator[](const json_pointer&) for unchecked access by reference

    @since version 2.0.2
    */
    template<class ValueType, typename std::enable_if<
                 std::is_convertible<basic_json_t, ValueType>::value, int>::type = 0>
    ValueType value(const json_pointer& ptr, const ValueType& default_value) const
    {
        // at only works for objects
        if (JSON_LIKELY(is_object()))
        {
            // if pointer resolves a value, return it or use default value
            JSON_TRY
            {
                return ptr.get_checked(this);
            }
            JSON_CATCH (out_of_range&)
            {
                return default_value;
            }
        }

        JSON_THROW(type_error::create(306, "cannot use value() with " + std::string(type_name())));
    }

    /*!
    @brief overload for a default value of type const char*
    @copydoc basic_json::value(const json_pointer&, ValueType) const
    */
    string_t value(const json_pointer& ptr, const char* default_value) const
    {
        return value(ptr, string_t(default_value));
    }

    /*!
    @brief access the first element

    Returns a reference to the first element in the container. For a JSON
    container `c`, the expression `c.front()` is equivalent to `*c.begin()`.

    @return In case of a structured type (array or object), a reference to the
    first element is returned. In case of number, string, or boolean values, a
    reference to the value is returned.

    @complexity Constant.

    @pre The JSON value must not be `null` (would throw `std::out_of_range`)
    or an empty array or object (undefined behavior, **guarded by
    assertions**).
    @post The JSON value remains unchanged.

    @throw invalid_iterator.214 when called on `null` value

    @liveexample{The following code shows an example for `front()`.,front}

    @sa @ref back() -- access the last element

    @since version 1.0.0
    */
    reference front()
    {
        return *begin();
    }

    /*!
    @copydoc basic_json::front()
    */
    const_reference front() const
    {
        return *cbegin();
    }

    /*!
    @brief access the last element

    Returns a reference to the last element in the container. For a JSON
    container `c`, the expression `c.back()` is equivalent to
    @code {.cpp}
    auto tmp = c.end();
    --tmp;
    return *tmp;
    @endcode

    @return In case of a structured type (array or object), a reference to the
    last element is returned. In case of number, string, or boolean values, a
    reference to the value is returned.

    @complexity Constant.

    @pre The JSON value must not be `null` (would throw `std::out_of_range`)
    or an empty array or object (undefined behavior, **guarded by
    assertions**).
    @post The JSON value remains unchanged.

    @throw invalid_iterator.214 when called on a `null` value. See example
    below.

    @liveexample{The following code shows an example for `back()`.,back}

    @sa @ref front() -- access the first element

    @since version 1.0.0
    */
    reference back()
    {
        auto tmp = end();
        --tmp;
        return *tmp;
    }

    /*!
    @copydoc basic_json::back()
    */
    const_reference back() const
    {
        auto tmp = cend();
        --tmp;
        return *tmp;
    }

    /*!
    @brief remove element given an iterator

    Removes the element specified by iterator @a pos. The iterator @a pos must
    be valid and dereferenceable. Thus the `end()` iterator (which is valid,
    but is not dereferenceable) cannot be used as a value for @a pos.

    If called on a primitive type other than `null`, the resulting JSON value
    will be `null`.

    @param[in] pos iterator to the element to remove
    @return Iterator following the last removed element. If the iterator @a
    pos refers to the last element, the `end()` iterator is returned.

    @tparam IteratorType an @ref iterator or @ref const_iterator

    @post Invalidates iterators and references at or after the point of the
    erase, including the `end()` iterator.

    @throw type_error.307 if called on a `null` value; example: `"cannot use
    erase() with null"`
    @throw invalid_iterator.202 if called on an iterator which does not belong
    to the current JSON value; example: `"iterator does not fit current
    value"`
    @throw invalid_iterator.205 if called on a primitive type with invalid
    iterator (i.e., any iterator which is not `begin()`); example: `"iterator
    out of range"`

    @complexity The complexity depends on the type:
    - objects: amortized constant
    - arrays: linear in distance between @a pos and the end of the container
    - strings: linear in the length of the string
    - other types: constant

    @liveexample{The example shows the result of `erase()` for different JSON
    types.,erase__IteratorType}

    @sa @ref erase(IteratorType, IteratorType) -- removes the elements in
    the given range
    @sa @ref erase(const typename object_t::key_type&) -- removes the element
    from an object at the given key
    @sa @ref erase(const size_type) -- removes the element from an array at
    the given index

    @since version 1.0.0
    */
    template<class IteratorType, typename std::enable_if<
                 std::is_same<IteratorType, typename basic_json_t::iterator>::value or
                 std::is_same<IteratorType, typename basic_json_t::const_iterator>::value, int>::type
             = 0>
    IteratorType erase(IteratorType pos)
    {
        // make sure iterator fits the current value
        if (JSON_UNLIKELY(this != pos.m_object))
        {
            JSON_THROW(invalid_iterator::create(202, "iterator does not fit current value"));
        }

        IteratorType result = end();

        switch (m_type)
        {
            case value_t::boolean:
            case value_t::number_float:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::string:
            {
                if (JSON_UNLIKELY(not pos.m_it.primitive_iterator.is_begin()))
                {
                    JSON_THROW(invalid_iterator::create(205, "iterator out of range"));
                }

                if (is_string())
                {
                    AllocatorType<string_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, m_value.string);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, m_value.string, 1);
                    m_value.string = nullptr;
                }

                m_type = value_t::null;
                assert_invariant();
                break;
            }

            case value_t::object:
            {
                result.m_it.object_iterator = m_value.object->erase(pos.m_it.object_iterator);
                break;
            }

            case value_t::array:
            {
                result.m_it.array_iterator = m_value.array->erase(pos.m_it.array_iterator);
                break;
            }

            default:
                JSON_THROW(type_error::create(307, "cannot use erase() with " + std::string(type_name())));
        }

        return result;
    }

    /*!
    @brief remove elements given an iterator range

    Removes the element specified by the range `[first; last)`. The iterator
    @a first does not need to be dereferenceable if `first == last`: erasing
    an empty range is a no-op.

    If called on a primitive type other than `null`, the resulting JSON value
    will be `null`.

    @param[in] first iterator to the beginning of the range to remove
    @param[in] last iterator past the end of the range to remove
    @return Iterator following the last removed element. If the iterator @a
    second refers to the last element, the `end()` iterator is returned.

    @tparam IteratorType an @ref iterator or @ref const_iterator

    @post Invalidates iterators and references at or after the point of the
    erase, including the `end()` iterator.

    @throw type_error.307 if called on a `null` value; example: `"cannot use
    erase() with null"`
    @throw invalid_iterator.203 if called on iterators which does not belong
    to the current JSON value; example: `"iterators do not fit current value"`
    @throw invalid_iterator.204 if called on a primitive type with invalid
    iterators (i.e., if `first != begin()` and `last != end()`); example:
    `"iterators out of range"`

    @complexity The complexity depends on the type:
    - objects: `log(size()) + std::distance(first, last)`
    - arrays: linear in the distance between @a first and @a last, plus linear
      in the distance between @a last and end of the container
    - strings: linear in the length of the string
    - other types: constant

    @liveexample{The example shows the result of `erase()` for different JSON
    types.,erase__IteratorType_IteratorType}

    @sa @ref erase(IteratorType) -- removes the element at a given position
    @sa @ref erase(const typename object_t::key_type&) -- removes the element
    from an object at the given key
    @sa @ref erase(const size_type) -- removes the element from an array at
    the given index

    @since version 1.0.0
    */
    template<class IteratorType, typename std::enable_if<
                 std::is_same<IteratorType, typename basic_json_t::iterator>::value or
                 std::is_same<IteratorType, typename basic_json_t::const_iterator>::value, int>::type
             = 0>
    IteratorType erase(IteratorType first, IteratorType last)
    {
        // make sure iterator fits the current value
        if (JSON_UNLIKELY(this != first.m_object or this != last.m_object))
        {
            JSON_THROW(invalid_iterator::create(203, "iterators do not fit current value"));
        }

        IteratorType result = end();

        switch (m_type)
        {
            case value_t::boolean:
            case value_t::number_float:
            case value_t::number_integer:
            case value_t::number_unsigned:
            case value_t::string:
            {
                if (JSON_LIKELY(not first.m_it.primitive_iterator.is_begin()
                                or not last.m_it.primitive_iterator.is_end()))
                {
                    JSON_THROW(invalid_iterator::create(204, "iterators out of range"));
                }

                if (is_string())
                {
                    AllocatorType<string_t> alloc;
                    std::allocator_traits<decltype(alloc)>::destroy(alloc, m_value.string);
                    std::allocator_traits<decltype(alloc)>::deallocate(alloc, m_value.string, 1);
                    m_value.string = nullptr;
                }

                m_type = value_t::null;
                assert_invariant();
                break;
            }

            case value_t::object:
            {
                result.m_it.object_iterator = m_value.object->erase(first.m_it.object_iterator,
                                              last.m_it.object_iterator);
                break;
            }

            case value_t::array:
            {
                result.m_it.array_iterator = m_value.array->erase(first.m_it.array_iterator,
                                             last.m_it.array_iterator);
                break;
            }

            default:
                JSON_THROW(type_error::create(307, "cannot use erase() with " + std::string(type_name())));
        }

        return result;
    }

    /*!
    @brief remove element from a JSON object given a key

    Removes elements from a JSON object with the key value @a key.

    @param[in] key value of the elements to remove

    @return Number of elements removed. If @a ObjectType is the default
    `std::map` type, the return value will always be `0` (@a key was not
    found) or `1` (@a key was found).

    @post References and iterators to the erased elements are invalidated.
    Other references and iterators are not affected.

    @throw type_error.307 when called on a type other than JSON object;
    example: `"cannot use erase() with null"`

    @complexity `log(size()) + count(key)`

    @liveexample{The example shows the effect of `erase()`.,erase__key_type}

    @sa @ref erase(IteratorType) -- removes the element at a given position
    @sa @ref erase(IteratorType, IteratorType) -- removes the elements in
    the given range
    @sa @ref erase(const size_type) -- removes the element from an array at
    the given index

    @since version 1.0.0
    */
    size_type erase(const typename object_t::key_type& key)
    {
        // this erase only works for objects
        if (JSON_LIKELY(is_object()))
        {
            return m_value.object->erase(key);
        }

        JSON_THROW(type_error::create(307, "cannot use erase() with " + std::string(type_name())));
    }

    /*!
    @brief remove element from a JSON array given an index

    Removes element from a JSON array at the index @a idx.

    @param[in] idx index of the element to remove

    @throw type_error.307 when called on a type other than JSON object;
    example: `"cannot use erase() with null"`
    @throw out_of_range.401 when `idx >= size()`; example: `"array index 17
    is out of range"`

    @complexity Linear in distance between @a idx and the end of the container.

    @liveexample{The example shows the effect of `erase()`.,erase__size_type}

    @sa @ref erase(IteratorType) -- removes the element at a given position
    @sa @ref erase(IteratorType, IteratorType) -- removes the elements in
    the given range
    @sa @ref erase(const typename object_t::key_type&) -- removes the element
    from an object at the given key

    @since version 1.0.0
    */
    void erase(const size_type idx)
    {
        // this erase only works for arrays
        if (JSON_LIKELY(is_array()))
        {
            if (JSON_UNLIKELY(idx >= size()))
            {
                JSON_THROW(out_of_range::create(401, "array index " + std::to_string(idx) + " is out of range"));
            }

            m_value.array->erase(m_value.array->begin() + static_cast<difference_type>(idx));
        }
        else
        {
            JSON_THROW(type_error::create(307, "cannot use erase() with " + std::string(type_name())));
        }
    }

    /// @}


    ////////////
    // lookup //
    ////////////

    /// @name lookup
    /// @{

    /*!
    @brief find an element in a JSON object

    Finds an element in a JSON object with key equivalent to @a key. If the
    element is not found or the JSON value is not an object, end() is
    returned.

    @note This method always returns @ref end() when executed on a JSON type
          that is not an object.

    @param[in] key key value of the element to search for.

    @return Iterator to an element with key equivalent to @a key. If no such
    element is found or the JSON value is not an object, past-the-end (see
    @ref end()) iterator is returned.

    @complexity Logarithmic in the size of the JSON object.

    @liveexample{The example shows how `find()` is used.,find__key_type}

    @since version 1.0.0
    */
    template<typename KeyT>
    iterator find(KeyT&& key)
    {
        auto result = end();

        if (is_object())
        {
            result.m_it.object_iterator = m_value.object->find(std::forward<KeyT>(key));
        }

        return result;
    }

    /*!
    @brief find an element in a JSON object
    @copydoc find(KeyT&&)
    */
    template<typename KeyT>
    const_iterator find(KeyT&& key) const
    {
        auto result = cend();

        if (is_object())
        {
            result.m_it.object_iterator = m_value.object->find(std::forward<KeyT>(key));
        }

        return result;
    }

    /*!
    @brief returns the number of occurrences of a key in a JSON object

    Returns the number of elements with key @a key. If ObjectType is the
    default `std::map` type, the return value will always be `0` (@a key was
    not found) or `1` (@a key was found).

    @note This method always returns `0` when executed on a JSON type that is
          not an object.

    @param[in] key key value of the element to count

    @return Number of elements with key @a key. If the JSON value is not an
    object, the return value will be `0`.

    @complexity Logarithmic in the size of the JSON object.

    @liveexample{The example shows how `count()` is used.,count}

    @since version 1.0.0
    */
    template<typename KeyT>
    size_type count(KeyT&& key) const
    {
        // return 0 for all nonobject types
        return is_object() ? m_value.object->count(std::forward<KeyT>(key)) : 0;
    }

    /// @}


    ///////////////
    // iterators //
    ///////////////

    /// @name iterators
    /// @{

    /*!
    @brief returns an iterator to the first element

    Returns an iterator to the first element.

    @image html range-begin-end.svg "Illustration from cppreference.com"

    @return iterator to the first element

    @complexity Constant.

    @requirement This function helps `basic_json` satisfying the
    [Container](https://en.cppreference.com/w/cpp/named_req/Container)
    requirements:
    - The complexity is constant.

    @liveexample{The following code shows an example for `begin()`.,begin}

    @sa @ref cbegin() -- returns a const iterator to the beginning
    @sa @ref end() -- returns an iterator to the end
    @sa @ref cend() -- returns a const iterator to the end

    @since version 1.0.0
    */
    iterator begin() noexcept
    {
        iterator result(this);
        result.set_begin();
        return result;
    }

    /*!
    @copydoc basic_json::cbegin()
    */
    const_iterator begin() const noexcept
    {
        return cbegin();
    }

    /*!
    @brief returns a const iterator to the first element

    Returns a const iterator to the first element.

    @image html range-begin-end.svg "Illustration from cppreference.com"

    @return const iterator to the first element

    @complexity Constant.

    @requirement This function helps `basic_json` satisfying the
    [Container](https://en.cppreference.com/w/cpp/named_req/Container)
    requirements:
    - The complexity is constant.
    - Has the semantics of `const_cast<const basic_json&>(*this).begin()`.

    @liveexample{The following code shows an example for `cbegin()`.,cbegin}

    @sa @ref begin() -- returns an iterator to the beginning
    @sa @ref end() -- returns an iterator to the end
    @sa @ref cend() -- returns a const iterator to the end

    @since version 1.0.0
    */
    const_iterator cbegin() const noexcept
    {
        const_iterator result(this);
        result.set_begin();
        return result;
    }

    /*!
    @brief returns an iterator to one past the last element

    Returns an iterator to one past the last element.

    @image html range-begin-end.svg "Illustration from cppreference.com"

    @return iterator one past the last element

    @complexity Constant.

    @requirement This function helps `basic_json` satisfying the
    [Container](https://en.cppreference.com/w/cpp/named_req/Container)
    requirements:
    - The complexity is constant.

    @liveexample{The following code shows an example for `end()`.,end}

    @sa @ref cend() -- returns a const iterator to the end
    @sa @ref begin() -- returns an iterator to the beginning
    @sa @ref cbegin() -- returns a const iterator to the beginning

    @since version 1.0.0
    */
    iterator end() noexcept
    {
        iterator result(this);
        result.set_end();
        return result;
    }

    /*!
    @copydoc basic_json::cend()
    */
    const_iterator end() const noexcept
    {
        return cend();
    }

    /*!
    @brief returns a const iterator to one past the last element

    Returns a const iterator to one past the last element.

    @image html range-begin-end.svg "Illustration from cppreference.com"

    @return const iterator one past the last element

    @complexity Constant.

    @requirement This function helps `basic_json` satisfying the
    [Container](https://en.cppreference.com/w/cpp/named_req/Container)
    requirements:
    - The complexity is constant.
    - Has the semantics of `const_cast<const basic_json&>(*this).end()`.

    @liveexample{The following code shows an example for `cend()`.,cend}

    @sa @ref end() -- returns an iterator to the end
    @sa @ref begin() -- returns an iterator to the beginning
    @sa @ref cbegin() -- returns a const iterator to the beginning

    @since version 1.0.0
    */
    const_iterator cend() const noexcept
    {
        const_iterator result(this);
        result.set_end();
        return result;
    }

    /*!
    @brief returns an iterator to the reverse-beginning

    Returns an iterator to the reverse-beginning; that is, the last element.

    @image html range-rbegin-rend.svg "Illustration from cppreference.com"

    @complexity Constant.

    @requirement This function helps `basic_json` satisfying the
    [ReversibleContainer](https://en.cppreference.com/w/cpp/named_req/ReversibleContainer)
    requirements:
    - The complexity is constant.
    - Has the semantics of `reverse_iterator(end())`.

    @liveexample{The following code shows an example for `rbegin()`.,rbegin}

    @sa @ref crbegin() -- returns a const reverse iterator to the beginning
    @sa @ref rend() -- returns a reverse iterator to the end
    @sa @ref crend() -- returns a const reverse iterator to the end

    @since version 1.0.0
    */
    reverse_iterator rbegin() noexcept
    {
        return reverse_iterator(end());
    }

    /*!
    @copydoc basic_json::crbegin()
    */
    const_reverse_iterator rbegin() const noexcept
    {
        return crbegin();
    }

    /*!
    @brief returns an iterator to the reverse-end

    Returns an iterator to the reverse-end; that is, one before the first
    element.

    @image html range-rbegin-rend.svg "Illustration from cppreference.com"

    @complexity Constant.

    @requirement This function helps `basic_json` satisfying the
    [ReversibleContainer](https://en.cppreference.com/w/cpp/named_req/ReversibleContainer)
    requirements:
    - The complexity is constant.
    - Has the semantics of `reverse_iterator(begin())`.

    @liveexample{The following code shows an example for `rend()`.,rend}

    @sa @ref crend() -- returns a const reverse iterator to the end
    @sa @ref rbegin() -- returns a reverse iterator to the beginning
    @sa @ref crbegin() -- returns a const reverse iterator to the beginning

    @since version 1.0.0
    */
    reverse_iterator rend() noexcept
    {
        return reverse_iterator(begin());
    }

    /*!
    @copydoc basic_json::crend()
    */
    const_reverse_iterator rend() const noexcept
    {
        return crend();
    }

    /*!
    @brief returns a const reverse iterator to the last element

    Returns a const iterator to the reverse-beginning; that is, the last
    element.

    @image html range-rbegin-rend.svg "Illustration from cppreference.com"

    @complexity Constant.

    @requirement This function helps `basic_json` satisfying the
    [ReversibleContainer](https://en.cppreference.com/w/cpp/named_req/ReversibleContainer)
    requirements:
    - The complexity is constant.
    - Has the semantics of `const_cast<const basic_json&>(*this).rbegin()`.

    @liveexample{The following code shows an example for `crbegin()`.,crbegin}

    @sa @ref rbegin() -- returns a reverse iterator to the beginning
    @sa @ref rend() -- returns a reverse iterator to the end
    @sa @ref crend() -- returns a const reverse iterator to the end

    @since version 1.0.0
    */
    const_reverse_iterator crbegin() const noexcept
    {
        return const_reverse_iterator(cend());
    }

    /*!
    @brief returns a const reverse iterator to one before the first

    Returns a const reverse iterator to the reverse-end; that is, one before
    the first element.

    @image html range-rbegin-rend.svg "Illustration from cppreference.com"

    @complexity Constant.

    @requirement This function helps `basic_json` satisfying the
    [ReversibleContainer](https://en.cppreference.com/w/cpp/named_req/ReversibleContainer)
    requirements:
    - The complexity is constant.
    - Has the semantics of `const_cast<const basic_json&>(*this).rend()`.

    @liveexample{The following code shows an example for `crend()`.,crend}

    @sa @ref rend() -- returns a reverse iterator to the end
    @sa @ref rbegin() -- returns a reverse iterator to the beginning
    @sa @ref crbegin() -- returns a const reverse iterator to the beginning

    @since version 1.0.0
    */
    const_reverse_iterator crend() const noexcept
    {
        return const_reverse_iterator(cbegin());
    }

  public:
    /*!
    @brief wrapper to access iterator member functions in range-based for

    This function allows to access @ref iterator::key() and @ref
    iterator::value() during range-based for loops. In these loops, a
    reference to the JSON values is returned, so there is no access to the
    underlying iterator.

    For loop without iterator_wrapper:

    @code{cpp}
    for (auto it = j_object.begin(); it != j_object.end(); ++it)
    {
        std::cout << "key: " << it.key() << ", value:" << it.value() << '\n';
    }
    @endcode

    Range-based for loop without iterator proxy:

    @code{cpp}
    for (auto it : j_object)
    {
        // "it" is of type json::reference and has no key() member
        std::cout << "value: " << it << '\n';
    }
    @endcode

    Range-based for loop with iterator proxy:

    @code{cpp}
    for (auto it : json::iterator_wrapper(j_object))
    {
        std::cout << "key: " << it.key() << ", value:" << it.value() << '\n';
    }
    @endcode

    @note When iterating over an array, `key()` will return the index of the
          element as string (see example).

    @param[in] ref  reference to a JSON value
    @return iteration proxy object wrapping @a ref with an interface to use in
            range-based for loops

    @liveexample{The following code shows how the wrapper is used,iterator_wrapper}

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes in the JSON value.

    @complexity Constant.

    @note The name of this function is not yet final and may change in the
    future.

    @deprecated This stream operator is deprecated and will be removed in
                future 4.0.0 of the library. Please use @ref items() instead;
                that is, replace `json::iterator_wrapper(j)` with `j.items()`.
    */
    JSON_DEPRECATED
    static iteration_proxy<iterator> iterator_wrapper(reference ref) noexcept
    {
        return ref.items();
    }

    /*!
    @copydoc iterator_wrapper(reference)
    */
    JSON_DEPRECATED
    static iteration_proxy<const_iterator> iterator_wrapper(const_reference ref) noexcept
    {
        return ref.items();
    }

    /*!
    @brief helper to access iterator member functions in range-based for

    This function allows to access @ref iterator::key() and @ref
    iterator::value() during range-based for loops. In these loops, a
    reference to the JSON values is returned, so there is no access to the
    underlying iterator.

    For loop without `items()` function:

    @code{cpp}
    for (auto it = j_object.begin(); it != j_object.end(); ++it)
    {
        std::cout << "key: " << it.key() << ", value:" << it.value() << '\n';
    }
    @endcode

    Range-based for loop without `items()` function:

    @code{cpp}
    for (auto it : j_object)
    {
        // "it" is of type json::reference and has no key() member
        std::cout << "value: " << it << '\n';
    }
    @endcode

    Range-based for loop with `items()` function:

    @code{cpp}
    for (auto it : j_object.items())
    {
        std::cout << "key: " << it.key() << ", value:" << it.value() << '\n';
    }
    @endcode

    @note When iterating over an array, `key()` will return the index of the
          element as string (see example). For primitive types (e.g., numbers),
          `key()` returns an empty string.

    @return iteration proxy object wrapping @a ref with an interface to use in
            range-based for loops

    @liveexample{The following code shows how the function is used.,items}

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes in the JSON value.

    @complexity Constant.

    @since version 3.1.0.
    */
    iteration_proxy<iterator> items() noexcept
    {
        return iteration_proxy<iterator>(*this);
    }

    /*!
    @copydoc items()
    */
    iteration_proxy<const_iterator> items() const noexcept
    {
        return iteration_proxy<const_iterator>(*this);
    }

    /// @}


    //////////////
    // capacity //
    //////////////

    /// @name capacity
    /// @{

    /*!
    @brief checks whether the container is empty.

    Checks if a JSON value has no elements (i.e. whether its @ref size is `0`).

    @return The return value depends on the different types and is
            defined as follows:
            Value type  | return value
            ----------- | -------------
            null        | `true`
            boolean     | `false`
            string      | `false`
            number      | `false`
            object      | result of function `object_t::empty()`
            array       | result of function `array_t::empty()`

    @liveexample{The following code uses `empty()` to check if a JSON
    object contains any elements.,empty}

    @complexity Constant, as long as @ref array_t and @ref object_t satisfy
    the Container concept; that is, their `empty()` functions have constant
    complexity.

    @iterators No changes.

    @exceptionsafety No-throw guarantee: this function never throws exceptions.

    @note This function does not return whether a string stored as JSON value
    is empty - it returns whether the JSON container itself is empty which is
    false in the case of a string.

    @requirement This function helps `basic_json` satisfying the
    [Container](https://en.cppreference.com/w/cpp/named_req/Container)
    requirements:
    - The complexity is constant.
    - Has the semantics of `begin() == end()`.

    @sa @ref size() -- returns the number of elements

    @since version 1.0.0
    */
    bool empty() const noexcept
    {
        switch (m_type)
        {
            case value_t::null:
            {
                // null values are empty
                return true;
            }

            case value_t::array:
            {
                // delegate call to array_t::empty()
                return m_value.array->empty();
            }

            case value_t::object:
            {
                // delegate call to object_t::empty()
                return m_value.object->empty();
            }

            default:
            {
                // all other types are nonempty
                return false;
            }
        }
    }

    /*!
    @brief returns the number of elements

    Returns the number of elements in a JSON value.

    @return The return value depends on the different types and is
            defined as follows:
            Value type  | return value
            ----------- | -------------
            null        | `0`
            boolean     | `1`
            string      | `1`
            number      | `1`
            object      | result of function object_t::size()
            array       | result of function array_t::size()

    @liveexample{The following code calls `size()` on the different value
    types.,size}

    @complexity Constant, as long as @ref array_t and @ref object_t satisfy
    the Container concept; that is, their size() functions have constant
    complexity.

    @iterators No changes.

    @exceptionsafety No-throw guarantee: this function never throws exceptions.

    @note This function does not return the length of a string stored as JSON
    value - it returns the number of elements in the JSON value which is 1 in
    the case of a string.

    @requirement This function helps `basic_json` satisfying the
    [Container](https://en.cppreference.com/w/cpp/named_req/Container)
    requirements:
    - The complexity is constant.
    - Has the semantics of `std::distance(begin(), end())`.

    @sa @ref empty() -- checks whether the container is empty
    @sa @ref max_size() -- returns the maximal number of elements

    @since version 1.0.0
    */
    size_type size() const noexcept
    {
        switch (m_type)
        {
            case value_t::null:
            {
                // null values are empty
                return 0;
            }

            case value_t::array:
            {
                // delegate call to array_t::size()
                return m_value.array->size();
            }

            case value_t::object:
            {
                // delegate call to object_t::size()
                return m_value.object->size();
            }

            default:
            {
                // all other types have size 1
                return 1;
            }
        }
    }

    /*!
    @brief returns the maximum possible number of elements

    Returns the maximum number of elements a JSON value is able to hold due to
    system or library implementation limitations, i.e. `std::distance(begin(),
    end())` for the JSON value.

    @return The return value depends on the different types and is
            defined as follows:
            Value type  | return value
            ----------- | -------------
            null        | `0` (same as `size()`)
            boolean     | `1` (same as `size()`)
            string      | `1` (same as `size()`)
            number      | `1` (same as `size()`)
            object      | result of function `object_t::max_size()`
            array       | result of function `array_t::max_size()`

    @liveexample{The following code calls `max_size()` on the different value
    types. Note the output is implementation specific.,max_size}

    @complexity Constant, as long as @ref array_t and @ref object_t satisfy
    the Container concept; that is, their `max_size()` functions have constant
    complexity.

    @iterators No changes.

    @exceptionsafety No-throw guarantee: this function never throws exceptions.

    @requirement This function helps `basic_json` satisfying the
    [Container](https://en.cppreference.com/w/cpp/named_req/Container)
    requirements:
    - The complexity is constant.
    - Has the semantics of returning `b.size()` where `b` is the largest
      possible JSON value.

    @sa @ref size() -- returns the number of elements

    @since version 1.0.0
    */
    size_type max_size() const noexcept
    {
        switch (m_type)
        {
            case value_t::array:
            {
                // delegate call to array_t::max_size()
                return m_value.array->max_size();
            }

            case value_t::object:
            {
                // delegate call to object_t::max_size()
                return m_value.object->max_size();
            }

            default:
            {
                // all other types have max_size() == size()
                return size();
            }
        }
    }

    /// @}


    ///////////////
    // modifiers //
    ///////////////

    /// @name modifiers
    /// @{

    /*!
    @brief clears the contents

    Clears the content of a JSON value and resets it to the default value as
    if @ref basic_json(value_t) would have been called with the current value
    type from @ref type():

    Value type  | initial value
    ----------- | -------------
    null        | `null`
    boolean     | `false`
    string      | `""`
    number      | `0`
    object      | `{}`
    array       | `[]`

    @post Has the same effect as calling
    @code {.cpp}
    *this = basic_json(type());
    @endcode

    @liveexample{The example below shows the effect of `clear()` to different
    JSON types.,clear}

    @complexity Linear in the size of the JSON value.

    @iterators All iterators, pointers and references related to this container
               are invalidated.

    @exceptionsafety No-throw guarantee: this function never throws exceptions.

    @sa @ref basic_json(value_t) -- constructor that creates an object with the
        same value than calling `clear()`

    @since version 1.0.0
    */
    void clear() noexcept
    {
        switch (m_type)
        {
            case value_t::number_integer:
            {
                m_value.number_integer = 0;
                break;
            }

            case value_t::number_unsigned:
            {
                m_value.number_unsigned = 0;
                break;
            }

            case value_t::number_float:
            {
                m_value.number_float = 0.0;
                break;
            }

            case value_t::boolean:
            {
                m_value.boolean = false;
                break;
            }

            case value_t::string:
            {
                m_value.string->clear();
                break;
            }

            case value_t::array:
            {
                m_value.array->clear();
                break;
            }

            case value_t::object:
            {
                m_value.object->clear();
                break;
            }

            default:
                break;
        }
    }

    /*!
    @brief add an object to an array

    Appends the given element @a val to the end of the JSON value. If the
    function is called on a JSON null value, an empty array is created before
    appending @a val.

    @param[in] val the value to add to the JSON array

    @throw type_error.308 when called on a type other than JSON array or
    null; example: `"cannot use push_back() with number"`

    @complexity Amortized constant.

    @liveexample{The example shows how `push_back()` and `+=` can be used to
    add elements to a JSON array. Note how the `null` value was silently
    converted to a JSON array.,push_back}

    @since version 1.0.0
    */
    void push_back(basic_json&& val)
    {
        // push_back only works for null objects or arrays
        if (JSON_UNLIKELY(not(is_null() or is_array())))
        {
            JSON_THROW(type_error::create(308, "cannot use push_back() with " + std::string(type_name())));
        }

        // transform null object into an array
        if (is_null())
        {
            m_type = value_t::array;
            m_value = value_t::array;
            assert_invariant();
        }

        // add element to array (move semantics)
        m_value.array->push_back(std::move(val));
        // invalidate object
        val.m_type = value_t::null;
    }

    /*!
    @brief add an object to an array
    @copydoc push_back(basic_json&&)
    */
    reference operator+=(basic_json&& val)
    {
        push_back(std::move(val));
        return *this;
    }

    /*!
    @brief add an object to an array
    @copydoc push_back(basic_json&&)
    */
    void push_back(const basic_json& val)
    {
        // push_back only works for null objects or arrays
        if (JSON_UNLIKELY(not(is_null() or is_array())))
        {
            JSON_THROW(type_error::create(308, "cannot use push_back() with " + std::string(type_name())));
        }

        // transform null object into an array
        if (is_null())
        {
            m_type = value_t::array;
            m_value = value_t::array;
            assert_invariant();
        }

        // add element to array
        m_value.array->push_back(val);
    }

    /*!
    @brief add an object to an array
    @copydoc push_back(basic_json&&)
    */
    reference operator+=(const basic_json& val)
    {
        push_back(val);
        return *this;
    }

    /*!
    @brief add an object to an object

    Inserts the given element @a val to the JSON object. If the function is
    called on a JSON null value, an empty object is created before inserting
    @a val.

    @param[in] val the value to add to the JSON object

    @throw type_error.308 when called on a type other than JSON object or
    null; example: `"cannot use push_back() with number"`

    @complexity Logarithmic in the size of the container, O(log(`size()`)).

    @liveexample{The example shows how `push_back()` and `+=` can be used to
    add elements to a JSON object. Note how the `null` value was silently
    converted to a JSON object.,push_back__object_t__value}

    @since version 1.0.0
    */
    void push_back(const typename object_t::value_type& val)
    {
        // push_back only works for null objects or objects
        if (JSON_UNLIKELY(not(is_null() or is_object())))
        {
            JSON_THROW(type_error::create(308, "cannot use push_back() with " + std::string(type_name())));
        }

        // transform null object into an object
        if (is_null())
        {
            m_type = value_t::object;
            m_value = value_t::object;
            assert_invariant();
        }

        // add element to array
        m_value.object->insert(val);
    }

    /*!
    @brief add an object to an object
    @copydoc push_back(const typename object_t::value_type&)
    */
    reference operator+=(const typename object_t::value_type& val)
    {
        push_back(val);
        return *this;
    }

    /*!
    @brief add an object to an object

    This function allows to use `push_back` with an initializer list. In case

    1. the current value is an object,
    2. the initializer list @a init contains only two elements, and
    3. the first element of @a init is a string,

    @a init is converted into an object element and added using
    @ref push_back(const typename object_t::value_type&). Otherwise, @a init
    is converted to a JSON value and added using @ref push_back(basic_json&&).

    @param[in] init  an initializer list

    @complexity Linear in the size of the initializer list @a init.

    @note This function is required to resolve an ambiguous overload error,
          because pairs like `{"key", "value"}` can be both interpreted as
          `object_t::value_type` or `std::initializer_list<basic_json>`, see
          https://github.com/nlohmann/json/issues/235 for more information.

    @liveexample{The example shows how initializer lists are treated as
    objects when possible.,push_back__initializer_list}
    */
    void push_back(initializer_list_t init)
    {
        if (is_object() and init.size() == 2 and (*init.begin())->is_string())
        {
            basic_json&& key = init.begin()->moved_or_copied();
            push_back(typename object_t::value_type(
                          std::move(key.get_ref<string_t&>()), (init.begin() + 1)->moved_or_copied()));
        }
        else
        {
            push_back(basic_json(init));
        }
    }

    /*!
    @brief add an object to an object
    @copydoc push_back(initializer_list_t)
    */
    reference operator+=(initializer_list_t init)
    {
        push_back(init);
        return *this;
    }

    /*!
    @brief add an object to an array

    Creates a JSON value from the passed parameters @a args to the end of the
    JSON value. If the function is called on a JSON null value, an empty array
    is created before appending the value created from @a args.

    @param[in] args arguments to forward to a constructor of @ref basic_json
    @tparam Args compatible types to create a @ref basic_json object

    @throw type_error.311 when called on a type other than JSON array or
    null; example: `"cannot use emplace_back() with number"`

    @complexity Amortized constant.

    @liveexample{The example shows how `push_back()` can be used to add
    elements to a JSON array. Note how the `null` value was silently converted
    to a JSON array.,emplace_back}

    @since version 2.0.8
    */
    template<class... Args>
    void emplace_back(Args&& ... args)
    {
        // emplace_back only works for null objects or arrays
        if (JSON_UNLIKELY(not(is_null() or is_array())))
        {
            JSON_THROW(type_error::create(311, "cannot use emplace_back() with " + std::string(type_name())));
        }

        // transform null object into an array
        if (is_null())
        {
            m_type = value_t::array;
            m_value = value_t::array;
            assert_invariant();
        }

        // add element to array (perfect forwarding)
        m_value.array->emplace_back(std::forward<Args>(args)...);
    }

    /*!
    @brief add an object to an object if key does not exist

    Inserts a new element into a JSON object constructed in-place with the
    given @a args if there is no element with the key in the container. If the
    function is called on a JSON null value, an empty object is created before
    appending the value created from @a args.

    @param[in] args arguments to forward to a constructor of @ref basic_json
    @tparam Args compatible types to create a @ref basic_json object

    @return a pair consisting of an iterator to the inserted element, or the
            already-existing element if no insertion happened, and a bool
            denoting whether the insertion took place.

    @throw type_error.311 when called on a type other than JSON object or
    null; example: `"cannot use emplace() with number"`

    @complexity Logarithmic in the size of the container, O(log(`size()`)).

    @liveexample{The example shows how `emplace()` can be used to add elements
    to a JSON object. Note how the `null` value was silently converted to a
    JSON object. Further note how no value is added if there was already one
    value stored with the same key.,emplace}

    @since version 2.0.8
    */
    template<class... Args>
    std::pair<iterator, bool> emplace(Args&& ... args)
    {
        // emplace only works for null objects or arrays
        if (JSON_UNLIKELY(not(is_null() or is_object())))
        {
            JSON_THROW(type_error::create(311, "cannot use emplace() with " + std::string(type_name())));
        }

        // transform null object into an object
        if (is_null())
        {
            m_type = value_t::object;
            m_value = value_t::object;
            assert_invariant();
        }

        // add element to array (perfect forwarding)
        auto res = m_value.object->emplace(std::forward<Args>(args)...);
        // create result iterator and set iterator to the result of emplace
        auto it = begin();
        it.m_it.object_iterator = res.first;

        // return pair of iterator and boolean
        return {it, res.second};
    }

    /*!
    @brief inserts element

    Inserts element @a val before iterator @a pos.

    @param[in] pos iterator before which the content will be inserted; may be
    the end() iterator
    @param[in] val element to insert
    @return iterator pointing to the inserted @a val.

    @throw type_error.309 if called on JSON values other than arrays;
    example: `"cannot use insert() with string"`
    @throw invalid_iterator.202 if @a pos is not an iterator of *this;
    example: `"iterator does not fit current value"`

    @complexity Constant plus linear in the distance between @a pos and end of
    the container.

    @liveexample{The example shows how `insert()` is used.,insert}

    @since version 1.0.0
    */
    iterator insert(const_iterator pos, const basic_json& val)
    {
        // insert only works for arrays
        if (JSON_LIKELY(is_array()))
        {
            // check if iterator pos fits to this JSON value
            if (JSON_UNLIKELY(pos.m_object != this))
            {
                JSON_THROW(invalid_iterator::create(202, "iterator does not fit current value"));
            }

            // insert to array and return iterator
            iterator result(this);
            result.m_it.array_iterator = m_value.array->insert(pos.m_it.array_iterator, val);
            return result;
        }

        JSON_THROW(type_error::create(309, "cannot use insert() with " + std::string(type_name())));
    }

    /*!
    @brief inserts element
    @copydoc insert(const_iterator, const basic_json&)
    */
    iterator insert(const_iterator pos, basic_json&& val)
    {
        return insert(pos, val);
    }

    /*!
    @brief inserts elements

    Inserts @a cnt copies of @a val before iterator @a pos.

    @param[in] pos iterator before which the content will be inserted; may be
    the end() iterator
    @param[in] cnt number of copies of @a val to insert
    @param[in] val element to insert
    @return iterator pointing to the first element inserted, or @a pos if
    `cnt==0`

    @throw type_error.309 if called on JSON values other than arrays; example:
    `"cannot use insert() with string"`
    @throw invalid_iterator.202 if @a pos is not an iterator of *this;
    example: `"iterator does not fit current value"`

    @complexity Linear in @a cnt plus linear in the distance between @a pos
    and end of the container.

    @liveexample{The example shows how `insert()` is used.,insert__count}

    @since version 1.0.0
    */
    iterator insert(const_iterator pos, size_type cnt, const basic_json& val)
    {
        // insert only works for arrays
        if (JSON_LIKELY(is_array()))
        {
            // check if iterator pos fits to this JSON value
            if (JSON_UNLIKELY(pos.m_object != this))
            {
                JSON_THROW(invalid_iterator::create(202, "iterator does not fit current value"));
            }

            // insert to array and return iterator
            iterator result(this);
            result.m_it.array_iterator = m_value.array->insert(pos.m_it.array_iterator, cnt, val);
            return result;
        }

        JSON_THROW(type_error::create(309, "cannot use insert() with " + std::string(type_name())));
    }

    /*!
    @brief inserts elements

    Inserts elements from range `[first, last)` before iterator @a pos.

    @param[in] pos iterator before which the content will be inserted; may be
    the end() iterator
    @param[in] first begin of the range of elements to insert
    @param[in] last end of the range of elements to insert

    @throw type_error.309 if called on JSON values other than arrays; example:
    `"cannot use insert() with string"`
    @throw invalid_iterator.202 if @a pos is not an iterator of *this;
    example: `"iterator does not fit current value"`
    @throw invalid_iterator.210 if @a first and @a last do not belong to the
    same JSON value; example: `"iterators do not fit"`
    @throw invalid_iterator.211 if @a first or @a last are iterators into
    container for which insert is called; example: `"passed iterators may not
    belong to container"`

    @return iterator pointing to the first element inserted, or @a pos if
    `first==last`

    @complexity Linear in `std::distance(first, last)` plus linear in the
    distance between @a pos and end of the container.

    @liveexample{The example shows how `insert()` is used.,insert__range}

    @since version 1.0.0
    */
    iterator insert(const_iterator pos, const_iterator first, const_iterator last)
    {
        // insert only works for arrays
        if (JSON_UNLIKELY(not is_array()))
        {
            JSON_THROW(type_error::create(309, "cannot use insert() with " + std::string(type_name())));
        }

        // check if iterator pos fits to this JSON value
        if (JSON_UNLIKELY(pos.m_object != this))
        {
            JSON_THROW(invalid_iterator::create(202, "iterator does not fit current value"));
        }

        // check if range iterators belong to the same JSON object
        if (JSON_UNLIKELY(first.m_object != last.m_object))
        {
            JSON_THROW(invalid_iterator::create(210, "iterators do not fit"));
        }

        if (JSON_UNLIKELY(first.m_object == this))
        {
            JSON_THROW(invalid_iterator::create(211, "passed iterators may not belong to container"));
        }

        // insert to array and return iterator
        iterator result(this);
        result.m_it.array_iterator = m_value.array->insert(
                                         pos.m_it.array_iterator,
                                         first.m_it.array_iterator,
                                         last.m_it.array_iterator);
        return result;
    }

    /*!
    @brief inserts elements

    Inserts elements from initializer list @a ilist before iterator @a pos.

    @param[in] pos iterator before which the content will be inserted; may be
    the end() iterator
    @param[in] ilist initializer list to insert the values from

    @throw type_error.309 if called on JSON values other than arrays; example:
    `"cannot use insert() with string"`
    @throw invalid_iterator.202 if @a pos is not an iterator of *this;
    example: `"iterator does not fit current value"`

    @return iterator pointing to the first element inserted, or @a pos if
    `ilist` is empty

    @complexity Linear in `ilist.size()` plus linear in the distance between
    @a pos and end of the container.

    @liveexample{The example shows how `insert()` is used.,insert__ilist}

    @since version 1.0.0
    */
    iterator insert(const_iterator pos, initializer_list_t ilist)
    {
        // insert only works for arrays
        if (JSON_UNLIKELY(not is_array()))
        {
            JSON_THROW(type_error::create(309, "cannot use insert() with " + std::string(type_name())));
        }

        // check if iterator pos fits to this JSON value
        if (JSON_UNLIKELY(pos.m_object != this))
        {
            JSON_THROW(invalid_iterator::create(202, "iterator does not fit current value"));
        }

        // insert to array and return iterator
        iterator result(this);
        result.m_it.array_iterator = m_value.array->insert(pos.m_it.array_iterator, ilist.begin(), ilist.end());
        return result;
    }

    /*!
    @brief inserts elements

    Inserts elements from range `[first, last)`.

    @param[in] first begin of the range of elements to insert
    @param[in] last end of the range of elements to insert

    @throw type_error.309 if called on JSON values other than objects; example:
    `"cannot use insert() with string"`
    @throw invalid_iterator.202 if iterator @a first or @a last does does not
    point to an object; example: `"iterators first and last must point to
    objects"`
    @throw invalid_iterator.210 if @a first and @a last do not belong to the
    same JSON value; example: `"iterators do not fit"`

    @complexity Logarithmic: `O(N*log(size() + N))`, where `N` is the number
    of elements to insert.

    @liveexample{The example shows how `insert()` is used.,insert__range_object}

    @since version 3.0.0
    */
    void insert(const_iterator first, const_iterator last)
    {
        // insert only works for objects
        if (JSON_UNLIKELY(not is_object()))
        {
            JSON_THROW(type_error::create(309, "cannot use insert() with " + std::string(type_name())));
        }

        // check if range iterators belong to the same JSON object
        if (JSON_UNLIKELY(first.m_object != last.m_object))
        {
            JSON_THROW(invalid_iterator::create(210, "iterators do not fit"));
        }

        // passed iterators must belong to objects
        if (JSON_UNLIKELY(not first.m_object->is_object()))
        {
            JSON_THROW(invalid_iterator::create(202, "iterators first and last must point to objects"));
        }

        m_value.object->insert(first.m_it.object_iterator, last.m_it.object_iterator);
    }

    /*!
    @brief updates a JSON object from another object, overwriting existing keys

    Inserts all values from JSON object @a j and overwrites existing keys.

    @param[in] j  JSON object to read values from

    @throw type_error.312 if called on JSON values other than objects; example:
    `"cannot use update() with string"`

    @complexity O(N*log(size() + N)), where N is the number of elements to
                insert.

    @liveexample{The example shows how `update()` is used.,update}

    @sa https://docs.python.org/3.6/library/stdtypes.html#dict.update

    @since version 3.0.0
    */
    void update(const_reference j)
    {
        // implicitly convert null value to an empty object
        if (is_null())
        {
            m_type = value_t::object;
            m_value.object = create<object_t>();
            assert_invariant();
        }

        if (JSON_UNLIKELY(not is_object()))
        {
            JSON_THROW(type_error::create(312, "cannot use update() with " + std::string(type_name())));
        }
        if (JSON_UNLIKELY(not j.is_object()))
        {
            JSON_THROW(type_error::create(312, "cannot use update() with " + std::string(j.type_name())));
        }

        for (auto it = j.cbegin(); it != j.cend(); ++it)
        {
            m_value.object->operator[](it.key()) = it.value();
        }
    }

    /*!
    @brief updates a JSON object from another object, overwriting existing keys

    Inserts all values from from range `[first, last)` and overwrites existing
    keys.

    @param[in] first begin of the range of elements to insert
    @param[in] last end of the range of elements to insert

    @throw type_error.312 if called on JSON values other than objects; example:
    `"cannot use update() with string"`
    @throw invalid_iterator.202 if iterator @a first or @a last does does not
    point to an object; example: `"iterators first and last must point to
    objects"`
    @throw invalid_iterator.210 if @a first and @a last do not belong to the
    same JSON value; example: `"iterators do not fit"`

    @complexity O(N*log(size() + N)), where N is the number of elements to
                insert.

    @liveexample{The example shows how `update()` is used__range.,update}

    @sa https://docs.python.org/3.6/library/stdtypes.html#dict.update

    @since version 3.0.0
    */
    void update(const_iterator first, const_iterator last)
    {
        // implicitly convert null value to an empty object
        if (is_null())
        {
            m_type = value_t::object;
            m_value.object = create<object_t>();
            assert_invariant();
        }

        if (JSON_UNLIKELY(not is_object()))
        {
            JSON_THROW(type_error::create(312, "cannot use update() with " + std::string(type_name())));
        }

        // check if range iterators belong to the same JSON object
        if (JSON_UNLIKELY(first.m_object != last.m_object))
        {
            JSON_THROW(invalid_iterator::create(210, "iterators do not fit"));
        }

        // passed iterators must belong to objects
        if (JSON_UNLIKELY(not first.m_object->is_object()
                          or not last.m_object->is_object()))
        {
            JSON_THROW(invalid_iterator::create(202, "iterators first and last must point to objects"));
        }

        for (auto it = first; it != last; ++it)
        {
            m_value.object->operator[](it.key()) = it.value();
        }
    }

    /*!
    @brief exchanges the values

    Exchanges the contents of the JSON value with those of @a other. Does not
    invoke any move, copy, or swap operations on individual elements. All
    iterators and references remain valid. The past-the-end iterator is
    invalidated.

    @param[in,out] other JSON value to exchange the contents with

    @complexity Constant.

    @liveexample{The example below shows how JSON values can be swapped with
    `swap()`.,swap__reference}

    @since version 1.0.0
    */
    void swap(reference other) noexcept (
        std::is_nothrow_move_constructible<value_t>::value and
        std::is_nothrow_move_assignable<value_t>::value and
        std::is_nothrow_move_constructible<json_value>::value and
        std::is_nothrow_move_assignable<json_value>::value
    )
    {
        std::swap(m_type, other.m_type);
        std::swap(m_value, other.m_value);
        assert_invariant();
    }

    /*!
    @brief exchanges the values

    Exchanges the contents of a JSON array with those of @a other. Does not
    invoke any move, copy, or swap operations on individual elements. All
    iterators and references remain valid. The past-the-end iterator is
    invalidated.

    @param[in,out] other array to exchange the contents with

    @throw type_error.310 when JSON value is not an array; example: `"cannot
    use swap() with string"`

    @complexity Constant.

    @liveexample{The example below shows how arrays can be swapped with
    `swap()`.,swap__array_t}

    @since version 1.0.0
    */
    void swap(array_t& other)
    {
        // swap only works for arrays
        if (JSON_LIKELY(is_array()))
        {
            std::swap(*(m_value.array), other);
        }
        else
        {
            JSON_THROW(type_error::create(310, "cannot use swap() with " + std::string(type_name())));
        }
    }

    /*!
    @brief exchanges the values

    Exchanges the contents of a JSON object with those of @a other. Does not
    invoke any move, copy, or swap operations on individual elements. All
    iterators and references remain valid. The past-the-end iterator is
    invalidated.

    @param[in,out] other object to exchange the contents with

    @throw type_error.310 when JSON value is not an object; example:
    `"cannot use swap() with string"`

    @complexity Constant.

    @liveexample{The example below shows how objects can be swapped with
    `swap()`.,swap__object_t}

    @since version 1.0.0
    */
    void swap(object_t& other)
    {
        // swap only works for objects
        if (JSON_LIKELY(is_object()))
        {
            std::swap(*(m_value.object), other);
        }
        else
        {
            JSON_THROW(type_error::create(310, "cannot use swap() with " + std::string(type_name())));
        }
    }

    /*!
    @brief exchanges the values

    Exchanges the contents of a JSON string with those of @a other. Does not
    invoke any move, copy, or swap operations on individual elements. All
    iterators and references remain valid. The past-the-end iterator is
    invalidated.

    @param[in,out] other string to exchange the contents with

    @throw type_error.310 when JSON value is not a string; example: `"cannot
    use swap() with boolean"`

    @complexity Constant.

    @liveexample{The example below shows how strings can be swapped with
    `swap()`.,swap__string_t}

    @since version 1.0.0
    */
    void swap(string_t& other)
    {
        // swap only works for strings
        if (JSON_LIKELY(is_string()))
        {
            std::swap(*(m_value.string), other);
        }
        else
        {
            JSON_THROW(type_error::create(310, "cannot use swap() with " + std::string(type_name())));
        }
    }

    /// @}

  public:
    //////////////////////////////////////////
    // lexicographical comparison operators //
    //////////////////////////////////////////

    /// @name lexicographical comparison operators
    /// @{

    /*!
    @brief comparison: equal

    Compares two JSON values for equality according to the following rules:
    - Two JSON values are equal if (1) they are from the same type and (2)
      their stored values are the same according to their respective
      `operator==`.
    - Integer and floating-point numbers are automatically converted before
      comparison. Note than two NaN values are always treated as unequal.
    - Two JSON null values are equal.

    @note Floating-point inside JSON values numbers are compared with
    `json::number_float_t::operator==` which is `double::operator==` by
    default. To compare floating-point while respecting an epsilon, an alternative
    [comparison function](https://github.com/mariokonrad/marnav/blob/master/src/marnav/math/floatingpoint.hpp#L34-#L39)
    could be used, for instance
    @code {.cpp}
    template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value, T>::type>
    inline bool is_same(T a, T b, T epsilon = std::numeric_limits<T>::epsilon()) noexcept
    {
        return std::abs(a - b) <= epsilon;
    }
    @endcode

    @note NaN values never compare equal to themselves or to other NaN values.

    @param[in] lhs  first JSON value to consider
    @param[in] rhs  second JSON value to consider
    @return whether the values @a lhs and @a rhs are equal

    @exceptionsafety No-throw guarantee: this function never throws exceptions.

    @complexity Linear.

    @liveexample{The example demonstrates comparing several JSON
    types.,operator__equal}

    @since version 1.0.0
    */
    friend bool operator==(const_reference lhs, const_reference rhs) noexcept
    {
        const auto lhs_type = lhs.type();
        const auto rhs_type = rhs.type();

        if (lhs_type == rhs_type)
        {
            switch (lhs_type)
            {
                case value_t::array:
                    return (*lhs.m_value.array == *rhs.m_value.array);

                case value_t::object:
                    return (*lhs.m_value.object == *rhs.m_value.object);

                case value_t::null:
                    return true;

                case value_t::string:
                    return (*lhs.m_value.string == *rhs.m_value.string);

                case value_t::boolean:
                    return (lhs.m_value.boolean == rhs.m_value.boolean);

                case value_t::number_integer:
                    return (lhs.m_value.number_integer == rhs.m_value.number_integer);

                case value_t::number_unsigned:
                    return (lhs.m_value.number_unsigned == rhs.m_value.number_unsigned);

                case value_t::number_float:
                    return (lhs.m_value.number_float == rhs.m_value.number_float);

                default:
                    return false;
            }
        }
        else if (lhs_type == value_t::number_integer and rhs_type == value_t::number_float)
        {
            return (static_cast<number_float_t>(lhs.m_value.number_integer) == rhs.m_value.number_float);
        }
        else if (lhs_type == value_t::number_float and rhs_type == value_t::number_integer)
        {
            return (lhs.m_value.number_float == static_cast<number_float_t>(rhs.m_value.number_integer));
        }
        else if (lhs_type == value_t::number_unsigned and rhs_type == value_t::number_float)
        {
            return (static_cast<number_float_t>(lhs.m_value.number_unsigned) == rhs.m_value.number_float);
        }
        else if (lhs_type == value_t::number_float and rhs_type == value_t::number_unsigned)
        {
            return (lhs.m_value.number_float == static_cast<number_float_t>(rhs.m_value.number_unsigned));
        }
        else if (lhs_type == value_t::number_unsigned and rhs_type == value_t::number_integer)
        {
            return (static_cast<number_integer_t>(lhs.m_value.number_unsigned) == rhs.m_value.number_integer);
        }
        else if (lhs_type == value_t::number_integer and rhs_type == value_t::number_unsigned)
        {
            return (lhs.m_value.number_integer == static_cast<number_integer_t>(rhs.m_value.number_unsigned));
        }

        return false;
    }

    /*!
    @brief comparison: equal
    @copydoc operator==(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator==(const_reference lhs, const ScalarType rhs) noexcept
    {
        return (lhs == basic_json(rhs));
    }

    /*!
    @brief comparison: equal
    @copydoc operator==(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator==(const ScalarType lhs, const_reference rhs) noexcept
    {
        return (basic_json(lhs) == rhs);
    }

    /*!
    @brief comparison: not equal

    Compares two JSON values for inequality by calculating `not (lhs == rhs)`.

    @param[in] lhs  first JSON value to consider
    @param[in] rhs  second JSON value to consider
    @return whether the values @a lhs and @a rhs are not equal

    @complexity Linear.

    @exceptionsafety No-throw guarantee: this function never throws exceptions.

    @liveexample{The example demonstrates comparing several JSON
    types.,operator__notequal}

    @since version 1.0.0
    */
    friend bool operator!=(const_reference lhs, const_reference rhs) noexcept
    {
        return not (lhs == rhs);
    }

    /*!
    @brief comparison: not equal
    @copydoc operator!=(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator!=(const_reference lhs, const ScalarType rhs) noexcept
    {
        return (lhs != basic_json(rhs));
    }

    /*!
    @brief comparison: not equal
    @copydoc operator!=(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator!=(const ScalarType lhs, const_reference rhs) noexcept
    {
        return (basic_json(lhs) != rhs);
    }

    /*!
    @brief comparison: less than

    Compares whether one JSON value @a lhs is less than another JSON value @a
    rhs according to the following rules:
    - If @a lhs and @a rhs have the same type, the values are compared using
      the default `<` operator.
    - Integer and floating-point numbers are automatically converted before
      comparison
    - In case @a lhs and @a rhs have different types, the values are ignored
      and the order of the types is considered, see
      @ref operator<(const value_t, const value_t).

    @param[in] lhs  first JSON value to consider
    @param[in] rhs  second JSON value to consider
    @return whether @a lhs is less than @a rhs

    @complexity Linear.

    @exceptionsafety No-throw guarantee: this function never throws exceptions.

    @liveexample{The example demonstrates comparing several JSON
    types.,operator__less}

    @since version 1.0.0
    */
    friend bool operator<(const_reference lhs, const_reference rhs) noexcept
    {
        const auto lhs_type = lhs.type();
        const auto rhs_type = rhs.type();

        if (lhs_type == rhs_type)
        {
            switch (lhs_type)
            {
                case value_t::array:
                    return (*lhs.m_value.array) < (*rhs.m_value.array);

                case value_t::object:
                    return *lhs.m_value.object < *rhs.m_value.object;

                case value_t::null:
                    return false;

                case value_t::string:
                    return *lhs.m_value.string < *rhs.m_value.string;

                case value_t::boolean:
                    return lhs.m_value.boolean < rhs.m_value.boolean;

                case value_t::number_integer:
                    return lhs.m_value.number_integer < rhs.m_value.number_integer;

                case value_t::number_unsigned:
                    return lhs.m_value.number_unsigned < rhs.m_value.number_unsigned;

                case value_t::number_float:
                    return lhs.m_value.number_float < rhs.m_value.number_float;

                default:
                    return false;
            }
        }
        else if (lhs_type == value_t::number_integer and rhs_type == value_t::number_float)
        {
            return static_cast<number_float_t>(lhs.m_value.number_integer) < rhs.m_value.number_float;
        }
        else if (lhs_type == value_t::number_float and rhs_type == value_t::number_integer)
        {
            return lhs.m_value.number_float < static_cast<number_float_t>(rhs.m_value.number_integer);
        }
        else if (lhs_type == value_t::number_unsigned and rhs_type == value_t::number_float)
        {
            return static_cast<number_float_t>(lhs.m_value.number_unsigned) < rhs.m_value.number_float;
        }
        else if (lhs_type == value_t::number_float and rhs_type == value_t::number_unsigned)
        {
            return lhs.m_value.number_float < static_cast<number_float_t>(rhs.m_value.number_unsigned);
        }
        else if (lhs_type == value_t::number_integer and rhs_type == value_t::number_unsigned)
        {
            return lhs.m_value.number_integer < static_cast<number_integer_t>(rhs.m_value.number_unsigned);
        }
        else if (lhs_type == value_t::number_unsigned and rhs_type == value_t::number_integer)
        {
            return static_cast<number_integer_t>(lhs.m_value.number_unsigned) < rhs.m_value.number_integer;
        }

        // We only reach this line if we cannot compare values. In that case,
        // we compare types. Note we have to call the operator explicitly,
        // because MSVC has problems otherwise.
        return operator<(lhs_type, rhs_type);
    }

    /*!
    @brief comparison: less than
    @copydoc operator<(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator<(const_reference lhs, const ScalarType rhs) noexcept
    {
        return (lhs < basic_json(rhs));
    }

    /*!
    @brief comparison: less than
    @copydoc operator<(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator<(const ScalarType lhs, const_reference rhs) noexcept
    {
        return (basic_json(lhs) < rhs);
    }

    /*!
    @brief comparison: less than or equal

    Compares whether one JSON value @a lhs is less than or equal to another
    JSON value by calculating `not (rhs < lhs)`.

    @param[in] lhs  first JSON value to consider
    @param[in] rhs  second JSON value to consider
    @return whether @a lhs is less than or equal to @a rhs

    @complexity Linear.

    @exceptionsafety No-throw guarantee: this function never throws exceptions.

    @liveexample{The example demonstrates comparing several JSON
    types.,operator__greater}

    @since version 1.0.0
    */
    friend bool operator<=(const_reference lhs, const_reference rhs) noexcept
    {
        return not (rhs < lhs);
    }

    /*!
    @brief comparison: less than or equal
    @copydoc operator<=(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator<=(const_reference lhs, const ScalarType rhs) noexcept
    {
        return (lhs <= basic_json(rhs));
    }

    /*!
    @brief comparison: less than or equal
    @copydoc operator<=(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator<=(const ScalarType lhs, const_reference rhs) noexcept
    {
        return (basic_json(lhs) <= rhs);
    }

    /*!
    @brief comparison: greater than

    Compares whether one JSON value @a lhs is greater than another
    JSON value by calculating `not (lhs <= rhs)`.

    @param[in] lhs  first JSON value to consider
    @param[in] rhs  second JSON value to consider
    @return whether @a lhs is greater than to @a rhs

    @complexity Linear.

    @exceptionsafety No-throw guarantee: this function never throws exceptions.

    @liveexample{The example demonstrates comparing several JSON
    types.,operator__lessequal}

    @since version 1.0.0
    */
    friend bool operator>(const_reference lhs, const_reference rhs) noexcept
    {
        return not (lhs <= rhs);
    }

    /*!
    @brief comparison: greater than
    @copydoc operator>(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator>(const_reference lhs, const ScalarType rhs) noexcept
    {
        return (lhs > basic_json(rhs));
    }

    /*!
    @brief comparison: greater than
    @copydoc operator>(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator>(const ScalarType lhs, const_reference rhs) noexcept
    {
        return (basic_json(lhs) > rhs);
    }

    /*!
    @brief comparison: greater than or equal

    Compares whether one JSON value @a lhs is greater than or equal to another
    JSON value by calculating `not (lhs < rhs)`.

    @param[in] lhs  first JSON value to consider
    @param[in] rhs  second JSON value to consider
    @return whether @a lhs is greater than or equal to @a rhs

    @complexity Linear.

    @exceptionsafety No-throw guarantee: this function never throws exceptions.

    @liveexample{The example demonstrates comparing several JSON
    types.,operator__greaterequal}

    @since version 1.0.0
    */
    friend bool operator>=(const_reference lhs, const_reference rhs) noexcept
    {
        return not (lhs < rhs);
    }

    /*!
    @brief comparison: greater than or equal
    @copydoc operator>=(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator>=(const_reference lhs, const ScalarType rhs) noexcept
    {
        return (lhs >= basic_json(rhs));
    }

    /*!
    @brief comparison: greater than or equal
    @copydoc operator>=(const_reference, const_reference)
    */
    template<typename ScalarType, typename std::enable_if<
                 std::is_scalar<ScalarType>::value, int>::type = 0>
    friend bool operator>=(const ScalarType lhs, const_reference rhs) noexcept
    {
        return (basic_json(lhs) >= rhs);
    }

    /// @}

    /*!
    @brief serialize to stream
    @deprecated This stream operator is deprecated and will be removed in
                future 4.0.0 of the library. Please use
                @ref operator<<(std::ostream&, const basic_json&)
                instead; that is, replace calls like `j >> o;` with `o << j;`.
    @since version 1.0.0; deprecated since version 3.0.0
    */
    JSON_DEPRECATED
    friend std::ostream& operator>>(const basic_json& j, std::ostream& o)
    {
        return o << j;
    }

    /// @}


    /////////////////////
    // deserialization //
    /////////////////////

    /// @name deserialization
    /// @{

    /*!
    @brief deserialize from a compatible input

    This function reads from a compatible input. Examples are:
    - an array of 1-byte values
    - strings with character/literal type with size of 1 byte
    - input streams
    - container with contiguous storage of 1-byte values. Compatible container
      types include `std::vector`, `std::string`, `std::array`,
      `std::valarray`, and `std::initializer_list`. Furthermore, C-style
      arrays can be used with `std::begin()`/`std::end()`. User-defined
      containers can be used as long as they implement random-access iterators
      and a contiguous storage.

    @pre Each element of the container has a size of 1 byte. Violating this
    precondition yields undefined behavior. **This precondition is enforced
    with a static assertion.**

    @pre The container storage is contiguous. Violating this precondition
    yields undefined behavior. **This precondition is enforced with an
    assertion.**
    @pre Each element of the container has a size of 1 byte. Violating this
    precondition yields undefined behavior. **This precondition is enforced
    with a static assertion.**

    @warning There is no way to enforce all preconditions at compile-time. If
             the function is called with a noncompliant container and with
             assertions switched off, the behavior is undefined and will most
             likely yield segmentation violation.

    @param[in] i  input to read from
    @param[in] cb  a parser callback function of type @ref parser_callback_t
    which is used to control the deserialization by filtering unwanted values
    (optional)

    @return result of the deserialization

    @throw parse_error.101 if a parse error occurs; example: `""unexpected end
    of input; expected string literal""`
    @throw parse_error.102 if to_unicode fails or surrogate error
    @throw parse_error.103 if to_unicode fails

    @complexity Linear in the length of the input. The parser is a predictive
    LL(1) parser. The complexity can be higher if the parser callback function
    @a cb has a super-linear complexity.

    @note A UTF-8 byte order mark is silently ignored.

    @liveexample{The example below demonstrates the `parse()` function reading
    from an array.,parse__array__parser_callback_t}

    @liveexample{The example below demonstrates the `parse()` function with
    and without callback function.,parse__string__parser_callback_t}

    @liveexample{The example below demonstrates the `parse()` function with
    and without callback function.,parse__istream__parser_callback_t}

    @liveexample{The example below demonstrates the `parse()` function reading
    from a contiguous container.,parse__contiguouscontainer__parser_callback_t}

    @since version 2.0.3 (contiguous containers)
    */
    static basic_json parse(detail::input_adapter&& i,
                            const parser_callback_t cb = nullptr,
                            const bool allow_exceptions = true)
    {
        basic_json result;
        parser(i, cb, allow_exceptions).parse(true, result);
        return result;
    }

    static bool accept(detail::input_adapter&& i)
    {
        return parser(i).accept(true);
    }

    static bool sax_parse(detail::input_adapter&& i, json_sax_t* sax,
                          input_format_t format = input_format_t::json,
                          const bool strict = true)
    {
        assert(sax);
        switch (format)
        {
            case input_format_t::json:
                return parser(std::move(i)).sax_parse(sax, strict);
            default:
                return binary_reader(std::move(i)).sax_parse(format, sax, strict);
        }
    }

    /*!
    @brief deserialize from an iterator range with contiguous storage

    This function reads from an iterator range of a container with contiguous
    storage of 1-byte values. Compatible container types include
    `std::vector`, `std::string`, `std::array`, `std::valarray`, and
    `std::initializer_list`. Furthermore, C-style arrays can be used with
    `std::begin()`/`std::end()`. User-defined containers can be used as long
    as they implement random-access iterators and a contiguous storage.

    @pre The iterator range is contiguous. Violating this precondition yields
    undefined behavior. **This precondition is enforced with an assertion.**
    @pre Each element in the range has a size of 1 byte. Violating this
    precondition yields undefined behavior. **This precondition is enforced
    with a static assertion.**

    @warning There is no way to enforce all preconditions at compile-time. If
             the function is called with noncompliant iterators and with
             assertions switched off, the behavior is undefined and will most
             likely yield segmentation violation.

    @tparam IteratorType iterator of container with contiguous storage
    @param[in] first  begin of the range to parse (included)
    @param[in] last  end of the range to parse (excluded)
    @param[in] cb  a parser callback function of type @ref parser_callback_t
    which is used to control the deserialization by filtering unwanted values
    (optional)
    @param[in] allow_exceptions  whether to throw exceptions in case of a
    parse error (optional, true by default)

    @return result of the deserialization

    @throw parse_error.101 in case of an unexpected token
    @throw parse_error.102 if to_unicode fails or surrogate error
    @throw parse_error.103 if to_unicode fails

    @complexity Linear in the length of the input. The parser is a predictive
    LL(1) parser. The complexity can be higher if the parser callback function
    @a cb has a super-linear complexity.

    @note A UTF-8 byte order mark is silently ignored.

    @liveexample{The example below demonstrates the `parse()` function reading
    from an iterator range.,parse__iteratortype__parser_callback_t}

    @since version 2.0.3
    */
    template<class IteratorType, typename std::enable_if<
                 std::is_base_of<
                     std::random_access_iterator_tag,
                     typename std::iterator_traits<IteratorType>::iterator_category>::value, int>::type = 0>
    static basic_json parse(IteratorType first, IteratorType last,
                            const parser_callback_t cb = nullptr,
                            const bool allow_exceptions = true)
    {
        basic_json result;
        parser(detail::input_adapter(first, last), cb, allow_exceptions).parse(true, result);
        return result;
    }

    template<class IteratorType, typename std::enable_if<
                 std::is_base_of<
                     std::random_access_iterator_tag,
                     typename std::iterator_traits<IteratorType>::iterator_category>::value, int>::type = 0>
    static bool accept(IteratorType first, IteratorType last)
    {
        return parser(detail::input_adapter(first, last)).accept(true);
    }

    template<class IteratorType, typename std::enable_if<
                 std::is_base_of<
                     std::random_access_iterator_tag,
                     typename std::iterator_traits<IteratorType>::iterator_category>::value, int>::type = 0>
    static bool sax_parse(IteratorType first, IteratorType last, json_sax_t* sax)
    {
        return parser(detail::input_adapter(first, last)).sax_parse(sax);
    }

    /*!
    @brief deserialize from stream
    @deprecated This stream operator is deprecated and will be removed in
                version 4.0.0 of the library. Please use
                @ref operator>>(std::istream&, basic_json&)
                instead; that is, replace calls like `j << i;` with `i >> j;`.
    @since version 1.0.0; deprecated since version 3.0.0
    */
    JSON_DEPRECATED
    friend std::istream& operator<<(basic_json& j, std::istream& i)
    {
        return operator>>(i, j);
    }

    /*!
    @brief deserialize from stream

    Deserializes an input stream to a JSON value.

    @param[in,out] i  input stream to read a serialized JSON value from
    @param[in,out] j  JSON value to write the deserialized input to

    @throw parse_error.101 in case of an unexpected token
    @throw parse_error.102 if to_unicode fails or surrogate error
    @throw parse_error.103 if to_unicode fails

    @complexity Linear in the length of the input. The parser is a predictive
    LL(1) parser.

    @note A UTF-8 byte order mark is silently ignored.

    @liveexample{The example below shows how a JSON value is constructed by
    reading a serialization from a stream.,operator_deserialize}

    @sa parse(std::istream&, const parser_callback_t) for a variant with a
    parser callback function to filter values while parsing

    @since version 1.0.0
    */
    friend std::istream& operator>>(std::istream& i, basic_json& j)
    {
        parser(detail::input_adapter(i)).parse(false, j);
        return i;
    }

    /// @}

    ///////////////////////////
    // convenience functions //
    ///////////////////////////

    /*!
    @brief return the type as string

    Returns the type name as string to be used in error messages - usually to
    indicate that a function was called on a wrong JSON type.

    @return a string representation of a the @a m_type member:
            Value type  | return value
            ----------- | -------------
            null        | `"null"`
            boolean     | `"boolean"`
            string      | `"string"`
            number      | `"number"` (for all number types)
            object      | `"object"`
            array       | `"array"`
            discarded   | `"discarded"`

    @exceptionsafety No-throw guarantee: this function never throws exceptions.

    @complexity Constant.

    @liveexample{The following code exemplifies `type_name()` for all JSON
    types.,type_name}

    @sa @ref type() -- return the type of the JSON value
    @sa @ref operator value_t() -- return the type of the JSON value (implicit)

    @since version 1.0.0, public since 2.1.0, `const char*` and `noexcept`
    since 3.0.0
    */
    const char* type_name() const noexcept
    {
        {
            switch (m_type)
            {
                case value_t::null:
                    return "null";
                case value_t::object:
                    return "object";
                case value_t::array:
                    return "array";
                case value_t::string:
                    return "string";
                case value_t::boolean:
                    return "boolean";
                case value_t::discarded:
                    return "discarded";
                default:
                    return "number";
            }
        }
    }


  private:
    //////////////////////
    // member variables //
    //////////////////////

    /// the type of the current element
    value_t m_type = value_t::null;

    /// the value of the current element
    json_value m_value = {};

    //////////////////////////////////////////
    // binary serialization/deserialization //
    //////////////////////////////////////////

    /// @name binary serialization/deserialization support
    /// @{

  public:
    /*!
    @brief create a JSON value from an input in CBOR format

    Deserializes a given input @a i to a JSON value using the CBOR (Concise
    Binary Object Representation) serialization format.

    The library maps CBOR types to JSON value types as follows:

    CBOR type              | JSON value type | first byte
    ---------------------- | --------------- | ----------
    Integer                | number_unsigned | 0x00..0x17
    Unsigned integer       | number_unsigned | 0x18
    Unsigned integer       | number_unsigned | 0x19
    Unsigned integer       | number_unsigned | 0x1A
    Unsigned integer       | number_unsigned | 0x1B
    Negative integer       | number_integer  | 0x20..0x37
    Negative integer       | number_integer  | 0x38
    Negative integer       | number_integer  | 0x39
    Negative integer       | number_integer  | 0x3A
    Negative integer       | number_integer  | 0x3B
    Negative integer       | number_integer  | 0x40..0x57
    UTF-8 string           | string          | 0x60..0x77
    UTF-8 string           | string          | 0x78
    UTF-8 string           | string          | 0x79
    UTF-8 string           | string          | 0x7A
    UTF-8 string           | string          | 0x7B
    UTF-8 string           | string          | 0x7F
    array                  | array           | 0x80..0x97
    array                  | array           | 0x98
    array                  | array           | 0x99
    array                  | array           | 0x9A
    array                  | array           | 0x9B
    array                  | array           | 0x9F
    map                    | object          | 0xA0..0xB7
    map                    | object          | 0xB8
    map                    | object          | 0xB9
    map                    | object          | 0xBA
    map                    | object          | 0xBB
    map                    | object          | 0xBF
    False                  | `false`         | 0xF4
    True                   | `true`          | 0xF5
    Nill                   | `null`          | 0xF6
    Half-Precision Float   | number_float    | 0xF9
    Single-Precision Float | number_float    | 0xFA
    Double-Precision Float | number_float    | 0xFB

    @warning The mapping is **incomplete** in the sense that not all CBOR
             types can be converted to a JSON value. The following CBOR types
             are not supported and will yield parse errors (parse_error.112):
             - byte strings (0x40..0x5F)
             - date/time (0xC0..0xC1)
             - bignum (0xC2..0xC3)
             - decimal fraction (0xC4)
             - bigfloat (0xC5)
             - tagged items (0xC6..0xD4, 0xD8..0xDB)
             - expected conversions (0xD5..0xD7)
             - simple values (0xE0..0xF3, 0xF8)
             - undefined (0xF7)

    @warning CBOR allows map keys of any type, whereas JSON only allows
             strings as keys in object values. Therefore, CBOR maps with keys
             other than UTF-8 strings are rejected (parse_error.113).

    @note Any CBOR output created @ref to_cbor can be successfully parsed by
          @ref from_cbor.

    @param[in] i  an input in CBOR format convertible to an input adapter
    @param[in] strict  whether to expect the input to be consumed until EOF
                       (true by default)
    @param[in] allow_exceptions  whether to throw exceptions in case of a
    parse error (optional, true by default)

    @return deserialized JSON value

    @throw parse_error.110 if the given input ends prematurely or the end of
    file was not reached when @a strict was set to true
    @throw parse_error.112 if unsupported features from CBOR were
    used in the given input @a v or if the input is not valid CBOR
    @throw parse_error.113 if a string was expected as map key, but not found

    @complexity Linear in the size of the input @a i.

    @liveexample{The example shows the deserialization of a byte vector in CBOR
    format to a JSON value.,from_cbor}

    @sa http://cbor.io
    @sa @ref to_cbor(const basic_json&) for the analogous serialization
    @sa @ref from_msgpack(detail::input_adapter, const bool, const bool) for the
        related MessagePack format
    @sa @ref from_ubjson(detail::input_adapter, const bool, const bool) for the
        related UBJSON format

    @since version 2.0.9; parameter @a start_index since 2.1.1; changed to
           consume input adapters, removed start_index parameter, and added
           @a strict parameter since 3.0.0; added @allow_exceptions parameter
           since 3.2.0
    */
    static basic_json from_cbor(detail::input_adapter&& i,
                                const bool strict = true,
                                const bool allow_exceptions = true)
    {
        basic_json result;
        detail::json_sax_dom_parser<basic_json> sdp(result, allow_exceptions);
        const bool res = binary_reader(detail::input_adapter(i)).sax_parse(input_format_t::cbor, &sdp, strict);
        return res ? result : basic_json(value_t::discarded);
    }

    /*!
    @copydoc from_cbor(detail::input_adapter, const bool, const bool)
    */
    template<typename A1, typename A2,
             detail::enable_if_t<std::is_constructible<detail::input_adapter, A1, A2>::value, int> = 0>
    static basic_json from_cbor(A1 && a1, A2 && a2,
                                const bool strict = true,
                                const bool allow_exceptions = true)
    {
        basic_json result;
        detail::json_sax_dom_parser<basic_json> sdp(result, allow_exceptions);
        const bool res = binary_reader(detail::input_adapter(std::forward<A1>(a1), std::forward<A2>(a2))).sax_parse(input_format_t::cbor, &sdp, strict);
        return res ? result : basic_json(value_t::discarded);
    }

    /*!
    @brief create a JSON value from an input in MessagePack format

    Deserializes a given input @a i to a JSON value using the MessagePack
    serialization format.

    The library maps MessagePack types to JSON value types as follows:

    MessagePack type | JSON value type | first byte
    ---------------- | --------------- | ----------
    positive fixint  | number_unsigned | 0x00..0x7F
    fixmap           | object          | 0x80..0x8F
    fixarray         | array           | 0x90..0x9F
    fixstr           | string          | 0xA0..0xBF
    nil              | `null`          | 0xC0
    false            | `false`         | 0xC2
    true             | `true`          | 0xC3
    float 32         | number_float    | 0xCA
    float 64         | number_float    | 0xCB
    uint 8           | number_unsigned | 0xCC
    uint 16          | number_unsigned | 0xCD
    uint 32          | number_unsigned | 0xCE
    uint 64          | number_unsigned | 0xCF
    int 8            | number_integer  | 0xD0
    int 16           | number_integer  | 0xD1
    int 32           | number_integer  | 0xD2
    int 64           | number_integer  | 0xD3
    str 8            | string          | 0xD9
    str 16           | string          | 0xDA
    str 32           | string          | 0xDB
    array 16         | array           | 0xDC
    array 32         | array           | 0xDD
    map 16           | object          | 0xDE
    map 32           | object          | 0xDF
    negative fixint  | number_integer  | 0xE0-0xFF

    @warning The mapping is **incomplete** in the sense that not all
             MessagePack types can be converted to a JSON value. The following
             MessagePack types are not supported and will yield parse errors:
              - bin 8 - bin 32 (0xC4..0xC6)
              - ext 8 - ext 32 (0xC7..0xC9)
              - fixext 1 - fixext 16 (0xD4..0xD8)

    @note Any MessagePack output created @ref to_msgpack can be successfully
          parsed by @ref from_msgpack.

    @param[in] i  an input in MessagePack format convertible to an input
                  adapter
    @param[in] strict  whether to expect the input to be consumed until EOF
                       (true by default)
    @param[in] allow_exceptions  whether to throw exceptions in case of a
    parse error (optional, true by default)

    @return deserialized JSON value

    @throw parse_error.110 if the given input ends prematurely or the end of
    file was not reached when @a strict was set to true
    @throw parse_error.112 if unsupported features from MessagePack were
    used in the given input @a i or if the input is not valid MessagePack
    @throw parse_error.113 if a string was expected as map key, but not found

    @complexity Linear in the size of the input @a i.

    @liveexample{The example shows the deserialization of a byte vector in
    MessagePack format to a JSON value.,from_msgpack}

    @sa http://msgpack.org
    @sa @ref to_msgpack(const basic_json&) for the analogous serialization
    @sa @ref from_cbor(detail::input_adapter, const bool, const bool) for the
        related CBOR format
    @sa @ref from_ubjson(detail::input_adapter, const bool, const bool) for
        the related UBJSON format

    @since version 2.0.9; parameter @a start_index since 2.1.1; changed to
           consume input adapters, removed start_index parameter, and added
           @a strict parameter since 3.0.0; added @allow_exceptions parameter
           since 3.2.0
    */
    static basic_json from_msgpack(detail::input_adapter&& i,
                                   const bool strict = true,
                                   const bool allow_exceptions = true)
    {
        basic_json result;
        detail::json_sax_dom_parser<basic_json> sdp(result, allow_exceptions);
        const bool res = binary_reader(detail::input_adapter(i)).sax_parse(input_format_t::msgpack, &sdp, strict);
        return res ? result : basic_json(value_t::discarded);
    }

    /*!
    @copydoc from_msgpack(detail::input_adapter, const bool, const bool)
    */
    template<typename A1, typename A2,
             detail::enable_if_t<std::is_constructible<detail::input_adapter, A1, A2>::value, int> = 0>
    static basic_json from_msgpack(A1 && a1, A2 && a2,
                                   const bool strict = true,
                                   const bool allow_exceptions = true)
    {
        basic_json result;
        detail::json_sax_dom_parser<basic_json> sdp(result, allow_exceptions);
        const bool res = binary_reader(detail::input_adapter(std::forward<A1>(a1), std::forward<A2>(a2))).sax_parse(input_format_t::msgpack, &sdp, strict);
        return res ? result : basic_json(value_t::discarded);
    }

    /*!
    @brief create a JSON value from an input in UBJSON format

    Deserializes a given input @a i to a JSON value using the UBJSON (Universal
    Binary JSON) serialization format.

    The library maps UBJSON types to JSON value types as follows:

    UBJSON type | JSON value type                         | marker
    ----------- | --------------------------------------- | ------
    no-op       | *no value, next value is read*          | `N`
    null        | `null`                                  | `Z`
    false       | `false`                                 | `F`
    true        | `true`                                  | `T`
    float32     | number_float                            | `d`
    float64     | number_float                            | `D`
    uint8       | number_unsigned                         | `U`
    int8        | number_integer                          | `i`
    int16       | number_integer                          | `I`
    int32       | number_integer                          | `l`
    int64       | number_integer                          | `L`
    string      | string                                  | `S`
    char        | string                                  | `C`
    array       | array (optimized values are supported)  | `[`
    object      | object (optimized values are supported) | `{`

    @note The mapping is **complete** in the sense that any UBJSON value can
          be converted to a JSON value.

    @param[in] i  an input in UBJSON format convertible to an input adapter
    @param[in] strict  whether to expect the input to be consumed until EOF
                       (true by default)
    @param[in] allow_exceptions  whether to throw exceptions in case of a
    parse error (optional, true by default)

    @return deserialized JSON value

    @throw parse_error.110 if the given input ends prematurely or the end of
    file was not reached when @a strict was set to true
    @throw parse_error.112 if a parse error occurs
    @throw parse_error.113 if a string could not be parsed successfully

    @complexity Linear in the size of the input @a i.

    @liveexample{The example shows the deserialization of a byte vector in
    UBJSON format to a JSON value.,from_ubjson}

    @sa http://ubjson.org
    @sa @ref to_ubjson(const basic_json&, const bool, const bool) for the
             analogous serialization
    @sa @ref from_cbor(detail::input_adapter, const bool, const bool) for the
        related CBOR format
    @sa @ref from_msgpack(detail::input_adapter, const bool, const bool) for
        the related MessagePack format

    @since version 3.1.0; added @allow_exceptions parameter since 3.2.0
    */
    static basic_json from_ubjson(detail::input_adapter&& i,
                                  const bool strict = true,
                                  const bool allow_exceptions = true)
    {
        basic_json result;
        detail::json_sax_dom_parser<basic_json> sdp(result, allow_exceptions);
        const bool res = binary_reader(detail::input_adapter(i)).sax_parse(input_format_t::ubjson, &sdp, strict);
        return res ? result : basic_json(value_t::discarded);
    }

    /*!
    @copydoc from_ubjson(detail::input_adapter, const bool, const bool)
    */
    template<typename A1, typename A2,
             detail::enable_if_t<std::is_constructible<detail::input_adapter, A1, A2>::value, int> = 0>
    static basic_json from_ubjson(A1 && a1, A2 && a2,
                                  const bool strict = true,
                                  const bool allow_exceptions = true)
    {
        basic_json result;
        detail::json_sax_dom_parser<basic_json> sdp(result, allow_exceptions);
        const bool res = binary_reader(detail::input_adapter(std::forward<A1>(a1), std::forward<A2>(a2))).sax_parse(input_format_t::ubjson, &sdp, strict);
        return res ? result : basic_json(value_t::discarded);
    }

    /// @}

    //////////////////////////
    // JSON Pointer support //
    //////////////////////////

    /// @name JSON Pointer functions
    /// @{

    /*!
    @brief access specified element via JSON Pointer

    Uses a JSON pointer to retrieve a reference to the respective JSON value.
    No bound checking is performed. Similar to @ref operator[](const typename
    object_t::key_type&), `null` values are created in arrays and objects if
    necessary.

    In particular:
    - If the JSON pointer points to an object key that does not exist, it
      is created an filled with a `null` value before a reference to it
      is returned.
    - If the JSON pointer points to an array index that does not exist, it
      is created an filled with a `null` value before a reference to it
      is returned. All indices between the current maximum and the given
      index are also filled with `null`.
    - The special value `-` is treated as a synonym for the index past the
      end.

    @param[in] ptr  a JSON pointer

    @return reference to the element pointed to by @a ptr

    @complexity Constant.

    @throw parse_error.106   if an array index begins with '0'
    @throw parse_error.109   if an array index was not a number
    @throw out_of_range.404  if the JSON pointer can not be resolved

    @liveexample{The behavior is shown in the example.,operatorjson_pointer}

    @since version 2.0.0
    */
    reference operator[](const json_pointer& ptr)
    {
        return ptr.get_unchecked(this);
    }

    /*!
    @brief access specified element via JSON Pointer

    Uses a JSON pointer to retrieve a reference to the respective JSON value.
    No bound checking is performed. The function does not change the JSON
    value; no `null` values are created. In particular, the the special value
    `-` yields an exception.

    @param[in] ptr  JSON pointer to the desired element

    @return const reference to the element pointed to by @a ptr

    @complexity Constant.

    @throw parse_error.106   if an array index begins with '0'
    @throw parse_error.109   if an array index was not a number
    @throw out_of_range.402  if the array index '-' is used
    @throw out_of_range.404  if the JSON pointer can not be resolved

    @liveexample{The behavior is shown in the example.,operatorjson_pointer_const}

    @since version 2.0.0
    */
    const_reference operator[](const json_pointer& ptr) const
    {
        return ptr.get_unchecked(this);
    }

    /*!
    @brief access specified element via JSON Pointer

    Returns a reference to the element at with specified JSON pointer @a ptr,
    with bounds checking.

    @param[in] ptr  JSON pointer to the desired element

    @return reference to the element pointed to by @a ptr

    @throw parse_error.106 if an array index in the passed JSON pointer @a ptr
    begins with '0'. See example below.

    @throw parse_error.109 if an array index in the passed JSON pointer @a ptr
    is not a number. See example below.

    @throw out_of_range.401 if an array index in the passed JSON pointer @a ptr
    is out of range. See example below.

    @throw out_of_range.402 if the array index '-' is used in the passed JSON
    pointer @a ptr. As `at` provides checked access (and no elements are
    implicitly inserted), the index '-' is always invalid. See example below.

    @throw out_of_range.403 if the JSON pointer describes a key of an object
    which cannot be found. See example below.

    @throw out_of_range.404 if the JSON pointer @a ptr can not be resolved.
    See example below.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes in the JSON value.

    @complexity Constant.

    @since version 2.0.0

    @liveexample{The behavior is shown in the example.,at_json_pointer}
    */
    reference at(const json_pointer& ptr)
    {
        return ptr.get_checked(this);
    }

    /*!
    @brief access specified element via JSON Pointer

    Returns a const reference to the element at with specified JSON pointer @a
    ptr, with bounds checking.

    @param[in] ptr  JSON pointer to the desired element

    @return reference to the element pointed to by @a ptr

    @throw parse_error.106 if an array index in the passed JSON pointer @a ptr
    begins with '0'. See example below.

    @throw parse_error.109 if an array index in the passed JSON pointer @a ptr
    is not a number. See example below.

    @throw out_of_range.401 if an array index in the passed JSON pointer @a ptr
    is out of range. See example below.

    @throw out_of_range.402 if the array index '-' is used in the passed JSON
    pointer @a ptr. As `at` provides checked access (and no elements are
    implicitly inserted), the index '-' is always invalid. See example below.

    @throw out_of_range.403 if the JSON pointer describes a key of an object
    which cannot be found. See example below.

    @throw out_of_range.404 if the JSON pointer @a ptr can not be resolved.
    See example below.

    @exceptionsafety Strong guarantee: if an exception is thrown, there are no
    changes in the JSON value.

    @complexity Constant.

    @since version 2.0.0

    @liveexample{The behavior is shown in the example.,at_json_pointer_const}
    */
    const_reference at(const json_pointer& ptr) const
    {
        return ptr.get_checked(this);
    }

    /*!
    @brief return flattened JSON value

    The function creates a JSON object whose keys are JSON pointers (see [RFC
    6901](https://tools.ietf.org/html/rfc6901)) and whose values are all
    primitive. The original JSON value can be restored using the @ref
    unflatten() function.

    @return an object that maps JSON pointers to primitive values

    @note Empty objects and arrays are flattened to `null` and will not be
          reconstructed correctly by the @ref unflatten() function.

    @complexity Linear in the size the JSON value.

    @liveexample{The following code shows how a JSON object is flattened to an
    object whose keys consist of JSON pointers.,flatten}

    @sa @ref unflatten() for the reverse function

    @since version 2.0.0
    */
    basic_json flatten() const
    {
        basic_json result(value_t::object);
        json_pointer::flatten("", *this, result);
        return result;
    }

    /*!
    @brief unflatten a previously flattened JSON value

    The function restores the arbitrary nesting of a JSON value that has been
    flattened before using the @ref flatten() function. The JSON value must
    meet certain constraints:
    1. The value must be an object.
    2. The keys must be JSON pointers (see
       [RFC 6901](https://tools.ietf.org/html/rfc6901))
    3. The mapped values must be primitive JSON types.

    @return the original JSON from a flattened version

    @note Empty objects and arrays are flattened by @ref flatten() to `null`
          values and can not unflattened to their original type. Apart from
          this example, for a JSON value `j`, the following is always true:
          `j == j.flatten().unflatten()`.

    @complexity Linear in the size the JSON value.

    @throw type_error.314  if value is not an object
    @throw type_error.315  if object values are not primitive

    @liveexample{The following code shows how a flattened JSON object is
    unflattened into the original nested JSON object.,unflatten}

    @sa @ref flatten() for the reverse function

    @since version 2.0.0
    */
    basic_json unflatten() const
    {
        return json_pointer::unflatten(*this);
    }

    /// @}

    //////////////////////////
    // JSON Patch functions //
    //////////////////////////

    /// @name JSON Patch functions
    /// @{

    /*!
    @brief applies a JSON patch

    [JSON Patch](http://jsonpatch.com) defines a JSON document structure for
    expressing a sequence of operations to apply to a JSON) document. With
    this function, a JSON Patch is applied to the current JSON value by
    executing all operations from the patch.

    @param[in] json_patch  JSON patch document
    @return patched document

    @note The application of a patch is atomic: Either all operations succeed
          and the patched document is returned or an exception is thrown. In
          any case, the original value is not changed: the patch is applied
          to a copy of the value.

    @throw parse_error.104 if the JSON patch does not consist of an array of
    objects

    @throw parse_error.105 if the JSON patch is malformed (e.g., mandatory
    attributes are missing); example: `"operation add must have member path"`

    @throw out_of_range.401 if an array index is out of range.

    @throw out_of_range.403 if a JSON pointer inside the patch could not be
    resolved successfully in the current JSON value; example: `"key baz not
    found"`

    @throw out_of_range.405 if JSON pointer has no parent ("add", "remove",
    "move")

    @throw other_error.501 if "test" operation was unsuccessful

    @complexity Linear in the size of the JSON value and the length of the
    JSON patch. As usually only a fraction of the JSON value is affected by
    the patch, the complexity can usually be neglected.

    @liveexample{The following code shows how a JSON patch is applied to a
    value.,patch}

    @sa @ref diff -- create a JSON patch by comparing two JSON values

    @sa [RFC 6902 (JSON Patch)](https://tools.ietf.org/html/rfc6902)
    @sa [RFC 6901 (JSON Pointer)](https://tools.ietf.org/html/rfc6901)

    @since version 2.0.0
    */
    basic_json patch(const basic_json& json_patch) const
    {
        // make a working copy to apply the patch to
        basic_json result = *this;

        // the valid JSON Patch operations
        enum class patch_operations {add, remove, replace, move, copy, test, invalid};

        const auto get_op = [](const std::string & op)
        {
            if (op == "add")
            {
                return patch_operations::add;
            }
            if (op == "remove")
            {
                return patch_operations::remove;
            }
            if (op == "replace")
            {
                return patch_operations::replace;
            }
            if (op == "move")
            {
                return patch_operations::move;
            }
            if (op == "copy")
            {
                return patch_operations::copy;
            }
            if (op == "test")
            {
                return patch_operations::test;
            }

            return patch_operations::invalid;
        };

        // wrapper for "add" operation; add value at ptr
        const auto operation_add = [&result](json_pointer & ptr, basic_json val)
        {
            // adding to the root of the target document means replacing it
            if (ptr.is_root())
            {
                result = val;
            }
            else
            {
                // make sure the top element of the pointer exists
                json_pointer top_pointer = ptr.top();
                if (top_pointer != ptr)
                {
                    result.at(top_pointer);
                }

                // get reference to parent of JSON pointer ptr
                const auto last_path = ptr.pop_back();
                basic_json& parent = result[ptr];

                switch (parent.m_type)
                {
                    case value_t::null:
                    case value_t::object:
                    {
                        // use operator[] to add value
                        parent[last_path] = val;
                        break;
                    }

                    case value_t::array:
                    {
                        if (last_path == "-")
                        {
                            // special case: append to back
                            parent.push_back(val);
                        }
                        else
                        {
                            const auto idx = json_pointer::array_index(last_path);
                            if (JSON_UNLIKELY(static_cast<size_type>(idx) > parent.size()))
                            {
                                // avoid undefined behavior
                                JSON_THROW(out_of_range::create(401, "array index " + std::to_string(idx) + " is out of range"));
                            }
                            else
                            {
                                // default case: insert add offset
                                parent.insert(parent.begin() + static_cast<difference_type>(idx), val);
                            }
                        }
                        break;
                    }

                    default:
                    {
                        // if there exists a parent it cannot be primitive
                        assert(false);  // LCOV_EXCL_LINE
                    }
                }
            }
        };

        // wrapper for "remove" operation; remove value at ptr
        const auto operation_remove = [&result](json_pointer & ptr)
        {
            // get reference to parent of JSON pointer ptr
            const auto last_path = ptr.pop_back();
            basic_json& parent = result.at(ptr);

            // remove child
            if (parent.is_object())
            {
                // perform range check
                auto it = parent.find(last_path);
                if (JSON_LIKELY(it != parent.end()))
                {
                    parent.erase(it);
                }
                else
                {
                    JSON_THROW(out_of_range::create(403, "key '" + last_path + "' not found"));
                }
            }
            else if (parent.is_array())
            {
                // note erase performs range check
                parent.erase(static_cast<size_type>(json_pointer::array_index(last_path)));
            }
        };

        // type check: top level value must be an array
        if (JSON_UNLIKELY(not json_patch.is_array()))
        {
            JSON_THROW(parse_error::create(104, 0, "JSON patch must be an array of objects"));
        }

        // iterate and apply the operations
        for (const auto& val : json_patch)
        {
            // wrapper to get a value for an operation
            const auto get_value = [&val](const std::string & op,
                                          const std::string & member,
                                          bool string_type) -> basic_json &
            {
                // find value
                auto it = val.m_value.object->find(member);

                // context-sensitive error message
                const auto error_msg = (op == "op") ? "operation" : "operation '" + op + "'";

                // check if desired value is present
                if (JSON_UNLIKELY(it == val.m_value.object->end()))
                {
                    JSON_THROW(parse_error::create(105, 0, error_msg + " must have member '" + member + "'"));
                }

                // check if result is of type string
                if (JSON_UNLIKELY(string_type and not it->second.is_string()))
                {
                    JSON_THROW(parse_error::create(105, 0, error_msg + " must have string member '" + member + "'"));
                }

                // no error: return value
                return it->second;
            };

            // type check: every element of the array must be an object
            if (JSON_UNLIKELY(not val.is_object()))
            {
                JSON_THROW(parse_error::create(104, 0, "JSON patch must be an array of objects"));
            }

            // collect mandatory members
            const std::string op = get_value("op", "op", true);
            const std::string path = get_value(op, "path", true);
            json_pointer ptr(path);

            switch (get_op(op))
            {
                case patch_operations::add:
                {
                    operation_add(ptr, get_value("add", "value", false));
                    break;
                }

                case patch_operations::remove:
                {
                    operation_remove(ptr);
                    break;
                }

                case patch_operations::replace:
                {
                    // the "path" location must exist - use at()
                    result.at(ptr) = get_value("replace", "value", false);
                    break;
                }

                case patch_operations::move:
                {
                    const std::string from_path = get_value("move", "from", true);
                    json_pointer from_ptr(from_path);

                    // the "from" location must exist - use at()
                    basic_json v = result.at(from_ptr);

                    // The move operation is functionally identical to a
                    // "remove" operation on the "from" location, followed
                    // immediately by an "add" operation at the target
                    // location with the value that was just removed.
                    operation_remove(from_ptr);
                    operation_add(ptr, v);
                    break;
                }

                case patch_operations::copy:
                {
                    const std::string from_path = get_value("copy", "from", true);
                    const json_pointer from_ptr(from_path);

                    // the "from" location must exist - use at()
                    basic_json v = result.at(from_ptr);

                    // The copy is functionally identical to an "add"
                    // operation at the target location using the value
                    // specified in the "from" member.
                    operation_add(ptr, v);
                    break;
                }

                case patch_operations::test:
                {
                    bool success = false;
                    JSON_TRY
                    {
                        // check if "value" matches the one at "path"
                        // the "path" location must exist - use at()
                        success = (result.at(ptr) == get_value("test", "value", false));
                    }
                    JSON_CATCH (out_of_range&)
                    {
                        // ignore out of range errors: success remains false
                    }

                    // throw an exception if test fails
                    if (JSON_UNLIKELY(not success))
                    {
                        JSON_THROW(other_error::create(501, "unsuccessful: " + val.dump()));
                    }

                    break;
                }

                case patch_operations::invalid:
                {
                    // op must be "add", "remove", "replace", "move", "copy", or
                    // "test"
                    JSON_THROW(parse_error::create(105, 0, "operation value '" + op + "' is invalid"));
                }
            }
        }

        return result;
    }

    /*!
    @brief creates a diff as a JSON patch

    Creates a [JSON Patch](http://jsonpatch.com) so that value @a source can
    be changed into the value @a target by calling @ref patch function.

    @invariant For two JSON values @a source and @a target, the following code
    yields always `true`:
    @code {.cpp}
    source.patch(diff(source, target)) == target;
    @endcode

    @note Currently, only `remove`, `add`, and `replace` operations are
          generated.

    @param[in] source  JSON value to compare from
    @param[in] target  JSON value to compare against
    @param[in] path    helper value to create JSON pointers

    @return a JSON patch to convert the @a source to @a target

    @complexity Linear in the lengths of @a source and @a target.

    @liveexample{The following code shows how a JSON patch is created as a
    diff for two JSON values.,diff}

    @sa @ref patch -- apply a JSON patch
    @sa @ref merge_patch -- apply a JSON Merge Patch

    @sa [RFC 6902 (JSON Patch)](https://tools.ietf.org/html/rfc6902)

    @since version 2.0.0
    */
    static basic_json diff(const basic_json& source, const basic_json& target,
                           const std::string& path = "")
    {
        // the patch
        basic_json result(value_t::array);

        // if the values are the same, return empty patch
        if (source == target)
        {
            return result;
        }

        if (source.type() != target.type())
        {
            // different types: replace value
            result.push_back(
            {
                {"op", "replace"}, {"path", path}, {"value", target}
            });
        }
        else
        {
            switch (source.type())
            {
                case value_t::array:
                {
                    // first pass: traverse common elements
                    std::size_t i = 0;
                    while (i < source.size() and i < target.size())
                    {
                        // recursive call to compare array values at index i
                        auto temp_diff = diff(source[i], target[i], path + "/" + std::to_string(i));
                        result.insert(result.end(), temp_diff.begin(), temp_diff.end());
                        ++i;
                    }

                    // i now reached the end of at least one array
                    // in a second pass, traverse the remaining elements

                    // remove my remaining elements
                    const auto end_index = static_cast<difference_type>(result.size());
                    while (i < source.size())
                    {
                        // add operations in reverse order to avoid invalid
                        // indices
                        result.insert(result.begin() + end_index, object(
                        {
                            {"op", "remove"},
                            {"path", path + "/" + std::to_string(i)}
                        }));
                        ++i;
                    }

                    // add other remaining elements
                    while (i < target.size())
                    {
                        result.push_back(
                        {
                            {"op", "add"},
                            {"path", path + "/" + std::to_string(i)},
                            {"value", target[i]}
                        });
                        ++i;
                    }

                    break;
                }

                case value_t::object:
                {
                    // first pass: traverse this object's elements
                    for (auto it = source.cbegin(); it != source.cend(); ++it)
                    {
                        // escape the key name to be used in a JSON patch
                        const auto key = json_pointer::escape(it.key());

                        if (target.find(it.key()) != target.end())
                        {
                            // recursive call to compare object values at key it
                            auto temp_diff = diff(it.value(), target[it.key()], path + "/" + key);
                            result.insert(result.end(), temp_diff.begin(), temp_diff.end());
                        }
                        else
                        {
                            // found a key that is not in o -> remove it
                            result.push_back(object(
                            {
                                {"op", "remove"}, {"path", path + "/" + key}
                            }));
                        }
                    }

                    // second pass: traverse other object's elements
                    for (auto it = target.cbegin(); it != target.cend(); ++it)
                    {
                        if (source.find(it.key()) == source.end())
                        {
                            // found a key that is not in this -> add it
                            const auto key = json_pointer::escape(it.key());
                            result.push_back(
                            {
                                {"op", "add"}, {"path", path + "/" + key},
                                {"value", it.value()}
                            });
                        }
                    }

                    break;
                }

                default:
                {
                    // both primitive type: replace value
                    result.push_back(
                    {
                        {"op", "replace"}, {"path", path}, {"value", target}
                    });
                    break;
                }
            }
        }

        return result;
    }

    /// @}

    ////////////////////////////////
    // JSON Merge Patch functions //
    ////////////////////////////////

    /// @name JSON Merge Patch functions
    /// @{

    /*!
    @brief applies a JSON Merge Patch

    The merge patch format is primarily intended for use with the HTTP PATCH
    method as a means of describing a set of modifications to a target
    resource's content. This function applies a merge patch to the current
    JSON value.

    The function implements the following algorithm from Section 2 of
    [RFC 7396 (JSON Merge Patch)](https://tools.ietf.org/html/rfc7396):

    ```
    define MergePatch(Target, Patch):
      if Patch is an Object:
        if Target is not an Object:
          Target = {} // Ignore the contents and set it to an empty Object
        for each Name/Value pair in Patch:
          if Value is null:
            if Name exists in Target:
              remove the Name/Value pair from Target
          else:
            Target[Name] = MergePatch(Target[Name], Value)
        return Target
      else:
        return Patch
    ```

    Thereby, `Target` is the current object; that is, the patch is applied to
    the current value.

    @param[in] patch  the patch to apply

    @complexity Linear in the lengths of @a patch.

    @liveexample{The following code shows how a JSON Merge Patch is applied to
    a JSON document.,merge_patch}

    @sa @ref patch -- apply a JSON patch
    @sa [RFC 7396 (JSON Merge Patch)](https://tools.ietf.org/html/rfc7396)

    @since version 3.0.0
    */
    void merge_patch(const basic_json& patch)
    {
        if (patch.is_object())
        {
            if (not is_object())
            {
                *this = object();
            }
            for (auto it = patch.begin(); it != patch.end(); ++it)
            {
                if (it.value().is_null())
                {
                    erase(it.key());
                }
                else
                {
                    operator[](it.key()).merge_patch(it.value());
                }
            }
        }
        else
        {
            *this = patch;
        }
    }

    /// @}
};
} // namespace nlohmann

///////////////////////
// nonmember support //
///////////////////////

// specialization of std::swap, and std::hash
namespace std
{
/*!
@brief exchanges the values of two JSON objects

@since version 1.0.0
*/
template<>
inline void swap<nlohmann::json>(nlohmann::json& j1, nlohmann::json& j2) noexcept(
    is_nothrow_move_constructible<nlohmann::json>::value and
    is_nothrow_move_assignable<nlohmann::json>::value
)
{
    j1.swap(j2);
}

/// specialization for std::less<value_t>
/// @note: do not remove the space after '<',
///        see https://github.com/nlohmann/json/pull/679
template<>
struct less< ::nlohmann::detail::value_t>
{
    /*!
    @brief compare two value_t enum values
    @since version 3.0.0
    */
    bool operator()(nlohmann::detail::value_t lhs,
                    nlohmann::detail::value_t rhs) const noexcept
    {
        return nlohmann::detail::operator<(lhs, rhs);
    }
};

} // namespace std

/*!
@brief user-defined string literal for JSON values

This operator implements a user-defined string literal for JSON objects. It
can be used by adding `"_json"` to a string literal and returns a JSON object
if no parse error occurred.

@param[in] s  a string representation of a JSON object
@param[in] n  the length of string @a s
@return a JSON object

@since version 1.0.0
*/
inline nlohmann::json operator "" _json(const char* s, std::size_t n)
{
    return nlohmann::json::parse(s, s + n);
}

/*!
@brief user-defined string literal for JSON pointer

This operator implements a user-defined string literal for JSON Pointers. It
can be used by adding `"_json_pointer"` to a string literal and returns a JSON pointer
object if no parse error occurred.

@param[in] s  a string representation of a JSON Pointer
@param[in] n  the length of string @a s
@return a JSON pointer object

@since version 2.0.0
*/
inline nlohmann::json::json_pointer operator "" _json_pointer(const char* s, std::size_t n)
{
    return nlohmann::json::json_pointer(std::string(s, n));
}

// #include <nlohmann/detail/macro_unscope.hpp>


// restore GCC/clang diagnostic settings
#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic pop
#endif
#if defined(__clang__)
    #pragma GCC diagnostic pop
#endif

// clean up
#undef JSON_CATCH
#undef JSON_THROW
#undef JSON_TRY
#undef JSON_LIKELY
#undef JSON_UNLIKELY
#undef JSON_DEPRECATED
#undef JSON_HAS_CPP_14
#undef JSON_HAS_CPP_17
#undef NLOHMANN_BASIC_JSON_TPL_DECLARATION
#undef NLOHMANN_BASIC_JSON_TPL
#undef NLOHMANN_JSON_HAS_HELPER


#endif

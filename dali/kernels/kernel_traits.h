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

#ifndef DALI_KERNELS_KERNEL_TRAITS_H_
#define DALI_KERNELS_KERNEL_TRAITS_H_

#include <type_traits>
#include <tuple>
#include "dali/kernels/kernel_params.h"
#include "dali/kernels/util.h"
#include "dali/kernels/tuple_helpers.h"
#include "dali/kernels/kernel_req.h"

namespace dali {
namespace kernels {

namespace detail {

template <typename T>
struct is_input : std::false_type {};

template <typename StorageBackend, typename T, int dim>
struct is_input<InList<StorageBackend, T, dim>> : std::true_type {};

template <typename StorageBackend, typename T, int dim>
struct is_input<InTensor<StorageBackend, T, dim>> : std::true_type {};

template <typename T>
struct is_output : std::false_type {};

template <typename StorageBackend, typename T, int dim>
struct is_output<TensorListView<StorageBackend, T, dim>>
: std::integral_constant<bool, !std::is_const<T>::value> {};

template <typename StorageBackend, typename T, int dim>
struct is_output<TensorView<StorageBackend, T, dim>>
: std::integral_constant<bool, !std::is_const<T>::value> {};

template <typename T>
struct is_kernel_arg : std::true_type {};

template <typename StorageBackend, typename T, int dim>
struct is_kernel_arg<TensorListView<StorageBackend, T, dim>> : std::false_type {};

template <typename StorageBackend, typename T, int dim>
struct is_kernel_arg<TensorView<StorageBackend, T, dim>> : std::false_type {};

template <>
struct is_kernel_arg<KernelContext> : std::false_type {};

template <typename T, template <typename> class Predicate, bool value = Predicate<
  typename std::remove_const<typename std::remove_reference<T>::type>::type
  >::value>
struct filter {
  using type = std::tuple<T>;
};

template <typename T, template <typename> class Predicate>
struct filter<T, Predicate, false> {
  using type = std::tuple<>;
};


template <typename F>
struct input_lists;

template <typename... Args>
struct input_lists<void(Args...)> {
  using type = dali::detail::tuple_cat_t<
    typename filter<Args, is_input>::type...
  >;
};

template <typename F>
struct output_lists;

template <typename... Args>
struct output_lists<void(Args...)> {
  using type = dali::detail::tuple_cat_t<
    typename filter<Args, is_output>::type...
  >;
};

template <typename F>
struct kernel_arg_params;

template <typename... Args>
struct kernel_arg_params<void(Args...)> {
  using type = dali::detail::tuple_cat_t<
    typename filter<Args, is_kernel_arg>::type...
  >;
};

template <typename Kernel>
struct kernel_run_inputs {
  using type = typename detail::input_lists<decltype(Kernel::Run)>::type;
};


template <typename Kernel>
struct kernel_run_outputs {
  using type = typename detail::output_lists<decltype(Kernel::Run)>::type;
};


template <typename Kernel>
struct kernel_run_args {
  using type = typename detail::kernel_arg_params<decltype(Kernel::Run)>::type;
};


IMPL_HAS_NESTED_TYPE(Inputs)
IMPL_HAS_NESTED_TYPE(Outputs)
IMPL_HAS_NESTED_TYPE(Args)

template <typename Kernel, bool has_explicit_def = has_type_Inputs<Kernel>::value>
struct KernelInputs {
  using type = typename Kernel::Inputs;
};

template <typename Kernel>
struct KernelInputs<Kernel, false> {
  using type = typename kernel_run_inputs<Kernel>::type;
};

template <typename Kernel, bool has_explicit_def = has_type_Outputs<Kernel>::value>
struct KernelOutputs {
  using type = typename Kernel::Outputs;
};

template <typename Kernel>
struct KernelOutputs<Kernel, false> {
  using type = typename kernel_run_outputs<Kernel>::type;
};

template <typename Kernel, bool has_explicit_def = has_type_Args<Kernel>::value>
struct KernelArgs {
  using type = typename Kernel::Args;
};

template <typename Kernel>
struct KernelArgs<Kernel, false> {
  using type = typename kernel_run_args<Kernel>::type;
};

}  // namespace detail

/// @brief Tells what inputs a kernel takes
///
/// If there's a type `Kernel::Inputs`, then this type is returned.
/// Otherwise, it's a tuple of all `InList` parameters from `Kernel::Run` signature.
template <typename Kernel>
using kernel_inputs = typename detail::KernelInputs<Kernel>::type;

/// @brief Tells what outputs a kernel produces
///
/// If there's a type `Kernel::Outputs`, then this type is returned.
/// Otherwise, it's a tuple of all `OutList` parameters from `Kernel::Run` signature.
template <typename Kernel>
using kernel_outputs = typename detail::KernelOutputs<Kernel>::type;

/// @brief Tells what extra arguments a kernel takes
///
/// If there's a type `Kernel::Args`, then this type is returned.
/// Otherwise returns all parameters to `Kernel::Run` that are neither
/// `InList`, `OutList` or KernelContext.
template <typename Kernel>
using kernel_args = typename detail::KernelArgs<Kernel>::type;

namespace detail {

IMPL_HAS_UNIQUE_FUNCTION(Run)
IMPL_HAS_UNIQUE_FUNCTION(GetRequirements)

template <typename Kernel>
std::is_same<void, decltype(apply_all(Kernel::Run,
    std::declval<KernelContext&>(),
    std::declval<kernel_outputs<Kernel>>(),
    std::declval<kernel_inputs<Kernel>>(),
    std::declval<kernel_args<Kernel>>() ))>
  IsKernelRunnable(Kernel*);

std::false_type IsKernelRunnable(...);

template <typename Kernel>
std::is_same<KernelRequirements, decltype(apply_all(Kernel::GetRequirements,
    std::declval<KernelContext&>(),
    std::declval<kernel_inputs<Kernel>>(),
    std::declval<kernel_args<Kernel>>() ))>
  CanGetRequirements(Kernel*);

std::false_type CanGetRequirements(...);

template <typename Kernel,
    bool assert_,
    bool is_runnable = decltype(IsKernelRunnable(static_cast<Kernel*>(nullptr)))::value,
    bool can_get_requirements = decltype(CanGetRequirements(static_cast<Kernel*>(nullptr)))::value
    >
struct check_kernel_params : std::integral_constant<bool, is_runnable && can_get_requirements> {
    static_assert(!assert_ || is_runnable,
      "Kernel::Run method cannot be run with inferred arguments.\n"
      "Check argument order (context, [outputs], [inputs], [arguments]).\n"
      "Check return type = void");

    static_assert(!assert_ || can_get_requirements,
      "Kernel::GetRequirements method cannot be run with inferred arguments.\n"
      "Check argument order (context, [inputs], [arguments]).\n"
      "Check return type = KernelRequirements");
};

template <typename Kernel, bool assert_,
    bool has_get_requirements = has_unique_function_GetRequirements<Kernel>::value>
struct check_kernel_has_get_requirements : std::false_type {
  static_assert(!assert_ || has_get_requirements,
  "Kernel class must have a unique, static GetRequirements function");
};

template <typename Kernel, bool assert_>
struct check_kernel_has_get_requirements<Kernel, assert_, true>
 : check_kernel_params<Kernel, assert_> {};

template <typename Kernel, bool assert_, bool has_run = has_unique_function_Run<Kernel>::value>
struct check_kernel_has_run : std::false_type {
  static_assert(!assert_ || has_run, "Kernel class must have a unique, static Run function");
};

template <typename Kernel, bool assert_>
struct check_kernel_has_run<Kernel, assert_, true>
 : check_kernel_has_get_requirements<Kernel, assert_> {};

}  // namespace detail

template <typename Kernel, bool assert_ = true>
using check_kernel = detail::check_kernel_has_run<Kernel, assert_>;

template <typename Kernel>
using is_kernel = check_kernel<Kernel, false>;

}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_KERNEL_TRAITS_H_

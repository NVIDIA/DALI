// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_MATH_EXPRESSIONS_BROADCASTING_H_
#define DALI_OPERATORS_MATH_EXPRESSIONS_BROADCASTING_H_

#include <utility>
#include "dali/core/api_helper.h"
#include "dali/core/small_vector.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {
namespace expr {

/**
 * @brief Returns the number of dimensions of the broadcast shape
 */
DLL_PUBLIC int BroadcastNdim(span<const TensorShape<>*> shapes);
DLL_PUBLIC int BroadcastNdim(span<const TensorListShape<>*> shapes);

/**
 * @brief Verifies that all shapes have the same number of samples
 */
DLL_PUBLIC void CheckNumSamples(span<const TensorListShape<>*> shapes);

/**
 * @brief Calculates the resulting shape of broadcasting two or more compatible shapes.
 *        It is allowed that the shape has different number of dimensions, in which case
 *        the shape with fewer dimensions will be prepended with leading dimensions
 *        with extent 1.
 *        Example: lhs=(10, 10, 3), rhs=(1, 3) -> result=(10, 10, 3)
 *
 * @param result resulting shape (should have as many dimensions as the max dimension of inputs)
 * @param lhs shape of left hand side operand
 * @param rhs shape of right hand side operand
 */
DLL_PUBLIC void BroadcastShape(TensorShape<>& result, span<const TensorShape<>*> shapes);
DLL_PUBLIC void BroadcastShape(TensorListShape<>& result, span<const TensorListShape<>*> shapes);

template <typename Shape>
void BroadcastShape(Shape &result, const Shape& a, const Shape& b) {
  std::array<const Shape*, 2> arr = {&a, &b};
  BroadcastShape(result, make_span(arr));
}

template <typename Shape>
void BroadcastShape(Shape &result, const Shape &a, const Shape &b, const Shape &c) {
  std::array<const Shape*, 3> arr = {&a, &b, &c};
  BroadcastShape(result, make_span(arr));
}

/**
 * @brief Verifies whether two shapes or more can be broadcast
 *        Two shapes can be broadcast if all the extents are either
 *        equal or one of them is equal to one for the `ndim` rightmost
 *        dimensions, being `ndim` the minimum number of dimensions of
 *        the two.
 *        If the shapes are identical (no broadcasting needed), true will
 *        be returned.
 */
DLL_PUBLIC bool CanBroadcast(span<const TensorShape<>*> shapes);
DLL_PUBLIC bool CanBroadcast(span<const TensorListShape<>*> shapes);

template <typename Shape>
bool CanBroadcast(const Shape &a, const Shape &b) {
  std::array<const Shape*, 2> arr = {&a, &b};
  return CanBroadcast(make_span(arr));
}

template <typename Shape>
bool CanBroadcast(const Shape &a, const Shape &b, const Shape& c) {
  std::array<const Shape*, 3> arr = {&a, &b, &c};
  return CanBroadcast(make_span(arr));
}


/**
 * @brief Returns true if the shapes require broadcasting
 *        Two or more shapes require broadcasting if they are not scalar-like
 *        and their shapes are not equal.
 * @remarks This function does not check whether the shapes can be broadcast
 *          For this, use CanBroadcast
 */
DLL_PUBLIC bool NeedBroadcasting(span<const TensorShape<>*> shapes);
DLL_PUBLIC bool NeedBroadcasting(span<const TensorListShape<>*> shapes);

template <typename Shape>
bool NeedBroadcasting(const Shape &a, const Shape &b) {
  std::array<const Shape*, 2> arr = {&a, &b};
  return NeedBroadcasting(make_span(arr));
}

template <typename Shape>
bool NeedBroadcasting(const Shape &a, const Shape &b, const Shape& c) {
  std::array<const Shape*, 3> arr = {&a, &b, &c};
  return NeedBroadcasting(make_span(arr));
}

/**
 * @brief Calculates strides to cover a possibly broadcast shape.
 *        The stride for those broadcast dimensions is set to 0
 */
DLL_PUBLIC TensorShape<> StridesForBroadcasting(const TensorShape<> &out_sh,
                                                const TensorShape<> &in_sh,
                                                const TensorShape<> &in_strides);

/**
 * @brief Expands a shape to have at least ndim dimensions, adding leading dimensions
 *        with extent 1 if necessary. `sh` is expected to have no more than ndim dimensions.
 */
DLL_PUBLIC void ExpandToNDims(TensorShape<> &sh, int ndim);

/**
 * @brief It simplifies shapes for arithmetic op execution with broadcasting.
 *        It detects and collapses adjacent dimensions into groups of dimensions
 *        that either broadcast or not.
 * @param shapes span of shapes to broadcast
 * @remarks For shapes that don't need broadcasting, it results in a 1D shape.
 */
void SimplifyShapesForBroadcasting(span<TensorShape<> *> shapes);
DLL_PUBLIC void SimplifyShapesForBroadcasting(TensorShape<> &a, TensorShape<> &b);
DLL_PUBLIC void SimplifyShapesForBroadcasting(TensorShape<> &a, TensorShape<> &b, TensorShape<> &c);

/**
 * @brief Throws an error when the number of dimensions in the simplified shapes exceeds 6
 */
DLL_PUBLIC void CheckBroadcastingSimplifiedDim(int ndim);

}  // namespace expr
}  // namespace dali

#endif  // DALI_OPERATORS_MATH_EXPRESSIONS_BROADCASTING_H_

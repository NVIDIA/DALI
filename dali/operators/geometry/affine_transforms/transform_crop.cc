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

#include "dali/operators/geometry/affine_transforms/transform_base_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(transforms__Crop)
  .DocStr(R"code(Produces an affine transform matrix that maps a reference coordinate space to another one.

This transform can be used to adjust coordinates after a crop operation so that a ``from_start`` point will
be mapped to ``to_start`` and ``from_end`` will be mapped to ``to_end``.

If another transform matrix is passed as an input, the operator applies the transformation to the matrix provided.

.. note::
    The output of this operator can be fed directly to ``CoordTransform`` and ``WarpAffine`` operators.
)code")
  .AddOptionalArg(
    "from_start",
    R"code(The lower bound of the original coordinate space.

.. note::
    If left empty, a vector of zeros will be assumed.
    If a single value is provided, it will be repeated to match the number of dimensions
)code",
    std::vector<float>{0.0}, true)
  .AddOptionalArg(
    "from_end",
    R"code(The upper bound of the original coordinate space.

.. note::
    If left empty, a vector of ones will be assumed.
    If a single value is provided, it will be repeated to match the number of dimensions
)code",
    std::vector<float>{1.0}, true)
  .AddOptionalArg(
    "to_start",
    R"code(The lower bound of the destination coordinate space.

.. note::
    If left empty, a vector of zeros will be assumed.
    If a single value is provided, it will be repeated to match the number of dimensions
)code",
    std::vector<float>{0.0}, true)
  .AddOptionalArg(
    "to_end",
    R"code(The upper bound of the destination coordinate space.

.. note::
    If left empty, a vector of ones will be assumed.
    If a single value is provided, it will be repeated to match the number of dimensions
)code",
    std::vector<float>{1.0}, true)
  .AddOptionalArg(
    "absolute",
    R"code(If set to true, start and end coordinates will be swapped if start > end.)code",
    false)
  .NumInput(0, 1)
  .NumOutput(1)
  .AddParent("TransformAttr");

/**
 * @brief Scale transformation.
 */
class TransformCropCPU
    : public TransformBaseOp<CPUBackend, TransformCropCPU> {
 public:
  using SupportedDims = dims<1, 2, 3, 4, 5, 6>;

  explicit TransformCropCPU(const OpSpec &spec) :
      TransformBaseOp<CPUBackend, TransformCropCPU>(spec),
      from_start_("from_start", spec),
      from_end_("from_end", spec),
      to_start_("to_start", spec),
      to_end_("to_end", spec) {
  }

  template <typename T, int mat_dim>
  void DefineTransforms(span<affine_mat_t<T, mat_dim>> matrices) {
    constexpr int ndim = mat_dim - 1;
    assert(matrices.size() <= static_cast<int>(from_start_.size()));
    assert(matrices.size() <= static_cast<int>(from_end_.size()));
    assert(matrices.size() <= static_cast<int>(to_start_.size()));
    assert(matrices.size() <= static_cast<int>(to_end_.size()));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      vec<ndim> from_start = as_vec<ndim>(from_start_[i]);
      vec<ndim> from_end = as_vec<ndim>(from_end_[i]);
      vec<ndim> to_start = as_vec<ndim>(to_start_[i]);
      vec<ndim> to_end = as_vec<ndim>(to_end_[i]);
      mat = affine_mat_t<T, mat_dim>::identity();

      if (absolute_) {
        for (int d = 0; d < ndim; d++) {
          if (from_start[d] > from_end[d]) std::swap(from_start[d], from_end[d]);
          if (to_start[d] > to_end[d]) std::swap(to_start[d], to_end[d]);
        }
      }
      vec<ndim> from_extent = from_end - from_start;
      vec<ndim> to_extent = to_end - to_start;

      for (int d = 0; d < ndim; d++) {
        float scale = to_extent[d] / from_extent[d];
        mat(d, d) = scale;
        mat(d, ndim) = to_start[d] - scale * from_start[d];
      }
    }
  }

  void ProcessArgs(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    from_start_.Acquire(spec, ws, nsamples_, true);
    from_end_.Acquire(spec, ws, nsamples_, true);
    to_start_.Acquire(spec, ws, nsamples_, true);
    to_end_.Acquire(spec, ws, nsamples_, true);
    auto sizes = std::array<ptrdiff_t, 4>{
        from_start_[0].num_elements(), from_end_[0].num_elements(),
        to_start_[0].num_elements(), to_end_[0].num_elements()};
    ndim_ = *std::max_element(sizes.begin(), sizes.end());
    DALI_ENFORCE(std::all_of(sizes.begin(), sizes.end(),
        [&](size_t sz){ return static_cast<int>(sz) == ndim_ || sz == 1; }),
      "Arguments ``from_start``, ``from_end``, ``to_start`` and ``to_end`` should"
      " have the same number of dimensions or be a single element which will be broadcast"
      " to all dimensions");
    absolute_ = spec.GetArgument<bool>("absolute");
  }

  bool IsConstantTransform() const {
    return !from_start_.IsArgInput() && !from_end_.IsArgInput() &&
           !to_start_.IsArgInput() && !to_end_.IsArgInput();
  }

 private:
  ArgValue<float, 1> from_start_;
  ArgValue<float, 1> from_end_;
  ArgValue<float, 1> to_start_;
  ArgValue<float, 1> to_end_;
  bool absolute_ = false;
};

DALI_REGISTER_OPERATOR(transforms__Crop, TransformCropCPU, CPU);

}  // namespace dali

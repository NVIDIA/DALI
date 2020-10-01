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

DALI_SCHEMA(TransformCrop)
  .DocStr(R"code(Produces an affine transform matrix that maps a reference to another one.

This transform can be used to adjust coordinates after a crop operation so that a ``from_start`` point will
be mapped to ``to_start`` and ``from_end`` will be mapped to ``to_end``.

If another transform matrix is passed as an input, the operator applies the transformation to the matrix provided.

.. note::
    The output of this operator can be fed directly to the ``MT`` argument of ``CoordTransform`` operator.
)code")
  .AddArg(
    "from_start",
    R"code(The lower bound of the original coordinate space.)code",
    DALI_FLOAT_VEC, true)
  .AddArg(
    "from_end",
    R"code(The upper bound of the original coordinate space.)code",
    DALI_FLOAT_VEC, true)
  .AddArg(
    "to_start",
    R"code(The lower bound of the destination coordinate space.)code",
    DALI_FLOAT_VEC, true)
  .AddArg(
    "to_end",
    R"code(The upper bound of the destination coordinate space.)code",
    DALI_FLOAT_VEC, true)
  .AddOptionalArg(
    "absolute",
    R"code(If set to true, start and end coordinates will be swapped if start > end.)code",
    false, true)
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
    assert(matrices.size() == static_cast<int>(from_start_.size()));
    assert(matrices.size() == static_cast<int>(from_end_.size()));
    assert(matrices.size() == static_cast<int>(to_start_.size()));
    assert(matrices.size() == static_cast<int>(to_end_.size()));
    for (int i = 0; i < matrices.size(); i++) {
      auto &mat = matrices[i];
      const vec<ndim> &from_start = *reinterpret_cast<vec<ndim>*>(from_start_[i].data());
      const vec<ndim> &from_end = *reinterpret_cast<vec<ndim>*>(from_end_[i].data());
      const vec<ndim> &to_start = *reinterpret_cast<vec<ndim>*>(to_start_[i].data());
      const vec<ndim> &to_end = *reinterpret_cast<vec<ndim>*>(to_end_[i].data());
      mat = affine_mat_t<T, mat_dim>::identity();
      vec<ndim> from_extent = from_end - from_start;
      vec<ndim> to_extent = to_end - to_start;
      vec<ndim> offset = to_start - from_start;

      if (absolute_) {
        for (int d = 0; d < ndim; d++) {
          from_extent[d] = std::abs(from_extent[d]);
          to_extent[d] = std::abs(to_extent[d]);
        }
      }

      for (int d = 0; d < ndim; d++) {
        constexpr float kEps = std::numeric_limits<float>::epsilon();
        DALI_ENFORCE(std::abs(from_extent[d]) > kEps,
          make_string("from_end[d] should be different from from_start[d] for all "
                      "dimensions. Got: from_start=",
                      from_start, " and from_end=", from_end));
        DALI_ENFORCE(std::abs(to_extent[d]) > kEps,
          make_string("to_end[d] should be different from to_start[d] for all "
                      "dimensions. Got: to_start=",
                      to_start, " and to_end=", to_end));
        float scale = to_extent[d] / from_extent[d];
        mat(d, d) = scale; 
        mat(d, ndim) = -scale * offset[d];
      }
    }
  }

  void ProcessArgs(const OpSpec &spec, const workspace_t<CPUBackend> &ws) {
    int repeat = IsConstantTransform() ? 0 : nsamples_;
    assert(from_start_.IsDefined());
    assert(from_end_.IsDefined());
    assert(to_start_.IsDefined());
    assert(to_end_.IsDefined());
    from_start_.Read(spec, ws, repeat);
    from_end_.Read(spec, ws, repeat);
    to_start_.Read(spec, ws, repeat);
    to_end_.Read(spec, ws, repeat);
    absolute_ = spec.GetArgument<bool>("absolute");
    if (from_start_[0].size() != from_end_[0].size() &&
        from_start_[0].size() != to_start_[0].size() &&
        from_start_[0].size() != to_end_[0].size()) {
      DALI_FAIL("Arguments ``from_start``, ``from_end``, ``to_start`` and ``to_end`` should"
                " have the same number of dimensions");
    }
    ndim_ = from_start_[0].size();
  }

  bool IsConstantTransform() const {
    return !from_start_.IsArgInput() && !from_end_.IsArgInput() &&
           !to_start_.IsArgInput() && !to_end_.IsArgInput();
  }

 private:
  Argument<std::vector<float>> from_start_;
  Argument<std::vector<float>> from_end_;
  Argument<std::vector<float>> to_start_;
  Argument<std::vector<float>> to_end_;
  bool absolute_ = false;
};

DALI_REGISTER_OPERATOR(TransformCrop, TransformCropCPU, CPU);

}  // namespace dali

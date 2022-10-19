// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_REMAP_ROTATE_PARAMS_H_
#define DALI_OPERATORS_IMAGE_REMAP_ROTATE_PARAMS_H_

#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/imgproc/warp/affine.h"
#include "dali/kernels/imgproc/warp/mapping_traits.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/operators/image/remap/warp_param_provider.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/format.h"

namespace dali {

template <int spatial_ndim>
using RotateParams = kernels::AffineMapping<spatial_ndim>;

inline std::tuple<ivec2, ivec2> RotatedCanvasSize(TensorShape<2> input_size, double angle) {
  double eps = 1e-2;
  double abs_cos = std::abs(std::cos(angle));
  double abs_sin = std::abs(std::sin(angle));
  int w = input_size[1];
  int h = input_size[0];
  int w_out = std::ceil(abs_cos * w + abs_sin * h - eps);
  int h_out = std::ceil(abs_cos * h + abs_sin * w - eps);
  ivec2 parity;
  if (abs_sin <= abs_cos) {
    // if rotated by less than +/-45deg (or more than +/-135deg),
    // maintain size parity to reduce blur
    parity[0] = w % 2;
    parity[1] = h % 2;
  } else {
    parity[0] = h % 2;
    parity[1] = w % 2;
  }
  return {{w_out, h_out}, parity};
}

inline std::tuple<ivec3, ivec3> RotatedCanvasSize(TensorShape<3> input_shape, vec3 axis,
                                                  double angle) {
  ivec3 in_size = kernels::shape2vec(input_shape);
  float eps = 1e-2f;
  mat3 M = sub<3, 3>(rotation3D(axis, angle));


  mat3 absM;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      absM(i, j) = std::abs(M(i, j));

  ivec3 out_size = ceil_int(absM * vec3(in_size) - eps);
  ivec3 parity;

  // This vector contains indices of dimensions that contribute (relatively) most
  // to the output size.
  // For example, if
  //    out_size.x = 0.4 * in_size.x + 0.7 * in_size.y  + 0.1 * in_size.z
  // then the dominant_src_axis.x is y (1)
  ivec3 dominant_src_axis = { 0, 1, 2 };

  for (int i = 0; i < 3; i++) {
    float maxv = absM(i, dominant_src_axis[i]);
    for (int j = 0; j < 3; j++) {
      if (absM(i, j) > maxv) {
        maxv = absM(i, j);
        dominant_src_axis[i] = j;
      }
    }
  }

  // Here we attempt to keep parity of the output size same as it was for the dominant
  // source dimension - this helps reduce the blur in central area, especially for
  // small angles.
  for (int i = 0; i < 3; i++) {
    parity[i] = in_size[dominant_src_axis[i]] % 2;
  }

  return {out_size, parity};
}

template <typename Backend, int spatial_ndim_, typename BorderType>
class RotateParamProvider
: public WarpParamProvider<Backend, spatial_ndim_, RotateParams<spatial_ndim_>, BorderType> {
 protected:
  static constexpr int spatial_ndim = spatial_ndim_;
  using MappingParams = RotateParams<spatial_ndim>;
  using Base = WarpParamProvider<Backend, spatial_ndim, MappingParams, BorderType>;
  using SpatialShape = typename Base::SpatialShape;
  using Base::ws_;
  using Base::spec_;
  using Base::sequence_extents_;
  using Base::params_gpu_;
  using Base::params_cpu_;
  using Base::num_samples_;
  using Base::out_sizes_;

  void SetParams() override {
    input_shape_ = convert_dim<spatial_ndim + 1>(ws_->template Input<Backend>(0).shape());
    Collect(angles_, "angle", true);

    // For 2D, assume positive CCW rotation when (0,0) denotes top-left corner.
    // For 3D, we just follow the mathematical formula for rotation around arbitrary axis.
    if (spatial_ndim == 2)
      for (auto &a : angles_) a = -a;

    if (spatial_ndim == 3)
      Collect(axes_, "axis", true);
  }

  template <typename T>
  void CopyIgnoreShape(vector<T> &out, const TensorListView<StorageCPU, const T> &TL) {
    int64_t n = TL.num_elements();
    out.resize(n);
    if (!n)
      return;
    int64_t sample_size = TL.shape[0].num_elements();
    int s = 0;  // sample index
    int64_t ofs = 0;  // offset within sample
    for (int64_t i = 0; i < n; i++) {
      if (ofs == sample_size) {
        ofs = 0;
        s++;
        assert(s < TL.num_samples());
        sample_size = TL.shape[s].num_elements();
      }
      out[i] = TL.data[s][ofs++];
    }
  }

  template <typename T, int N>
  void CopyIgnoreShape(vector<vec<N, T>> &out, const TensorListView<StorageCPU, const T> &TL) {
    int64_t n = TL.num_elements() / N;
    out.resize(n);
    if (!n)
      return;
    int64_t sample_size = TL.shape[0].num_elements();
    int s = 0;  // sample index
    int64_t ofs = 0;  // offset within sample
    for (int64_t i = 0; i < n; i++) {
      for (int j = 0; j < N; j++) {
        if (ofs == sample_size) {
          ofs = 0;
          s++;
          assert(s < TL.num_samples());
          sample_size = TL.shape[s].num_elements();
        }
        out[i][j] = TL.data[s][ofs++];
      }
    }
  }

  template <typename T>
  enable_if_t<is_scalar<T>::value>
  Collect(std::vector<T> &v, const std::string &name, bool required) {
    if (spec_->HasTensorArgument(name)) {
      auto arg_view = dali::view<const T>(ws_->ArgumentInput(name));
      int n = arg_view.num_elements();
      DALI_ENFORCE(n == num_samples_, make_string(
        "Unexpected number of elements in argument `", name, "`: ", n,
        "; expected: ", num_samples_));
      CopyIgnoreShape(v, arg_view);
    } else {
      T scalar;
      v.clear();

      if (required)
        scalar = spec_->template GetArgument<T>(name);

      if (required || spec_->TryGetArgument(scalar, name))
        v.resize(num_samples_, scalar);
    }
  }

  template <int N, typename T>
  void Collect(std::vector<vec<N, T>> &v, const std::string &name, bool required) {
    if (spec_->HasTensorArgument(name)) {
      auto arg_view = dali::view<const T>(ws_->ArgumentInput(name));
      int n = arg_view.num_elements();
      DALI_ENFORCE(n == N * num_samples_, make_string(
        "Unexpected number of elements in argument `", name, "`: ", n,
        "; expected: ", num_samples_));
      CopyIgnoreShape(v, arg_view);
    } else {
      v.clear();

      std::vector<T> tmp;

      if (!spec_->TryGetArgument(tmp, name)) {
        if (required)
          DALI_FAIL(make_string("Argument `", name, "` is required"));
        return;
      }

      DALI_ENFORCE(static_cast<int>(tmp.size()) == N,
        make_string("Argument `", name, "` must be a ", N, "D vector"));

      vec<N, T> fill;
      for (int i = 0; i < N; i++)
        fill[i] = tmp[i];

      v.resize(num_samples_, fill);
    }
  }

  void AdjustParams() override {
    AdjustParams(std::integral_constant<int, spatial_ndim>());
  }

  void AdjustParams(std::integral_constant<int, 2>) {
    using kernels::shape2vec;
    using kernels::skip_dim;
    assert(input_shape_.num_samples() == num_samples_);
    assert(static_cast<int>(out_sizes_.size()) == num_samples_);

    auto *params = this->template AllocParams<mm::memory_kind::host>();
    for (int i = 0; i < num_samples_; i++) {
      ivec2 in_size = shape2vec(skip_dim<2>(input_shape_[i]));
      ivec2 out_size = shape2vec(out_sizes_[i]);

      float a = deg2rad(angles_[i]);
      mat3 M = translation(in_size*0.5f) * rotation2D(-a) * translation(-out_size*0.5f);
      params[i] = sub<2, 3>(M);
    }
  }

  void AdjustParams(std::integral_constant<int, 3>) {
    using kernels::shape2vec;
    using kernels::skip_dim;
    assert(input_shape_.num_samples() == num_samples_);
    assert(static_cast<int>(out_sizes_.size()) == num_samples_);

    auto *params = this->template AllocParams<mm::memory_kind::host>();
    for (int i = 0; i < num_samples_; i++) {
      ivec3 in_size = shape2vec(skip_dim<3>(input_shape_[i]));
      ivec3 out_size = shape2vec(out_sizes_[i]);

      vec3 axis = axes_[i];
      float a = deg2rad(angles_[i]);
      // NOTE: This is a destination-to-source transform - hence, angle is reversed
      mat4 M = translation(in_size*0.5f) * rotation3D(axis, -a) * translation(-out_size*0.5f);
      params[i] = sub<3, 4>(M);
    }
  }

  void InferSize() override {
    assert(sequence_extents_);
    const auto &sequence_extents = *sequence_extents_;
    VALUE_SWITCH(sequence_extents.sample_dim(), ndims_unfolded, (0, 1),
      (InferSize(sequence_extents.template to_static<ndims_unfolded>());),
      (DALI_FAIL("Unsupported number of frame extents")));
  }

  template <int ndims_unfolded>
  void InferSize(const TensorListShape<ndims_unfolded> sequence_extents) {
    assert(sequence_extents.num_elements() == num_samples_);
    assert(static_cast<int>(out_sizes_.size()) == num_samples_);
    constexpr auto ndim_tag = std::integral_constant<int, spatial_ndim>();
    for (int frame_idx = 0, seq_idx = 0; seq_idx < sequence_extents.num_samples(); seq_idx++) {
      auto num_frames = volume(sequence_extents[seq_idx]);
      if (num_frames == 0) {
        continue;
      }
      ivec<spatial_ndim> acc_parity, acc_shape;
      std::tie(acc_shape, acc_parity) = InferSize(ndim_tag, frame_idx);
      // acc_shape is (extentwise) max of output shapes of all frames in the sample
      for (int i = 1; i < num_frames; i++) {
        ivec<spatial_ndim> parity, shape;
        std::tie(shape, parity) = InferSize(ndim_tag, frame_idx + i);
        acc_parity += parity;
        acc_shape = max(acc_shape, shape);
      }
      // do the correction of shape extents parity by a majority vote, so that at least half
      // of the frames in the sequence have the desired parity of each extent
      acc_shape += (acc_shape % 2) ^ (2 * acc_parity > num_frames);
      // set the output shape to all frames
      auto shape = kernels::vec2shape(acc_shape);
      for (int i = 0; i < num_frames; i++) {
        out_sizes_[frame_idx++] = shape;
      }
    }
  }

  auto InferSize(std::integral_constant<int, 2>, int sample_idx) {
    auto in_shape = kernels::skip_dim<2>(input_shape_[sample_idx]);
    return RotatedCanvasSize(in_shape, deg2rad(angles_[sample_idx]));
  }

  auto InferSize(std::integral_constant<int, 3>, int sample_idx) {
    auto in_shape = kernels::skip_dim<3>(input_shape_[sample_idx]);
    return RotatedCanvasSize(in_shape, axes_[sample_idx], deg2rad(angles_[sample_idx]));
  }

  bool ShouldInferSize() const override {
    return !this->HasExplicitSize() && !this->KeepOriginalSize();
  }

  bool KeepOriginalSize() const override {
    return spec_->template GetArgument<bool>("keep_size");
  }

  std::vector<float> angles_;
  std::vector<vec3> axes_;
  TensorListShape<spatial_ndim + 1> input_shape_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_REMAP_ROTATE_PARAMS_H_

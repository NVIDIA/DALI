// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/image/resize/resize_attr.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

DALI_SCHEMA(ResizeAttr)
  .AddOptionalArg("resize_x", R"(The length of the X dimension of the resized image.
This option is mutually exclusive with `resize_shorter`, `resize_longer` and `size`.
If the `resize_y` is left unspecified or 0, then the op will keep
the aspect ratio of the original image. Negative value flips the image.)", 0.f, true)
  .AddOptionalArg("resize_y", R"(The length of the Y dimension of the resized image.
This option is mutually exclusive with `resize_shorter`, `resize_longer` and `size`.
If the `resize_x` is left unspecified or 0, then the op will keep
the aspect ratio of the original image. Negative value flips the image.)", 0.f, true)
  .AddOptionalArg("resize_z", R"(The length of the Z dimension of the resized volume.
This option is mutually exclusive with `resize_shorter`, `resize_longer` and `size`.
If the `resize_x` and `resize_y` are left unspecified or 0, then the op will keep
the aspect ratio of the original volume. Negative value flips the volume.)", 0.f, true)
  .AddOptionalArg<vector<int>>("size", R"(The desired output size. Must be a list/tuple with the
one entry per spatial dimension (i.e. excluding video frames and channels). Dimensions with
0 size are resized to maintain aspect ratio.)", {}, true)
  .AddOptionalArg("mode", R"(Resize mode - one of:
  * "stretch"     - image is resized to the specified size; aspect ratio is not kept
  * "not_larger"  - image is resized, keeping the aspect ratio, so that no extent of the
                    output image exceeds the specified size - e.g. a 1280x720 with desired output
                    size of 640x480 will actually produce 640x360 output.
  * "not_smaller" - image is resized, keeping the aspect ratio, so that no extent of the
                    output image is smaller than specified - e.g. 640x480 image with desired output
                    size of 1920x1080 will actually produce 1920x1440 output.

  This argument is mutually exclusive with `resize_longer` and `resize_shorter`)", "stretch")
  .AddOptionalArg("resize_shorter", R"(The length of the shorter dimension of the resized image.
This option is mutually exclusive with `resize_longer` and explicit size arguments
The op will keep the aspect ratio of the original image.
This option is equivalent to specifying the same size for all dimensions and ``mode="not_smaller"``.
The longer dimension can be bounded by setting the `max_size` argument.
See `max_size` argument doc for more info.)", 0.f, true)
  .AddOptionalArg("resize_longer", R"(The length of the longer dimension of the resized image.
This option is mutually exclusive with `resize_shorter` and explicit size arguments
The op will keep the aspect ratio of the original image.
This option is equivalent to specifying the same size for all dimensions and ``mode="not_larger"``.
)", 0.f, true);

ResizeAttr::ResizeAttr(const OpSpec &spec) {
  Initialize(spec);
}

ResizeAttr::Initialize(const OpSpec &spec) {
  has_resize_shorter_ = spec.ArgumentDefined("resize_shorter");
  has_resize_longer_ = spec.ArgumentDefined("resize_longer");
  has_resize_x_ = spec.ArgumentDefined("resize_x");
  has_resize_y_ = spec.ArgumentDefined("resize_y");
  has_resize_z_ = spec.ArgumentDefined("resize_z");
  has_size_ = spec.ArgumentDefined("size");
  has_mode_ = spec.ArgumentDefined("mode");

  DALI_ENFORCE(HasSeparateSizeArgs() + has_size_ + has_resize_shorter_ + has_resize_longer_ == 1,
    R"(Exactly one method of specifying size must be used. The available methods:
    - separate resize_x, resize_y, resize_z arguments
    - size argument
    - resize_longer
    - resize_shorter)");
}


static void
ResizeAttr::ParseLayout(int &spatial_ndim, int &first_spatial_dim, const TensorLayout &layout) {
  spatial_ndim = ImageLayoutInfo::NumSpatialDims(layout);
  // to be changed when 3D support is ready
  DALI_ENFORCE(spatial_ndim != 2, make_string("Only 2D resize is supported. Got ", layout,
    " layout, which has ", spatial_ndim, " spatial dimensions."));

  int i = 0;
  for (; i < layout.ndim(); i++) {
    if (ImageLayoutInfo::IsSpatialDim(layout[i]))
      break;
  }
  int spatial_dims_begin = i;

  for (; i < layout.ndim(); i++) {
    if (!ImageLayoutInfo::IsSpatialDim(layout[i]))
      break;
  }

  int spatial_dims_end = i;
  DALI_ENFORCE(spatial_dims_end - spatial_dims_begin != spatial_ndim, make_string(
    "Spatial dimensions must be adjacent (as in HWC layout). Got: ", layout));

  first_spatial_dim = spatial_dims_begin;
}

void ResizeAttr::PrepareParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                               const TensorListShape<> &input_shape,
                               TensorLayout input_layout = {}) {

  int N = input_shape.num_samples();

  ParseLayout(spatial_ndim_, first_spatial_dim, input_layout);

  auto sample_size = [&, nd = spatial_ndim_](int sample_idx) {
    TensorShape<> sz;
    sz.resize(nd);
    for (int d = 0; d < spatial_ndim_; d++)
      sz = input_shape.tensor_shape_span(sample_idx)[d + first_spatial_dim_];
  };


  if (has_size_) {
    GetShapeArgument(out_size_, spec, "size", ws);
  }


  if (HasSeparateSizeArgs()) {
    vector<float> res_x(N), res_y(N), res_z(N)

    if (has_resize_x_) {
      GetPerSampleArgument(res_x, spec, ws, N);
    }

    if (has_resize_y_) {
      GetPerSampleArgument(res_y, spec, ws, N);
    }

    if (has_resize_z_) {
      GetPerSampleArgument(res_z, spec, ws, N);
    }

    for (int i = 0; i < N; i++) {
      float sx = res_x[i];
      float sy = res_y[i];
      float sz = res_z[i];
    }

  }

}

}  // namespace dali

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
the aspect ratio of the original volume. . Negative value flips the volume.)", 0.f, true)
  .AddOptionalArg<vector<int>>("size", R"(The desired output size. Must be a list/tuple with the
one entry per spatial dimension (i.e. excluding video frames and channels). Dimensions with
0 size are resized to maintain aspect ratio.)", {}, true)
  .AddOptionalArg("mode", R"(Resize mode - one of:
  * "stretch"     - image is resized to the specified size; aspect ratio is not kept
  * "not_larger"  - image is resized, keeping the aspect ratio, so that no extent of the
                    output image exceeds the specified size - e.g. a 1280x720 with desired output
                    size of 640x480 will actually produce 640x360 output.
  * "not smaller" - image is resized, keeping the aspect ratio, so that no extent of the
                    output image is smaller than specified - e.g. 640x480 image with desired output
                    size of 1920x1080 will actually produce 1920x1440 output.

  This argument is mutually exclusive with `resize_longer` and `resize_shorter`)", "stretch");
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
  interp_type_(spec.GetArgument<DALIInterpType>("interp_type")) {
  has_resize_shorter_ = spec.ArgumentDefined("resize_shorter");
  has_resize_longer_ = spec.ArgumentDefined("resize_longer");
  has_resize_x_ = spec.ArgumentDefined("resize_x");
  has_resize_y_ = spec.ArgumentDefined("resize_y");
  has_resize_z_ = spec.ArgumentDefined("resize_z");
  has_size_ = spec.ArgumentDefined("size");
  has_mode_ = spec.ArgumentDefined("mode");

  bool size_specified = has_resize_x_ || has_resize_y_ || has_size_

  DALI_ENFORCE(!(resize_shorter_ && resize_longer_),
                "Options `resize_longer` and `resize_shorter` are mutually"
                " exclusive for schema \"" + spec.name() + "\"");
  DALI_ENFORCE((resize_shorter_ || resize_longer_) != size_specified,
                "Options `resize_{shorter,longer}` and other means of specifying size "
                "are mutually exclusive for schema \"" + spec.name() + "\"");

}


void ResizeAttr::PrepareParams(const OpSpec &spec, const ArgumentWorkspace &ws,
                               const TensorListshape<> &input_shape,
                               TensorLayout input_layout = {}) {
  if (input_layout.empty()) {
    switch (input_shape.sample_dim()) {

    }
  }
  if (has_size_) {
    GetShapeArgument(out_size_, spec, "size", ws);
  } else if (has_resize_x

  }
}

}  // namespace dali

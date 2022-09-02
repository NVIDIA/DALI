// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/imgcodec/peek_image_shape.h"
#include "dali/operators/imgcodec/operator_utils.h"
#include "dali/imgcodec/registry.h"
#include "dali/imgcodec/util/output_shape.h"

namespace dali {
namespace imgcodec {

DALI_SCHEMA(experimental__PeekImageShape)
  .DocStr(R"code(Obtains the shape of the encoded image.)code")
  .NumInput(1)
  .NumOutput(1)
  .AddOptionalTypeArg("dtype",
    R"code(Data type, to which the sizes are converted.)code", DALI_INT64)
  .AddOptionalArg("adjust_orientation",
    R"code(Use the EXIF orientation metadata when calculating the shape.)code", true)
  .AddOptionalArg("image_type",
    R"code(Color format of the image.)code", DALI_RGB);

ImgcodecPeekImageShape::ImgcodecPeekImageShape(const OpSpec &spec)
    : Operator<CPUBackend>(spec) {
  output_type_ = spec.GetArgument<DALIDataType>("dtype");
  switch (output_type_) {
  case DALI_INT32:
  case DALI_UINT32:
  case DALI_INT64:
  case DALI_UINT64:
  case DALI_FLOAT:
  case DALI_FLOAT64:
    break;
  default:
    {
      DALI_FAIL(make_string(
        "Operator PeekImageShape can return the output as one of the following:\n"
        "int32, uint32, int64, uint64, float or double;\n"
        "requested: ", output_type_));
      break;
    }
  }
  use_orientation_ = spec.GetArgument<bool>("adjust_orientation");
  image_type_ = spec.GetArgument<DALIImageType>("image_type");
}

bool ImgcodecPeekImageShape::CanInferOutputs() const {
  return true;
}

bool ImgcodecPeekImageShape::SetupImpl(std::vector<OutputDesc> &output_desc,
                                       const HostWorkspace &ws) {
  const auto &input = ws.template Input<CPUBackend>(0);
  size_t batch_size = input.num_samples();
  output_desc.resize(1);
  output_desc[0] = {uniform_list_shape<1>(input.num_samples(), { 3 }), output_type_};
  return true;
}

void ImgcodecPeekImageShape::RunImpl(HostWorkspace &ws) {
  auto &thread_pool = ws.GetThreadPool();
  const auto &input = ws.template Input<CPUBackend>(0);
  auto &output = ws.template Output<CPUBackend>(0);
  DALI_ENFORCE(input.type() == DALI_UINT8 && input.sample_dim() == 1,
                "The input must be a raw, undecoded file stored as a flat uint8 array.");

  for (int i = 0; i < input.num_samples(); i++) {
    thread_pool.AddWork([i, &input, &output, this] (int tid) {
      auto src = SampleAsImageSource(input[i], input.GetMeta(i).GetSourceInfo());
      auto *format = ImageFormatRegistry::instance().GetImageFormat(&src);
      DALI_ENFORCE(format, make_string("Cannot parse the image: ", src.SourceInfo()));
      auto info = format->Parser()->GetInfo(&src);
      TensorShape<> shape;
      DecodeParams params;
      params.format = image_type_;
      params.use_orientation = use_orientation_;
      OutputShape(shape, info, params, {});
      TYPE_SWITCH(output_type_, type2id, Type,
                  (int32_t, uint32_t, int64_t, uint64_t, float, double), (
        auto out = view<Type, 1>(output[i]);
        for (int i = 0; i < 3; ++i) {
          out.data[i] = shape[i];
        }
      ), (DALI_FAIL(make_string("Unsupported type for shapes: ", output_type_))));  // NOLINT
    }, 0);
  }
  thread_pool.RunAll();
}


DALI_REGISTER_OPERATOR(experimental__PeekImageShape, ImgcodecPeekImageShape, CPU);

}  // namespace imgcodec
}  // namespace dali

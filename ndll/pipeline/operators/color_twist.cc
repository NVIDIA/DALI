// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/color_twist.h"
#include <vector>
#include <string>

namespace ndll {

NDLL_OPERATOR_SCHEMA(ColorTwist)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("image_type", "Input/output image type", NDLL_RGB);

NDLL_REGISTER_OPERATOR(ColorIntensity, ColorIntensity<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA_WITH_PARENT(ColorIntensity, ColorTwist)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("R_level", "Intensity of red channel", 1.f)
    .AddOptionalArg("G_level", "Intensity of green channel", 1.f)
    .AddOptionalArg("B_level", "Intensity of blue channel", 1.f)
    .AddOptionalArg("GRAY_level", "Intensity of gray level", 1.f);


NDLL_REGISTER_OPERATOR(ColorOffset, ColorOffset<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA_WITH_PARENT(ColorOffset, ColorTwist)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("R_level", "Intensity of red channel", 0.f)
    .AddOptionalArg("G_level", "Intensity of green channel", 0.f)
    .AddOptionalArg("B_level", "Intensity of blue channel", 0.f)
    .AddOptionalArg("GRAY_level", "Intensity of gray level", 0.f);


NDLL_REGISTER_OPERATOR(HueSaturation, HueSaturation<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA_WITH_PARENT(HueSaturation, ColorTwist)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue", "Hue parameter", 0.f)
    .AddOptionalArg("saturation", "Color saturation level", 1.f);


NDLL_REGISTER_OPERATOR(ColorContrast, ColorContrast<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA_WITH_PARENT(ColorContrast, ColorTwist)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("slope", "Slope of color contrast", 1.f)
    .AddOptionalArg("bias",  "Bias of color contrast", 0.f);

template <>
void ColorTwist<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  auto output = ws->Output<CPUBackend>(idx);

  const uint8 * input_ptrs;
  uint8 * output_ptrs;
  vector<NDLLSize> size(1);

  DataDependentSetupCPU(input, output, "ColorTwist", &input_ptrs, &output_ptrs, &size);

  float matr[4][4];
  NDLL_CALL(BatchedColorTwist(&input_ptrs, size.data(),
                              &output_ptrs, 1, C_,
                              twistFunc_, twistMatr(matr) ? matr : NULL));
}

NDLLError_t BatchedColorTwist(const uint8 **in_batch, const NDLLSize *sizes, uint8 **out_batch,
                              int N, int C, colorTwistFunc func, const Npp32f aTwist[][4]) {
  NDLL_ASSERT(N > 0);
  NDLL_ASSERT(C == 1 || C == 3);
  NDLL_ASSERT(sizes != nullptr);

  if (!func)            // If transformation function is not defined,
    aTwist = NULL;      // we will not use twist matrix

  for (int i = 0; i < N; ++i) {
    NDLL_ASSERT(in_batch[i] != nullptr);
    NDLL_ASSERT(out_batch[i] != nullptr);

    const int nStep = sizes[i].width * C;
    if (aTwist)
      NDLL_CHECK_NPP(func(in_batch[i], nStep, out_batch[i], nStep, sizes[i], aTwist));
    else
      CUDA_CALL(cudaMemcpy(out_batch[i], in_batch[i],
                           nStep * sizes[i].height, cudaMemcpyDefault));
  }

  return NDLLSuccess;
}

bool brightness_matrix(const float *pScale, float pMatr[][4]) {
  /*
   Resulting matrix:
    r_scale,  0.0,    0.0,    0.0,
      0.0,  g_scale,  0.0,    0.0,
      0.0,    0.0,  b_scale,  0.0,
      0.0,    0.0,    0.0,    1.0
   */
  int i = 3;
  while (i-- > 0 && pScale[i] == 1.0) {}
  if (i < 0)
    return false;   // identical color transformation

  for (i = 0; i < 3; ++i) {
    pMatr[i][i] = pScale[i];
    for (size_t j = i + 1; j < 4; ++j)
      pMatr[i][j] = pMatr[j][i] = 0.0;
  }

  pMatr[3][3] = 1;
  return true;
}

bool color_offset_matrix(const float *pOffset, float pMatr[][4]) {
  /*
   Resulting matrix:
      1.0,    0.0,    0.0,  r_offset,
      0.0,    1.0,    0.0,  g_offset,
      0.0,    0.0,    1.0,  b_offset,
      0.0,    0.0,    0.0,    1.0
   */
  int i = 3;
  while (i-- > 0 && pOffset[i] == 0.0) {}
  if (i < 0)
    return false;   // identical color transformation


  for (i = 0; i < 3; ++i) {
    pMatr[i][i] = 1.0;
    for (size_t j = i + 1; j < 3; ++j)
      pMatr[i][j] = pMatr[j][i] = 0.0;

    pMatr[i][3] = pOffset[i];
    pMatr[3][i] = 0.0;
  }

  pMatr[3][3] = 1;
  return true;
}

bool contrast_matrix(float slop, float bias, float pMatr[][4]) {
  /*
   Resulting matrix:
     slop,    0.0,    0.0,   bias,
      0.0,   slop,    0.0,   bias,
      0.0,    0.0,   slop,   bias,
      0.0,    0.0,    0.0,    1.0
   */
  if (slop == 1. && bias == 0.)
    return false;   // identical color transformation

  for (int i = 0; i < 3; ++i) {
    pMatr[i][i] = slop;
    for (size_t j = i + 1; j < 3; ++j)
      pMatr[i][j] = pMatr[j][i] = 0.0;

    pMatr[i][3] = bias;
    pMatr[3][i] = 0.0;
  }

  pMatr[3][3] = 1;
  return true;
}

bool hue_saturation_matrix(float hue, float saturation, float transf_matr[][4]) {
/*
    Form a color saturation and hue transformation matrix for RGB images.

    Single matrix transform for both hue and saturation change. Matrix taken
    from https://beesbuzz.biz/code/hsv_color_transforms.php. Derived by
    transforming first to HSV, then do the modification, and transfom back to RGB.

    Args:
            hue: (float) hue rotation in degrees, 0.0 is identity
    saturation: (float) saturation multiplier, 1.0 is identity

    Returns:
            transform_mat: tensor(float) augmented transformation matrix (4, 4)
*/

  if (hue == 0. && saturation == 1.)
    return false;   // identical color transformation

  const float const_mat[] = {.299, .587, .114, 0.0,
                             .299, .587, .114, 0.0,
                             .299, .587, .114, 0.0,
                             .0,   .0,   .0,   1.0};

  const float sch_mat[] = { .701, -.587, -.114, 0.0,
                            -.299,  .413, -.114, 0.0,
                            -.300, -.588, .886,  0.0,
                            .0,    .0,   .0,    0.0};


  const float ssh_mat[] = { .168,   .330, -.497, 0.0,
                            -.328,   .035,  .292, 0.0,
                            1.25, -1.05,  -.203, 0.0,
                            .0,    .0,    .0,   0.0};

  const double angle = hue * M_PI / 180.0;
  const float sch = saturation * cos(angle);
  const float ssh = saturation * sin(angle);

  float *pMatr = reinterpret_cast<float *>(transf_matr);
  for (size_t i = 0; i < sizeof(const_mat) / sizeof(const_mat[0]); ++i)
    pMatr[i] = const_mat[i] + sch * sch_mat[i] + ssh * ssh_mat[i];

  return true;
}

}  // namespace ndll

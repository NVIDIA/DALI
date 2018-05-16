// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/color_twist.h"
#include <vector>
#include <string>

namespace ndll {

NDLL_OPERATOR_SCHEMA(ColorTwist)
    .DocStr("ColorTwist - used as a base Schema for Color Transformations")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("image_type", "Input/output image type", NDLL_RGB);

NDLL_OPERATOR_SCHEMA(ColorIntensity)
    .DocStr("Changes the intensity of RGB/Gray images")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("RGB_level", "Intensity of RGB channels", vector<float>{1.f, 1.f, 1.f})
    .AddOptionalArg("GRAY_level", "Intensity of gray level", 1.f)
    .AddParent("ColorTwist");

NDLL_OPERATOR_SCHEMA(ColorOffset)
    .DocStr("Sets the offset for RGB/Gray channels")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("RGB_level", "Intensity of RGB channels", vector<float>{0.f, 0.f, 0.f})
    .AddOptionalArg("GRAY_level", "Intensity of gray level", 0.f)
    .AddParent("ColorTwist");

NDLL_OPERATOR_SCHEMA(HueSaturation)
    .DocStr("Changes the hue/saturation levels of the image")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue", "Hue parameter", 0.f)
    .AddOptionalArg("saturation", "Color saturation level", 1.f)
    .AddParent("ColorTwist");

NDLL_OPERATOR_SCHEMA(ColorContrast)
    .DocStr("Changes the color contrast of the image")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("slope", "Slope of color contrast", 1.f)
    .AddOptionalArg("bias",  "Bias of color contrast", 0.f)
    .AddParent("ColorTwist");

/*  Following operators are not implemented for CPU
NDLL_REGISTER_OPERATOR_FOR_DEVICE(ColorIntensity, CU);
NDLL_REGISTER_OPERATOR_FOR_DEVICE(ColorOffset, CPU);
NDLL_REGISTER_OPERATOR_FOR_DEVICE(HueSaturation, CPU);
NDLL_REGISTER_OPERATOR_FOR_DEVICE(ColorContrast, CPU);
*/

template <>
void ColorTwist<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  NDLL_NOT_IMPLEMENED_OPERATOR;
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

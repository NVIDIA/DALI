// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

#include <nppdefs.h>
#include "ndll/pipeline/operators/color_twist.h"

namespace ndll {

NDLL_OPERATOR_SCHEMA(ColorTwist)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("image_type", "Input/output image type", NDLL_RGB);

NDLL_REGISTER_OPERATOR(ColorIntensity, ColorIntensity<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(ColorIntensity, ColorIntensity<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA_WITH_PARENT(ColorIntensity, ColorTwist)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("R_level", "Intensity of red channel", 1.f)
    .AddOptionalArg("G_level", "Intensity of green channel", 1.f)
    .AddOptionalArg("B_level", "Intensity of blue channel", 1.f)
    .AddOptionalArg("GRAY_level", "Intensity of gray level", 1.f);


NDLL_REGISTER_OPERATOR(ColorOffset, ColorOffset<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(ColorOffset, ColorOffset<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA_WITH_PARENT(ColorOffset, ColorTwist)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("R_level", "Intensity of red channel", 0.f)
    .AddOptionalArg("G_level", "Intensity of green channel", 0.f)
    .AddOptionalArg("B_level", "Intensity of blue channel", 0.f)
    .AddOptionalArg("GRAY_level", "Intensity of gray level", 0.f);


NDLL_REGISTER_OPERATOR(HueSaturation, HueSaturation<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(HueSaturation, HueSaturation<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA_WITH_PARENT(HueSaturation, ColorTwist)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("hue", "Hue parameter", 0.f)
    .AddOptionalArg("saturation", "Color saturation level", 1.f);


NDLL_REGISTER_OPERATOR(ColorContrast, ColorContrast<CPUBackend>, CPU);
NDLL_REGISTER_OPERATOR(ColorContrast, ColorContrast<GPUBackend>, GPU);

NDLL_OPERATOR_SCHEMA_WITH_PARENT(ColorContrast, ColorTwist)
    .DocStr("Foo")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("slope", "Slope of color contrast", 1.f)
    .AddOptionalArg("bias",  "Bias of color contrast", 0.f);

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

#if !NEW_RESIZE_IMPLEMENTED

void DataDependentSetupCPU(const Tensor<CPUBackend> &input,
                           Tensor<CPUBackend> *output, const char *pOpName,
                           const uint8 **ppInRaster, uint8 **ppOutRaster,
                           vector<NDLLSize> *pSizes, const NDLLSize *out_size) {
  NDLL_ENFORCE(input.ndim() == 3);
  NDLL_ENFORCE(IsType<uint8>(input.type()), "Expects input data in uint8.");

  const vector<Index> &shape = input.shape();
  const int C = shape[2];
  NDLL_ENFORCE(C == 1 || C == 3,
               string(pOpName ? pOpName : "Operation") +
               " supports only hwc rgb & grayscale inputs.");

  if (out_size)
    output->Resize({out_size->height, out_size->width, C});
  else
    output->Resize(shape);

  output->set_type(input.type());

  if (!ppInRaster)
    return;

  *ppInRaster = input.template data<uint8>();
  if (ppOutRaster)
    *ppOutRaster = static_cast<uint8 *>(output->raw_mutable_data());

  if (pSizes) {
    (*pSizes)[0].height = shape[0];
    (*pSizes)[0].width = shape[1];
  }
}

void CollectPointersForExecution(size_t batch_size,
            const TensorList<GPUBackend> &input, vector<const uint8 *> *inPtrs,
            TensorList<GPUBackend> *output, vector<uint8 *> *outPtrs) {
  if (!inPtrs || !outPtrs)
    return;

  // Collect the pointers for execution
  for (size_t i = 0; i < batch_size; ++i) {
    (*inPtrs)[i] = input.template tensor<uint8>(i);
    (*outPtrs)[i] = output->template mutable_tensor<uint8>(i);
  }
}

bool DataDependentSetupGPU(const TensorList<GPUBackend> &input, TensorList<GPUBackend> *output,
                           size_t batch_size, bool reshapeBatch, vector<const uint8 *> *inPtrs,
                           vector<uint8 *> *outPtrs, vector<NDLLSize> *pSizes,
                           ResizeParamDescr *pResizeDescr) {
  NDLL_ENFORCE(IsType<uint8>(input.type()),
               "Expected input data stored in uint8.");

#if NEW_RESIZE_IMPLEMENTED
  auto pResize = pResizeDescr ? pResizeDescr->pResize_ : NULL;
  auto pResizeParam = pResizeDescr ? pResizeDescr->pResizeParam_ : NULL;
  auto pMirroring = pResizeDescr ? pResizeDescr->pMirroring_ : NULL;
  auto pTotalSize = pResizeDescr ? pResizeDescr->pTotalSize_ : NULL;

      // Set all elements to 0, if we will use them
  if (pTotalSize)
    memset(pTotalSize, 0, pResizeDescr->nBatchSlice_ * sizeof(pTotalSize[0]));
#endif

  bool newResize = false;
  vector<Dims> output_shape(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    // Verify the inputs
    const auto &input_shape = input.tensor_shape(i);
    NDLL_ENFORCE(input_shape.size() == 3,
                 "Expects 3-dimensional image input.");

    NDLL_ENFORCE(input_shape[2] == 1 || input_shape[2] == 3,
                 "Not valid color type argument (1 or 3)");

#if NEW_RESIZE_IMPLEMENTED
    // Collect the output shapes
    if (pResize) {
      // We are resizing
      const auto input_size = pResize->size(input_t, i);
      const auto out_size = pResize->size(output_t, i);
      pResize->SetSize(input_size, input_shape, pResize->newSizes(i), out_size);

      if (pResizeParam) {
        // NewResize is used
        const int H0 = input_size->height;
        const int W0 = input_size->width;
        const int H1 = out_size->height;
        const int W1 = out_size->width;

        int cropY, cropX;
        const bool doingCrop = pResize->CropNeeded(*out_size);
        if (doingCrop)
          pResize->DefineCrop(out_size, &cropX, &cropY);
        else
          cropY = cropX = 0;

        auto resizeParam = pResizeParam + i * (pMirroring ? N_GRID_PARAMS : 1);
        if (pMirroring) {
          const int lcmH = lcm(H0, H1);
          const int lcmW = lcm(W0, W1);

          const int sy0 = lcmH / H0;
          const int sy1 = lcmH / H1;
          const int sx0 = lcmW / W0;
          const int sx1 = lcmW / W1;

          if (!newResize) {
            newResize = resizeParam[0].x != sx0 || resizeParam[0].y != sy0 ||
                        resizeParam[1].x != sx1 || resizeParam[1].y != sy1 ||
                        resizeParam[2].x != cropX || resizeParam[2].y != cropY;
          }

          if (newResize) {
            resizeParam[0] = {sx0, sy0};
            resizeParam[1] = {sx1, sy1};
            resizeParam[2] = {cropX, cropY};
          }

          if (pTotalSize) {
            // We need to check for overflow
            const size_t idx = i % pResizeDescr->nBatchSlice_;
            if (pTotalSize[idx] < UINT_MAX - sx0 * sy0)
              pTotalSize[idx] += sx0 * sy0;
            else
              pTotalSize[idx] = UINT_MAX;
          }

          if (pMirroring)
            pResize->MirrorNeeded(pMirroring + i);
        } else {
          resizeParam[0] = {W1, H1};
        }
      }

      // Collect the output shapes
      output_shape[i] = {out_size->height, out_size->width, input_shape[2]};
    } else {
      output_shape[i] = input_shape;
    }
#else
    output_shape[i] = input_shape;
#endif
    if (pSizes) {
      (*pSizes)[i].height = input_shape[0];
      (*pSizes)[i].width = input_shape[1];
      if (reshapeBatch) {
        // When batch is reshaped: only one "image" will be used
        (*pSizes)[i].height *= batch_size;
        pSizes = NULL;
      }
    }
  }

  // Resize the output
  output->Resize(output_shape);
  output->set_type(input.type());

  CollectPointersForExecution(reshapeBatch ? 1 : batch_size, input, inPtrs, output, outPtrs);
  return newResize;
}
#endif

}  // namespace ndll

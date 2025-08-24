// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "bbox_rotate.h"
#include "dali/core/geom/vec.h"

#if defined(__x86_64__) || defined(_M_X64)
// x86_64 (amd64) specific SIMD code (SSE, AVX, etc.)
#include <xmmintrin.h>
#elif defined(__aarch64__)
// ARM64 (aarch64) specific SIMD code (NEON, etc.)
#include <arm_neon.h>
#else
#error "Unsupported architecture"
#endif

namespace dali {

DALI_SCHEMA(BBoxRotate)
    .DocStr(
        R"code(Transforms bounding boxes so that the boxes remain in the same place in the image after
the image is rotated. Boxes that land outside the image with `keep_size=True` will be removed from the
output, a mask of valid boxes will be returned that can be used to remove other data such as class labels.

Box coordinates must be (0,1) normalized and the image is assumed to be center rotated with `fn.rotate`.
)code")
    .NumInput(1, 2)
    .InputDox(
        0, "bboxes", "2D TensorList of float",
        R"code(Relative coordinates of the bounding boxes that are represented as a 2D tensor, where the
first dimension refers to the index of the bounding box, and the second dimension refers to the index
of the coordinate.
)code")
    .InputDox(
        1, "labels", "1D TensorList of int",
        R"code(Class labels for the bounding boxes. These should be provided if the `keep_size` argument is
set to True as boxes may be truncated, this ensures the corresponding labels will also be removed.
)code")
    .NumOutput(1)
    .AdditionalOutputsFn([](const OpSpec& spec) {
      return spec.NumRegularInput() - 1;  // +1 if labels are provided
    })
    .AddArg("ltrb", R"code(True for ``ltrb`` or False for ``xywh``.)code", DALI_BOOL, false)
    .AddArg("angle", R"code(Rotation angle in degrees.)code", DALI_FLOAT, true)
    .AddOptionalArg(
        "keep_size",
        R"code(If true, the bounding box output coordinates will assume the image canvas size was also kept 
(see `nvidia.dali.fn.rotate`).)code",
        false, false)
    .AddOptionalArg("mode",
                    R"code(Mode of the bounding box transformation. Possible values are:
- ``expand``: expands the bounding box to definitely enclose the target, but may be larger than rotated target.
- ``fixed``: retains the original size of the bounding box, but may be smaller than the rotated target. 
- ``halfway``: halfway between the expanded size and the original size.
)code",
                    dali::string("expand"), false)
    .AddOptionalArg(
        "remove_threshold",
        R"code(Relative area remaining threshold for removing boxes that fall outside the image boundaries
after rotation. Should be between [0-1] where 0 means no removal even if box lies outside, 1 means removal 
if any part of the box is outside.)code",
        0.1f, false);

using Mode = BBoxRotate<CPUBackend>::Mode;

/**
 * @brief Expands the bounding boxes to all corners x1y1, x2y1, x1y2, x2y2 in flattened xCoords and
 *        yCoords.
 *
 * @note Not sure if I can use the 16-byte aligned version as an assumption.
 *
 * @tparam ltrb If true, the input boxes are in left-top-right-bottom format, otherwise in
 *         xywh format.
 * @param inBoxes Input bounding boxes.
 * @param xCoords Output flattened x coordinates of the corners.
 * @param yCoords Output flattened y coordinates of the corners.
 * @param numBoxes Number of bounding boxes.
 */
template <bool ltrb>
void expandToAllCorners(dali::span<const vec<4>> inBoxes, dali::span<float> xCoords,
                        dali::span<float> yCoords) {
  for (int i = 0; i < inBoxes.size(); ++i) {
    auto box = inBoxes[i];
    // Convert to x2y2
    if constexpr (!ltrb) {
      box.z += box.x;
      box.w += box.y;
    }
    const auto offset = 4 * i;
#if defined(__x86_64__) || defined(_M_X64)
    // Copy X coordinates
    const __m128 xvec = _mm_setr_ps(box.x, box.z, box.x, box.z);
    _mm_storeu_ps(&xCoords[offset], xvec);
    // Copy Y coordinates
    const __m128 yvec = _mm_setr_ps(box.y, box.y, box.w, box.w);
    _mm_storeu_ps(&yCoords[offset], yvec);
#elif defined(__aarch64__)
    // Copy X coordinates
    float32x4_t xvec = {box.x, box.z, box.x, box.z};
    vst1q_f32(&xCoords[offset], xvec);
    // Copy Y coordinates
    float32x4_t yvec = {box.y, box.y, box.w, box.w};
    vst1q_f32(&yCoords[offset], yvec);
#endif
  }
}

/**
 * @brief Rotates the corners of the bounding boxes by a given angle.
 *
 * @param xCoords Flattened x coordinates of the corners.
 * @param yCoords Flattened y coordinates of the corners.
 * @param angle Rotation angle in radians.
 */
void rotateCorners(dali::span<float> xCoords, dali::span<float> yCoords, float angle) {
  float cos_a = std::cos(angle);
  float sin_a = std::sin(angle);

  for (int i = 0; i < xCoords.size(); ++i) {
    float x = xCoords[i];
    float y = yCoords[i];
    xCoords[i] = x * cos_a - y * sin_a;
    yCoords[i] = x * sin_a + y * cos_a;
  }
}

/**
 * @brief Converts corner coordinates to ltrb bounding boxes.
 * @param xCoords Flattened x coordinates of the corners.
 * @param yCoords Flattened y coordinates of the corners.
 * @param outBoxes Output bounding boxes.
 */
void cornersToBoxes(dali::span<float> xCoords, dali::span<float> yCoords,
                    dali::span<vec<4>> outBoxes) {
  for (int i = 0; i < outBoxes.size(); ++i) {
    const auto offset = 4 * i;
    vec<4> xvec(xCoords[offset]);
    vec<4> yvec(yCoords[offset]);
    auto [x1, x2] = std::minmax_element(xvec.begin(), xvec.end());
    auto [y1, y2] = std::minmax_element(yvec.begin(), yvec.end());
    outBoxes[i] = vec<4>(*x1, *y1, *x2, *y2);
  }
}

/**
 * @brief Applies expansion correction to the bounding box sizes by shrinking them by the amount the
 * rotation expanded them.
 *
 * @param boxes boxes to be corrected inplace.
 * @param expansion general expansion factor of the rotation
 * @param halfway whether to only correct halfway, not the full expansion
 */
void applyExpansionCorrectionToBoxSize(dali::span<vec<4>> boxes, float expansion, bool halfway) {
  float correction = expansion - 1.0f;
  if (halfway) {
    correction *= 0.5f;
  }
  for (auto& box : boxes) {
    box.x += correction;
    box.y += correction;
    box.z -= correction;
    box.w -= correction;
  }
}

/**
 * @brief Clips and removes boxes that fall below a certain threshold of area change.
 * @param boxes normalized xyxy boxes
 * @param remove_threshold threshold of area change to remove boxes
 * @param inputLabels optional input labels to filter synchronously with truncated boxes
 * @param outputLabels option result of the filtered labels
 * @return the new number of valid boxes
 */
int clipAndRemoveBoxes(dali::span<vec<4>> boxes, float remove_threshold,
                       std::optional<ConstSampleView<CPUBackend>> inputLabels,
                       std::optional<SampleView<CPUBackend>> outputLabels) {
  int outIdx = 0;
  for (int inIdx = 0; inIdx < boxes.size(); ++inIdx) {
    auto box = boxes[inIdx];
    const float old_area = (box.z - box.x) * (box.w - box.y);
    box.x = std::clamp(box.x, 0.0f, 1.0f);
    box.y = std::clamp(box.y, 0.0f, 1.0f);
    box.z = std::clamp(box.z, 0.0f, 1.0f);
    box.w = std::clamp(box.w, 0.0f, 1.0f);
    const float new_area = (box.z - box.x) * (box.w - box.y);
    if (new_area / old_area > remove_threshold) {
      boxes[outIdx] = box;
      if (inputLabels && outputLabels) {
        outputLabels.value().mutable_data<int>()[outIdx] = inputLabels.value().data<int>()[inIdx];
      }
      ++outIdx;
    }
  }
  return outIdx;
}

int rotateBoxesKernel(ConstSampleView<CPUBackend> inBoxTensor, SampleView<CPUBackend> rotateBuffer,
                      SampleView<CPUBackend> outBoxTensor, float angle,
                      std::optional<ConstSampleView<CPUBackend>> inputLabels,
                      std::optional<SampleView<CPUBackend>> outputLabels, bool ltrb, bool keep_size,
                      Mode mode, float remove_threshold) {
  const auto numBoxes = inBoxTensor.shape()[0];
  auto inBoxes = dali::span(inBoxTensor.data<vec<4>>(), numBoxes);
  auto xCoords = dali::span(rotateBuffer.mutable_data<float>(), 4 * numBoxes);
  auto yCoords = dali::span(xCoords.end(), 4 * numBoxes);
  if (ltrb) {
    expandToAllCorners<true>(inBoxes, xCoords, yCoords);
  } else {
    expandToAllCorners<false>(inBoxes, xCoords, yCoords);
  }

  // Center the coordinates on zero
  std::transform(xCoords.begin(), xCoords.end(), xCoords.begin(),
                 [](float elem) { return elem - 0.5f; });
  std::transform(yCoords.begin(), yCoords.end(), yCoords.begin(),
                 [](float elem) { return elem - 0.5f; });

  // Apply rotation
  rotateCorners(xCoords, yCoords, angle);

  const float expansion = std::sin(angle) + std::cos(angle);
  // Move coordinates back to [0-1]
  if (keep_size) {
    std::transform(xCoords.begin(), xCoords.end(), xCoords.begin(),
                   [](float elem) { return elem + 0.5f; });
    std::transform(yCoords.begin(), yCoords.end(), yCoords.begin(),
                   [](float elem) { return elem + 0.5f; });
  } else {
    std::transform(xCoords.begin(), xCoords.end(), xCoords.begin(),
                   [expansion](float elem) { return elem / expansion + 0.5f; });
    std::transform(yCoords.begin(), yCoords.end(), yCoords.begin(),
                   [expansion](float elem) { return elem / expansion + 0.5f; });
  }

  // Convert back to bounding boxes
  auto outBoxes = dali::span(outBoxTensor.mutable_data<vec<4>>(), numBoxes);
  cornersToBoxes(xCoords, yCoords, outBoxes);

  // Apply correction to expansion factor if required
  if (mode != Mode::Expand) {
    applyExpansionCorrectionToBoxSize(outBoxes, expansion, mode == Mode::Halfway);
  }

  // TODO Clip to image coordiantes, optionally removing labels that fall outside the new ROI with
  // the box.
  const auto numOutBoxes =
      clipAndRemoveBoxes(outBoxes, remove_threshold, inputLabels, outputLabels);

  if (!ltrb) {                                            // Convert to xywh format if needed
    outBoxes = dali::span(outBoxes.data(), numOutBoxes);  // handle if some boxes culled
    std::for_each(outBoxes.begin(), outBoxes.end(), [](vec<4>& box) {
      box.z -= box.x;
      box.w -= box.y;
    });
  }

  return numOutBoxes;
}

template <>
void BBoxRotate<CPUBackend>::RunImpl(Workspace& ws) {
  const auto& inputBoxes = ws.Input<CPUBackend>(0);

  auto& tpool = ws.GetThreadPool();
  for (int sampleIdx = 0; sampleIdx < inputBoxes.num_samples(); ++sampleIdx) {
    tpool.AddWork([&, sampleIdx](int) {
      std::optional<ConstSampleView<CPUBackend>> inputLabels;
      std::optional<SampleView<CPUBackend>> outputLabels;
      if (ws.NumInput() == 2) {
        inputLabels = ws.Input<CPUBackend>(1)[sampleIdx];
        outputLabels = ws.Output<CPUBackend>(1)[sampleIdx];
      }
      float angle;
      if (spec_.HasTensorArgument("angle")) {
        angle = ws.ArgumentInput("angle")[sampleIdx].data<float>()[0];
      } else {
        angle = spec_.GetArgument<float>("angle");
      }
      angle *= M_PI / 180.0f;  // Convert to radians
      const auto numOutBoxes =
          rotateBoxesKernel(inputBoxes[sampleIdx], bbox_rotate_buffer_[sampleIdx],
                            ws.Output<CPUBackend>(0)[sampleIdx], angle, inputLabels, outputLabels,
                            use_ltrb_, keep_size_, mode_, remove_threshold_);
      ws.Output<CPUBackend>(0).ResizeSample(sampleIdx, dali::TensorShape<2>(numOutBoxes, 4));
      if (outputLabels.has_value()) {
        ws.Output<CPUBackend>(1).ResizeSample(sampleIdx, dali::TensorShape<1>(numOutBoxes));
      }
    });
  }
  tpool.RunAll();
}

DALI_REGISTER_OPERATOR(BBoxRotate, BBoxRotate<CPUBackend>, CPU);

}  // namespace dali
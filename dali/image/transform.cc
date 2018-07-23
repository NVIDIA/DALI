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

#include "dali/image/transform.h"

#include "dali/util/image.h"
#include "dali/util/ocv.h"

namespace dali {

// Note: User is responsible for avoiding division by 0 w/ the std deviation
DALIError_t ResizeCropMirrorHost(const uint8 *img, int H, int W, int C,
    int rsz_h, int rsz_w, int crop_y, int crop_x, int crop_h,
    int crop_w, int mirror, uint8 *out_img, DALIInterpType type,
    uint8 *workspace) {
  DALI_ASSERT(img != nullptr);
  DALI_ASSERT(out_img != nullptr);
  DALI_ASSERT(H > 0);
  DALI_ASSERT(W > 0);
  DALI_ASSERT((C == 3) || (C == 1));
  DALI_ASSERT(rsz_h > 0);
  DALI_ASSERT(rsz_w > 0);
  // Crop must be valid
  DALI_ASSERT(crop_y >= 0);
  DALI_ASSERT(crop_x >= 0);
  DALI_ASSERT(crop_h > 0);
  DALI_ASSERT(crop_w > 0);
  DALI_ASSERT((crop_y + crop_h) <= rsz_h);
  DALI_ASSERT((crop_x + crop_w) <= rsz_w);
  int channel_flag = C == 3 ? CV_8UC3 : CV_8UC1;
  const cv::Mat cv_img = CreateMatFromPtr(H, W, channel_flag, img);

  cv::Mat rsz_img;
  if (workspace != nullptr) {
    // We have a temporary buffer allocated by the user. For this function we
    // need a buffer of rsz_h*rsz_w*C bytes. Wrap this buffer w/ a cv::Mat
    rsz_img = CreateMatFromPtr(rsz_h, rsz_w, channel_flag, workspace);
  }

  int ocv_type;
  DALI_FORWARD_ERROR(OCVInterpForDALIInterp(type, &ocv_type));
  cv::resize(cv_img, rsz_img,
      cv::Size(rsz_w, rsz_h),
      0, 0, ocv_type);

  // Do the crop/mirror into the output buffer
  if (mirror) {
    for (int i = 0; i < crop_h; ++i) {
      for (int j = 0; j < crop_w; ++j) {
        int pxl_off = i*crop_w*C + j*C;
        int mirror_j = (crop_w-j-1);
        int src_off = (i+crop_y)*rsz_w*C + (mirror_j+crop_x)*C;
        for (int k = 0; k < C; ++k) {
          out_img[pxl_off + k] = rsz_img.ptr()[src_off + k];
        }
      }
    }
  } else {
    for (int i = 0; i < crop_h; ++i) {
      for (int j = 0; j < crop_w; ++j) {
        int pxl_off = i*crop_w*C + j*C;
        int src_off = (i+crop_y)*rsz_w*C + (j+crop_x)*C;
        for (int k = 0; k < C; ++k) {
          out_img[pxl_off + k] = rsz_img.ptr()[src_off + k];
        }
      }
    }
  }
  return DALISuccess;
}

DALIError_t FastResizeCropMirrorHost(const uint8 *img, int H, int W, int C,
    int rsz_h, int rsz_w, int crop_y, int crop_x, int crop_h,
    int crop_w, int mirror, uint8 *out_img, DALIInterpType type,
    uint8 *workspace) {
  DALI_ASSERT(img != nullptr);
  DALI_ASSERT(out_img != nullptr);
  DALI_ASSERT(H > 0);
  DALI_ASSERT(W > 0);
  DALI_ASSERT((C == 3) || (C == 1));
  DALI_ASSERT(rsz_h > 0);
  DALI_ASSERT(rsz_w > 0);
  // Crop must be valid
  DALI_ASSERT(crop_y >= 0);
  DALI_ASSERT(crop_x >= 0);
  DALI_ASSERT(crop_h > 0);
  DALI_ASSERT(crop_w > 0);
  DALI_ASSERT((crop_y + crop_h) <= rsz_h);
  DALI_ASSERT((crop_x + crop_w) <= rsz_w);

  // FAST RESIZE: We are going to do a crop, so we back-project the crop into the
  // input image, get an ROI on this region, and then resize to the crop dimensions
  // this effectively does the resize+crop in one step, then we just mirror.
  int roi_w, roi_h, roi_x, roi_y;
  roi_w = static_cast<int>(static_cast<float>(crop_w) / rsz_w * W);
  roi_h = static_cast<int>(static_cast<float>(crop_h) / rsz_h * H);
  roi_x = static_cast<int>(static_cast<float>(crop_x) / rsz_w * W + 0.5f);
  roi_y = static_cast<int>(static_cast<float>(crop_y) / rsz_h * H + 0.5f);

  int channel_flag = C == 3 ? CV_8UC3 : CV_8UC1;
  const cv::Mat cv_img = CreateMatFromPtr(roi_h, roi_w, channel_flag,
      img + (roi_y*W + roi_x)*C, W*C);

  // Note: We only need an intermediate buffer if we are going to mirror the image.
  // If we are not going to mirror the image, wrap the output pointer in a cv::Mat
  // and do the resize directly into that buffer to avoid the unnescessary memcpy.
  // Then just return.
  int ocv_type;
  DALI_FORWARD_ERROR(OCVInterpForDALIInterp(type, &ocv_type));
  if (!mirror) {
    cv::Mat cv_out_img = CreateMatFromPtr(crop_h, crop_w, channel_flag, out_img);
    cv::resize(cv_img, cv_out_img,
        cv::Size(crop_w, crop_h),
        0, 0, ocv_type);
    return DALISuccess;
  }

  cv::Mat rsz_img;
  if (workspace != nullptr) {
    // We have a tmp buffer allocated by the user. For this function the tmp
    // buffer only needs to be crop_w x crop_h x C. Wrap this buffer w/ a cv::Mat
    // header
    rsz_img = CreateMatFromPtr(crop_h, crop_w, channel_flag, workspace);
  }

  cv::resize(cv_img, rsz_img,
      cv::Size(crop_w, crop_h),
      0, 0, ocv_type);

  // Mirror the image into the output image buffer
  for (int i = 0; i < crop_h; ++i) {
    for (int j = 0; j < crop_w; ++j) {
      int pxl_off = i*crop_w*C + j*C;
      int mirror_j = (crop_w-j-1);
      int src_off = i*crop_w*C + mirror_j*C;
      for (int k = 0; k < C; ++k) {
        out_img[pxl_off + k] = rsz_img.ptr()[src_off + k];
      }
    }
  }
  return DALISuccess;
}

}  // namespace dali

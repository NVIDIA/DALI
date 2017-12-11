// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/image/transform.h"

#include "ndll/util/image.h"
#include "ndll/util/ocv.h"

namespace ndll {

// Note: User is responsible for avoiding division by 0 w/ the std deviation
NDLLError_t ResizeCropMirrorHost(const uint8 *img, int H, int W, int C,
    int rsz_h, int rsz_w, int crop_y, int crop_x, int crop_h,
    int crop_w, bool mirror, uint8 *out_img, NDLLInterpType type,
    uint8 *workspace) {
  // TODO(tgale): Figure out which ones of these we actually want to check.
  // We need a better way of error checking so that the user actually knows
  // what they did wrong.
  NDLL_ASSERT(img != nullptr);
  NDLL_ASSERT(out_img != nullptr);
  NDLL_ASSERT(H > 0);
  NDLL_ASSERT(W > 0);
  NDLL_ASSERT((C == 3) || (C == 1));
  NDLL_ASSERT(rsz_h > 0);
  NDLL_ASSERT(rsz_w > 0);
  // Crop must be valid
  NDLL_ASSERT(crop_y >= 0);
  NDLL_ASSERT(crop_x >= 0);
  NDLL_ASSERT(crop_h > 0);
  NDLL_ASSERT(crop_w > 0);
  NDLL_ASSERT((crop_y + crop_h) <= rsz_h);
  NDLL_ASSERT((crop_x + crop_w) <= rsz_w);
  // Note: OpenCV can't take a const pointer to wrap even when the cv::Mat is const. This
  // is kinda icky to const_cast away the const-ness, but there isn't another way
  // (that I know of) without making the input argument non-const.
  int channel_flag = C == 3 ? CV_8UC3 : CV_8UC1;
  const cv::Mat cv_img = cv::Mat(H, W, channel_flag, const_cast<uint8*>(img));

  cv::Mat rsz_img;
  if (workspace != nullptr) {
    // We have a temporary buffer allocated by the user. For this function we
    // need a buffer of rsz_h*rsz_w*C bytes. Wrap this buffer w/ a cv::Mat
    rsz_img = cv::Mat(rsz_h, rsz_w, channel_flag, workspace);
  }

  int ocv_type;
  NDLL_FORWARD_ERROR(OCVInterpForNDLLInterp(type, &ocv_type));
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
  return NDLLSuccess;
}

NDLLError_t FastResizeCropMirrorHost(const uint8 *img, int H, int W, int C,
    int rsz_h, int rsz_w, int crop_y, int crop_x, int crop_h,
    int crop_w, bool mirror, uint8 *out_img, NDLLInterpType type,
    uint8 *workspace) {
  // TODO(tgale): Figure out which ones of these we actually want to check.
  // We need a better way of error checking so that the user actually knows
  // what they did wrong.
  NDLL_ASSERT(img != nullptr);
  NDLL_ASSERT(out_img != nullptr);
  NDLL_ASSERT(H > 0);
  NDLL_ASSERT(W > 0);
  NDLL_ASSERT((C == 3) || (C == 1));
  NDLL_ASSERT(rsz_h > 0);
  NDLL_ASSERT(rsz_w > 0);
  // Crop must be valid
  NDLL_ASSERT(crop_y >= 0);
  NDLL_ASSERT(crop_x >= 0);
  NDLL_ASSERT(crop_h > 0);
  NDLL_ASSERT(crop_w > 0);
  NDLL_ASSERT((crop_y + crop_h) <= rsz_h);
  NDLL_ASSERT((crop_x + crop_w) <= rsz_w);

  // FAST RESIZE: We are going to do a crop, so we back-project the crop into the
  // input image, get an ROI on this region, and then resize to the crop dimensions
  // this effectively does the resize+crop in one step, then we just mirror.
  int roi_w, roi_h, roi_x, roi_y;
  roi_w = static_cast<int>(static_cast<float>(crop_w) / rsz_w * W);
  roi_h = static_cast<int>(static_cast<float>(crop_h) / rsz_h * H);
  roi_x = static_cast<int>(static_cast<float>(crop_x) / rsz_w * W);
  roi_y = static_cast<int>(static_cast<float>(crop_y) / rsz_h * H);

  // Note: OpenCV can't take a const pointer to wrap even when the cv::Mat is const. This
  // is kinda icky to const_cast away the const-ness, but there isn't another way
  // (that I know of) without making the input argument non-const.
  int channel_flag = C == 3 ? CV_8UC3 : CV_8UC1;
  const cv::Mat cv_img = cv::Mat(roi_h, roi_w, channel_flag,
      const_cast<uint8*>(img) + roi_y*W*C + roi_x*C, W*C);

  // Note: We only need an intermediate buffer if we are going to mirror the image.
  // If we are not going to mirror the image, wrap the output pointer in a cv::Mat
  // and do the resize directly into that buffer to avoid the unnescessary memcpy.
  // Then just return.
  if (!mirror) {
    cv::Mat cv_out_img(crop_h, crop_w, channel_flag, out_img);
    cv::resize(cv_img, cv_out_img,
        cv::Size(crop_w, crop_h),
        0, 0, cv::INTER_LINEAR);
    return NDLLSuccess;
  }

  cv::Mat rsz_img;
  if (workspace != nullptr) {
    // We have a tmp buffer allocated by the user. For this function the tmp
    // buffer only needs to be crop_w x crop_h x C. Wrap this buffer w/ a cv::Mat
    // header
    rsz_img = cv::Mat(crop_h, crop_w, channel_flag, workspace);
  }

  int ocv_type;
  NDLL_FORWARD_ERROR(OCVInterpForNDLLInterp(type, &ocv_type));
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
  return NDLLSuccess;
}

}  // namespace ndll

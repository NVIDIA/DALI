#include "ndll/image/transform.h"

#include <opencv2/opencv.hpp>

namespace ndll {

// Note: User is responsible for avoiding division by 0 w/ the std deviation
NDLLError_t ResizeCropMirrorHost(const uint8 *image, int H, int W, int C,
    int rsz_h, int rsz_w, int crop_y, int crop_x, int crop_h,
    int crop_w, bool mirror, uint8 *out_img) {
  // TODO(tgale): Figure out which ones of these we actually want to check.
  // We need a better way of error checking so that the user actually knows
  // what they did wrong.
  NDLL_ASSERT(image != nullptr);
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
  // is kinda icky to const_cast away the const-ness, but there isn't another way without
  // making the input argument non-const.
  const cv::Mat cv_img = cv::Mat(H, W, C == 3 ? CV_8UC3 : CV_8UC1, const_cast<uint8*>(image));

  // Note: We need an intermediate buffer to work in for the resize.
  // This means that we will have host memory allocations inside this
  // function, which is not ideal. We could add an option to pass in
  // a pointer to workspace that we can wrap with this so the user
  // can allocate the buffer if they want.
  cv::Mat rsz_img;
  cv::resize(cv_img, rsz_img,
      cv::Size(rsz_w, rsz_h),
      0, 0, cv::INTER_LINEAR);
  
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

NDLLError_t FastResizeCropMirrorHost(const uint8 *image, int H, int W, int C,
    int rsz_h, int rsz_w, int crop_y, int crop_x, int crop_h,
    int crop_w, bool mirror, uint8 *out_img) {
  // TODO(tgale): Figure out which ones of these we actually want to check.
  // We need a better way of error checking so that the user actually knows
  // what they did wrong.
  NDLL_ASSERT(image != nullptr);
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
  // is kinda icky to const_cast away the const-ness, but there isn't another way without
  // making the input argument non-const.
  const cv::Mat cv_img = cv::Mat(H, W, C == 3 ? CV_8UC3 : CV_8UC1, const_cast<uint8*>(image));

  // MODIFIED RESIZE: We are going to do a crop, so we back-project the crop into the
  // input image, get an ROI on this region, and then resize to the crop dimensions
  // this effectively does the resize+crop in one step, then we just mirror.
  int roi_w, roi_h, roi_x, roi_y;
  roi_w = int(float(crop_w) / rsz_w * W);
  roi_h = int(float(crop_h) / rsz_h * H);
  roi_x = int(float(crop_x) / rsz_w * W);
  roi_y = int(float(crop_y) / rsz_h * H);
  cv::Rect roi(roi_x, roi_y, roi_w, roi_h);
  cv_img = cv_img(roi);
  
  // Note: We need an intermediate buffer to work in for the resize.
  // This means that we will have host memory allocations inside this
  // function, which is not ideal. We could add an option to pass in
  // a pointer to workspace that we can wrap with this so the user
  // can allocate the buffer if they want.
  cv::Mat rsz_img;
  cv::resize(cv_img, rsz_img,
      cv::Size(crop_w, crop_h),
      0, 0, cv::INTER_LINEAR);

  // TODO(tgale): The code above this should be allset, but below we'll need
  // to just iterate over the image and mirror it like normal because the
  // rsz_img is already the crop dims. Currently I just removed the crop offsets
  // from being added to the src_off ptr, but I'll need to look at this closer
  // to make sure it works
  
  // Do the crop/mirror into the output buffer. 
  if (mirror) {
    for (int i = 0; i < crop_h; ++i) {
      for (int j = 0; j < crop_w; ++j) {
        int pxl_off = i*crop_w*C + j*C;
        int mirror_j = (crop_w-j-1);
        int src_off = i*rsz_w*C + mirror_j*C;
        for (int k = 0; k < C; ++k) {
          out_img[pxl_off + k] = rsz_img.ptr()[src_off + k];
        }
      }
    }
  } else {
    for (int i = 0; i < crop_h; ++i) {
      for (int j = 0; j < crop_w; ++j) {
        int pxl_off = i*crop_w*C + j*C;
        int src_off = i*rsz_w*C + j*C;
        for (int k = 0; k < C; ++k) {
          out_img[pxl_off + k] = rsz_img.ptr()[src_off + k];
        }
      }
    }
  }
  return NDLLSuccess;
}

} // namespace ndll

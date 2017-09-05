#include "ndll/image/transform.h"

#include <opencv2/opencv.hpp>

namespace ndll {

NDLLError_t ResizeCropMirrorHost(const uint8 *image, int H, int W, int C,
    int rsz_h, int rsz_w, int crop_x, int crop_y, int crop_h, int crop_w,
    bool mirror, uint8 *out_img) {
  // TODO(tgale): Validate input parameters for crop & resize
  NDLL_ASSERT((C == 3) || (C == 1));

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
} // namespace ndll

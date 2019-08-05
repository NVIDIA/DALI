#ifndef DALI_BRIGHTNESS_CONTRAST_TEST_UTILS_H
#define DALI_BRIGHTNESS_CONTRAST_TEST_UTILS_H

namespace dali {
namespace kernels {
namespace brightness_contrast {
namespace test {

/**
 * Creates cv::Mat based on provided arguments.
 * This mat is for roi-testing purposes only. Particularly, it doesn't care
 * about image type (i.e. number of channels), so don't try to imshow it.
 *
 * @param ptr Can't be const, due to cv::Mat API
 * @param roi
 * @param rows height of the input image
 * @param cols width of the input image
 */
template <int nchannels, class T, class Roi>
cv::Mat_<cv::Vec<T, nchannels>> to_mat(T *ptr, Roi roi, int rows, int cols) {
  static_assert(std::is_same<Roi, Box<2, int>>::value, "Roi is supposed to be `Box<2, int>`");
  auto roi_w = roi.extent().x;
  auto roi_h = roi.extent().y;
  //TODO
//  assert(roi.hi.x < cols && roi.hi.y < rows);  // Roi overflows the image
  cv::Mat mat(rows, cols, CV_MAKETYPE(cv::DataDepth<std::remove_const_t<T>>::value, nchannels),
              const_cast<T *>(ptr));
  cv::Rect rect(roi.lo.x, roi.lo.y, roi_w, roi_h);
  cv::Mat_<cv::Vec<T, nchannels>> out_copy;
  mat(rect).copyTo(out_copy);
  assert(out_copy.isContinuous());
  return out_copy;
}

}  // namespace test
}  // namespace brightness_contrast
}  // namespace kernels
}  // namespace dali

#endif  // DALI_BRIGHTNESS_CONTRAST_TEST_UTILS_H

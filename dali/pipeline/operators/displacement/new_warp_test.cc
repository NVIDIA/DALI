#include <gtest/gtest.h>
#include "warp.h"
#include "dali/kernels/imgproc/warp/affine.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/operators/operator.h"
#include <opencv2/core.hpp>
#include "dali/core/geom/vec.h"

namespace dali {

template <int dim>
struct Roi {
  ivec<dim> lo, hi;
  auto extent() const { return hi - lo; }
};

template <int nchannels, class T, class Roi>
cv::Mat_<cv::Vec<T, nchannels>> to_mat(const T *ptr, Roi roi, int rows, int cols) {
  auto roi_w = roi.extent().x;
  auto roi_h = roi.extent().y;
  cv::Mat mat(rows, cols,
              CV_MAKETYPE(cv::DataDepth<std::remove_const_t<T>>::value, nchannels),
              const_cast<T*>(ptr));
  cv::Rect rect(roi.lo.x, roi.lo.y, roi_w, roi_h);
  cv::Mat_<cv::Vec<T, nchannels>> out_copy;
  mat(rect).copyTo(out_copy);
  return out_copy;
}

TEST(WTF, StaticSwitch) {
  float data[5][5][3] = {};
  float z = 0;
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++)
      for (int k = 0; k < 3; k++)
        data[i][j][k] = z++;
  auto tmp = to_mat<3>(&data[0][0][0], Roi<2>{{1,2},{4,4}}, 5, 5);
  ASSERT_EQ(tmp.cols, 3);
  ASSERT_EQ(tmp.rows, 2);
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        EXPECT_EQ(tmp(i, j)[k], data[i+2][j+1][k]) << "@ " << i << " " << j << " " << k;
}
}

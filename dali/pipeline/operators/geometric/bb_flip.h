#ifndef DALI_BB_FLIP_H
#define DALI_BB_FLIP_H

#include <dali/pipeline/operators/operator.h>
#include <dali/pipeline/operators/common.h>

namespace dali {

class BbFlip : public Operator<CPUBackend> {
 public:
  explicit BbFlip(const OpSpec &spec);

  virtual ~BbFlip() = default;
  DISABLE_COPY_MOVE_ASSIGN(BbFlip);

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx) override;

 private:
  const int BB_TYPE_SIZE = 4; /// Bounding box is always vector of 4 floats

  /**
   * Bounding box can be represented in two ways:
   * 1. Upper-left corner, width, height (`wh_type`)
   *    (x1, y1,  w,  h)
   * 2. Upper-left and Lower-right corners (`two-point type`)
   *    (x1, y1, x2, y2)
   *
   * Both of them have coordinates in image coordinate system
   * (i.e. 0.0-1.0)
   *
   * If `coordinates_type_wh_` is true, then we deal with 1st type. Otherwise, the 2nd one.
   */
  bool coordinates_type_wh_;
};

} // namespace dali

#endif //DALI_BB_FLIP_H

#ifndef NDLL_STAGE_H_
#define NDLL_STAGE_H_

#include "ndll/common.h"
namespace ndll {

class Stage {
public:
  Stage() {}
  virtual ~Stage() = default;

  /**
   * @brief Runs this computation for a single image 
   * in the prefetch stage (on cpu)
   */
  virtual void RunPrefetch() = 0;

  /**
   * @brief Runs this computation on a batch of images
   * in the forward stage (on gpu)
   */
  virtual void RunForward() = 0;
  
  DISABLE_COPY_ASSIGN(Stage);
private:
};

} // namespace ndll

#endif // NDLL_STAGE_H_

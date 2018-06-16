// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PIPELINE_OPERATORS_RESIZE_RANDOM_RESIZED_CROP_H_
#define DALI_PIPELINE_OPERATORS_RESIZE_RANDOM_RESIZED_CROP_H_

#include <vector>
#include <random>
#include <memory>
#include <utility>

#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/op_spec.h"

namespace dali {

template <typename Backend>
class RandomResizedCrop : public Operator<Backend> {
 public:
  explicit inline RandomResizedCrop(const OpSpec &spec) :
    Operator<Backend>(spec),
    params_(new Params()),
    size_(spec.GetRepeatedArgument<int>("size")),
    num_attempts_(spec.GetArgument<int>("num_attempts")),
    interp_type_(spec.GetArgument<DALIInterpType>("interp_type")) {
    InitParams(spec);
  }

  virtual inline ~RandomResizedCrop() = default;

  DISABLE_COPY_MOVE_ASSIGN(RandomResizedCrop);

  USE_OPERATOR_MEMBERS();

 protected:
  void RunImpl(Workspace<Backend> * ws, const int idx) override;
  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

 private:
  struct CropInfo {
    int x, y;
    int w, h;
  };

  void InitParams(const OpSpec &spec);

  bool TryCrop(int H, int W,
               std::uniform_real_distribution<float> *ratio_dis,
               std::uniform_real_distribution<float> *area_dis,
               std::uniform_real_distribution<float> *uniform,
               std::mt19937 *gen,
               CropInfo * crop) {
      float scale  = (*area_dis)(*gen);
      float ratio  = (*ratio_dis)(*gen);
      float swap   = (*uniform)(*gen);

      size_t original_area = H * W;
      float target_area = scale * original_area;

      int w = static_cast<int>(round(sqrtf(target_area * ratio)));
      int h = static_cast<int>(round(sqrtf(target_area / ratio)));

      if (swap < 0.5f) {
        std::swap(w, h);
      }

      if (w <= W && h <= H) {
        float rand_x = (*uniform)(*gen);
        float rand_y = (*uniform)(*gen);

        crop->w = w;
        crop->h = h;
        crop->x = static_cast<int>(rand_x * (W - w));
        crop->y = static_cast<int>(rand_y * (H - h));
        return true;
      } else {
        return false;
      }
  }

  // To be filled by actual implementations
  struct Params {};

  unique_ptr<Params> params_;

  std::vector<int> size_;
  int num_attempts_;
  DALIInterpType interp_type_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_RESIZE_RANDOM_RESIZED_CROP_H_

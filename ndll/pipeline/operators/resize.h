// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_RESIZE_H_
#define NDLL_PIPELINE_OPERATORS_RESIZE_H_

#include <random>
#include <utility>
#include <vector>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class Resize : public Operator<Backend> {
 public:
  explicit inline Resize(const OpSpec &spec) :
    Operator<Backend>(spec),
    rand_gen_(time(nullptr)),
    random_resize_(spec.GetArgument<bool>("random_resize")),
    warp_resize_(spec.GetArgument<bool>("warp_resize")),
    resize_a_(spec.GetArgument<int>("resize_a")),
    resize_b_(spec.GetArgument<int>("resize_b")),
    image_type_(spec.GetArgument<NDLLImageType>("image_type")),
    color_(IsColor(image_type_)), C_(color_ ? 3 : 1),
    type_(spec.GetArgument<NDLLInterpType>("interp_type")) {
    // Validate input parameters
    NDLL_ENFORCE(resize_a_ > 0 && resize_b_ > 0);
    NDLL_ENFORCE(resize_a_ <= resize_b_);

    // Resize per-image data
    input_ptrs_.resize(batch_size_);
    output_ptrs_.resize(batch_size_);
    input_sizes_.resize(batch_size_);
    output_sizes_.resize(batch_size_);

    // Per set-of-samples random numbers
    per_sample_rand_.resize(batch_size_);
  }

  virtual inline ~Resize() = default;

 protected:
  void SetupSharedSampleParams(Workspace<Backend>* ws) override;

  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  inline void DataDependentSetup(Workspace<Backend> *ws, const int idx);

  std::mt19937 rand_gen_;

  // Resize meta-data
  bool random_resize_;
  bool warp_resize_;
  int resize_a_, resize_b_;

  // Input/output channels meta-data
  NDLLImageType image_type_;
  bool color_;
  int C_;

  // Interpolation type
  NDLLInterpType type_;

  // store per-thread data for same resize on multiple data
  std::vector<std::pair<int, int>> per_sample_rand_;

  vector<const uint8*> input_ptrs_;
  vector<uint8*> output_ptrs_;

  vector<NDLLSize> input_sizes_, output_sizes_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_RESIZE_H_

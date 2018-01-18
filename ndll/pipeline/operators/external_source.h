// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_EXTERNAL_SOURCE_H_
#define NDLL_PIPELINE_OPERATORS_EXTERNAL_SOURCE_H_

#include <string>
#include <vector>

#include "ndll/pipeline/operator.h"

namespace ndll {

/**
 * @brief Provides in-graph access to data fed in from outside of ndll.
 * For now, we do a copy from the passed in data into our data to avoid
 * potential scoping and data corruption issues.
 */
template <typename Backend>
class ExternalSource : public Operator<Backend> {
 public:
  inline explicit ExternalSource(const OpSpec &spec) :
    Operator<Backend>(spec) {
    output_name_ = spec.Output(0);
  }

  virtual inline ~ExternalSource() = default;

  inline string name() const override {
    return "ExternalSource (" + output_name_ + ")";
  }

  /**
   * @brief Sets the data that should be passed out of the op
   * on the next iteration.
   */
  inline void SetDataSource(const TensorList<Backend> &tl) {
    // Note: If we create a GPU source, we will need to figure
    // out what stream we want to do this copy in. CPU we can
    // pass anything as it is ignored.
    tl_data_.Copy(tl, 0);
    data_in_tl_ = true;
  }

  /**
   * @brief Sets the data that should be passed out of the op
   * on the next iteration.
   */
  inline void SetDataSource(const vector<Tensor<Backend>> &t) {
    // Note: If we create a GPU source, we will need to figure
    // out what stream we want to do this copy in. CPU we can
    // pass anything as it is ignored.
    t_data_.resize(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
      t_data_[i].Copy(t[i], 0);
    }
    data_in_tl_ = false;
  }

  DISABLE_COPY_MOVE_ASSIGN(ExternalSource);

 protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws, const int idx) override {
    // Wrap the output tensor around our data
    auto output = ws->Output<Backend>(idx);
    if (data_in_tl_) {
      output->ShareData(&tl_data_, ws->data_idx());
    } else {
      NDLL_ENFORCE_VALID_INDEX((size_t)ws->data_idx(), t_data_.size());
      auto &data = t_data_[ws->data_idx()];
      output->ShareData(&data);
    }
  }

  inline void RunBatchedGPU(DeviceWorkspace *ws, const int idx) override {
    NDLL_ENFORCE(data_in_tl_, "Cannot feed non-contiguous data in gpu op.");
    auto output = ws->Output<Backend>(idx);
    output->ShareData(&tl_data_);
  }

  string output_name_;
  TensorList<Backend> tl_data_;
  vector<Tensor<Backend>> t_data_;
  bool data_in_tl_ = true;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_EXTERNAL_SOURCE_H_

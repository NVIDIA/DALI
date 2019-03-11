// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_OPERATORS_UTIL_EXTERNAL_SOURCE_H_
#define DALI_PIPELINE_OPERATORS_UTIL_EXTERNAL_SOURCE_H_

#include <atomic>
#include <string>
#include <vector>
#include <condition_variable>
#include <mutex>

#include "dali/pipeline/operators/operator.h"

namespace dali {

/**
 * @brief Provides in-graph access to data fed in from outside of dali.
 * For now, we do a copy from the passed in data into our data to avoid
 * potential scoping and data corruption issues.
 */
template <typename Backend>
class ExternalSource : public Operator<Backend> {
 public:
  inline explicit ExternalSource(const OpSpec &spec) :
    Operator<Backend>(spec),
    samples_processed_(0),
    busy_(false) {
    output_name_ = spec.Output(0);
  }

  inline ~ExternalSource() override = default;

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
    std::unique_lock<std::mutex> lock(busy_m_);
    cv_.wait(lock,  [this]{return !this->busy_;});
    tl_data_.Copy(tl, 0);
    data_in_tl_ = true;
    busy_ = true;
  }

  /**
   * @brief Sets the data that should be passed out of the op
   * on the next iteration.
   */
  inline void SetDataSource(const vector<Tensor<Backend>> &t) {
    // Note: If we create a GPU source, we will need to figure
    // out what stream we want to do this copy in. CPU we can
    // pass anything as it is ignored.

    std::unique_lock<std::mutex> lock(busy_m_);
    cv_.wait(lock,  [this]{return !this->busy_;});

    t_data_.resize(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
      t_data_[i].Copy(t[i], 0);
    }
    data_in_tl_ = false;
    busy_ = true;
  }

  DISABLE_COPY_MOVE_ASSIGN(ExternalSource);

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  string output_name_;
  TensorList<Backend> tl_data_;
  std::vector<Tensor<Backend>> t_data_;
  bool data_in_tl_;

  int samples_processed_;

  bool busy_;
  std::condition_variable cv_;
  std::mutex busy_m_;
  std::mutex samples_processed_m_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_UTIL_EXTERNAL_SOURCE_H_

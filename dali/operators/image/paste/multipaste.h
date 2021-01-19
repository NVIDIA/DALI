// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_
#define DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/static_switch.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/core/format.h"
#include "dali/util/crop_window.h"
#include "dali/pipeline/data/types.h"


namespace dali {

template <typename Backend>
class MultiPasteOp : public Operator<Backend> {
 public:
  ~MultiPasteOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(MultiPasteOp);

 protected:
  using Coords = TensorView<typename compute_to_storage<Backend>::type, const int, 1>;

  explicit MultiPasteOp(const OpSpec &spec)
      : Operator<Backend>(spec)
      , output_type_arg_(spec.GetArgument<DALIDataType>("dtype"))
      , output_type_(DALI_NO_TYPE)
      , input_type_(DALI_NO_TYPE)
      , output_size_("output_size", spec)
      , in_idx_("in_ids", spec)
      , in_anchors_("in_anchors", spec)
      , in_shapes_("shapes", spec)
      , out_anchors_("out_anchors", spec) {
    zero_anchors_ = Coords(zeros_, {2});

    if (std::is_same<Backend, GPUBackend>::value) {
      kernel_manager_.Resize(1, 1);
    } else {
      kernel_manager_.Resize(num_threads_, max_batch_size_);
    }
  }

  bool CanInferOutputs() const override {
    return true;
  }

  template<int ndim = 2>
  bool Intersects(const Coords& anchors1, const Coords& shapes1,
                  const Coords& anchors2, const Coords& shapes2) const {
    for (int i = 0; i < ndim; i++) {
      if (anchors1.data[i] + shapes1.data[i] <= anchors2.data[i]
          || anchors2.data[i] + shapes2.data[i] <= anchors1.data[i]) {
        return false;
      }
    }
    return true;
  }

  void AcquireArguments(const OpSpec &spec, const workspace_t<Backend> &ws) {
    const auto &images = ws.template InputRef<Backend>(0);

    auto curr_batch_size = ws.GetRequestedBatchSize(0);
    output_size_.Acquire(spec, ws, curr_batch_size, true);
    in_idx_.Acquire(spec, ws, curr_batch_size, false);
    if (out_anchors_.IsDefined()) {
      out_anchors_.Acquire(spec, ws, curr_batch_size, false);
    }
    if (in_anchors_.IsDefined()) {
      in_anchors_.Acquire(spec, ws, curr_batch_size, false);
    }
    if (in_shapes_.IsDefined()) {
      in_shapes_.Acquire(spec, ws, curr_batch_size, false);
    }
    input_type_ = ws.template InputRef<Backend>(0).type().id();
    output_type_ =
        output_type_arg_ != DALI_NO_TYPE
        ? output_type_arg_
        : input_type_;

    for (int i = 0; i < curr_batch_size; i++) {
      raw_input_size_mem_.push_back(static_cast<int>(images.shape()[i].data()[0]));
      raw_input_size_mem_.push_back(static_cast<int>(images.shape()[i].data()[1]));
    }

    for (int i = 0; i < curr_batch_size; i++) {
      const int64_t n_paste = in_idx_[i].shape[0];

      if (in_anchors_.IsDefined()) {
        DALI_ENFORCE(in_anchors_[i].shape[0] == n_paste,
                     "in_anchors must be same length as in_idx");
      }
      if (in_shapes_.IsDefined()) {
        DALI_ENFORCE(in_shapes_[i].shape[0] == n_paste, "in_shapes must be same length as in_idx");
      }
      if (out_anchors_.IsDefined()) {
        DALI_ENFORCE(out_anchors_[i].shape[0] == n_paste,
                     "out_anchors must be same length as in_idx");
      }

      bool found_intersection = false;

      for (int j = 0; j < n_paste; j++) {
        auto out_anchor = GetInAnchors(i, j);
        auto j_idx = in_idx_[i].data[j];
        const auto &shape = GetShape(i, j, Coords(raw_input_size_mem_.data() + 2 * j_idx,
                                                  dali::TensorShape<>(2)));
        for (int k = 0; k < j; k++) {
          auto k_idx = in_idx_[i].data[k];
          if (Intersects(out_anchor, shape, GetInAnchors(i, k), GetShape(
                  i, k, Coords(raw_input_size_mem_.data() + 2 * k_idx, dali::TensorShape<>(2))))) {
            found_intersection = true;
            break;
          }
        }
        if (found_intersection) {
          break;
        }
      }
      no_intersections_.push_back(!found_intersection);
    }
  }

  inline Coords GetInAnchors(int sample_num, int paste_num) const {
    return in_anchors_.IsDefined()
           ? subtensor(in_anchors_[sample_num], paste_num)
           : zero_anchors_;
  }

  inline Coords GetShape(int sample_num, int paste_num, Coords default_shape) const {
    return in_shapes_.IsDefined()
           ? subtensor(in_shapes_[sample_num], paste_num)
           : default_shape;
  }

  inline Coords GetOutAnchors(int sample_num, int paste_num) const {
    return out_anchors_.IsDefined()
           ? subtensor(out_anchors_[sample_num], paste_num)
           : zero_anchors_;
  }

  USE_OPERATOR_MEMBERS();
  DALIDataType output_type_arg_, output_type_, input_type_;

  ArgValue<int, 1> output_size_;

  ArgValue<int, 1> in_idx_;
  ArgValue<int, 2> in_anchors_;
  ArgValue<int, 2> in_shapes_;
  ArgValue<int, 2> out_anchors_;

  kernels::KernelManager kernel_manager_;

  const int zeros_[2] = {0, 0};
  Coords zero_anchors_;

  vector<bool> no_intersections_;
  vector<int> raw_input_size_mem_;
};


class MultiPasteCpu : public MultiPasteOp<CPUBackend> {
 public:
  explicit MultiPasteCpu(const OpSpec &spec) : MultiPasteOp(spec) {}

  using Operator<CPUBackend>::RunImpl;

  ~MultiPasteCpu() override = default;

  DISABLE_COPY_MOVE_ASSIGN(MultiPasteCpu);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;

  void RunImpl(workspace_t<CPUBackend> &ws) override;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_PASTE_MULTIPASTE_H_

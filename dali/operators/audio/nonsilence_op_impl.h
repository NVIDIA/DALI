// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_NONSILENCE_OP_IMPL_H
#define DALI_NONSILENCE_OP_IMPL_H

#include <utility>
#include <vector>
#include <dali/kernels/kernel_manager.h>
#include <gtest/gtest_prod.h>
#include <dali/pipeline/data/views.h>
#include <dali/kernels/signal/decibel/to_decibels_cpu.h>
#include "dali/core/convert.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/kernels/signal/moving_mean_square.h"

namespace dali {



//class DLL_PUBLIC NonsilenceOperatorCpuImpl {
// private:
//  using MmsKernel=kernels::signal::MovingMeanSquareCpu<float>;
//  using ToDbKernel=kernels::signal::ToDecibelsCpu<float>;
//  using MmsArgs= kernels::signal::MovingMeanSquareArgs;
//  using DbArgs=  kernels::signal::ToDecibelsArgs<float>;
//
// public:
//
//  template<typename InputType, int ndims = 1>
//  struct Args {
//    TensorView<StorageCPU, const InputType, ndims> input;
//    float cutoff_db;
//    float reference_db;
//    bool reference_max;
//    int window_length;
//    int reset_inteval;
//  };
//
//  explicit  NonsilenceOperatorCpuImpl(int batch_size) : batch_size_(batch_size){}
//
//  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t <CPUBackend> &ws) {
//    nthreads_ = 1;
//    nsamples_ = 1;
//
//    TypeInfo output_type;
//    output_type.SetType<detail::OutputType>(TypeTable::GetTypeID<detail::OutputType>());
//    TensorShape<> scalar_shape = {1};
//
//    output_desc.resize(detail::kNumOutputs);
//    for (int i = 0; i < detail::kNumOutputs; i++) {
//      output_desc[i].shape = uniform_list_shape(batch_size_, scalar_shape);
//      output_desc[i].type = output_type;
//    }
//    return true;
//  }
//
//
//  template<typename InputType>
//  void RunImpl(workspace_t <CPUBackend> &ws) {
//
//  }
//
//
//  /**
//   *
//   * @return (begin_idx, length)
//   */
//  template<typename InputType>
//  std::pair<int, int>
//  DetectNonsilenceRegion(int thread_id, int sample_id, const Args<InputType> &args) {
//    SetupKernels(nthreads_, nsamples_);
//    DbArgs db_args;
//    db_args.s_ref=args.reference_db;
//    db_args.ref_max=args.reference_max;
//    RunKernels(thread_id, sample_id, args.input, {args.window_length, args.reset_inteval}, db_args);
//    auto dbs = view_as_tensor<float>(to_db_kernel_.outputs_[sample_id]);
//    return LeadTrailThresh(make_cspan(dbs.data, dbs.num_elements()), args.cutoff_db);
//  }
//
//
//  /**
//   * Performs leading and trailing thresholding.
//   *
//   * Returns index of a first value above the threshold in the buffer and
//   * the length to the point, where last value above the threshold appears in the buffer.
//   * @return (begin_idx, length)
//   */
//  template<typename T>
//  static std::pair<int, int> LeadTrailThresh(span<const T> buffer, T cutoff) {
//    assert(buffer.size() > 0);
//    int begin = -1;
//    int end = buffer.size();
//    while (begin < end && buffer[++begin] < cutoff);  // NOLINT
//    if (begin == end) return {-1, 0};
//    while (buffer[--end] < cutoff);  // NOLINT
//    return {begin, end - begin + 1};
//  }
//
//
//  void SetupKernels(int nthreads, int nsamples) {
//    mms_kernel_.Setup(nthreads, nsamples);
//    to_db_kernel_.Setup(nthreads, nsamples);
//  }
//
//
//  template<typename InputType>
//  void RunKernels(int thread_id, int sample_id, TensorView<StorageCPU, const InputType, 1> in,
//                  const MmsArgs &mms_args, const DbArgs &db_args) {
//    mms_kernel_.Run(thread_id, sample_id, in, mms_args);
//    auto db_in = view_as_tensor<const float>(mms_kernel_.outputs_[sample_id]).to_static<1>();
//    to_db_kernel_.Run(thread_id, sample_id, db_in, db_args);
//  }
//
//
//  int nthreads_ = -1, nsamples_ = -1;
//  int batch_size_;
//
//
//  template<typename Kernel>
//  struct YetAnotherKernelManager {
//    kernels::KernelManager manager_;
//    std::vector<Tensor < CPUBackend>> outputs_;
//
//
//    void Setup(int nthreads, int nsamples) {
//      manager_.Resize<Kernel>(nthreads, nsamples);
//      outputs_.resize(nsamples);
//    }
//
//
//    template<typename InputType, typename Args>
//    void Run(int thread_id, int sample_id, TensorView<StorageCPU, const InputType, 1> in,
//             const Args &args) {
//      kernels::KernelContext kctx;
//      auto reqs = manager_.Setup<Kernel>(sample_id, kctx, in, args);
//      outputs_[sample_id].Resize(reqs.output_shapes[0][sample_id]);
//      auto out = view_as_tensor<float>(outputs_[sample_id]);
//      manager_.Run<Kernel>(thread_id, sample_id, kctx, out.template to_static<1>(), in, args);
//    }
//  };
//
//  YetAnotherKernelManager<MmsKernel> mms_kernel_;
//  YetAnotherKernelManager<ToDbKernel> to_db_kernel_;
//};


}  // namespace dali



#endif //DALI_NONSILENCE_OP_IMPL_H

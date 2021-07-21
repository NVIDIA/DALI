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

#ifndef DALI_PIPELINE_OPERATOR_BATCH_SIZE_PROVIDER_H_
#define DALI_PIPELINE_OPERATOR_BATCH_SIZE_PROVIDER_H_

namespace dali {

/**
 * BatchSizeProvider is an Operator, that determines the batch size used in
 * a single iteration in the pipeline. In general, this Operator will also
 * read data into the pipeline, like e.g. ExternalSource or any Reader.
 *
 * The usage is similar to a unidirectional iterator over a list of batch sizes
 */
class BatchSizeProvider {
  /*
   * Typically, you'll need to inherit from this interface virtually:
   *
   * template<typename Backend>
   * class MyOperatorBsp : public OperatorBase<Backend>, virtual public BatchSizeProvider {};
   */
 public:
  /**
   * Returns next batch size.
   *
   * Implementation shall assure that it's possible to call NextBatchSize()
   * multiple times for the same batch, before Advance() invocation.
   *
   * When there's no next batch size available, the implementation shall throw std::out_of_range.
   */
  virtual int NextBatchSize() = 0;

  /**
   * Advances to next batch.
   *
   * When there's no further data available, Advance() shall throw std::out_of_range
   */
  virtual void Advance() = 0;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BATCH_SIZE_PROVIDER_H_

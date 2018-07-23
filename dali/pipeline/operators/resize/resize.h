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

#ifndef DALI_PIPELINE_OPERATORS_RESIZE_RESIZE_H_
#define DALI_PIPELINE_OPERATORS_RESIZE_RESIZE_H_

#include <random>
#include <utility>
#include <vector>

#include "dali/common.h"
#include "dali/pipeline/operators/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/operators/fused/resize_crop_mirror.h"

namespace dali {

typedef enum {
  input_t,
  output_t
} io_type;

typedef std::pair<int, int> resize_t;

class ResizeAttr;
typedef NppiPoint MirroringInfo;

class ResizeParamDescr {
 public:
  ResizeParamDescr(ResizeAttr *pntr, NppiPoint *pOutResize = NULL, MirroringInfo *pMirror = NULL,
                        size_t pTotalSize[] = NULL, size_t batchSliceNumb = 0) :
                        pResize_(pntr), pResizeParam_(pOutResize), pMirroring_(pMirror),
                        pTotalSize_(pTotalSize), nBatchSlice_(batchSliceNumb) {}
  ResizeAttr *pResize_;
  NppiPoint *pResizeParam_;
  MirroringInfo *pMirroring_;
  size_t *pTotalSize_;
  size_t nBatchSlice_;
};

class ResizeAttr : protected ResizeCropMirrorAttr {
 public:
  explicit inline ResizeAttr(const OpSpec &spec) : ResizeCropMirrorAttr(spec),
            color_(IsColor(image_type_)), C_(color_ ? 3 : 1) {
  }

  void SetSize(DALISize *in_size, const vector<Index> &shape, int idx,
               DALISize *out_size, TransformMeta const * meta = NULL) const;

  inline vector<DALISize> &sizes(io_type type)            { return sizes_[type]; }
  inline DALISize *size(io_type type, size_t idx)         { return sizes(type).data() + idx; }
  void DefineCrop(DALISize *out_size, int *pCropX, int *pCropY, int idx = -1) const;
  void MirrorNeeded(NppiPoint *pntr, int idx = -1) const  {
      pntr->x = per_sample_meta_[idx].mirror;
      pntr->y = 0;  // Vertical mirroring not yet implemented for ResizeCropMirror
  }

 protected:
  virtual uint ResizeInfoNeeded() const                   { return 0; }

  inline vector<const uint8*> *inputImages()              { return &input_ptrs_; }
  inline vector<uint8 *> *outputImages()                  { return &output_ptrs_; }

  // Input/output channels meta-data
  bool color_;
  int C_;

  // store per-thread data for same resize on multiple data
  std::vector<TransformMeta> per_sample_meta_;

  vector<const uint8*> input_ptrs_;
  vector<uint8*> output_ptrs_;

  vector<DALISize> sizes_[2];
};

template <typename Backend>
class Resize : public Operator<Backend>, protected ResizeAttr {
 public:
  explicit inline Resize(const OpSpec &spec) :
    Operator<Backend>(spec), ResizeAttr(spec) {
      resizeParam_.resize(batch_size_ * 2);
      // Resize per-image data
      input_ptrs_.resize(batch_size_);
      output_ptrs_.resize(batch_size_);
      sizes_[0].resize(batch_size_);
      sizes_[1].resize(batch_size_);

      // Per set-of-sample TransformMeta
      per_sample_meta_.resize(batch_size_);
  }

  virtual inline ~Resize() = default;

 protected:
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  void SetupSharedSampleParams(Workspace<Backend> *ws) override;

  inline void DataDependentSetup(Workspace<Backend> *ws, const int idx);

  vector<NppiPoint> resizeParam_;
  USE_OPERATOR_MEMBERS();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_RESIZE_RESIZE_H_

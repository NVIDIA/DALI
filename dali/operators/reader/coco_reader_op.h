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

#ifndef DALI_OPERATORS_READER_COCO_READER_OP_H_
#define DALI_OPERATORS_READER_COCO_READER_OP_H_

#include <fstream>
#include <istream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/operators/reader/loader/coco_loader.h"
#include "dali/operators/reader/reader_op.h"

namespace dali {

class COCOReader : public DataReader<CPUBackend, ImageLabelWrapper> {
 public:
  explicit COCOReader(const OpSpec& spec);
  void RunImpl(SampleWorkspace &ws) override;

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, ImageLabelWrapper);

 private:
  CocoLoader& LoaderImpl() { return dynamic_cast<CocoLoader&>(*loader_); }

  bool output_polygon_masks_ = false;
  bool output_pixelwise_masks_ = false;
  bool output_image_ids_ = false;

  bool legacy_polygon_format_ = false;

  void PixelwiseMasks(int image_id, int* masks_output);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_COCO_READER_OP_H_

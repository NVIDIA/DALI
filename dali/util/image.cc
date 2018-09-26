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

#include "dali/util/image.h"

namespace dali {

void LoadFromFile(const string &file_name, uint8 **image, int *h, int *w, int *c) {
  std::ifstream file(file_name + ".txt");
  DALI_ENFORCE(file.is_open());

  file >> *h;
  file >> *w;
  file >> *c;

  // lol at this multiplication
  int size = (*h)*(*w)*(*c);
  *image = new uint8[size];
  int tmp = 0;
  for (int i = 0; i < size; ++i) {
    file >> tmp;
    (*image)[i] = (uint8)tmp;
  }
}

int idxHWC(int h, int w, int c, int i, int j, int k) {
  return (i * w + j) * c + k;
}

int idxCHW(int h, int w, int c, int i, int j, int k) {
  return (k * h + i) * w + j;
}

void WriteHWCImage(const uint8 *img, int h, int w, int c, const string &file_name) {
  WriteImageScaleBias(img, h, w, c, 0.f, 1.0f, file_name, idxHWC);
}

void WriteBatch(const TensorList<CPUBackend> &tl, const string &suffix, float bias, float scale) {
  const auto type = tl.type();
  const auto layout = tl.GetLayout();

  DALI_IMAGE_TYPE_SWITCH(type.id(), imgType,
    if (layout == DALI_NCHW)
      WriteCHWBatch<imgType>(tl, bias, scale, suffix);
    else
      WriteHWCBatch<imgType>(tl, bias, scale, suffix);
  )
}

}  // namespace dali

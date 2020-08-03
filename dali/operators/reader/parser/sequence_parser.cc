// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>
#include <memory>

#include "dali/operators/reader/parser/sequence_parser.h"
#include "dali/image/image_factory.h"

namespace dali {

void SequenceParser::Parse(const TensorSequence& data, SampleWorkspace* ws) {
  auto& sequence = ws->Output<CPUBackend>(0);
  sequence.SetLayout("FHWC");
  sequence.set_type(TypeInfo::Create<uint8_t>());
  Index seq_length = data.tensors.size();

  // Decode first frame, obtain it's size and allocate output
  {
    auto file_name = data.tensors[0].GetSourceInfo();
    std::unique_ptr<Image> img;

    try {
      img = ImageFactory::CreateImage(
        data.tensors[0].data<uint8_t>(), data.tensors[0].size(), image_type_);
      img->Decode();
    } catch (std::exception &e) {
      DALI_FAIL(e.what() + ". File: " + file_name);
    }
    const auto decoded = img->GetImage();

    const auto shape = img->GetShape();
    const Index h = shape[0];
    const Index w = shape[1];
    const Index c = shape[2];
    const auto frame_size = volume(shape);

    // Calculate shape of sequence tensor, that is Frames x (Frame Shape)
    auto seq_shape = std::vector<Index>{seq_length, h, w, c};
    sequence.Resize(seq_shape);
    // Take a view tensor for first frame and copy it to target sequence
    auto view_0 = sequence.SubspaceTensor(0);
    std::memcpy(view_0.raw_mutable_data(), decoded.get(), frame_size);
  }

  // Decode and copy rest of the frames
  for (Index frame = 1; frame < seq_length; frame++) {
    auto view_tensor = sequence.SubspaceTensor(frame);
    auto file_name = data.tensors[frame].GetSourceInfo();
    std::unique_ptr<Image> img;
    try {
      img = ImageFactory::CreateImage(data.tensors[frame].data<uint8_t>(),
                                      data.tensors[frame].size(), image_type_);
      img->Decode();
    } catch (std::exception &e) {
      DALI_FAIL(e.what() + ". File: " + file_name);
    }
    img->GetImage(view_tensor.mutable_data<uint8_t>());
    DALI_ENFORCE(view_tensor.shares_data(),
                 "Buffer view was invalidated after image decoding, frames do not match in "
                 "dimensions");
  }
}

}  // namespace dali

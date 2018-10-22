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

#ifndef DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_H_
#define DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_H_

#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/image/image_factory.h"
#include "dali/pipeline/operators/operator.h"

namespace dali {

class HostDecoder : public Operator<CPUBackend> {
 public:
  explicit inline HostDecoder(const OpSpec &spec)
      : Operator<CPUBackend>(spec),
        output_type_(spec.GetArgument<DALIImageType>("output_type")),
        c_(IsColor(output_type_) ? 3 : 1),
        decode_sequences_(spec.GetArgument<bool>("decode_sequences")) {}

  virtual inline ~HostDecoder() = default;
  DISABLE_COPY_MOVE_ASSIGN(HostDecoder);

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx) override {
    const auto &input = ws->Input<CPUBackend>(0);
    auto *output = ws->Output<CPUBackend>(0);
    // Verify input
    DALI_ENFORCE(input.ndim() == 1, "Input must be 1D encoded jpeg string.");
    DALI_ENFORCE(IsType<uint8>(input.type()), "Input must be stored as uint8 data.");

    if (!decode_sequences_) {
      DALI_ENFORCE(ws->NumInput() == 1, "When decoding singular images one input is expected");
      DecodeSingle(input.data<uint8_t>(), input.size(), output);
    } else {
      DALI_ENFORCE(ws->NumInput() == 2, "When decoding sequences two inputs are expected");
      const auto &metadata = ws->Input<CPUBackend>(1);
      DALI_ENFORCE(metadata.ndim() == 1, "Metadata must be 1D encoded offsets.");
      DALI_ENFORCE(IsType<Index>(metadata.type()) == 1, "Metadata must be stored as int64 data.");
      const auto *metadata_ptr = metadata.data<Index>();
      const auto *input_ptr = input.data<uint8_t>();
      Index frame_count = metadata_ptr[0];
      for (Index frame = 0; frame < frame_count; frame++) {
        auto frame_size = metadata_ptr[frame + 1];
        if (frame == 0) {
          // First frame to check sizes
          Tensor<CPUBackend> tmp;
          DecodeSingle(input_ptr, frame_size, &tmp);

          // Calculate shape of sequence tensor, that is Frames x (Frame Shape)
          auto frames_x_shape = std::vector<Index>();
          frames_x_shape.push_back(frame_count);
          auto frame_shape = tmp.shape();
          frames_x_shape.insert(frames_x_shape.end(), frame_shape.begin(), frame_shape.end());
          output->Resize(frames_x_shape);
          output->set_type(TypeInfo::Create<uint8_t>());
          // Take a view tensor for first frame and
          auto view_0 = output->SubspaceTensor(frame);
          std::memcpy(view_0.raw_mutable_data(), tmp.raw_data(), tmp.size());
        } else {
          // Rest of frames
          auto view_tensor = output->SubspaceTensor(frame);
          DecodeSingle(input_ptr, frame_size, &view_tensor);
          DALI_ENFORCE(view_tensor.shares_data(),
                       "Buffer view was invalidated after image decoding, frames do not match in "
                       "dimensions");
        }
        input_ptr += frame_size;
      }
    }
  }

  /**
   * @brief Decode single stream of bytes into 3-dimensional tensor of shape HWC representing image
   *
   * @param data input data stream
   * @param size size of input data
   * @param output decoded image
   */
  void DecodeSingle(const uint8_t *data, size_t size, Tensor<CPUBackend> *output) {
    auto img = ImageFactory::CreateImage(data, size, output_type_);
    img->Decode();
    const auto decoded = img->GetImage();
    const auto hwc = img->GetImageDims();
    const auto h = std::get<0>(hwc);
    const auto w = std::get<1>(hwc);
    const auto c = std::get<2>(hwc);

    output->Resize({static_cast<int>(h), static_cast<int>(w), static_cast<int>(c)});
    unsigned char *out_data = output->mutable_data<uint8_t>();
    std::memcpy(out_data, decoded.get(), h * w * c);
  }

  DALIImageType output_type_;
  int c_;
  bool decode_sequences_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DECODER_HOST_DECODER_H_

// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/video/reader/video_reader_decoder_resize_op.h"

#include <memory>
#include <vector>

namespace dali {

VideoReaderDecoderResize::VideoReaderDecoderResize(const OpSpec &spec)
    : VideoReaderDecoder<GPUBackend>(spec),
      ResizeBase<GPUBackend>(spec) {
  ResizeBase<GPUBackend>::InitializeGPU(spec.GetArgument<int>("minibatch_size"),
                                        spec.GetArgument<int64_t>("temp_buffer_hint"));
}

bool VideoReaderDecoderResize::SetupImpl(std::vector<OutputDesc> &output_desc,
                                         const Workspace &ws) {
  VideoReaderDecoder<GPUBackend>::SetupImpl(output_desc, ws);
  // Output 0 is described by the base reader with the decoded (not resized) shape.
  input_shape_ = output_desc[0].shape;
  int num_samples = input_shape_.num_samples();

  resize_attr_.PrepareResizeParams(spec_, ws, input_shape_, "FHWC");
  resampling_attr_.PrepareFilterParams(spec_, ws, num_samples);
  resample_params_.resize(resize_attr_.params_.size());
  resampling_attr_.GetResamplingParams(make_span(resample_params_),
                                       make_cspan(resize_attr_.params_));
  resize_attr_.GetResizedShape(output_shape_, input_shape_);

  output_desc[0].shape = output_shape_;
  return true;
}

void VideoReaderDecoderResize::ResizeVideoOutput(Workspace &ws) {
  auto &video_output = ws.Output<GPUBackend>(0);
  int batch_size = GetCurrBatchSize();
  assert(batch_size == video_output.num_samples());

  // Each sequence is resized separately, like in the legacy readers.video_resize: the resize
  // parameters are per sequence and are applied to all of its frames.
  for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
    auto &sample_data = GetSample(sample_id).data_;

    TensorList<GPUBackend> input;
    input.ShareData(sample_data.get_data_ptr(), sample_data.nbytes(), sample_data.is_pinned(),
                    uniform_list_shape(1, sample_data.shape()), sample_data.type(),
                    sample_data.device_id());

    TensorList<GPUBackend> output;
    TensorListShape<> sample_out_shape = uniform_list_shape(1, video_output.tensor_shape(sample_id));
    output.ShareData(std::shared_ptr<void>(video_output.raw_mutable_tensor(sample_id),
                                           [](void *) {}),
                     sample_out_shape.num_elements() * video_output.type_info().size(),
                     video_output.is_pinned(), sample_out_shape, video_output.type(),
                     video_output.device_id());

    TensorListShape<> resized_shape;
    SetupResize(resized_shape, output.type(), input.shape(), input.type(),
                make_cspan(&resample_params_[sample_id], 1), resize_attr_.first_spatial_dim_);
    assert(resized_shape == output.shape());
    RunResize(ws, output, input);
    video_output.SetSourceInfo(sample_id, sample_data.GetMeta().GetSourceInfo());
  }
  video_output.SetLayout("FHWC");
}

void VideoReaderDecoderResize::RunImpl(Workspace &ws) {
  ResizeVideoOutput(ws);
  WriteMetadataOutputs(ws);
}

DALI_SCHEMA(experimental__readers__VideoResize)
    .DocStr(R"code(Loads, decodes and resizes video files.

This operator combines the features of :meth:`nvidia.dali.fn.experimental.readers.video` and
:meth:`nvidia.dali.fn.resize`. The frames are resized on the GPU right after decoding, without
materializing the full-resolution sequences in the operator's output.

The outputs of the operator are: video, [labels], [frame_num], [timestamps]. They are the same as
the ones of :meth:`nvidia.dali.fn.experimental.readers.video`, except that the height and width of
``video`` are the ones requested by the resize arguments. The output data type follows ``dtype``.
)code")
    .NumInput(0)
    .OutputFn(detail::VideoReaderDecoderOutputFn)
    .AddParent("experimental__readers__Video")
    .AddParent("ResizeAttr")
    .AddParent("ResamplingFilterAttr")
    .OutputNDim(0, 4)
    .OutputLayout(0, "FHWC");

DALI_REGISTER_OPERATOR(experimental__readers__VideoResize, VideoReaderDecoderResize, GPU);

}  // namespace dali

// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/operators/reader/video_reader_op.h"

namespace dali {

void VideoReader::Prefetch() {
  DomainTimeRange tr("[DALI][VideoReader] Prefetch #" + to_string(curr_batch_producer_),
                      DomainTimeRange::kRed);
  DataReader<GPUBackend, SequenceWrapper>::Prefetch();
  auto &curr_batch = prefetched_batch_queue_[curr_batch_producer_];
  auto &curr_tensor_list = prefetched_batch_tensors_[curr_batch_producer_];

  // resize the current batch
  TensorListShape<4> tmp_shapes;
  tmp_shapes.resize(curr_batch.size());
  auto ref_type = curr_batch[0]->dtype;
  for (size_t data_idx = 0; data_idx < curr_batch.size(); ++data_idx) {
    auto &sample = curr_batch[data_idx];
    assert(ref_type == sample->dtype);
    tmp_shapes.set_tensor_shape(data_idx, sample->shape());
  }

  curr_tensor_list.Resize(tmp_shapes, TypeTable::GetTypeInfo(ref_type));

  // ask for frames
  for (size_t data_idx = 0; data_idx < curr_tensor_list.ntensor(); ++data_idx) {
    auto &sample = curr_batch[data_idx];
    sample->sequence.ShareData(&curr_tensor_list, static_cast<int>(data_idx));
    sample->read_sample_f();
    // data has been read, decouple sequence from the wrapped memory
    sample->sequence.Reset();
  }
  // make sure that frames have been processed
  for (size_t data_idx = 0; data_idx < curr_tensor_list.ntensor(); ++data_idx) {
    auto &sample = curr_batch[data_idx];
    // We have to wait for all kernel recorded in sequence's event are completed
    LOG_LINE << "Waiting for sequence..";
    sample->wait();
    LOG_LINE << "Got sequence\n";
  }
}


DALI_REGISTER_OPERATOR(readers__Video, VideoReader, GPU);

DALI_SCHEMA(readers__Video)
  .DocStr(R"code(Loads and decodes video files using FFmpeg and NVDECODE, which is
the hardware-accelerated video decoding feature in the NVIDIA(R) GPU.

The video streams can be in most of the container file formats. FFmpeg is used to parse video
containers and returns a batch of sequences of ``sequence_length`` frames with shape
``(N, F, H, W, C)``, where ``N`` is the batch size, and ``F`` is the number of frames).
This class only supports constant frame rate videos.

.. note::
  Containers which doesn't support indexing, like mpeg, requires DALI to seek to the sequence  when
  each new sequence needs to be decoded.)code")
  .NumInput(0)
  .OutputFn(detail::VideoReaderOutputFn)
  .AddOptionalArg("filenames",
      R"code(File names of the video files to load.

This option is mutually exclusive with ``file_list`` and ``file_root``.)code",
      std::vector<std::string>{})
    .AddOptionalArg<vector<int>>("labels", R"(Labels associated with the files listed in
``filenames`` argument.

If an empty list is provided, sequential 0-based indices are used as labels. If not provided,
no labels will be yielded.)", nullptr)
  .AddOptionalArg("file_root",
      R"code(Path to a directory that contains the data files.

This option is mutually exclusive with ``filenames`` and ``file_list``.)code",
      std::string())
  .AddOptionalArg("file_list",
      R"code(Path to the file with a list of ``file label [start_frame [end_frame]]`` values.

Positive value means the exact frame, negative counts as a Nth frame from the end (it follows
python array indexing schema), equal values for the start and end frame would yield an empty
sequence and a warning. This option is mutually exclusive with ``filenames``
and ``file_root``.)code",
      std::string())
  .AddOptionalArg("enable_frame_num",
      R"code(If the ``file_list`` or ``filenames`` argument is passed, returns the frame number
output.)code",
      false)
  .AddOptionalArg("enable_timestamps",
      R"code(If the ``file_list`` or ``filenames`` argument is passed, returns the timestamps
output. )code",
      false)
  .AddArg("sequence_length",
      R"code(Frames to load per sequence.)code",
      DALI_INT32)
  .AddOptionalArg("step",
      R"code(Frame interval between each sequence.

When the value is less than 0, ``step`` is set to ``sequence_length``.)code",
      -1)
  .AddOptionalArg("channels",
      R"code(Number of channels.)code",
      3)
  .AddOptionalArg("additional_decode_surfaces",
      R"code(Additional decode surfaces to use beyond minimum required.

This argument is ignored when the decoder cannot determine the minimum number of
decode surfaces

.. note::

  This can happen when the driver is an older version.

This parameter can be used to trade off memory usage with performance.)code",
      2)
  .AddOptionalArg("normalized",
      R"code(Gets the output as normalized data.)code",
      false)
  .AddOptionalArg("image_type",
      R"(The color space of the output frames (RGB or YCbCr).)",
      DALI_RGB)
  .AddOptionalArg("dtype",
      R"(Output data type.

Supported types: ``UINT8`` or ``FLOAT``.)",
      DALI_UINT8)
  .AddOptionalArg("stride",
      R"code(Distance between consecutive frames in the sequence.)code", 1u, false)
  .AddOptionalArg("skip_vfr_check",
      R"code(Skips the check for the variable frame rate (VFR) videos.

Use this flag to suppress false positive detection of VFR videos.

.. warning::

  When the dataset indeed contains VFR files, setting this flag may cause the decoder to
  malfunction.)code", false)
  .AddOptionalArg("file_list_frame_num",
      R"code(If the start/end timestamps are provided in file_list, you can interpret them
as frame numbers instead of as timestamps.

If floating point values have been provided, the start frame number will be rounded up and
the end frame number will be rounded down.

Frame numbers start from 0.)code", false)
  .AddOptionalArg("file_list_include_preceding_frame",
      R"code(Changes the behavior how ``file_list`` start and end frame timestamps are translated
to a frame number.

If the start/end timestamps are provided in file_list as timestamps, the start frame is
calculated as ``ceil(start_time_stamp * FPS)`` and the end as ``floor(end_time_stamp * FPS)``.
If this argument is set to True, the equation changes to ``floor(start_time_stamp * FPS)`` and
``ceil(end_time_stamp * FPS)`` respectively. In effect, the first returned frame is not later, and
the end frame not earlier, than the provided timestamps. This behavior is more aligned with how the visible
timestamps are correlated with displayed video frames.

.. note::

  When ``file_list_frame_num`` is set to True, this option does not take any effect.

.. warning::

  This option is available for legacy behavior compatibility.
)code", false)
  .AddOptionalArg("pad_sequences",
      R"code(Allows creation of incomplete sequences if there is an insufficient number
of frames at the very end of the video.

Redundant frames are zeroed. Corresponding time stamps and frame numbers are set to -1.)code", false)
  .AddParent("LoaderBase");


// Deprecated alias
DALI_REGISTER_OPERATOR(VideoReader, VideoReader, GPU);

DALI_SCHEMA(VideoReader)
    .DocStr("Legacy alias for :meth:`readers.video`.")
    .NumInput(0)
    .OutputFn(detail::VideoReaderOutputFn)
    .AddParent("readers__Video")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "readers__Video",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;

}  // namespace dali

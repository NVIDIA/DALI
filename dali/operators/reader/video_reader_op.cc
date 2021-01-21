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


DALI_REGISTER_OPERATOR(VideoReader, VideoReader, GPU);

DALI_SCHEMA(VideoReader)
  .DocStr(R"code(Loads and decodes video files using FFmpeg and NVDECODE, which is
the hardware-accelerated video decoding feature in the NVIDIA(R) GPU.

The video streams can be in most of the container file formats. FFmpeg is used to parse video
containers and returns a batch of sequences of ``sequence_length`` frames with shape
``(N, F, H, W, C)``, where ``N`` is the batch size, and ``F`` is the number of frames).
This class only supports constant frame rate videos.)code")
  .NumInput(0)
  .OutputFn(detail::VideoReaderOutputFn)
  .AddOptionalArg("filenames",
      R"code(File names of the video files to load.

This option is mutually exclusive with ``filenames`` and ``file_root``.)code",
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
  .AddOptionalArg("interleave_size",
      R"code(Reorganizes the video sequences.
When ``interleave_size`` is greater 0 it reorganizes the video sequences into different
interleaved batches by selecting from the provided videos in a round robin order.

For example if an ``interleave_size`` of 2 is specified and 4 different videos are provided the video
sequences will be interleaved into the following order:

.. code-block:: python

  video 0 sequence 0, video 1 sequence 0,
  video 0 sequence 1, video 1 sequence 1,
                   ...
  video 0 sequence n, video 1 sequence n,
  video 2 sequence 0, video 3 sequence 0,
  video 2 sequence 1, video 3 sequence 1,
                   ...
  video 2 sequence m, video 3 sequence m
 

If ``interleave_size > 0`` it affects how video sequences are shuffled. As only the order of the
videos is shuffled but not the sequences. Otherwise, the default shuffle behavior is kept.)code", 0)
  .AddOptionalArg("interleave_mode", R"code(
Determines how to reorganize video sequences when ``interleave_size > 0``. If a continuous mode has
been selected boundaries between different videos are disregarded except for the last ones.
Otherwise, boundaries in between video frames are kept.

Here is a list of supported modes:

* | ``"shorten_continuous"`` - interleaves all sequences while disregarding boundary in between
  | different videos except for the last ones. In this case, the last videos are shortened to the
  | length of the shortest video in the interleaved batch.
* | ``"repeat_continuous"`` - interleaves all sequences while disregarding boundary in between
  | different videos except for the last ones. In this case, the shorter videos are repeated until
  | all other video sequences have been read.
* | ``"clamp_continuous"`` - interleaves all sequences while disregarding boundary in between
  | different videos except for the last ones. In this case, the last sequence of the shorter
  | videos is repeated until all other video sequences have been read.
* | ``"shorten"`` - interleaves all sequences while keeping the boundary in between different
  | videos. In this case, all videos are shortened to the shortest video of the interleave batch.
* | ``"repeat"`` - interleaves all sequences while keeping the boundary in between different
  | videos. In this case, all shorter videos are repeated until all other video sequences of
  | the interleaved batch have been read.
* | ``"clamp"`` - interleaves all sequences while keeping the boundary in between different
  | videos. In this case, the last sequence of all shorter videos is repeated until all other
  | video sequences of the interleave batch have been read.)code", "shorten")
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
  .AddParent("LoaderBase");
}  // namespace dali

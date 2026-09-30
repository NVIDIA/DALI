// Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/video/reader/video_reader_decoder_op.h"

#include <string>
#include <vector>

namespace dali {

namespace detail {

int VideoReaderDecoderOutputFn(const OpSpec &spec) {
  // Must match VideoReaderDecoder::has_labels_: an empty file_list/file_root is "not provided".
  bool has_labels = spec.HasArgument("labels") ||
                    !spec.GetArgument<std::string>("file_list").empty() ||
                    !spec.GetArgument<std::string>("file_root").empty();
  bool has_frame_num =
      ParseFrameNumPolicy(spec.GetArgument<std::string>("enable_frame_num")) !=
      FrameNumPolicy::None;
  return 1 + has_labels + has_frame_num + spec.GetArgument<bool>("enable_timestamps");
}

boundary::BoundaryType GetReaderBoundaryType(const OpSpec &spec) {
  if (spec.HasArgument("pad_sequences")) {
    DALI_ENFORCE(!spec.HasArgument("pad_mode"),
                 "The deprecated argument `pad_sequences` cannot be combined with `pad_mode`. "
                 "Use `pad_mode` only (`pad_sequences=True` is `pad_mode='constant'`).");
    return spec.GetArgument<bool>("pad_sequences") ? boundary::BoundaryType::CONSTANT
                                                   : boundary::BoundaryType::ISOLATED;
  }
  return GetBoundaryType(spec);
}

FileListOptions GetFileListOptions(const OpSpec &spec) {
  const bool has_frame_num = spec.HasArgument("file_list_frame_num");
  const bool has_preceding = spec.HasArgument("file_list_include_preceding_frame");
  DALI_ENFORCE(!(has_frame_num && spec.HasArgument("file_list_format")),
               "The deprecated argument `file_list_frame_num` cannot be combined with "
               "`file_list_format`. Use `file_list_format` only (`file_list_frame_num=True` is "
               "`file_list_format='frames'`).");
  DALI_ENFORCE(!(has_preceding && spec.HasArgument("file_list_rounding")),
               "The deprecated argument `file_list_include_preceding_frame` cannot be combined "
               "with `file_list_rounding`. Use `file_list_rounding` only "
               "(`file_list_include_preceding_frame=True` is "
               "`file_list_rounding='start_down_end_up'`).");

  FileListOptions opts;
  std::string format_str;
  if (has_frame_num) {
    format_str = spec.GetArgument<bool>("file_list_frame_num") ? "frames" : "timestamps";
  } else {
    format_str = spec.GetArgument<std::string>("file_list_format");
  }
  if (format_str == "frames") {
    opts.format = FileListFormat::kFrames;
  } else if (format_str == "timestamps") {
    opts.format = FileListFormat::kTimestamps;
  } else {
    DALI_FAIL(make_string("Invalid file_list_format: ", format_str));
  }

  // Not explicitly given when has_preceding is set (enforced above), so this is the default.
  auto rounding_str = spec.GetArgument<std::string>("file_list_rounding");
  // Legacy: "When file_list_frame_num is set to True, this option does not take any effect."
  if (has_preceding && spec.GetArgument<bool>("file_list_include_preceding_frame") &&
      opts.format == FileListFormat::kTimestamps) {
    rounding_str = "start_down_end_up";
  }
  if (rounding_str == "start_down_end_up") {
    opts.rounding = FileListRounding::kStartDownEndUp;
  } else if (rounding_str == "start_up_end_down") {
    opts.rounding = FileListRounding::kStartUpEndDown;
  } else if (rounding_str == "all_up") {
    opts.rounding = FileListRounding::kAllUp;
  } else if (rounding_str == "all_down") {
    opts.rounding = FileListRounding::kAllDown;
  } else {
    DALI_FAIL(make_string("Invalid file_list_rounding: ", rounding_str));
  }

  opts.include_end = spec.GetArgument<bool>("file_list_include_end");
  return opts;
}

}  // namespace detail

DALI_SCHEMA(experimental__readers__Video)
    .DocStr(R"code(Loads and decodes video files from disk.

The operator supports most common video container formats using libavformat (FFmpeg).
The operator utilizes either libavcodec (FFmpeg) or NVIDIA Video Codec SDK (NVDEC) for decoding the frames.

The following video codecs are supported by both CPU and GPU backends:

* VP8
* VP9
* MJPEG

The following codecs are supported by the GPU backend only:

* AV1
* MPEG-4
* H.264/AVC
* H.265/HEVC

The outputs of the operator are: video, [labels], [frame_num], [timestamps].

* ``video``: A sequence of frames with shape ``(F, H, W, C)`` where ``F`` is the number of frames in the sequence
  (can vary between samples), ``H`` is the frame height in pixels, ``W`` is the frame width in pixels, and ``C`` is
  the number of color channels.
* ``labels``: Label associated with the sample. Only available when using ``labels`` with ``filenames``, or when
  using ``file_list`` or ``file_root``.
* ``frame_num``: Frame number information. Shape and content depend on ``enable_frame_num``:

  * ``"scalar"`` or ``True``: Index of the first frame in the decoded sequence, shape ``(1,)``.
  * ``"sequence"``: Frame index of each decoded frame, shape ``(F,)``. Padded frames (e.g. when
    using ``pad_mode='constant'``) have index ``-1``.
* ``timestamps``: Time in seconds of each frame in the sequence. Only available when ``enable_timestamps=True``.
)code")
    .NumInput(0)
    .OutputFn(detail::VideoReaderDecoderOutputFn)
    .AddOptionalArg("filenames",
                    R"code(Absolute paths to the video files to load.

This option is mutually exclusive with `file_root` and `file_list`.)code",
                    std::vector<std::string>{})
    .AddOptionalArg("file_root",
                    R"code(Path to a directory that contains the data files.

This option is mutually exclusive with `filenames` and `file_list`.)code",
                    std::string())
    .AddOptionalArg("file_list",
                    R"code(Path to the file with a list of ``file label [start [end]]`` values.

``start`` and ``end`` are optional and can be used to specify the start and end of the video to load.
The values can be interpreted differently depending on the ``file_list_format``. A missing
``end`` means "until the end of the video", and negative values count from the end of the video.

This option is mutually exclusive with `filenames` and `file_root`.)code",
                    std::string())
    .AddOptionalArg("file_list_format",
        R"code(How to interpret start/end values in file_list:

* ``frames``: Use exact frame numbers (0-based). Negative values count from end.
* ``timestamps``: Use timestamps in seconds, relative to the first frame of the video. Negative
  values count from the end. A ``start`` past the end of the video is an error; an ``end`` past
  the end of the video is clamped.

Default: ``timestamps``.)code",
        "timestamps")
    .AddOptionalArg("file_list_rounding",
        R"code(How to handle non-exact frame matches:

* ``start_up_end_down`` (default): Round start up and end down. Matches the rounding
  convention of the legacy ``readers.video`` operator's ``file_list_include_preceding_frame``
  default (``False``).
* ``start_down_end_up``: Round start down and end up
* ``all_up``: Round both up
* ``all_down``: Round both down)code",
        "start_up_end_down")
    .AddOptionalArg("file_list_include_end",
        R"code(If set to True, the ``end`` value of a `file_list` entry is treated as inclusive,
i.e. the frame at ``end`` is included in the selected range.

By default (False), ``end`` acts as an exclusive bound, which matches the behavior of the
legacy ``readers.video`` operator. Setting this to True selects one more frame (unless the range
already reaches the end of the video) than ``readers.video`` would for the same `file_list`.)code",
        false)
    .AddOptionalArg<vector<int>>("labels", R"(Labels associated with the files listed in
`filenames` argument.

If an empty list is provided, sequential 0-based indices are used as labels. If not provided,
no labels will be yielded.)",
                                 nullptr)
    .AddArg("sequence_length", R"code(Frames to load per sequence.)code", DALI_INT32)
    .AddOptionalArg("enable_frame_num",
                    R"code(Determines what frame number information is returned as an additional output.

* ``"none"`` or ``False`` (default): No frame number output.
* ``"scalar"`` or ``True``: Returns the index of the first frame in the decoded sequence, shape ``(1,)``.
* ``"sequence"``: Returns the frame index of each decoded frame, shape ``(F,)``. For padded
  frames (e.g. when using ``pad_mode='constant'``), the index is ``-1``.)code",
                    std::string("none"))
    .AddOptionalArg("enable_timestamps",
                    R"code(If set, returns the timestamp of the frames in the decoded sequence
as an additional output.)code",
                    false)
    .AddOptionalArg("step",
                    R"code(Frame interval between each sequence.

When the value is less than 0, `step` is set to `sequence_length`.)code",
                    -1)
    .AddOptionalArg("stride", R"code(Distance between consecutive frames in the sequence.)code", 1u,
                    false)
    .AddOptionalArg("uniform_sample",
        R"code(If set to True, uniformly samples ``sequence_length`` frames from the full video
(or from the video range defined by ``file_list``), regardless of the video length.

The sampled frame indices correspond to ``numpy.linspace(start, end-1, sequence_length)``
rounded to the nearest integer using ``floor(x + 0.5)`` (rounds half away from zero,
matching C++ ``std::round`` — not NumPy's default banker's rounding).

If ``sequence_length`` exceeds the number of frames in the video, frames are repeated
rather than padded. For example, sampling 5 frames from a 3-frame video yields indices
``[0, 1, 1, 2, 2]``. A single-frame video always produces a sequence of identical frames.

When enabled, each video file produces exactly one sample per epoch.
The ``stride``, ``step``, and ``pad_mode`` arguments are ignored.)code",
        false)
    .AddOptionalArg<std::string>(
        "pad_mode",
        R"code(How to handle videos with insufficient frames when using start_frame/sequence_length/stride:

* ``'none'``: Return shorter sequences if not enough frames: ABC -> ABC
* ``'constant'``: Pad with a fixed value (specified by ``pad_value``): ABC -> ABCPPP
* ``'edge'`` or ``'repeat'``: Repeat the last valid frame: ABC -> ABCCCC
* ``'reflect_1001'`` or ``'symmetric'``: Reflect padding, including the last element: ABC -> ABCCBA
* ``'reflect_101'`` or ``'reflect'``: Reflect padding, not including the last element: ABC -> ABCBA

Not relevant when using ``frames`` argument.)code",
        "none", true)
    .AddOptionalArg("fill_value",
                    R"code(Value(s) used to pad missing frames when ``pad_mode='constant'``'.

Each value must be in range [0, 255].
If a single value is provided, it will be used for all channels.
Otherwise, the number of values must match the number of channels in the video.
With ``dtype=FLOAT`` the values keep this 8-bit scale, or are divided by 255 when `normalized`
is set, so that padding matches decoded pixels of the same value.)code",
                    std::vector<int>{
                        0,
                    })
    .AddOptionalArg("image_type", R"(The color space of the output frames (RGB or YCbCr).)",
                    DALI_RGB)
    .AddOptionalTypeArg("dtype",
                    R"code(Output data type. Supported types: ``UINT8`` or ``FLOAT``.

``FLOAT`` is only supported on the GPU backend.)code",
                    DALI_UINT8)
    .AddOptionalArg("normalized",
                    R"code(If set, and ``dtype`` is ``FLOAT``, the output is returned as
normalized data in the range ``[0.0, 1.0]``. Ignored when ``dtype`` is ``UINT8``.)code",
                    false)
    .AddOptionalArg("channels",
                    R"code(Number of channels in the output. Must match the actual number of
decoded channels (currently always ``3``); provided for compatibility with ``readers.video``.)code",
                    3)
    .AddOptionalArg("additional_decode_surfaces",
                    R"code(Additional decode-lookahead slots, beyond a baseline of 8, used by the GPU
decoder.

This value sizes the GPU decoder's host-side frame reorder buffer (and the initial surface
count hint given to the video parser), which bounds how many frames can be decoded ahead of
the one being returned. It does not directly set the number of NVDEC decode surfaces; that
count is determined by the codec's own requirements (``min_num_decode_surfaces``).

Must be non-negative. Only relevant for the GPU backend; ignored on CPU.)code",
                    2)
    .AddOptionalArg("require_constant_frame_rate",
                    R"code(If set, raises an error if the video has a variable
frame rate. Default: ``False`` (variable frame rate videos are decoded using decode-order frame
indexing).)code",
                    false)
    .AddOptionalArg("pad_sequences",
                    R"code(Legacy ``readers.video`` argument, kept for compatibility.

``True`` is equivalent to ``pad_mode='constant'`` (with the default ``fill_value=0`` the
missing frames are zeroed, as in ``readers.video``); ``False`` keeps `pad_mode` at its
default. Cannot be combined with `pad_mode`.)code",
                    false)
    .DeprecateArg("pad_sequences", "2.4", false,
                  "`pad_sequences` is deprecated; use `pad_mode='constant'` instead.")
    .AddOptionalArg("file_list_frame_num",
                    R"code(Legacy ``readers.video`` argument, kept for compatibility.

``True`` is equivalent to ``file_list_format='frames'`` and ``False`` to
``file_list_format='timestamps'``. Cannot be combined with `file_list_format`.)code",
                    false)
    .DeprecateArg("file_list_frame_num", "2.4", false,
                  "`file_list_frame_num` is deprecated; use `file_list_format` instead.")
    .AddOptionalArg("file_list_include_preceding_frame",
                    R"code(Legacy ``readers.video`` argument, kept for compatibility.

When `file_list` entries are timestamps, ``True`` is equivalent to
``file_list_rounding='start_down_end_up'``. Has no effect with frame-number entries (as in
``readers.video``). Cannot be combined with `file_list_rounding`.)code",
                    false)
    .DeprecateArg("file_list_include_preceding_frame", "2.4", false,
                  "`file_list_include_preceding_frame` is deprecated; use `file_list_rounding` "
                  "instead.")
    .AddOptionalArg("skip_vfr_check",
                    R"code(Legacy ``readers.video`` argument, accepted for compatibility and
ignored.

This is intentionally NOT mapped onto `require_constant_frame_rate`: ``readers.video``
rejects variable frame rate (VFR) videos by default (``skip_vfr_check=False``) using a
heuristic, while this operator decodes VFR videos by default. Mapping the flag would change
behavior even for callers that never set it. To reject VFR videos, use
``require_constant_frame_rate=True``.)code",
                    false)
    .DeprecateArg("skip_vfr_check", "2.4", true,
                  "`skip_vfr_check` is ignored by experimental.readers.video, which decodes "
                  "variable frame rate videos by default. Use `require_constant_frame_rate=True` "
                  "to reject them.")
    .AddParent("LoaderBase")
    .OutputNDim(0, 4)
    .OutputLayout(0, "FHWC")
    .OutputDType(0, [](const OpSpec &spec) {
      return spec.GetArgument<DALIDataType>("dtype");
    });


DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<GPUBackend>, GPU);

}  // namespace dali

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

#include "dali/operators/reader/coco_reader_op.h"


namespace dali {
DALI_REGISTER_OPERATOR(COCOReader, COCOReader, CPU);
DALI_SCHEMA(COCOReader)
  .NumInput(0)
  .NumOutput(3)
  .DocStr(R"code(Reads data from a COCO dataset that is composed of a directory with
images and annotation files. For each image with m bboxes, the bboxes are returned as  ``(m,4)``
Tensor (``m * [x, y, w, h]`` or ``m * [left, top, right, bottom]``) and labels as ``(m,1)``
Tensor (``m * category_id``).)code")
  .AddOptionalArg(
    "meta_files_path",
    "Path to the directory with meta files that contain preprocessed COCO annotations.",
    std::string())
  .AddOptionalArg("annotations_file",
      R"code(List of paths to the JSON annotations files.)code",
      std::string())
  .AddOptionalArg("shuffle_after_epoch",
      R"code(If set to True, the reader shuffles the entire  dataset after each epoch.)code",
      false)
  .AddArg("file_root",
      R"code(Path to a directory that contains the data files.)code",
      DALI_STRING)
  .AddOptionalArg("ltrb",
      R"code(If set to True, bboxes are returned as [left, top, right, bottom].

If set to False, the bboxes are returned as else [x, y, width, height].)code",
      false)
  .AddOptionalArg("masks",
      R"code(If set to True, segmentation masks are read and returned as polygons,
which is a list of points (2 floats).

Each mask can be one or more polygons, and for a given sample, the polygons are represented by the
following tensors:

- ``masks_meta`` -> list of tuples (mask_idx, start_idx, end_idx)
- ``masks_coords``-> list of (x,y) coordinates

One mask can have one or more ``masks_meta`` values that have the same ``mask_idx.``
This means that the mask for that given index consists of several polygons.
``start_idx`` indicates the index of the first coordinates in ``masks_coords``.
Currently objects with ``iscrowd=1`` annotations are skipped because RLE masks are not suitable
for instance segmentation.)code",
      false)
  .AddOptionalArg("skip_empty",
      R"code(If true, reader will skip samples with no object instances in them)code",
      false)
  .AddOptionalArg("size_threshold",
      R"code(If the width or the height of a bounding box that represents an instance of an object
is lower than this value, the object will be skipped during reading.\n
The value is represented as an absolute value.)code",
      0.1f,
      false)
  .AddOptionalArg("ratio",
      R"code(If set to True, the returned bbox coordinates are relative to the image size.)code",
      false)
  .AddOptionalArg("file_list",
      R"code(Path to the file that contains a list of whitespace separated ``file id`` pairs.

To traverse the file_root directory and obtain files and labels, leave empty
this value empty.)code",
      std::string())
  .AddOptionalArg("save_img_ids",
      R"code(If set to True, the image IDs are also returned.)code",
      false)
  .AddOptionalArg("dump_meta_files",
      R"code(If set to True, the operator dumps the meta files in the folder that
is provided with ``dump_meta_files_path``.)code",
      false)
  .AddOptionalArg(
    "dump_meta_files_path", R"code(Path to the directory in which to save the meta files that
contain the preprocessed COCO annotations.)code",
    std::string())
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("masks")) * 2
           + static_cast<int>(spec.GetArgument<bool>("save_img_ids"));
  })
  .AddParent("LoaderBase");


void COCOReader::ValidateOptions(const OpSpec &spec) {
  DALI_ENFORCE(
    spec.HasArgument("meta_files_path") || spec.HasArgument("annotations_file"),
    "`meta_files_path` or `annotations_file` must be provided");
  DALI_ENFORCE(
    !spec.HasArgument("file_list"),
    "Argument `file_list` is no longer supported for `COCOReader`."
    "The same functionality can be implemented with meta files option.");
  DALI_ENFORCE(!skip_cached_images_,
    "COCOReader doesn't support `skip_cached_images` option");

  if (spec.HasArgument("meta_files_path")) {
    DALI_ENFORCE(
      !spec.HasArgument("annotations_file"),
      "`meta_files_path` and `annotations_file` cannot be both provided.");
    DALI_ENFORCE(
      !spec.HasArgument("skip_empty"),
      "When reading data from meta files `skip_empty` option is not supported.");
    DALI_ENFORCE(
      !spec.HasArgument("ratio"),
      "When reading data from meta files `ratio` option is not supported.");
    DALI_ENFORCE(
      !spec.HasArgument("ltrb"),
      "When reading data from meta files `ltrb` option is not supported.");
    DALI_ENFORCE(
      !spec.HasArgument("size_threshold"),
      "When reading data from meta files `size_threshold` option is not supported.");
    DALI_ENFORCE(
      !spec.HasArgument("dump_meta_files"),
      "When reading data from meta files `dump_meta_files` option is not supported.");
    DALI_ENFORCE(
      !spec.HasArgument("dump_meta_files_path"),
      "When reading data from meta files `dump_meta_files_path` option is not supported.");
  }

  if (spec.HasArgument("dump_meta_files")) {
    DALI_ENFORCE(
      spec.HasArgument("dump_meta_files_path"),
      "When dumping meta files `dump_meta_files_path` must be provided.");
  }
}

}  // namespace dali

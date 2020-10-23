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
#include <set>

namespace dali {

DALI_SCHEMA(COCOReader)
  .NumInput(0)
  .NumOutput(3)
  .DocStr(R"code(Reads data from a COCO dataset that is composed of a directory with
images and annotation JSON files.

This readers produces the following outputs::

    images, bounding_boxes, labels, ((polygons, vertices) | (pixelwise_masks)), (image_ids)

**images**

Each sample contains image data with layout ``HWC`` (height, width, channels).

**bounding_boxes**

Each sample can have an arbitrary ``M`` number of bounding boxes, each described by 4 coordinates::

    [[x_0, y_0, w_0, h_0],
     [x_1, y_1, w_1, h_1]
     ...
     [x_M, y_M, w_M, h_M]]

or in ``[l, t, r, b]`` format if requested (see ``ltrb`` argument).

**labels**

Each bounding box is associated with an integer label representing a category identifier::
        
    [label_0, label_1, ..., label_M]

**polygons** and **vertices** (Optional, present if ``polygon_masks`` is set to True)
    
If ``polygon_masks`` is enabled, two extra outputs describing masks by a set of polygons.
    
Each mask contains an arbitrary number of polygons ``P``, each associated with a mask index in the range [0, M) and 
composed by a group of ``V`` vertices. The output ``polygons`` describes the polygons as follows::

    [[mask_idx_0, start_vertex_idx_0, end_vertex_idx_0],
     [mask_idx_1, start_vertex_idx_1, end_vertex_idx_1],
     ...
     [mask_idx_P, start_vertex_idx_P, end_vertex_idx_P]]

where ``mask_idx`` is the index of the mask the polygon, in the range ``[0, M)``, and ``start_vertex_idx`` and  ``end_verted_idx``
define the range of indices of vertices, as they appear in the output ``vertices``, belonging to this polygon.

Each sample in ``vertices`` contains a list of vertices that composed the different polygons in the sample, as 2D coordinates::

    [[x_0, y_0],
     [x_1, y_1],
     ...
     [x_V, y_V]]

**pixelwise_masks** (Optional, present if argument ``pixelwise_masks`` is set to True)

Contains image-like data, same shape and layout as ``images``, representing a pixelwise segmentation mask.

**image_ids** (Optional, present if argument ``image_ids`` is set to True)

One element per sample, representing an image identifier.
)code")
  .AddOptionalArg("preprocessed_annotations",
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

If set to False, the bboxes are returned as [x, y, width, height].)code",
      false)
  .DeprecateArgInFavorOf("masks", "polygon_masks")  // deprecated since 0.28dev
  .AddOptionalArg("polygon_masks",
      R"code(If set to True, segmentation mask polygons are read in the form of two outputs:
``polygons`` and ``vertices``. This argument is mutually exclusive with ``pixelwise_masks``.

.. warning::
    Currently objects with ``iscrowd=1`` annotations are skipped.)code",
      false)
  .AddOptionalArg("pixelwise_masks",
      R"code(If true, segmentation masks are read and returned as pixel-wise masks. This argument is
mutually exclusive with ``polygon_masks``.)code",
      false)
  .AddOptionalArg("skip_empty",
      R"code(If true, reader will skip samples with no object instances in them)code",
      false)
  .AddOptionalArg("size_threshold",
      R"code(If the width or the height, in number of pixels, of a bounding box that represents an 
instance of an object is lower than this value, the object will be ignored.)code",
      0.1f,
      false)
  .DeprecateArgInFavorOf("ratio", "relative_coordinates")  // deprecated since 0.28dev
  .AddOptionalArg("relative_coordinates",
      R"code(If set to True, the returned bbox and mask polygon coordinates are relative to the image dimensions.)code",
      false)
  .DeprecateArgInFavorOf("save_img_ids", "image_ids")  // deprecated since 0.28dev
  .AddOptionalArg("image_ids",
      R"code(If set to True, the image IDs will be produced in an extra output.)code",
      false)
  .DeprecateArgInFavorOf("dump_meta_files", "save_preprocessed_annotations")  // deprecated since 0.28dev
  .AddOptionalArg("save_preprocessed_annotations",
      R"code(If set to True, the operator saves a set of files containing binary representations of the
preprocessed COCO annotations.)code",
      false)
  .DeprecateArgInFavorOf("dump_meta_files_path", "save_preprocessed_annotations_dir")  // deprecated since 0.28dev
  .AddOptionalArg("dump_meta_files_path",
      R"code(Path to the directory in which to save the preprocessed COCO annotations files.)code",
    std::string())
  .AdditionalOutputsFn([](const OpSpec& spec) {
    return static_cast<int>(spec.GetArgument<bool>("masks")) * 2
           + static_cast<int>(spec.GetArgument<bool>("pixelwise_masks"))
           + static_cast<int>(spec.GetArgument<bool>("save_img_ids"));
  })
  .AddParent("LoaderBase");

DALI_REGISTER_OPERATOR(COCOReader, COCOReader, CPU);

void COCOReader::ValidateOptions(const OpSpec &spec) {
  DALI_ENFORCE(
    spec.HasArgument("meta_files_path") || spec.HasArgument("annotations_file"),
    "`meta_files_path` or `annotations_file` must be provided");
  DALI_ENFORCE(!skip_cached_images_,
    "COCOReader doesn't support `skip_cached_images` option");

  if (spec.HasArgument("meta_files_path")) {
    for (const char* arg_name : {"annotations_file", "skip_empty", "ratio", "ltrb", 
                                 "size_threshold", "dump_meta_files", "dump_meta_files_path"}) {
      if (spec.HasArgument(arg_name))
        DALI_FAIL(make_string("When reading data from meta files, \"", arg_name, "\" is not supported."));
    }
  }

  if (output_polygon_masks_ && output_pixelwise_masks_) {
    DALI_FAIL("``pixelwise_masks`` and ``polygon_masks`` are mutually exclusive");
  }

  if (spec.HasArgument("dump_meta_files") != spec.HasArgument("dump_meta_files_path")) {
    DALI_FAIL("``dump_meta_files`` and ``dump_meta_files_path`` should be provided together"); 
  }
}

void COCOReader::PixelwiseMasks(int image_id, int* mask) {
  const auto &meta = masks_meta_[image_id];
  const auto &coords = mask_coords_[image_id];
  int h = heights_[image_id];
  int w = widths_[image_id];
  const int *labels_in = labels_.data() + offsets_[image_id];
  int labels_size = counts_[image_id];
  std::set<int> labels(labels_in, labels_in + labels_size);
  if (!labels.size()) {
    return;
  }

  // Create a run-length encoding for each polygon, indexed by label :
  std::map<int, std::vector<RLE> > frPoly;
  std::vector<double> in;
  for (uint polygon_idx = 0; polygon_idx < meta.size() / 3; polygon_idx++) {
    int mask_idx = meta[3 * polygon_idx];
    int start_idx = meta[3 * polygon_idx + 1];
    int end_idx = meta[3 * polygon_idx + 2];
    int label = *(labels_in + mask_idx);
    // Convert polygon to encoded mask
    in.resize(end_idx - start_idx);
    for (int i = 0; i < end_idx - start_idx; i++)
      in[i] = static_cast<double>(coords[start_idx + i]);
    RLE M;
    rleInit(&M, 0, 0, 0, 0);
    rleFrPoly(&M, in.data(), (end_idx - start_idx) / 2, h, w);
    frPoly[label].push_back(M);
  }

  // Reserve run-length encodings by labels
  RLE* R;
  rlesInit(&R, *labels.rbegin() + 1);

  // Create a run-length encoding for each compressed string representation
  for (uint ann_id = 0 ; ann_id < masks_rles_idx_[image_id].size(); ann_id++) {
    auto mask_idx = masks_rles_idx_[image_id][ann_id];
    const auto &str = masks_rles_[image_id][ann_id];
    int label = *(labels_in + mask_idx);
    rleFrString(&R[label], const_cast<char*>(str.c_str()), h, w);
  }

  // Merge each label (from multi-polygons annotations)
  uint lab_cnt = 0;
  for (const auto &rles : frPoly)
    rleMerge(rles.second.data(), &R[rles.first], rles.second.size(), 0);

  // Merge all the labels into a pair of vectors :
  // [2,2,2],[A,B,C] for [A,A,B,B,C,C]
  struct Encoding {
    uint m;
    std::unique_ptr<uint[]> cnts;
    std::unique_ptr<int[]> vals;
  };
  Encoding A;
  A.cnts = std::make_unique<uint[]>(h * w + 1);  // upper-bound
  A.vals = std::make_unique<int[]>(h * w + 1);

  // first copy the content of the first label to the output
  bool v = false;
  A.m = R[*labels.begin()].m;
  for (siz a = 0; a < R[*labels.begin()].m; a++) {
    A.cnts[a] = R[*labels.begin()].cnts[a];
    A.vals[a] = v ? *labels.begin() : 0;
    v = !v;
  }

  // then merge the other labels
  std::unique_ptr<uint[]> cnts = std::make_unique<uint[]>(h * w + 1);
  std::unique_ptr<int[]> vals = std::make_unique<int[]>(h * w + 1);
  for (auto label = ++labels.begin(); label != labels.end(); label++) {
    RLE B = R[*label];
    if (B.cnts == 0)
      continue;

    uint cnt_a = A.cnts[0];
    uint cnt_b = B.cnts[0];
    int next_val_a = A.vals[0];
    int val_a = next_val_a;
    int val_b = *label;
    bool next_vb = false;
    bool vb = next_vb;
    uint nb_seq_a, nb_seq_b;
    nb_seq_a = nb_seq_b = 1;
    int m = 0;

    int cnt_tot = 1;  // check if we advanced at all
    while (cnt_tot > 0) {
      uint c = std::min(cnt_a, cnt_b);
      cnt_tot = 0;
      // advance A
      cnt_a -= c;
      if (!cnt_a && nb_seq_a < A.m) {
        cnt_a = A.cnts[nb_seq_a];  // next sequence for A
        next_val_a = A.vals[nb_seq_a];
        nb_seq_a++;
      }
      cnt_tot += cnt_a;
      // advance B
      cnt_b -= c;
      if (!cnt_b && nb_seq_b < B.m) {
        cnt_b = B.cnts[nb_seq_b++];  // next sequence for B
        next_vb = !next_vb;
      }
      cnt_tot += cnt_b;

      if (val_a && vb)  // there's already a class at this pixel
                        // in this case, the last annotation wins (it's undefined by the spec)
        vals[m] = (!cnt_a) ? val_a : val_b;
      else if (val_a)
        vals[m] = val_a;
      else if (vb)
        vals[m] = val_b;
      else
        vals[m] = 0;
      cnts[m] = c;
      m++;

      // since we switched sequence for A or B, apply the new value from now on
      val_a = next_val_a;
      vb = next_vb;

      if (cnt_a == 0) break;
    }
    // copy back the buffers to the destination encoding
    A.m = m;
    for (int i = 0; i < m; i++) A.cnts[i] = cnts[i];
    for (int i = 0; i < m; i++) A.vals[i] = vals[i];
  }

  // Decode final pixelwise masks encoded via RLE
  memset(mask, 0, h * w * sizeof(int));
  int x = 0, y = 0;
  for (uint i = 0; i < A.m; i++)
    for (uint j = 0; j < A.cnts[i]; j++) {
      mask[x + y * w] = A.vals[i];
      if (++y >= h) {
        y = 0;
        x++;
      }
    }

  // Destroy RLEs
  rlesFree(&R, *labels.rbegin() + 1);
  for (auto rles : frPoly)
    for (auto rle : rles.second)
      rleFree(&rle);
}

}  // namespace dali

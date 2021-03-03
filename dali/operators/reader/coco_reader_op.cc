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

extern "C" {
#include "third_party/cocoapi/common/maskApi.h"
}

namespace dali {

namespace {

int COCOReaderOutputFn(const OpSpec &spec) {
  return OutPolygonMasksEnabled(spec) * 2 + OutPixelwiseMasksEnabled(spec) +
         OutImageIdsEnabled(spec);
}

}  // namespace

DALI_REGISTER_OPERATOR(readers__COCO, COCOReader, CPU);

DALI_SCHEMA(readers__COCO)
  .NumInput(0)
  .NumOutput(3)
  .DocStr(R"code(Reads data from a COCO dataset that is composed of a directory with
images and annotation JSON files.

This readers produces the following outputs::

    images, bounding_boxes, labels, ((polygons, vertices) | (pixelwise_masks)), (image_ids)

* **images**
  Each sample contains image data with layout ``HWC`` (height, width, channels).
* **bounding_boxes**
  Each sample can have an arbitrary ``M`` number of bounding boxes, each described by 4 coordinates::

    [[x_0, y_0, w_0, h_0],
     [x_1, y_1, w_1, h_1]
     ...
     [x_M, y_M, w_M, h_M]]

  or in ``[l, t, r, b]`` format if requested (see ``ltrb`` argument).
* **labels**
  Each bounding box is associated with an integer label representing a category identifier::

    [label_0, label_1, ..., label_M]

* **polygons** and **vertices** (Optional, present if ``polygon_masks`` is set to True)
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

* **pixelwise_masks** (Optional, present if argument ``pixelwise_masks`` is set to True)
  Contains image-like data, same shape and layout as ``images``, representing a pixelwise segmentation mask.
* **image_ids** (Optional, present if argument ``image_ids`` is set to True)
  One element per sample, representing an image identifier.)code")
  .AddOptionalArg("preprocessed_annotations",
    "Path to the directory with meta files that contain preprocessed COCO annotations.",
    std::string())
  .DeprecateArgInFavorOf("meta_files_path", "preprocessed_annotations")  // deprecated since 0.28dev
  .AddOptionalArg("annotations_file",
      R"code(List of paths to the JSON annotations files.)code",
      std::string())
  .AddOptionalArg("shuffle_after_epoch",
      R"code(If set to True, the reader shuffles the entire  dataset after each epoch.)code",
      false)
  .AddOptionalArg<string>("file_root",
      R"code(Path to a directory that contains the data files.

If a file list is not provided, this argument is required.)code",
      nullptr)
  .AddOptionalArg("ltrb",
      R"code(If set to True, bboxes are returned as [left, top, right, bottom].

If set to False, the bboxes are returned as [x, y, width, height].)code",
      false)
  .AddOptionalArg("polygon_masks",
      R"code(If set to True, segmentation mask polygons are read in the form of two outputs:
``polygons`` and ``vertices``. This argument is mutually exclusive with ``pixelwise_masks``.

.. warning::
    Currently objects with ``iscrowd=1`` annotations are skipped.)code",
      false)
  .AddOptionalArg("masks", R"code(Enable polygon masks.)code", false)
  .DeprecateArg("masks", false,
R"code(Use ``polygon_masks`` instead. Note that the polygon format has changed ``mask_id, start_coord, end_coord`` to ``mask_id, start_vertex, end_vertex`` where
start_coord and end_coord are total number of coordinates, effectly ``start_coord = 2 * start_vertex`` and ``end_coord = 2 * end_vertex``.
Example: A polygon with vertices ``[[x0, y0], [x1, y1], [x2, y2]]`` would be represented as ``[mask_id, 0, 6]`` when using the deprecated
argument ``masks``, but ``[mask_id, 0, 3]`` when using the new argument ``polygon_masks``.)code")  // deprecated since 0.28dev
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
  .AddOptionalArg("ratio",
      R"code(If set to True, the returned bbox and mask polygon coordinates are relative to the image dimensions.)code",
      false)
  .AddOptionalArg("image_ids",
      R"code(If set to True, the image IDs will be produced in an extra output.)code",
      false)
  .AddOptionalArg<vector<string>>("images", R"code(A list of image paths.

If provided, it specifies the images that will be read.
The images will be read in the same order as they appear in the list, and in case of
duplicates, multiple copies of the relevant samples will be produced.

If left unspecified or set to None, all images listed in the annotation file are read exactly once,
ordered by their image id.

The paths to be kept should match exactly those in the annotations file.

Note: This argument is mutually exclusive with ``preprocessed_annotations``.)code", nullptr)
  .DeprecateArgInFavorOf("save_img_ids", "image_ids")  // deprecated since 0.28dev
  .AddOptionalArg("save_preprocessed_annotations",
      R"code(If set to True, the operator saves a set of files containing binary representations of the
preprocessed COCO annotations.)code",
      false)
  .DeprecateArgInFavorOf("dump_meta_files",
                         "save_preprocessed_annotations")  // deprecated since 0.28dev
  .AddOptionalArg("save_preprocessed_annotations_dir",
      R"code(Path to the directory in which to save the preprocessed COCO annotations files.)code",
    std::string())
  .DeprecateArgInFavorOf("dump_meta_files_path",
                         "save_preprocessed_annotations_dir")  // deprecated since 0.28dev
  .AdditionalOutputsFn(COCOReaderOutputFn)
  .AddParent("LoaderBase");


// Deprecated alias
DALI_REGISTER_OPERATOR(COCOReader, COCOReader, CPU);

DALI_SCHEMA(COCOReader)
    .NumInput(0)
    .NumOutput(3)
    .DocStr("Legacy alias for :meth:`readers.coco`.")
    .AdditionalOutputsFn(COCOReaderOutputFn)
    .AddParent("readers__COCO")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "readers__COCO",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;

COCOReader::COCOReader(const OpSpec& spec): DataReader<CPUBackend, ImageLabelWrapper>(spec) {
  DALI_ENFORCE(!skip_cached_images_, "COCOReader doesn't support `skip_cached_images` option");
  output_polygon_masks_ = OutPolygonMasksEnabled(spec);
  legacy_polygon_format_ = spec.HasArgument("masks") && spec.GetArgument<bool>("masks");
  output_pixelwise_masks_ = OutPixelwiseMasksEnabled(spec);
  output_image_ids_ = OutImageIdsEnabled(spec);
  loader_ = InitLoader<CocoLoader>(spec);

  if (legacy_polygon_format_) {
    DALI_WARN("Warning: Using legacy format for polygons. "
              "Please use ``polygon_masks`` instead of ``masks`` argument.");
  }
}

void COCOReader::RunImpl(SampleWorkspace &ws) {
  const ImageLabelWrapper& image_label = GetSample(ws.data_idx());

  Index image_size = image_label.image.size();
  auto &image_output = ws.Output<CPUBackend>(0);
  int image_idx = image_label.label;

  image_output.Resize({image_size});
  image_output.SetSourceInfo(image_label.image.GetSourceInfo());
  std::memcpy(image_output.mutable_data<uint8_t>(), image_label.image.raw_data(), image_size);

  auto &loader_impl = LoaderImpl();
  auto bboxes = loader_impl.bboxes(image_idx);
  auto &boxes_output = ws.Output<CPUBackend>(1);
  boxes_output.Resize({bboxes.size(), 4});
  std::memcpy(boxes_output.mutable_data<float>(), bboxes.data(),
              bboxes.size() * sizeof(vec<4>));

  auto labels = loader_impl.labels(image_idx);
  auto &labels_output = ws.Output<CPUBackend>(2);
  labels_output.Resize({labels.size()});  // 0.28dev: changed shape from {N, 1} to {N}
  std::memcpy(labels_output.mutable_data<int>(), labels.data(),
              labels.size() * sizeof(int));

  int curr_out_idx = 3;
  if (output_polygon_masks_) {
    auto &polygons_output = ws.Output<CPUBackend>(curr_out_idx++);
    auto polygons = loader_impl.polygons(image_idx);
    polygons_output.Resize({polygons.size(), 3});
    std::memcpy(polygons_output.mutable_data<int>(),
                polygons.data(), polygons.size() * sizeof(ivec3));
    if (legacy_polygon_format_) {  // TODO(janton): remove this once we remove ``masks`` arg
      auto *poly_data = polygons_output.mutable_data<int>();
      for (int64_t i = 0; i < polygons.size(); i++) {
        poly_data[i * 3 + 1] *= 2;
        poly_data[i * 3 + 2] *= 2;
      }
    }
    auto &vertices_output = ws.Output<CPUBackend>(curr_out_idx++);
    auto vertices = loader_impl.vertices(image_idx);
    vertices_output.Resize({vertices.size(), 2});
    std::memcpy(vertices_output.mutable_data<float>(),
                vertices.data(), vertices.size() * sizeof(vec2));
  }

  if (output_pixelwise_masks_) {
    auto &masks_output = ws.Output<CPUBackend>(curr_out_idx++);
    auto masks_info = loader_impl.pixelwise_masks_info(image_idx);
    masks_output.Resize(masks_info.shape);
    masks_output.SetLayout("HWC");
    PixelwiseMasks(image_idx, masks_output.mutable_data<int>());
  }

  if (output_image_ids_) {
    auto &id_output = ws.Output<CPUBackend>(curr_out_idx++);
    id_output.Resize({1});
    *(id_output.mutable_data<int>()) = loader_impl.image_id(image_idx);
  }
}

void COCOReader::PixelwiseMasks(int image_idx, int* mask) {
  auto &loader_impl = LoaderImpl();
  auto pol = loader_impl.polygons(image_idx);
  auto ver = loader_impl.vertices(image_idx);
  auto masks_info = loader_impl.pixelwise_masks_info(image_idx);
  int h = masks_info.shape[0];
  int w = masks_info.shape[1];
  auto bboxes = loader_impl.bboxes(image_idx);
  auto labels_span = loader_impl.labels(image_idx);
  std::set<int> labels(labels_span.data(),
                       labels_span.data() + labels_span.size());
  if (!labels.size()) {
    return;
  }

  // Create a run-length encoding for each polygon, indexed by label :
  std::map<int, std::vector<RLE> > frPoly;
  std::vector<double> in;
  for (uint polygon_idx = 0; polygon_idx < pol.size(); polygon_idx++) {
    auto &polygon = pol[polygon_idx];
    int mask_idx = polygon[0];
    int start_idx = polygon[1];
    int end_idx = polygon[2];
    assert(mask_idx < labels_span.size());
    int label = labels_span[mask_idx];
    // Convert polygon to encoded mask
    int nver = end_idx - start_idx;
    auto pol_ver = span<const vec2>{ver.data() + start_idx, nver};
    in.resize(pol_ver.size() * 2);
    for (int i = 0, k = 0; i < pol_ver.size(); i++) {
      in[k++] = static_cast<double>(pol_ver[i].x);
      in[k++] = static_cast<double>(pol_ver[i].y);
    }
    RLE M;
    rleInit(&M, 0, 0, 0, 0);
    rleFrPoly(&M, in.data(), pol_ver.size(), h, w);
    frPoly[label].push_back(M);
  }

  // Reserve run-length encodings by labels
  RLE* R;
  rlesInit(&R, *labels.rbegin() + 1);

  // Mask was originally described in RLE format
  for (uint ann_id = 0 ; ann_id < masks_info.mask_indices.size(); ann_id++) {
    const auto &rle = masks_info.rles[ann_id];
    auto mask_idx = masks_info.mask_indices[ann_id];
    int label = labels_span[mask_idx];
    rleInit(&R[label], (*rle)->h, (*rle)->w, (*rle)->m, (*rle)->cnts);
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

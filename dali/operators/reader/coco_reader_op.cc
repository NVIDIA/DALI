// Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <set>
#include <stdexcept>
#include <variant>

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

    images, bounding_boxes, labels, ((polygons, vertices) | (pixelwise_masks)),
    (image_ids)

* **images**
  Each sample contains image data with layout ``HWC`` (height, width, channels).
* **bounding_boxes**
  Each sample can have an arbitrary ``M`` number of bounding boxes, each described by 4 coordinates::

    [[x_0, y_0, w_0, h_0],
     [x_1, y_1, w_1, h_1]
     ...
     [x_M, y_M, w_M, h_M]]

  or in ``[l, t, r, b]`` format if requested (see `ltrb` argument).
* **labels**
  Each bounding box is associated with an integer label representing a category identifier::

    [label_0, label_1, ..., label_M]

* **polygons** and **vertices** (Optional, present if `polygon_masks` is set to True)
  If `polygon_masks` is enabled, two extra outputs describing masks by a set of polygons.
  Each mask contains an arbitrary number of polygons ``P``, each associated with a mask index in the range [0, M) and
  composed by a group of ``V`` vertices. The output ``polygons`` describes the polygons as follows::

    [[mask_idx_0, start_vertex_idx_0, end_vertex_idx_0],
     [mask_idx_1, start_vertex_idx_1, end_vertex_idx_1],
     ...
     [mask_idx_P, start_vertex_idx_P, end_vertex_idx_P]]

  where ``mask_idx`` is the index of the mask the polygon, in the range ``[0, M)``, and ``start_vertex_idx`` and  ``end_vertex_idx``
  define the range of indices of vertices, as they appear in the output ``vertices``, belonging to this polygon.
  Each sample in ``vertices`` contains a list of vertices that composed the different polygons in the sample, as 2D coordinates::

    [[x_0, y_0],
     [x_1, y_1],
     ...
     [x_V, y_V]]

* **pixelwise_masks** (Optional, present if argument `pixelwise_masks` is set to True)
  Contains image-like data, same shape and layout as `images`, representing a pixelwise segmentation mask.
* **image_ids** (Optional, present if argument `image_ids` is set to True)
  One element per sample, representing an image identifier.)code")
  .AddOptionalArg("preprocessed_annotations",
    "Path to the directory with meta files that contain preprocessed COCO annotations.",
    std::string())
  .DeprecateArgInFavorOf("meta_files_path", "preprocessed_annotations", "0.28")
  .AddOptionalArg("annotations_file",
      R"code(List of paths to the JSON annotations files.)code",
      std::string())
  .AddOptionalArg("shuffle_after_epoch",
      R"code(If set to True, the reader shuffles the entire dataset after each epoch.

`stick_to_shard` and `random_shuffle` cannot be used when this argument is set to True.

See the Sharding and Shuffling documentation for details.)code",
      false)
  .AddOptionalArg<int64_t>("shuffle_after_epoch_seed",
      R"code(Random seed for the dataset shuffling performed after each epoch.

If not provided, a fixed default seed is used, which results in the same shuffling
pattern across different training runs. Providing a custom seed allows for different
shuffle patterns across training runs, which may be desirable for better statistical
properties.

.. note::
    When using multiple DALI pipelines (e.g., for multi-GPU training), all pipeline
    instances should use the same `shuffle_after_epoch_seed` to ensure a consistent
    global shuffle across all shards.

.. note::
    This argument has no effect unless `shuffle_after_epoch` is set to ``True``.)code",
      nullptr, false)
  .AddOptionalArg<string>("file_root",
      R"code(Path to a directory that contains the data files.

If a file list is not provided, this argument is required.)code",
      nullptr)
  .AddOptionalArg("ltrb",
      R"code(If set to True, bboxes are returned as [left, top, right, bottom].

If set to False, the bboxes are returned as [x, y, width, height].)code",
      false)
  .AddOptionalArg("include_iscrowd",
      R"code(If set to True annotations marked as ``iscrowd=1`` are included as well.)code",
      true)
  .AddOptionalArg("polygon_masks",
      R"code(If set to True, segmentation mask polygons are read in the form of two outputs:
``polygons`` and ``vertices``. This argument is mutually exclusive with `pixelwise_masks`.)code",
      false)
  .AddOptionalArg("masks", R"code(Enable polygon masks.)code", false)
  .DeprecateArg("masks", "0.28", false,
R"code(Use `polygon_masks` instead. Note that the polygon format has changed ``mask_id, start_coord, end_coord`` to ``mask_id, start_vertex, end_vertex`` where
start_coord and end_coord are total number of coordinates, effectly ``start_coord = 2 * start_vertex`` and ``end_coord = 2 * end_vertex``.
Example: A polygon with vertices ``[[x0, y0], [x1, y1], [x2, y2]]`` would be represented as ``[mask_id, 0, 6]`` when using the deprecated
argument ``masks``, but ``[mask_id, 0, 3]`` when using the new argument `polygon_masks`.)code")
  .AddOptionalArg("pixelwise_masks",
      R"code(If true, segmentation masks are read and returned as pixel-wise masks. This argument is
mutually exclusive with `polygon_masks`.)code",
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
  .AddOptionalArg("avoid_class_remapping",
      R"code(If set to True, lasses ID values are returned directly as they are defined in the manifest file.

Otherwise, classes' ID values are mapped to consecutive values in range 1-number of classes
disregarding exact values from the manifest (0 is reserved for a special background class.)code",
      false)
  .AddOptionalArg<vector<string>>("images", R"code(A list of image paths.

If provided, it specifies the images that will be read.
The images will be read in the same order as they appear in the list, and in case of
duplicates, multiple copies of the relevant samples will be produced.

If left unspecified or set to None, all images listed in the annotation file are read exactly once,
ordered by their image id.

The paths to be kept should match exactly those in the annotations file.

Note: This argument is mutually exclusive with `preprocessed_annotations`.)code", nullptr)
  .DeprecateArgInFavorOf("save_img_ids", "image_ids", "0.28")
  .AddOptionalArg("save_preprocessed_annotations",
      R"code(If set to True, the operator saves a set of files containing binary representations of the
preprocessed COCO annotations.)code",
      false)
  .DeprecateArgInFavorOf("dump_meta_files",
                         "save_preprocessed_annotations", "0.28")
  .AddOptionalArg("save_preprocessed_annotations_dir",
      R"code(Path to the directory in which to save the preprocessed COCO annotations files.)code",
    std::string())
  .DeprecateArgInFavorOf("dump_meta_files_path",
                         "save_preprocessed_annotations_dir", "0.28")
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
    .MakeDocHidden()
    .Deprecate(
        "1.0",
        "readers__COCO",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");

COCOReader::COCOReader(const OpSpec& spec)
    : DataReader<CPUBackend, ImageLabelWrapper, ImageLabelWrapper, true>(spec) {
  DALI_ENFORCE(!skip_cached_images_, "COCOReader doesn't support `skip_cached_images` option");
  output_polygon_masks_ = OutPolygonMasksEnabled(spec);
  legacy_polygon_format_ = spec.HasArgument("masks") && spec.GetArgument<bool>("masks");
  output_pixelwise_masks_ = OutPixelwiseMasksEnabled(spec);
  output_image_ids_ = OutImageIdsEnabled(spec);
  loader_ = InitLoader<CocoLoader>(spec);
  this->SetInitialSnapshot();

  if (legacy_polygon_format_) {
    DALI_WARN("Warning: Using legacy format for polygons. "
              "Please use ``polygon_masks`` instead of ``masks`` argument.");
  }
}

void COCOReader::RunImpl(SampleWorkspace &ws) {
  const ImageLabelWrapper& image_label = GetSample(ws.data_idx());

  Index image_size = image_label.image.size();
  auto &image_output = ws.Output<CPUBackend>(0);
  int image_idx = std::get<int32_t>(image_label.label);

  image_output.Resize({image_size}, DALI_UINT8);
  image_output.SetSourceInfo(image_label.image.GetSourceInfo());
  std::memcpy(image_output.mutable_data<uint8_t>(), image_label.image.raw_data(), image_size);

  auto &loader_impl = LoaderImpl();
  auto bboxes = loader_impl.bboxes(image_idx);
  auto &boxes_output = ws.Output<CPUBackend>(1);
  boxes_output.Resize({bboxes.size(), 4}, DALI_FLOAT);
  std::memcpy(boxes_output.mutable_data<float>(), bboxes.data(),
              bboxes.size() * sizeof(vec<4>));

  auto labels = loader_impl.labels(image_idx);
  auto &labels_output = ws.Output<CPUBackend>(2);
  labels_output.Resize({labels.size()}, DALI_INT32);  // 0.28dev: changed shape from {N, 1} to {N}
  std::memcpy(labels_output.mutable_data<int>(), labels.data(),
              labels.size() * sizeof(int));

  int curr_out_idx = 3;
  if (output_polygon_masks_) {
    auto &polygons_output = ws.Output<CPUBackend>(curr_out_idx++);
    auto polygons = loader_impl.polygons(image_idx);
    polygons_output.Resize({polygons.size(), 3}, DALI_INT32);
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
    vertices_output.Resize({vertices.size(), 2}, DALI_FLOAT);
    std::memcpy(vertices_output.mutable_data<float>(),
                vertices.data(), vertices.size() * sizeof(vec2));
  }

  if (output_pixelwise_masks_) {
    auto &masks_output = ws.Output<CPUBackend>(curr_out_idx++);
    auto masks_info = loader_impl.pixelwise_masks_info(image_idx);
    masks_output.Resize(masks_info.shape, DALI_INT32);
    masks_output.SetLayout("HWC");
    PixelwiseMasks(image_idx, masks_output.mutable_data<int>(),
                   image_label.image.GetSourceInfo());
  }

  if (output_image_ids_) {
    auto &id_output = ws.Output<CPUBackend>(curr_out_idx++);
    id_output.Resize({1}, DALI_INT32);
    *(id_output.mutable_data<int>()) = loader_impl.image_id(image_idx);
  }
}

void COCOReader::PixelwiseMasks(int image_idx, int* mask, const std::string &source) {
  bool ratio = spec_.GetArgument<bool>("ratio");
  auto &loader_impl = LoaderImpl();
  auto pol = loader_impl.polygons(image_idx);
  auto ver = loader_impl.vertices(image_idx);
  auto masks_info = loader_impl.pixelwise_masks_info(image_idx);
  int h = masks_info.shape[0];
  int w = masks_info.shape[1];
  const int64_t num_pixels = static_cast<int64_t>(h) * w;
  // Validate annotation RLEs before copying them into image-sized buffers. This also
  // covers compressed counts and preprocessed annotations, which bypass JSON parsing.
  for (int64_t mask_idx = 0; mask_idx < masks_info.rles.size(); mask_idx++) {
    const auto rle = masks_info.rles[mask_idx]->get();
    if (rle.h != static_cast<siz>(h) || rle.w != static_cast<siz>(w)) {
      throw std::invalid_argument(make_string(
          "Invalid COCO RLE mask: dimensions must match the image; got ", rle.h, "x", rle.w,
          ", expected ", h, "x", w, ". Image: \"", source, "\", RLE mask index ", mask_idx, "."));
    }
    if (rle.m == 0 || rle.m > static_cast<siz>(num_pixels + 1)) {
      throw std::invalid_argument(make_string(
          "Invalid COCO RLE mask: too many or missing runs; got ", rle.m,
          ", expected 1 to ", num_pixels + 1,
          ". Image: \"", source, "\", RLE mask index ", mask_idx, "."));
    }
    int64_t covered = 0;
    for (siz i = 0; i < rle.m; i++) {
      covered += rle.cnts[i];
      // Counts are nonnegative. Stop on invalid coverage before further additions can overflow.
      if (covered > num_pixels)
        break;
    }
    if (covered != num_pixels) {
      throw std::invalid_argument(make_string(
          "Invalid COCO RLE mask: the sum of run lengths must equal the number of pixels. "
          "Image: \"", source, "\", RLE mask index ", mask_idx, "."));
    }
  }
  auto bboxes = loader_impl.bboxes(image_idx);
  auto labels_span = loader_impl.labels(image_idx);
  std::set<int> labels(labels_span.data(),
                       labels_span.data() + labels_span.size());
  memset(mask, 0, num_pixels * sizeof(int));
  if (!labels.size()) {
    return;
  }

  // Create a run-length encoding for each polygon, indexed by label :
  std::map<int, std::vector<RLEMask>> frPoly;
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
      in[k++] = static_cast<double>(ratio ? pol_ver[i].x * w : pol_ver[i].x);
      in[k++] = static_cast<double>(ratio ? pol_ver[i].y * h : pol_ver[i].y);
    }
    RLE raw_mask;
    rleInit(&raw_mask, 0, 0, 0, 0);
    rleFrPoly(&raw_mask, in.data(), pol_ver.size(), h, w);
    RLEMask polygon_mask(raw_mask);
    frPoly[label].push_back(std::move(polygon_mask));
  }

  // Reserve run-length encodings by labels
  std::vector<RLEMask> R(*labels.rbegin() + 1);

  // Mask was originally described in RLE format
  for (uint ann_id = 0 ; ann_id < masks_info.mask_indices.size(); ann_id++) {
    const auto &rle = masks_info.rles[ann_id];
    auto mask_idx = masks_info.mask_indices[ann_id];
    int label = labels_span[mask_idx];
    R[label] = RLEMask((*rle)->h, (*rle)->w, make_cspan((*rle)->cnts, (*rle)->m));
  }

  // Merge each label (from multi-polygons annotations)
  for (const auto &rles : frPoly) {
    // The C API requires contiguous RLE structs; these views borrow the owned counts.
    std::vector<RLE> polygon_rles;
    polygon_rles.reserve(rles.second.size());
    for (const auto &rle : rles.second)
      polygon_rles.push_back(rle.get());
    RLE merged_mask{};
    rleMerge(polygon_rles.data(), &merged_mask, polygon_rles.size(), 0);
    R[rles.first] = RLEMask(merged_mask);
  }

  // Paint the labels in ascending order, so that a higher label wins where masks overlap
  // (the overlap is undefined by the spec). Each RLE alternates background and foreground runs,
  // starting with background, and walks the pixels in column-major order.
  // Zero-length runs and labels without a mask need no special handling.
  for (int label : labels) {
    const RLE rle = R[label].get();
    if (!rle.cnts)
      continue;
    int64_t begin = 0;
    for (siz i = 0; i < rle.m && begin < num_pixels; i++) {
      int64_t end = std::min<int64_t>(begin + rle.cnts[i], num_pixels);
      if (i % 2 == 1) {
        int64_t x = begin / h, y = begin % h;
        for (int64_t p = begin; p < end; p++) {
          mask[x + y * w] = label;
          if (++y == h) {
            y = 0;
            x++;
          }
        }
      }
      begin = end;
    }
  }
}

}  // namespace dali

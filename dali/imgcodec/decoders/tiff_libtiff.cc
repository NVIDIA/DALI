// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <tiffio.h>
#include "dali/imgcodec/decoders/tiff_libtiff.h"
#include "dali/imgcodec/util/convert.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/common/utils.h"

#define LIBTIFF_CALL_SUCCESS 1
#define LIBTIFF_CALL(call)                                \
  do {                                                    \
    int retcode = (call);                                 \
    DALI_ENFORCE(LIBTIFF_CALL_SUCCESS == retcode,         \
      "libtiff call failed with code "                    \
      + std::to_string(retcode) + ": " #call);            \
  } while (0)

namespace dali {
namespace imgcodec {

namespace detail {

class DecoderHelper {
 public:
  explicit DecoderHelper(ImageSource *in) : stream_(in->Open()), in_(in) {}

  static tmsize_t read(thandle_t handle, void *buffer, tmsize_t n) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    return helper->stream_->Read(buffer, n);
  }

  static tmsize_t write(thandle_t, void *, tmsize_t) {
    // Not used for decoding.
    return 0;
  }

  static toff_t seek(thandle_t handle, toff_t offset, int whence) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    helper->stream_->SeekRead(offset, whence);
    return helper->stream_->TellRead();
  }

  static int map(thandle_t handle, void **base, toff_t *size) {
    // This function will be used by LibTIFF only if input is InputKind::HostMemory.
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    if (helper->in_->Kind() != InputKind::HostMemory)
      std::logic_error("DecoderHelper::map can only be used for InputKind::HostMemory");
    *base = const_cast<void*>(helper->in_->RawData());
    *size = helper->in_->Size();
    return 0;
  }

  static toff_t size(thandle_t handle) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    return helper->stream_->Size();
  }

  static int close(thandle_t handle) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    delete helper;
    return 0;
  }

 private:
  std::shared_ptr<InputStream> stream_;
  ImageSource *in_;
};

std::unique_ptr<TIFF, void (*)(TIFF *)> OpenTiff(ImageSource *in) {
  TIFF *tiffptr;

  if (in->Kind() == InputKind::Filename) {
    tiffptr = TIFFOpen(in->Filename(), "r");
  } else {
    TIFFMapFileProc mapproc;
    if (in->Kind() == InputKind::HostMemory)
      mapproc = &DecoderHelper::map;
    else
      mapproc = nullptr;

    tiffptr = TIFFClientOpen("", "r", reinterpret_cast<thandle_t>(new DecoderHelper(in)),
                             &DecoderHelper::read,
                             &DecoderHelper::write,
                             &DecoderHelper::seek,
                             &DecoderHelper::close,
                             &DecoderHelper::size,
                             mapproc,
                             /* unmap */ 0);
  }

  DALI_ENFORCE(tiffptr != nullptr, make_string("Unable to open TIFF image: ", in->SourceInfo()));
  return {tiffptr, &TIFFClose};
}

struct TiffInfo {
  uint32_t image_width, image_height;
  uint16_t channels;

  uint32_t rows_per_strip;
  uint16_t bit_depth;
  uint16_t orientation;
  uint16_t compression;

  bool is_tiled;
  bool is_palette;
};

TiffInfo GetTiffInfo(TIFF *tiffptr) {
  TiffInfo info = {};

  LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_IMAGEWIDTH, &info.image_width));
  LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_IMAGELENGTH, &info.image_height));
  LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_SAMPLESPERPIXEL, &info.channels));
  LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_BITSPERSAMPLE, &info.bit_depth));
  LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_ORIENTATION, &info.orientation));
  LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_COMPRESSION, &info.compression));
  LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_ROWSPERSTRIP, &info.rows_per_strip));

  info.is_tiled = TIFFIsTiled(tiffptr);

  uint16_t photometric_interpretation;
  if (TIFFGetField(tiffptr, TIFFTAG_PHOTOMETRIC, &photometric_interpretation) &&
      photometric_interpretation == PHOTOMETRIC_PALETTE) {
    info.is_palette = true;
  }

  return info;
}

}  // namespace detail

DecodeResult LibTiffDecoderInstance::Decode(SampleView<CPUBackend> out, ImageSource *in,
                                            DecodeParams opts, const ROI &requested_roi) {
  auto tiff = detail::OpenTiff(in);
  auto info = detail::GetTiffInfo(tiff.get());

  // TODO(skarpinski) Support other bit depths
  DALI_ENFORCE(info.bit_depth == 8, "Not implemented: Only bit depth = 8 is supported");

  // TODO(skarpinski) Support palette images
  DALI_ENFORCE(!info.is_palette, "Not implemented: Palette images are not yet supported");

  // TODO(skarpinski) Support tiled images
  DALI_ENFORCE(!info.is_tiled, "Not implemented: Tiled images are not yet supported");

  unsigned out_channels = NumberOfChannels(opts.format, info.channels);

  // TODO(skarpinski) support different types
  using InType = uint8_t;
  using OutType = uint8_t;

  auto row_nbytes = TIFFScanlineSize(tiff.get());
  std::unique_ptr<InType, void(*)(void*)> row_buf{
    static_cast<InType *>(_TIFFmalloc(row_nbytes)), _TIFFfree};
  DALI_ENFORCE(row_buf.get() != nullptr, "Could not allocate memory");
  InType * const row_in  = row_buf.get();
  OutType * const img_out = out.mutable_data<OutType>();

  ROI roi;
  if (!requested_roi.use_roi()) {
    roi.begin = {0, 0};
    roi.end = out.shape().first(2);
  } else {
    roi = requested_roi;
  }

  // Need to read sequentially since not all the images support random access
  // From: http://www.libtiff.org/man/TIFFReadScanline.3t.html
  // Compression algorithm does not support random access. Data was requested in a non-sequential
  // order from a file that uses a compression algorithm and that has RowsPerStrip greater than one.
  // That is, data in the image is stored in a compressed form, and with multiple rows packed into a
  // strip. In this case, the library does not support random access to the data. The data should
  // either be accessed sequentially, or the file should be converted so that each strip is made up
  // of one row of data.
  const bool allow_random_row_access = (info.compression == COMPRESSION_NONE
                                        || info.rows_per_strip == 1);

  // If random access is not allowed, need to read sequentially all previous rows
  if (!allow_random_row_access) {
    for (int64_t y = 0; y < roi.begin[0]; y++) {
      LIBTIFF_CALL(TIFFReadScanline(tiff.get(), row_in, y, 0));
    }
  }

  uint64_t out_row_stride = roi.shape()[1] * out_channels;

  DALIImageType in_format;
  if (info.channels == 1)
    in_format = DALI_GRAY;
  else if (opts.format != DALI_ANY_DATA && info.channels >= 3)
    in_format = DALI_RGB;
  else
    in_format = DALI_ANY_DATA;

  TensorShape<> in_line_shape = {roi.shape()[1], info.channels};
  TensorShape<> in_line_strides = kernels::GetStrides(in_line_shape);
  TensorShape<> out_line_shape = {roi.shape()[1], out_channels};
  TensorShape<> out_line_strides = kernels::GetStrides(out_line_shape);

  for (int64_t roi_y = 0; roi_y < roi.shape()[0]; roi_y++) {
    LIBTIFF_CALL(TIFFReadScanline(tiff.get(), row_in, roi.begin[0] + roi_y, 0));

    Convert(img_out + (roi_y * out_row_stride), out_line_strides.data(), 1, opts.format,
            row_in + roi.begin[1] * info.channels, in_line_strides.data(), 1, in_format,
            out_line_shape.data(), 2);  // ndim = 2 because we convert a single row
  }

  return {true, nullptr};
}


}  // namespace imgcodec
}  // namespace dali

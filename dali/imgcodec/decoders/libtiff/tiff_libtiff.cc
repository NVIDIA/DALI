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

#include "dali/imgcodec/decoders/libtiff/tiff_libtiff.h"
#include <tiffio.h>
#include "dali/imgcodec/util/convert.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/common/utils.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/dynamic_scratchpad.h"

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
      return -1;
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
  uint16_t fill_order;

  bool is_tiled;
  bool is_palette;

  uint32_t tile_width, tile_height;
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
  LIBTIFF_CALL(TIFFGetFieldDefaulted(tiffptr, TIFFTAG_FILLORDER, &info.fill_order));

  info.is_tiled = TIFFIsTiled(tiffptr);
  if (info.is_tiled) {
    LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_TILEWIDTH, &info.tile_width));
    LIBTIFF_CALL(TIFFGetField(tiffptr, TIFFTAG_TILELENGTH, &info.tile_height));
  } else {
    // We will be reading data line-by-line and pretend that lines are tiles
    info.tile_width = info.image_width;
    info.tile_height = 1;
  }

  uint16_t photometric_interpretation;
  if (TIFFGetField(tiffptr, TIFFTAG_PHOTOMETRIC, &photometric_interpretation) &&
      photometric_interpretation == PHOTOMETRIC_PALETTE) {
    info.is_palette = true;
  }

  return info;
}

template <int depth>
struct depth2type;

template <>
struct depth2type<8> {
  using type = uint8_t;
};

template <>
struct depth2type<16> {
  using type = uint16_t;
};
template <>
struct depth2type<32> {
  using type = uint32_t;
};


/**
 * @brief Unpacks packed bits.
 *
 * @tparam OutputType Required output type
 * @tparam normalize If true, values will be upscaled to OutputType's dynamic range
 * @param nbits Number of bits per value
 * @param out Output array
 * @param in Pointer to the bits to unpack
 * @param n Number of values to unpack
 */
template<typename OutputType, bool normalize = true>
void DLL_PUBLIC UnpackBits(size_t nbits, OutputType *out, const void *in, size_t n) {
  // We don't care about endianness here, because we read byte-by-byte and:
  // 1) "The library attempts to hide bit- and byte-ordering differences between the image and the
  //    native machine by converting data to the native machine order."
  //    http://www.libtiff.org/man/TIFFReadScanline.3t.html
  // 2) We only support FILL_ORDER=1 (i.e. big endian), which is TIFF's default and the only fill
  //    order required in Baseline TIFF readers.
  //    https://www.awaresystems.be/imaging/tiff/tifftags/fillorder.html

  size_t out_type_bits = 8 * sizeof(OutputType);
  if (out_type_bits < nbits)
    throw std::logic_error("Unpacking bits failed: OutputType too small");
  if (n == 0)
    return;

  auto in_ptr = static_cast<const uint8_t *>(in);
  uint8_t buffer = *(in_ptr++);
  constexpr size_t buffer_capacity = 8 * sizeof(buffer);
  size_t bits_in_buffer = buffer_capacity;

  for (size_t i = 0; i < n; i++) {
    OutputType result = 0;
    size_t bits_to_read = nbits;
    while (bits_to_read > 0) {
      if (bits_in_buffer > bits_to_read) {
        // If we have enough bits in the buffer, we store them and finish
        result <<= bits_to_read;
        result |= buffer >> (buffer_capacity - bits_to_read);
        bits_in_buffer -= bits_to_read;
        buffer <<= bits_to_read;
        bits_to_read = 0;
      } else {
        // If we don't have enough bits, we store what we have and refill the buffer
        result <<= bits_in_buffer;
        result |= buffer >> (buffer_capacity - bits_in_buffer);
        bits_to_read -= bits_in_buffer;
        buffer = *(in_ptr++);
        bits_in_buffer = buffer_capacity;
      }
    }
    if (normalize) {
      double coeff = static_cast<double>((1ull << out_type_bits) - 1) / ((1ull << nbits) - 1);
      result *= coeff;
    }
    out[i] = result;
  }
}

}  // namespace detail

DecodeResult LibTiffDecoderInstance::Decode(DecodeContext ctx,
                                            SampleView<CPUBackend> out, ImageSource *in,
                                            DecodeParams opts, const ROI &requested_roi) {
  auto tiff = detail::OpenTiff(in);
  auto info = detail::GetTiffInfo(tiff.get());

  if (info.bit_depth > 32)
    return {false, make_exception_ptr(std::logic_error(
                      make_string("Unsupported bit depth: ", info.bit_depth)))};

  // TODO(skarpinski) Support palette images
  if (info.is_palette)
    return {false, make_exception_ptr(std::logic_error("Palette images are not yet supported"))};

  if (info.is_tiled && info.tile_width % 16 != 0 && info.tile_height % 16 != 0) {
    // http://www.libtiff.org/libtiff.html
    // (...) tile width and length must each be a multiple of 16 pixels
    return {false, make_exception_ptr(std::logic_error(
                      make_string("TIFF tile dimensions must be a multiple of 16")))};
  }

  if (info.is_tiled && (info.bit_depth != 8 && info.bit_depth != 16 && info.bit_depth != 32)) {
    return {false, make_exception_ptr(std::logic_error(
                      make_string("Unsupported bit depth in tiled TIFF: ", info.bit_depth)))};
  }

  // Other fill orders are rare and discouraged by TIFF specification, but can happen
  if (info.fill_order != FILLORDER_MSB2LSB)
    return {false, make_exception_ptr(std::logic_error("Only FILL_ORDER=1 is supported"))};

  unsigned out_channels = NumberOfChannels(opts.format, info.channels);

  size_t buf_nbytes;
  if (!info.is_tiled) {
    buf_nbytes = TIFFScanlineSize(tiff.get());
  } else {
    buf_nbytes = TIFFTileSize(tiff.get());
  }

  std::unique_ptr<void, void(*)(void*)> buf{_TIFFmalloc(buf_nbytes), _TIFFfree};
  DALI_ENFORCE(buf.get() != nullptr, "Could not allocate memory");

  ROI roi;
  if (!requested_roi.use_roi()) {
    roi.begin = {0, 0};
    roi.end = out.shape().first(2);
  } else {
    roi = requested_roi;
  }

  if (!info.is_tiled) {
    // Need to read sequentially since not all the images support random access
    // From: http://www.libtiff.org/man/TIFFReadScanline.3t.html
    // Compression algorithm does not support random access. Data was requested in a non-sequential
    // order from a file that uses a compression algorithm and that has RowsPerStrip greater than
    // one. That is, data in the image is stored in a compressed form, and with multiple rows packed
    // into a strip. In this case, the library does not support random access to the data. The data
    // should either be accessed sequentially, or the file should be converted so that each strip is
    // made up of one row of data.
    const bool allow_random_row_access = (info.compression == COMPRESSION_NONE
                                          || info.rows_per_strip == 1);

    // If random access is not allowed, need to read sequentially all previous rows
    if (!allow_random_row_access) {
      for (int64_t y = 0; y < roi.begin[0]; y++) {
        LIBTIFF_CALL(TIFFReadScanline(tiff.get(), buf.get(), y, 0));
      }
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

  TensorShape<> img_shape = {roi.shape()[0], roi.shape()[1], out_channels};
  TensorShape<> img_strides = kernels::GetStrides(img_shape);
  TensorShape<> tile_shape = {info.tile_height, info.tile_width, info.channels};
  TensorShape<> tile_strides = kernels::GetStrides(tile_shape);

  // We choose smallest possible type
  size_t in_type_bits = 0;
  if (info.bit_depth <= 8) in_type_bits = 8;
  else if (info.bit_depth <= 16) in_type_bits = 16;
  else if (info.bit_depth <= 32) in_type_bits = 32;
  else
    assert(false);

  TYPE_SWITCH(out.type(), type2id, OutType, (IMGCODEC_TYPES), (
    VALUE_SWITCH(in_type_bits, InTypeBits, (8, 16, 32), (
      using InputType = detail::depth2type<InTypeBits>::type;

      kernels::DynamicScratchpad scratchpad;
      InputType *in;
      if (info.bit_depth == InTypeBits) {
        in = static_cast<InputType *>(buf.get());
      } else {
        in = static_cast<InputType*>(scratchpad.Alloc(mm::memory_kind_id::host,
                  volume(tile_shape) * sizeof(InputType), sizeof(InputType)));
      }

      OutType *const img_out = out.mutable_data<OutType>();

      int64_t first_tile_y = roi.begin[0] - roi.begin[0] % info.tile_height;
      int64_t first_tile_x = roi.begin[1] - roi.begin[1] % info.tile_width;

      for (int64_t tile_y = first_tile_y; tile_y < roi.end[0]; tile_y += info.tile_height) {
        for (int64_t tile_x = first_tile_x; tile_x < roi.end[1]; tile_x += info.tile_width) {
          TensorShape<> tile_begin = {tile_y, tile_x};
          TensorShape<> tile_end = {tile_y + info.tile_height, tile_x + info.tile_width};

          tile_begin[0] = std::max(tile_begin[0], roi.begin[0]);
          tile_begin[1] = std::max(tile_begin[1], roi.begin[1]);
          tile_end[0] = std::min(tile_end[0], roi.end[0]);
          tile_end[1] = std::min(tile_end[1], roi.end[1]);

          TensorShape<> out_shape = {tile_end[0] - tile_begin[0], tile_end[1] - tile_begin[1],
                                     out_channels};

          if (info.is_tiled) {
            DALI_ENFORCE(TIFFReadTile(tiff.get(), buf.get(), tile_x, tile_y, 0, 0) > 0,
                         "TIFFReadTile failed");
          } else {
            LIBTIFF_CALL(TIFFReadScanline(tiff.get(), buf.get(), tile_y, 0));
          }

          if (info.bit_depth != InTypeBits) {
            detail::UnpackBits(info.bit_depth, in, buf.get(), volume(tile_shape));
          }

          OutType *out_ptr = img_out + (tile_begin[0] - roi.begin[0]) * img_strides[0]
                                     + (tile_begin[1] - roi.begin[1]) * img_strides[1];
          const InputType *in_ptr = in + (tile_begin[0] - tile_y) * tile_strides[0]
                                       + (tile_begin[1] - tile_x) * tile_strides[1];

          Convert(out_ptr, img_strides.data(), 2, opts.format,
                  in_ptr, tile_strides.data(), 2, in_format,
                  out_shape.data(), 3);
        }
      }
    ), DALI_FAIL(make_string("Unsupported bit depth: ", info.bit_depth)););  // NOLINT
  ), DALI_FAIL(make_string("Unsupported output type: ", out.type())));  // NOLINT

  return {true, nullptr};
}

}  // namespace imgcodec
}  // namespace dali

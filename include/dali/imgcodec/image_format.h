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

#ifndef DALI_IMGCODEC_IMAGE_FORMAT_H_
#define DALI_IMGCODEC_IMAGE_FORMAT_H_

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "dali/core/span.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/stream.h"
#include "dali/imgcodec/image_source.h"
#include "dali/imgcodec/image_orientation.h"

namespace dali {
namespace imgcodec {

struct ImageInfo {
  TensorShape<> shape;
  Orientation orientation = {};
};

class DLL_PUBLIC ImageParser {
 public:
  virtual ~ImageParser() = default;

  /**
   * @brief   Parses the encoded image to get image information (shape, ...)
   * @remarks ImageInfo will be valid only if `CanParse` returned true
   */
  virtual ImageInfo GetInfo(ImageSource *encoded) const = 0;

  /**
   * @brief Verifies whether the parser can understand an encoded image,
   *        that is, if it's in the format that this parser handles.
   */
  virtual bool CanParse(ImageSource *encoded) const = 0;

 protected:
  /**
   * @brief Reads first n bytes from the beginning of the encoded image
   */
  size_t ReadHeader(uint8_t *buffer, ImageSource *encoded, size_t n) const;
};

class ImageDecoderFactory;

class DLL_PUBLIC ImageFormat {
 public:
  ImageFormat(const char *name, shared_ptr<ImageParser> parser);

  /**
   * @brief Checks whether an encoded image matches this format
   */
  bool Matches(ImageSource *encoded) const;

  /**
   * @brief Gets a pointer to the image parser
   */
  const ImageParser *Parser() const;

  /**
   * @brief Returns a string representing the name of the format
   */
  const std::string& Name() const;

  /**
   * @brief Returns a set of decoder factories associated with this format
   */
  span<ImageDecoderFactory *const> Decoders() const;

  /**
   * @brief Registers a new decoder associated with this format, with a set priority
   *
   * A decoder is registered via a factory.
   *
   * @param factory   A decoder factory
   * @param priority  A float representing the priority of this codec.
   *                  The lower the number, the higher priority the codec has.
   */
  void RegisterDecoder(std::shared_ptr<ImageDecoderFactory> factory, float priority);

 private:
  std::string name_;
  std::shared_ptr<ImageParser> parser_;
  std::multimap<float, std::shared_ptr<ImageDecoderFactory>> decoders_;
  std::vector<ImageDecoderFactory*> decoder_ptrs_;
};

class DLL_PUBLIC ImageFormatRegistry {
 public:
  /**
   * @brief Registers a new image format
   *
   * @param format The image format to register
   */
  void RegisterFormat(std::shared_ptr<ImageFormat> format);
  /**
   * @brief Get the format of the image encoded in `image`
   *
   * This function tries to parse the image with parsers from all
   * registered formats and returns the first format that succeeded.
   *
   * @param image
   * @return const ImageFormat*
   */
  const ImageFormat *GetImageFormat(ImageSource *image) const;

  /**
   * @brief Gets a format by name by which it was registered.
   */
  ImageFormat *GetImageFormat(const char *name);

  /**
   * @brief Returns all registered image formats
   */
  span<ImageFormat *const> Formats() const;

  static ImageFormatRegistry &instance();

 private:
  std::vector<std::shared_ptr<ImageFormat>> formats_;
  std::vector<ImageFormat*> format_ptrs_;
  std::map<std::string, std::shared_ptr<ImageFormat>> by_name_;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_FORMAT_H_

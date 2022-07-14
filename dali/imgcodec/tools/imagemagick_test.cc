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

#include <getopt.h>
#include <iostream>
#include <regex>
#include <thread>
#include <filesystem>
#include <optional>

#include "dali/test/dali_test.h"
#include "dali/imgcodec/image_source.h"
#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/image_decoder.h"

#include "dali/imgcodec/parsers/bmp.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/imgcodec/parsers/jpeg2000.h"
#include "dali/imgcodec/parsers/png.h"
#include "dali/imgcodec/parsers/pnm.h"
#include "dali/imgcodec/parsers/tiff.h"
#include "dali/imgcodec/parsers/webp.h"

namespace dali {
namespace imgcodec {
namespace test {

void fail(const std::string &filename, const std::string &message) {
  std::cerr << "FAIL" << "\t" << filename << ": " << message << std::endl;
}

void ok(const std::string &filename) {
  std::cerr << "OK" << "\t" << filename << std::endl;
}

class ImgcodecTester {
 public:
  ImgcodecTester() {
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("jpeg", std::make_shared<JpegParser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("png", std::make_shared<PngParser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("bmp", std::make_shared<BmpParser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("tiff", std::make_shared<TiffParser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("pnm", std::make_shared<PnmParser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("jpeg2000", std::make_shared<Jpeg2000Parser>()));
    format_registry_.RegisterFormat(
        std::make_shared<ImageFormat>("webp", std::make_shared<WebpParser>()));
  }

  std::optional<TensorShape<>> shape_of(const std::string &filename) const {
    auto img = ImageSource::FromFilename(filename);
    auto fmt = this->format_registry_.GetImageFormat(&img);
    if (fmt == nullptr) {
      fail(filename, "Format not recognized by imgcodec");
      return {};
    }
    auto image_info = fmt->Parser()->GetInfo(&img);
    return image_info.shape;
  }

  ImageFormatRegistry format_registry_;
};

struct Options {
  std::string identify_path;
  std::string directory;
  std::regex filter;
  bool print;
  unsigned batch;
  ImgcodecTester imgcodec_tester;
};

Options default_options() {
  Options options = {
    .identify_path = "/usr/bin/identify",
    .directory = "",
    .filter = std::regex(".*"),
    .print = false,
    .batch = 1024,
  };
  return options;
}

std::vector<std::string> get_batch(const Options &options,
                                   std::filesystem::recursive_directory_iterator &it) {
  std::vector<std::string> filenames;
  filenames.reserve(options.batch);
  while (it != std::filesystem::end(it) && filenames.size() < options.batch) {
    const auto& entry = *(it++);
    if (entry.is_regular_file()) {
      const auto path = entry.path().string();
      if (std::regex_match(path, options.filter))
        filenames.push_back(path);
    }
  }
  return filenames;
}

void process(const Options &options, std::vector<std::string> filenames) {
  std::string cmd = options.identify_path + " -format  \"%w %h %[channels]\\n\"";
  for (const std::string &f : filenames) {
    cmd += " ";
    cmd += f;
  }

  FILE* pipe = popen(cmd.c_str(), "r");
  for (const std::string &filename : filenames) {
    int w, h, c;
    char tmp[16];
    std::string colors;
    if (fscanf(pipe, "%d %d %16s", &w, &h, tmp) != 3) {
      fail(filename, "Unable to parse ImageMagick's output");
      continue;
    }
    colors = tmp;

    if (colors == "srgb" || colors == "rgb") {
      c = 3;
    } else if (colors == "srgba" || colors == "rgba" || colors == "cmyk") {
      c = 4;
    } else if (colors == "gray") {
      c = 1;
    } else {
      fail(filename, "Unable to parse ImageMagick's output");
      continue;
    }

    TensorShape<> imagemagick_shape = {h, w, c};

    if (options.print) {
      std::cout << filename << "\t" << imagemagick_shape << std::endl;
    } else {
      auto imgcodec_shape = options.imgcodec_tester.shape_of(filename);
      if (imgcodec_shape) {
        if (imagemagick_shape == *imgcodec_shape) {
          ok(filename);
        } else {
          std::ostringstream ss;
          ss << "Expected " << imagemagick_shape << " but got " << *imgcodec_shape;
          fail(filename, ss.str());
        }
      }
    }
  }
}

void run(const Options &options) {
  auto directory_it = std::filesystem::recursive_directory_iterator(options.directory);

  while (directory_it != std::filesystem::end(directory_it)) {
    auto batch = get_batch(options, directory_it);
    process(options, batch);
  }
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali


void show_usage() {
  std::cerr << "???\n";
  // TODO(skarpinski)
}


int main(int argc, char **argv) {
  dali::imgcodec::test::Options options = dali::imgcodec::test::default_options();

  struct option long_options[] = {
    {"print", no_argument, nullptr, 'p'},
    {"identify", required_argument, nullptr, 'i'},
    {"filter", required_argument, nullptr, 'r'},
    {"batch", required_argument, nullptr, 'b'},
    {"help", no_argument, nullptr, 'h'},
    {0, 0, 0, 0}
  };

  int option_index = 0;
  char c;
  while ((c = getopt_long(argc, argv, "hpi:r:b:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'h':
        show_usage();
        return 0;
      case 'p':
        options.print = true;
        break;
      case 'i':
        options.identify_path = optarg;
        break;
      case 'r':
        options.filter = std::regex(optarg);
        break;
      case 'b':
        options.batch = std::stoi(optarg);
        break;
      case '?':
        break;
      case 0:
        break;
      default:
        show_usage();
        return 1;
    }
  }
  if (optind != argc - 1) {
    show_usage();
    return -1;
  }
  options.directory = argv[optind];

  dali::imgcodec::test::run(options);
}


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
#include <experimental/filesystem>
#include <optional>
#include <mutex>

#include "dali/test/dali_test.h"
#include "dali/pipeline/util/thread_pool.h"

#include "dali/imgcodec/image_source.h"
#include "dali/imgcodec/image_format.h"
#include "dali/imgcodec/image_decoder_interfaces.h"

#include "dali/imgcodec/parsers/bmp.h"
#include "dali/imgcodec/parsers/jpeg.h"
#include "dali/imgcodec/parsers/jpeg2000.h"
#include "dali/imgcodec/parsers/png.h"
#include "dali/imgcodec/parsers/pnm.h"
#include "dali/imgcodec/parsers/tiff.h"
#include "dali/imgcodec/parsers/webp.h"

namespace fs = std::experimental::filesystem;

namespace dali {
namespace imgcodec {
namespace test {

const char *help = R"(Usage: imagemagick_test [OPTIONS] PATH

Compares image shape returned by ImageMagick's `identify` tool with the shape
returned by DALI's imgcodec. If PATH is a directory, it is searched recursively
for matching files (see --filter). If PATH is a regular file, only this single
file is processed and the filters do not apply.

The following OPTIONS are available:
--help   -h           Show this help message
--filter=... -r=...   Regex to filter the files with, for example '.*\\.jpeg'
--print  -p           Only print ImageMagick's output and don't compare it with
--batch=N  -b=N       Runs ImageMagick on batches of N images (default is 1024)
--identify=... -i=... Path of `identify` tool, default is /usr/bin/identify
--jobs=N  -j=N        Number of concurrent jobs to run, defaults to number of cores
)";

void fail(std::ostream &stream, const std::string &filename, const std::string &message) {
  stream << "FAIL" << "\t" << filename << ": " << message << std::endl;
}

void ok(std::ostream &stream, const std::string &filename) {
  stream << "OK" << "\t" << filename << std::endl;
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
    if (fmt == nullptr) return {};
    auto image_info = fmt->Parser()->GetInfo(&img);
    return image_info.shape;
  }

  ImageFormatRegistry format_registry_;
};

struct Env {
  std::string identify_path;
  std::string directory;
  std::regex filter;
  bool print;
  unsigned batch;
  unsigned jobs;
  ImgcodecTester imgcodec_tester;
  std::mutex mutex;  // Mutex for synchronizing the output
};

std::vector<std::string> get_batch(const Env &env, fs::recursive_directory_iterator &it) {
  std::vector<std::string> filenames;
  filenames.reserve(env.batch);
  while (it != fs::end(it) && filenames.size() < env.batch) {
    const auto& entry = *(it++);
    if (is_regular_file(entry.path())) {
      const auto path = entry.path().string();
      if (std::regex_match(path, env.filter))
        filenames.push_back(std::move(path));
    }
  }
  return filenames;
}

std::optional<TensorShape<>> scan_imagemagick_shape(FILE *pipe) {
  int w, h, c;
  char tmp[16];
  std::string colors;
  if (fscanf(pipe, "%d %d %15s", &w, &h, tmp) != 3) return {};
  colors = tmp;

  if (colors == "srgb" || colors == "rgb") {
    c = 3;
  } else if (colors == "srgba" || colors == "rgba" || colors == "cmyk") {
    c = 4;
  } else if (colors == "gray") {
    c = 1;
  } else {
    return {};
  }

  TensorShape<> imagemagick_shape = {h, w, c};
  return imagemagick_shape;
}

void process(Env &env, std::vector<std::string> filenames) {
  std::ostringstream log, out;

  std::ostringstream cmd;
  cmd << env.identify_path << " -quiet -format  \"%w %h %[channels]\\n\"";
  for (const std::string &f : filenames) {
    cmd << " " << f;
  }
  FILE* pipe = popen(cmd.str().c_str(), "r");

  for (const std::string &filename : filenames) {
    auto imagemagick_shape = scan_imagemagick_shape(pipe);
    if (!imagemagick_shape) {
      fail(log, filename, "Unable to parse ImageMagick's output");
      continue;
    }

    if (env.print) {
      out << filename << "\t" << *imagemagick_shape << std::endl;
      continue;
    }
    try {
      auto imgcodec_shape = env.imgcodec_tester.shape_of(filename);

      if (!imgcodec_shape) {
        fail(log, filename, "Imgcodec failed to parse");
        continue;
      }

      if (*imagemagick_shape == *imgcodec_shape) {
        ok(log, filename);
      } else {
        std::ostringstream ss;
        ss << "Expected " << *imagemagick_shape << " but got " << *imgcodec_shape;
        fail(log, filename, ss.str());
      }
    }
    catch (const std::exception &e) {
      std::ostringstream ss;
      ss << "Imgcodec raised an exception: " << e.what();
      fail(log, filename, ss.str());
    }
  }

  pclose(pipe);

  {
    const std::lock_guard lock(env.mutex);
    std::cerr << log.str();
    std::cout << out.str();
  }
}

void run(Env &env) {
  if (!fs::is_directory(env.directory)) {
    process(env, {env.directory});
    return;
  }

  auto directory_it = fs::recursive_directory_iterator(env.directory);
  ThreadPool pool(env.jobs, -1, false, "imagemagick");

  while (directory_it != fs::end(directory_it)) {
    auto batch = get_batch(env, directory_it);
    pool.AddWork([=, &env](int tid){
      process(env, batch);
    });
  }
  pool.RunAll();
  pool.WaitForWork();
}

}  // namespace test
}  // namespace imgcodec
}  // namespace dali


int main(int argc, char **argv) {
  dali::imgcodec::test::Env env = {
    .identify_path = "/usr/bin/identify",
    .directory = "",
    .filter = std::regex(".*"),
    .print = false,
    .batch = 1024,
    .jobs = std::thread::hardware_concurrency(),
  };

  struct option long_options[] = {
    {"print", no_argument, nullptr, 'p'},
    {"identify", required_argument, nullptr, 'i'},
    {"filter", required_argument, nullptr, 'r'},
    {"batch", required_argument, nullptr, 'b'},
    {"help", no_argument, nullptr, 'h'},
    {"jobs", required_argument, nullptr, 'j'},
    {0, 0, 0, 0}
  };

  int option_index = 0;
  char c;

  while ((c = getopt_long(argc, argv, "hpi:r:b:j:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'h':
        std::cerr << dali::imgcodec::test::help;
        return 0;
      case 'p':
        env.print = true;
        break;
      case 'i':
        env.identify_path = optarg;
        break;
      case 'r':
        env.filter = std::regex(optarg);
        break;
      case 'b':
        env.batch = std::stoi(optarg);
        break;
      case 'j':
        env.jobs = std::stoi(optarg);
        break;
      case '?':
        break;
      case 0:
        break;
      default:
        std::cerr << dali::imgcodec::test::help;
        return 1;
    }
  }

  if (optind != argc - 1) {
    std::cerr << dali::imgcodec::test::help;
    return 1;
  }
  env.directory = argv[optind];
  dali::imgcodec::test::run(env);
}

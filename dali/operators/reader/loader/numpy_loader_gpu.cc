// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <dirent.h>
#include <errno.h>
#include <memory>
#include <set>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/numpy_loader_gpu.h"

namespace dali {

std::unique_ptr<CUFileStream> NumpyLoaderGPU::ParseHeader(std::unique_ptr<CUFileStream> file,
                                                          NumpyParseTarget& target) {
  // check if the file is actually a numpy file
  std::vector<uint8_t> token(11);
  int64_t nread = file->ReadCPU(token.data(), 10);
  DALI_ENFORCE(nread == 10, "Can not read header.");
  token[nread] = '\0';

  // check if heqder is too short
  std::string header = std::string(reinterpret_cast<char*>(token.data()));
  DALI_ENFORCE(header.find_first_of("NUMPY") != std::string::npos,
               "File is not a numpy file.");

  // extract header length
  uint16_t header_len = 0;
  memcpy(&header_len, &token[8], 2);
  DALI_ENFORCE((header_len + 10) % 16 == 0,
               "Error extracting header length.");

  // read header: the offset is a magic number
  int64 offset = (6+1+1+2);
  // the header_len can be 4GiB according to the NPYv2 file format
  // specification: https://numpy.org/neps/nep-0001-npy-format.html
  // while this allocation could be sizable, it is performed on the host.
  token.resize(header_len+1);
  file->Seek(offset);
  nread = file->ReadCPU(token.data(), header_len);
  DALI_ENFORCE(nread == header_len, "Can not read header.");
  token[header_len] = '\0';
  header = std::string(reinterpret_cast<char*>(token.data()));
  DALI_ENFORCE(header.find_first_of("{") != std::string::npos, "Header is corrupted.");
  offset += header_len;

  // store the file offset in the parse target
  target.data_offset = offset;

  // prepare file for later reads
  file->Seek(offset);

  // extract dictionary info from header
  std::smatch header_match;
  DALI_ENFORCE(std::regex_search(header, header_match, header_regex_),
               "Can not parse header.");

  // now extract header information
  // type
  std::string typestring = header_match[1].str();

  // < means LE, | means N/A, = means native. In all those cases, we can read
  bool little_endian =
    (typestring[0] == '<' || typestring[0] == '|' || typestring[0] == '=');
  DALI_ENFORCE(little_endian,
    "Big Endian files are not supported.");

  std::string tid = typestring.substr(1);
  // get type in a safe way
  target.type_info = TypeFromNumpyStr(tid);

  // check for data order
  if (header_match[2].str() == "False")
    target.fortran_order = false;
  else
    target.fortran_order = true;

  // set sizes
  std::string shapestring = header_match[3].str();
  std::regex shape_regex{R"(,+)"};  // split on comma
  std::sregex_token_iterator it{shapestring.begin(), shapestring.end(), shape_regex, -1};
  std::vector<std::string> shapevec{it, {}};

  // if shapevec size is 1 and shapevec[0] is the empty string,
  // the array is actually a scalar/singleton (denoted as ())
  // and thus the size needs to be set to one:
  if ( (shapevec.size() == 1) && (shapevec[0] == "") ) shapevec[0] = "1";

  // determine shapes
  size_t shapesize = shapevec.size();
  target.shape.resize(shapesize);
  // cheapest thing to do is to define the tensor in an reversed way
  if (target.fortran_order) {
    for (size_t i = 0; i < shapesize; ++i)
      target.shape[i] = static_cast<int64_t>(stoi(shapevec[shapesize-i-1]));
  } else {
    for (size_t i = 0; i < shapesize; ++i)
      target.shape[i] = static_cast<int64_t>(stoi(shapevec[i]));
  }

  return file;
}

// register tensor
void NumpyLoaderGPU::RegisterTensor(void *buffer, size_t total_size) {
  if (register_buffers_) {
    // get raw pointer
    auto dptr = static_cast<uint8_t*>(buffer);
    std::unique_lock<std::mutex> reg_lock(reg_mutex_);
    auto iter = reg_buff_.find(dptr);
    if (iter != reg_buff_.end()) {
      reg_lock.unlock();
      if (iter->second == total_size) {
        // the right buffer is registered already, return
        return;
      } else {
        cuFileBufDeregister(iter->first);
      }
    } else {
      reg_lock.unlock();
    }
    // no buffer found or with different size so register again
    cuFileBufRegister(dptr, total_size, 0);
    reg_lock.lock();
    reg_buff_[dptr] = total_size;
  }
}

std::unique_ptr<CUFileStream> NumpyLoaderGPU::ReadSampleHelper(std::unique_ptr<CUFileStream> file,
                                                               ImageFileWrapperGPU& imfile,
                                                               void *buffer,
                                                               Index offset,
                                                               size_t total_size) {
  // register the buffer (if needed)
  RegisterTensor(buffer, total_size);

  Index image_bytes = volume(imfile.shape) * imfile.type_info.size();

  // copy the image
  file->Read(static_cast<uint8_t*>(buffer), image_bytes, offset);

  return file;
}

// we need to implement that but we should split parsing and reading in this case
void NumpyLoaderGPU::ReadSample(ImageFileWrapperGPU& imfile) {
  // set the device:
  DeviceGuard g(device_id_);

  // extract image file
  auto image_file = images_[current_index_++];

  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(image_file);
  meta.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(image_file)) {
    meta.SetSkipSample(true);
    imfile.image.SetMeta(meta);
    imfile.image.Reset();
    imfile.image.set_type(TypeInfo::Create<uint8_t>());
    imfile.image.Resize({0});
    imfile.filename = "";
    return;
  }

  // set metadata
  imfile.image.SetMeta(meta);

  imfile.read_meta_f = [this, image_file, &imfile] () {
    // open file
    imfile.file_stream = CUFileStream::Open(file_root_ + "/" + image_file, read_ahead_, false);

    // read the header from file or cache
    NumpyParseTarget target;
    std::unique_lock<std::mutex> cache_lock(cache_mutex_);
    auto it = header_cache_.find(image_file);
    if (!cache_headers_ || it == header_cache_.end()) {
      cache_lock.unlock();
      imfile.file_stream = ParseHeader(std::move(imfile.file_stream), target);
      if (cache_headers_) {
        cache_lock.lock();
        header_cache_[image_file] = target;
        cache_lock.unlock();
      }
    } else {
      target = it->second;
      imfile.file_stream->Seek(target.data_offset);
      cache_lock.unlock();
    }
    imfile.type_info = target.type_info;
    imfile.shape = target.shape;
    imfile.meta = (target.fortran_order ? "transpose:true" : "transpose:false");
  };

  imfile.read_sample_f = [this, image_file, &imfile] (void *buffer, Index offset,
                                                      size_t total_size) {
    // read sample
    imfile.file_stream = ReadSampleHelper(std::move(imfile.file_stream), imfile,
                                          buffer, offset, total_size);
    // close the file handle
    imfile.file_stream->Close();
  };

  // set file path
  imfile.filename = file_root_ + "/" + image_file;
}

}  // namespace dali

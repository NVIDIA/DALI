// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_WEBDATASET_TAR_UTILS_H_
#define DALI_OPERATORS_READER_LOADER_WEBDATASET_TAR_UTILS_H_

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/util/file.h"

namespace dali {
namespace detail {
/**
 * @brief Used to access .tar archives through the given FileStream
 */
class DLL_PUBLIC TarArchive {
 public:
  TarArchive() = default;
  explicit TarArchive(std::unique_ptr<FileStream> stream_);
  TarArchive(TarArchive&&);
  ~TarArchive();
  TarArchive& operator=(TarArchive&&);

  /**
   * @brief Advances the archive to look at the next file in the tarball.
   *
   * @returns Whether it has got any more files to read (or has reached the end of the archive).
   */
  bool NextFile();
  /**
   * @brief Returns whether it has reached the end of archive.
   */
  bool EndOfArchive() const;
  /**
   * @brief Sets the offset to which the stream pointer should go.
   * @remark The offset must point to a file header; other values will cause undefined behaviour.
   */
  void SeekArchive(int64_t offset);
  /**
   * @brief Tells the offset of the beginning of the header of the current entry
   */
  int64_t TellArchive() const;

  enum EntryType {
    ENTRY_NONE = 0,
    ENTRY_FILE,
    ENTRY_DIR,
    ENTRY_HARDLINK,
    ENTRY_SYMLINK,
    ENTRY_CHARDEV,
    ENTRY_BLOCKDEV,
    ENTRY_FIFO
  };

  /**
   * @brief Returns the name of the file in the archive that is currently being viewed.
   * @remark The returned reference is invalidated upon a move operation or moving to the next file.
   */
  const std::string& GetFileName() const;
  /**
   * @brief Returns the size (in bytes) of the file in the archive that is currently being viewed.
   * @remark Will be 0 for any entry type other than a file
   */
  size_t GetFileSize() const;
  /**
   * @brief Returns the type of the entry in the archive
   */
  EntryType GetFileType() const;

  /**
   * @brief Reads the contents of the file and returns them.
   * Reads the contents of the file and returns them. In the case of success places the read head
   * at the end of file. In the other case keeps the read head intact. Depends on the behaviour of
   * the Get function of the provided FileStream.
   * @return The pointer to the full contents of the file, or nullptr if it cannot perform that.
   */
  std::shared_ptr<void> ReadFile();
  /**
   * @brief
   * Reads the given number of bytes into a given buffer, assuming that said buffer
   * is big enough to perform this operation. Returns the number of bytes actually read.
   * @param buffer the buffer to read the data into
   * @param count the maximum number of bytes to read
   * @returns the number of bytes actually read to the buffer
   */
  size_t Read(uint8_t* buffer, size_t count);
  /**
   * @brief Returns whether the file cursor is at the end of file.
   */
  bool EndOfFile() const;

 private:
  std::unique_ptr<FileStream> stream_;
  int instance_handle_ = -1;
  void* handle_ = nullptr;  // handle to the TAR struct
  friend ssize_t LibtarReadTarArchive(int, void*, size_t);

  std::string filename_;
  size_t filesize_ = 0;
  EntryType filetype_ = ENTRY_NONE;

  size_t readoffset_ = 0;
  int64_t current_header_ = 0;

  bool eof_ = true;  // when this is true the value of readoffset_ and stream_ offset is undefined
  void SetEof();

  void ParseHeader();
  void Close();  // resets objects to default values
};

}  // namespace detail
}  // namespace dali
#endif  // DALI_OPERATORS_READER_LOADER_WEBDATASET_TAR_UTILS_H_

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include "dali/common.h"
#include "dali/pipeline/operators/reader/loader/file_loader.h"
#include "dali/util/file.h"

namespace dali {

void filesystem::assemble_file_list(const std::string& path, int label,
                        std::vector<std::pair<std::string, int>> *file_label_pairs) {
  DIR *dir = opendir(path.c_str());
  struct dirent *entry;

  const std::vector<std::string> valid_extensions({".jpg", ".jpeg", ".png", ".bmp"});

  while ((entry = readdir(dir))) {
    std::string full_path = path + "/" + std::string{entry->d_name};
    struct stat s;
    stat(full_path.c_str(), &s);
    if (S_ISREG(s.st_mode)) {
      std::string full_path_lowercase = full_path;
      std::transform(full_path_lowercase.begin(), full_path_lowercase.end(),
                     full_path_lowercase.begin(), ::tolower);
      for (const std::string& s : valid_extensions) {
        size_t pos = full_path_lowercase.rfind(s);
        if (pos != std::string::npos && pos + s.size() == full_path_lowercase.size()) {
          file_label_pairs->push_back(std::make_pair(full_path, label));
          break;
        }
      }
    }
  }
  closedir(dir);
}

vector<std::pair<string, int>> filesystem::traverse_directories(const std::string& path) {
  // open the root
  DIR *dir = opendir(path.c_str());

  DALI_ENFORCE(dir != nullptr,
      "Directory " + path + " could not be opened.");

  struct dirent *entry;
  int dir_count = 0;

  std::vector<std::pair<std::string, int>> file_label_pairs;

  while ((entry = readdir(dir))) {
    struct stat s;
    std::string full_path = path + "/" + std::string(entry->d_name);
    int ret = stat(full_path.c_str(), &s);
    DALI_ENFORCE(ret == 0,
        "Could not access " + full_path + " during directory traversal.");
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
    if (S_ISDIR(s.st_mode)) {
      assemble_file_list(full_path, dir_count, &file_label_pairs);
      dir_count++;
    }
  }
  printf("read %lu files from %d directories\n", file_label_pairs.size(), dir_count);

  closedir(dir);

  return file_label_pairs;
}
void FileLoader::ReadSample(Tensor<CPUBackend>* tensor) {
  auto image_pair = image_label_pairs_[current_index_++];

  // handle wrap-around
  if (current_index_ == Size()) {
    current_index_ = 0;
  }

  FileStream *current_image = FileStream::Open(image_pair.first);
  Index image_size = current_image->Size();

  // resize tensor to hold [image, label]
  tensor->Resize({image_size + static_cast<Index>(sizeof(int))});

  // copy the image
  current_image->Read(tensor->mutable_data<uint8_t>(), image_size);

  // close the file handle
  current_image->Close();

  // copy the label
  *(reinterpret_cast<int*>(&tensor->mutable_data<uint8_t>()[image_size])) = image_pair.second;
}

Index FileLoader::Size() {
  return static_cast<Index>(image_label_pairs_.size());
}
}  // namespace dali

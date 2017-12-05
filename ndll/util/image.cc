#include "ndll/util/image.h"

namespace ndll {

void LoadJPEGS(const string image_folder, vector<string> *jpeg_names,
    vector<uint8*> *jpegs, vector<int> *jpeg_sizes) {
  const string image_list = image_folder + "/image_list.txt";
  std::ifstream file(image_list);
  NDLL_ENFORCE(file.is_open());
    
  string img;
  while(file >> img) {
    NDLL_ENFORCE(img.size());
    jpeg_names->push_back(image_folder + "/" + img);
  }

  for (auto img_name : *jpeg_names) {
    std::ifstream img_file(img_name);
    NDLL_ENFORCE(img_file.is_open());

    img_file.seekg(0, std::ios::end);
    int img_size = (int)img_file.tellg();
    img_file.seekg(0, std::ios::beg);

    jpegs->push_back(new uint8[img_size]);
    jpeg_sizes->push_back(img_size);
    img_file.read(reinterpret_cast<char*>((*jpegs)[jpegs->size()-1]), img_size);
  }
}

void LoadJPEGS(const vector<string> &jpeg_names,
    vector<uint8*> *jpegs, vector<int> *jpeg_sizes) {
  for (auto img_name : jpeg_names) {
    std::ifstream img_file(img_name);
    NDLL_ENFORCE(img_file.is_open());

    img_file.seekg(0, std::ios::end);
    int img_size = (int)img_file.tellg();
    img_file.seekg(0, std::ios::beg);

    jpegs->push_back(new uint8[img_size]);
    jpeg_sizes->push_back(img_size);
    img_file.read(reinterpret_cast<char*>((*jpegs)[jpegs->size()-1]), img_size);
  }
}

void LoadFromFile(string file_name, uint8 **image, int *h, int *w, int *c) {
  std::ifstream file(file_name + ".txt");
  NDLL_ENFORCE(file.is_open());

  file >> *h;
  file >> *w;
  file >> *c;

  // lol at this multiplication
  int size = (*h)*(*w)*(*c);
  *image = new uint8[size];
  int tmp = 0;
  for (int i = 0; i < size; ++i) {
    file >> tmp;
    (*image)[i] = (uint8)tmp;
  }
}

void WriteHWCImage(const uint8 *img, int h, int w, int c, string file_name) {
  NDLL_ENFORCE(img != nullptr);
  NDLL_ENFORCE(h >= 0);
  NDLL_ENFORCE(w >= 0);
  NDLL_ENFORCE(c >= 0);
  CUDA_CALL(cudaDeviceSynchronize());
  vector<uint8> tmp(h*w*c, 0);
  MemCopy(tmp.data(), img, h*w*c);
  std::ofstream file(file_name + ".ppm");
  NDLL_ENFORCE(file.is_open());

  file << "P3" << endl;
  file << w << " " << h << endl;
  file << "255" << endl;

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      for (int k = 0; k < c; ++k) {
        file << int(tmp[i*w*c + j*c + k]) << " ";
      }
    }
    file << endl;
  }
}

} // namespace ndll

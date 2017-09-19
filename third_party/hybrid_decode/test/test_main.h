#ifndef TEST_MAIN_H_
#define TEST_MAIN_H_

#include "debug.h"

#include <gtest/gtest.h>
#include <npp.h>

#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;

// The location & number of images to load for testing
// static const string IMAGE_FOLDER = "./images";
static const string IMAGE_FOLDER = "images";
static const int NUM_IMAGE = 1000;

class TestMain : public ::testing::Test {
public:
    void SetUp() {
        std::srand(std::time(0));

        // Load the images for testing
        loadJpegs(&jpegData_, &jpegLengths_);
        // get_ground_truth();
    }

    void TearDown() {
        for (auto &p : jpegData_) delete[] p;
        for (auto &p : groundTruth_) delete[] p;
    }

    // Stores all of the JPEGs for testing
    vector<unsigned char*> jpegData_;
    vector<int> jpegLengths_;
    vector<string> imageFiles_;

    vector<Npp8u*> groundTruth_;
    
    void loadJpegs(vector<unsigned char*> *jpegData, vector<int> *jpegLengths) {
        // Make sure memory is clear
        assert(jpegData->size() == 0);
        assert(jpegLengths->size() == 0);

        // load the list of images to read in
        imageFiles_.resize(NUM_IMAGE);
        readImageFileNames(&imageFiles_, IMAGE_FOLDER, IMAGE_FOLDER + "/image_list.txt");

        jpegData->resize(NUM_IMAGE);
        jpegLengths->resize(NUM_IMAGE);
        
        // Read in all the jpegs
        for (int i = 0; i < NUM_IMAGE; ++i) {
            loadJpeg(imageFiles_[i].c_str(), (*jpegData)[i], (*jpegLengths)[i]);
            ASSERT_TRUE(((*jpegData)[i] != nullptr))
                << "Input File Error: " << i << " "
                << imageFiles_[i] << endl;
        }
    }
    
    void get_ground_truth() {
        string folder = "test/";
        ASSERT_TRUE(imageFiles_.size() == NUM_IMAGE);

        for (int i = 0; i < NUM_IMAGE; ++i) {
            groundTruth_.push_back(load_image(folder + imageFiles_[i] + ".txt"));
        }
        
    }
    
    bool is_data_equal(Npp8u *a, Npp8u *b, int num) {
        Npp8u *tmpA = new Npp8u[num];
        Npp8u *tmpB = new Npp8u[num];
        CHECK_CUDA(cudaMemcpy(tmpA, a, num, cudaMemcpyDefault));
        CHECK_CUDA(cudaMemcpy(tmpB, b, num, cudaMemcpyDefault));

        for (int i = 0; i < num; ++i) {
            if (tmpA[i] != tmpB[i]) {
                cout << int(tmpA[i]) << " v. " << int(tmpB[i]) << endl;
                return false;
            }
        }
        return true;
    }

    Npp8u* load_image(string s) {
        ifstream f(s);
        assert(f.is_open());
        
        int h, w, c;
        f >> h;
        f >> w;
        f >> c;

        Npp8u *data = new Npp8u[h*w*c];
        int tmp = 0;
        for (int i = 0; i < h*w*c; ++i) {
            f >> tmp;
            data[i] = (Npp8u)tmp;
        }
        
        return data;
    }
    
    void dump_image(Npp8u *image, NppiSize size, int components, size_t stride, string fname) {
        CHECK_CUDA(cudaDeviceSynchronize());

        Npp8u *tmp = new Npp8u[size.width*size.height*components];
    
        CHECK_CUDA(cudaMemcpy2D(tmp, size.width*components,
                        image, stride,
                        components*size.width, size.height,
                        cudaMemcpyDeviceToHost));

        
        ofstream file(fname + ".txt");

        file << size.height << " " << size.width << " " << components << endl;
        for (int h = 0; h < size.height; ++h) {
            for (int w = 0; w < size.width; ++w) {
                for (int c = 0; c < components; ++c) {
                    file << unsigned(tmp[h*size.width*components + w*components + c]) << " ";
                }
            }
            file << endl;
        }
        delete[] tmp;

    }
    
    void dump_rgb_image(Npp8u *image, NppiSize size, size_t stride) {
        CHECK_CUDA(cudaDeviceSynchronize());

        Npp8u *tmp = new Npp8u[size.width*size.height*3];
    
        CHECK_CUDA(cudaMemcpy2D(tmp, size.width*3,
                        image, stride,
                        3*size.width, size.height,
                        cudaMemcpyDeviceToHost));

        static int r = 0;
        ofstream file("rgb" + std::to_string(r) + ".txt");
        r++;

        for (int h = 0; h < size.height; ++h) {
            for (int w = 0; w < size.width; ++w) {
                for (int c = 0; c < 3; ++c) {
                    file << unsigned(tmp[h*size.width*3 + w*3 + c]) << "-";
                }
                file << " ";
            }
            file << endl;
        }
        delete[] tmp;
    }

    void dump_planar_image(Npp8u *image_planes[3], Npp32s image_plane_steps[3],
            size_t heights[3], size_t widths[3]) {
        CHECK_CUDA(cudaDeviceSynchronize());
  
        static int num = 0; 
        for (int i = 0; i < 3; ++i) {
            ofstream file("out" + std::to_string(num) + ".txt");
            num++;
    
            // cout << "Host mem size: " << heights[i] * widths[i] << endl;
            Npp8u *tmp = new Npp8u[heights[i] * widths[i]];

            // cout << "dst: " << (long long) tmp << endl;
            // cout << "dpitch: " << widths[i] * sizeof(Npp8u) << endl;
            // cout << "src: " << (long long) image_planes[i] << endl;
            // cout << "spitch: " << image_plane_steps[i] * sizeof(Npp8u) << endl;
            // cout << "width (bytes): " << widths[i] * sizeof(Npp8u) << endl;
            // cout << "height : " << heights[i] << endl;

            CHECK_CUDA(cudaMemcpy(tmp, image_planes[i], widths[i] * heights[i],
                            cudaMemcpyDeviceToHost));
            
            // CHECK_CUDA(cudaMemcpy2D(tmp, widths[i] * sizeof(Npp8u),
            //                 image_planes[i], image_plane_steps[i] * sizeof(Npp8u),
            //                 widths[i] * sizeof(Npp8u), heights[i],
            //                 cudaMemcpyDeviceToHost));
    
            for (int j = 0; j < heights[i]; ++j) {
                for (int k = 0; k < widths[i]; ++k) {
                    file << unsigned(tmp[j * widths[i] + k]) << " ";
                }
                file << endl;
            }
            delete[] tmp;
        }
    }
    
protected:
    // Helper to load images
    void readImageFileNames(vector<string> *imageFiles,
            const string img_folder, const string imageList) {
        ifstream file(imageList);
        for (int i = 0; i < NUM_IMAGE; ++i) {
            string tmp;
            file >> tmp;
            ASSERT_TRUE(tmp.size()) << "ERROR: found empty string in image list at "
                                    << i << endl;
            (*imageFiles)[i] = img_folder + "/" + tmp;
        }
    }

    
    // Helper to load images
    void loadJpeg(const char *input_file, unsigned char *&pJpegData, int &nInputLength)
        {
            // Load file into CPU memory
            ifstream stream(input_file, ifstream::binary);

            ASSERT_TRUE(stream.good()) << "stream no good: " << input_file << endl;

            stream.seekg(0, std::ios::end);
            nInputLength = (int)stream.tellg();
            stream.seekg(0, std::ios::beg);

            pJpegData = new unsigned char[nInputLength];
            stream.read(reinterpret_cast<char *>(pJpegData), nInputLength);
        }

};

#endif // TEST_MAIN_H_

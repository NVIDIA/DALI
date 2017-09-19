#ifndef DEVICE_BUFFER_H_
#define DEVICE_BUFFER_H_

#include "debug.h"

class DeviceBuffer {
public:
    DeviceBuffer() : size_(0), data_(nullptr) {}
    ~DeviceBuffer() { CHECK_CUDA(cudaFree(data_)); }
    
    void resize(size_t n) {
        // only allocate if we don't have enough mem
        if (n > size_) {
            // clean up & allocate new buffer
            CHECK_CUDA(cudaFree(data_));
            CHECK_CUDA(cudaMalloc((void**)&data_, n));
            
            // Update new size
            size_ = n;
        }
    }

    size_t size() const { return size_; }
    
    unsigned char* data() { return data_; }
    const unsigned char* data() const { return data_; }

    operator unsigned char*() { return data_; }
    operator const unsigned char*() const { return data_; }
    operator void*() { return data_; }
    operator const void*() const { return data_; }
    
private:
    size_t size_;
    unsigned char *data_;
};

#endif // DEVICE_BUFFER_H_

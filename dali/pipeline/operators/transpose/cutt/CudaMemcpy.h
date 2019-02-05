/******************************************************************************
MIT License

Copyright (c) 2016 Antti-Pekka Hynninen
Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/
#ifndef CUDAMEMCPY_H
#define CUDAMEMCPY_H

#include <cuda_runtime.h>

template <typename T> void scalarCopy(const int n, const T* data_in, T* data_out, cudaStream_t stream);
template <typename T> void vectorCopy(const int n, T* data_in, T* data_out, cudaStream_t stream);
void memcpyFloat(const int n, float* data_in, float* data_out, cudaStream_t stream);

#endif // CUDAMEMCPY_H

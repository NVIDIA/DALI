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
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_INT_VECTOR_H
#define DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_INT_VECTOR_H

// Intel: Minimum SSE2 required for vectorization.
// SSE can't be used because it does not support integer operations. SSE defaults to scalar

#if defined(__SSE2__)
// Intel x86
#include <x86intrin.h>

#if defined(__AVX__)
#define USE_AVX
const int INT_VECTOR_LEN = 8;

#if defined(__AVX2__)
// #include <avx2intrin.h>
const char INT_VECTOR_TYPE[] = "AVX2";
#else
const char INT_VECTOR_TYPE[] = "AVX";
#endif

#else
#define USE_SSE
const int INT_VECTOR_LEN = 4;
const char INT_VECTOR_TYPE[] = "SSE2";
#endif

#elif defined(__ALTIVEC__)  // #if defined(__SSE2__)
#define USE_ALTIVEC
// IBM altivec
#include <altivec.h>
#undef bool
const int INT_VECTOR_LEN = 4;
const char INT_VECTOR_TYPE[] = "ALTIVEC";

#else // #if defined(__SSE2__)
// Nothing
const int INT_VECTOR_LEN = 1;
const char INT_VECTOR_TYPE[] = "SCALAR";
#endif

//
// Integer vector class for Intel and IBM CPU platforms
//
class int_vector {
private:

#if defined(USE_AVX)
  __m256i x;
#elif defined(USE_SSE)
  __m128i x;
#elif defined(USE_ALTIVEC)
  vector signed int x;
#else
  int x;
#endif

public:

  inline int_vector() {
  }

  inline int_vector(const int a) {
#if defined(USE_AVX)
    x = _mm256_set1_epi32(a);
#elif defined(USE_SSE)
    x = _mm_set1_epi32(a);
#elif defined(USE_ALTIVEC)
    x = (vector signed int){a, a, a, a};
#else
    x = a;
#endif    
  }

  inline int_vector(const int a[]) {
#if defined(USE_AVX)
    x = _mm256_set_epi32(a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0]);
#elif defined(USE_SSE)
    x = _mm_set_epi32(a[3], a[2], a[1], a[0]);
#elif defined(USE_ALTIVEC)
    x = vec_ld(0, a);
#else
    x = a[0];
#endif    
  }

#if defined(USE_AVX)
  inline int_vector(const __m256i ax) {
    x = ax;
  }
#elif defined(USE_SSE)
  inline int_vector(const __m128i ax) {
    x = ax;
  }
#elif defined(USE_ALTIVEC)
  inline int_vector(const vector signed int ax) {
    x = ax;
  }
#endif

  // 
  // Member functions
  //

  inline int_vector operator+=(const int_vector a) {
#if defined(USE_AVX)
    x = _mm256_add_epi32(x, a.x);
#elif defined(USE_SSE)
    x = _mm_add_epi32(x, a.x);
#elif defined(USE_ALTIVEC)
    x += a.x;
#else
    x += a.x;
#endif
    return *this;
  }

  inline int_vector operator-=(const int_vector a) {
#if defined(USE_AVX)
    x = _mm256_sub_epi32(x, a.x);
#elif defined(USE_SSE)
    x = _mm_sub_epi32(x, a.x);
#elif defined(USE_ALTIVEC)
    x -= a.x;
#else
    x -= a.x;
#endif
    return *this;
  }

  inline int_vector operator&=(const int_vector a) {
#if defined(USE_AVX)
    x = _mm256_and_si256(x, a.x);
#elif defined(USE_SSE)
    x = _mm_and_si128(x, a.x);
#elif defined(USE_ALTIVEC)
    x &= a.x;
#else
    x &= a.x;
#endif
    return *this;
  }

  inline int_vector operator|=(const int_vector a) {
#if defined(USE_AVX)
    x = _mm256_or_si256(x, a.x);
#elif defined(USE_SSE)
    x = _mm_or_si128(x, a.x);
#elif defined(USE_ALTIVEC)
    x |= a.x;
#else
    x |= a.x;
#endif
    return *this;
  }

  inline int_vector operator~() {
#if defined(USE_AVX)
    int_vector fullmask = int_vector(-1);
    return int_vector( _mm256_andnot_si256(x, fullmask.x) );
#elif defined(USE_SSE)
    int_vector fullmask = int_vector(-1);
    return int_vector( _mm_andnot_si128(x, fullmask.x) );
#elif defined(USE_ALTIVEC)
    return int_vector( ~x );
#else
    return ~x;
#endif
  }

  // Sign extended shift by a constant.
  // Note: 0 <= n <= 31. Otherwise results are unpredictable
  inline int_vector operator>>=(const int n) {
#if defined(USE_AVX)
    x = _mm256_srai_epi32(x, n);
#elif defined(USE_SSE)
    x = _mm_srai_epi32(x, n);
#elif defined(USE_ALTIVEC)
    x >>= n;
#else
    x >>= n;
#endif
    return *this;
  }

  // Sign extended shift by a constant
  // Note: 0 <= n <= 31. Otherwise results are unpredictable
  inline int_vector operator<<=(const int n) {
#if defined(USE_AVX)
    x = _mm256_slli_epi32(x, n);
#elif defined(USE_SSE)
    x = _mm_slli_epi32(x, n);
#elif defined(USE_ALTIVEC)
    x <<= n;
#else
    x <<= n;
#endif
    return *this;
  }

  // Copy contest to int array
  void copy(int* a) const {
#if defined(USE_AVX)
    _mm256_storeu_si256((__m256i *)a, x);
#elif defined(USE_SSE)
    _mm_storeu_si128((__m128i *)a, x);
#elif defined(USE_ALTIVEC)
     // void vec_stl (vector signed int, int, int *);
    vec_stl(x, 0, a);
#else
    a[0] = x;
#endif
  }

  //
  // Non-member functions
  //

  inline friend int_vector operator+(int_vector a, const int_vector b) {
    a += b;
    return a;
  }

  inline friend int_vector operator-(int_vector a, const int_vector b) {
    a -= b;
    return a;
  }

  inline friend int_vector operator&(int_vector a, const int_vector b) {
    a &= b;
    return a;
  }

  inline friend int_vector operator|(int_vector a, const int_vector b) {
    a |= b;
    return a;
  }

  inline friend int_vector operator>>(int_vector a, const int n) {
    a >>= n;
    return a;
  }

  inline friend int_vector operator<<(int_vector a, const int n) {
    a <<= n;
    return a;
  }

  // Returns 0xffffffff = -1 on the vector elements that are equal
  inline friend int_vector eq_mask(const int_vector a, const int_vector b) {
#if defined(USE_AVX)
    return int_vector(_mm256_cmpeq_epi32(a.x, b.x));
#elif defined(USE_SSE)
    return int_vector(_mm_cmpeq_epi32(a.x, b.x));
#elif defined(USE_ALTIVEC)
    return int_vector(a.x == b.x);
#else
    return int_vector((a.x == b.x)*(-1));
#endif
  }

  inline friend int_vector neq_mask(const int_vector a, const int_vector b) {
    return ~eq_mask(a, b);
  }

  // 0xffffffff => 1
  inline friend int_vector mask_to_bool(const int_vector a) {
#if defined(USE_AVX)
    return int_vector(_mm256_srli_epi32(a.x, 31));
#elif defined(USE_SSE)
    return int_vector(_mm_srli_epi32(a.x, 31));
#elif defined(USE_ALTIVEC)
    return int_vector((vector signed int)((vector unsigned int)a.x >> 31));
#else
    return ((unsigned int)a.x >> 31);
#endif
  }

  inline friend int_vector operator==(const int_vector a, const int_vector b) {
    return mask_to_bool(eq_mask(a, b));
  }

  inline friend int_vector operator!=(const int_vector a, const int_vector b) {
    return mask_to_bool(neq_mask(a, b));
  }

  // 1 => 0xffffffff
  inline friend int_vector bool_to_mask(const int_vector a) {
#if defined(USE_AVX)
    return neq_mask(a, int_vector(0));
#elif defined(USE_SSE)
    return neq_mask(a, int_vector(0));
#elif defined(USE_ALTIVEC)
    return neq_mask(a, int_vector(0));
#else
    return (a ? -1 : 0);
#endif
  }

  // Implicit type conversion
  // Returns true if any of the elements are != 0
  operator bool() const {
#if defined(USE_AVX)
    int_vector a = neq_mask(*this, int_vector(0));
    return (_mm256_movemask_epi8(a.x) != 0);
#elif defined(USE_SSE)
    int_vector a = neq_mask(*this, int_vector(0));
    return (_mm_movemask_epi8(a.x) != 0);
#elif defined(USE_ALTIVEC)
    return vec_any_ne(x, ((const vector signed int){0, 0, 0, 0}));
#else
    return x;
#endif
  }

  //
  // Helper functions
  //
  void print() {
    int vec[INT_VECTOR_LEN];
    this->copy(vec);
    for (int i=0;i < INT_VECTOR_LEN;i++) {
      printf("%d ", vec[i]);
    }
  }

};


#if defined(USE_ALTIVEC)
#undef vector
#undef pixel
#endif

#endif // DALI_PIPELINE_OPERATORS_TRANSPOSE_CUTT_INT_VECTOR_H
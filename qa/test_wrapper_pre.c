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

/*
 * It works together with test_wrapper_post.c. It should obtain pointers to free and malloc
 * that are later intercepted by sanitizer, as well as not intercepted versions of these provided
 * by test_wrapper_post.c together with the function that informs if the intercepted memory
 * management calls should be bypassed and the direct ones used instead
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>

static void* (*asan_malloc)(size_t);
static void* (*direct_malloc)(size_t);
static void (*asan_free)(void *);
static void (*direct_free)(void *);
static int (*use_direct_malloc)();

static void *get_dlsym(const char *sym) {
  void *rv;
  char *error;

  dlerror();
  rv = dlsym(RTLD_NEXT, sym);
  error = dlerror();
  if (error) {
    fprintf(stderr, "failed to find symbol %s: %s\n", sym, error);
    abort();
  }
  return rv;
}

static void trace_init() {
  asan_malloc = get_dlsym("malloc");
  direct_malloc = get_dlsym("direct_malloc");
  asan_free = get_dlsym("free");
  direct_free = get_dlsym("direct_free");
  use_direct_malloc = get_dlsym("use_direct_malloc");
}

void *malloc(size_t size) {
  if (!asan_malloc) {
    trace_init();
  }

  if (use_direct_malloc()) {
    return direct_malloc(size);
  } else {
    return asan_malloc(size);
  }
}

void free(void *p) {
  if (!asan_free) {
    trace_init();
  }

  // assume that allocation and free are done at the same level of _Unwind_Backtrace nesting
  // so we don't end up allocating from the nested call and releasing from unnested
  if (use_direct_malloc()) {
    direct_free(p);
  } else {
    asan_free(p);
  }
}

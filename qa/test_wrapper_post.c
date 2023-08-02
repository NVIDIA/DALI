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
 * It works together with test_wrapper_pre.c. It should obtain pointers to free and malloc
 * that are not intercepted by sanitizer, as well as _Unwind_Backtrace.
 * It also provides use_direct_malloc that allows learning if the program is inside of
 * the _Unwind_Backtrace function and any interception by sanitizer of the memory
 * allocation should be avoided as it leads to another call to _Unwind_Backtrace which
 * is not reentrant and holds a mutex which leads to a deadlock
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <unwind.h>

static void *(*real_malloc)(size_t);
static void (*real_free)(void *);

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
    real_malloc = get_dlsym("malloc");
    real_free = get_dlsym("free");
}

void *direct_malloc(size_t size) {
    if (!real_malloc) {
        trace_init();
    }

    return real_malloc(size);
}


void direct_free(void *p) {
    if (!real_free) {
        trace_init();
    }

    real_free(p);
}

_Unwind_Reason_Code (*orig_Unwind_Backtrace)(_Unwind_Trace_Fn trace, void * trace_argument);
void * (*orig_deregister_frame_info_bases)(const void *begin);
static __thread int unsafeness_depth;

static void unwind_init() {
    orig_Unwind_Backtrace = get_dlsym("_Unwind_Backtrace");
    orig_deregister_frame_info_bases =  get_dlsym("__deregister_frame_info_bases");
}

_Unwind_Reason_Code _Unwind_Backtrace(_Unwind_Trace_Fn trace, void * trace_argument) {
    if (!orig_Unwind_Backtrace) {
        unwind_init();
    }
    _Unwind_Reason_Code ret;
    ++unsafeness_depth;
    ret = orig_Unwind_Backtrace(trace, trace_argument);
    --unsafeness_depth;
    return ret;
}

void * __deregister_frame_info_bases(const void *begin) {
    if (!orig_deregister_frame_info_bases) {
        unwind_init();
    }
    void * ret;
    ++unsafeness_depth;
    ret = orig_deregister_frame_info_bases(begin);
    --unsafeness_depth;
    return ret;
}

int use_direct_malloc() {
    // as this lib is passed to LD_PREOLOAD, __thread variable is not created as thread library
    // is not available. So deffer the check of unsafeness_depth until orig_Unwind_Backtrace is
    // accessed and every thing is up and running
    return orig_Unwind_Backtrace && orig_deregister_frame_info_bases && unsafeness_depth;
}

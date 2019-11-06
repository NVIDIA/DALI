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
// limitations under the License.v

#ifndef DALI_DYNLINK_SND_H
#define DALI_DYNLINK_SND_H

#include <sndfile.h>
#include <dlfcn.h>

namespace dali {
namespace snd {


SNDFILE *(*sf_open)(const char *, int, SF_INFO *);

sf_count_t (*sf_readf_short)(SNDFILE *, short *, sf_count_t);

SNDFILE *(*sf_open_virtual)(SF_VIRTUAL_IO *, int, SF_INFO *, void *);

const char *(*sf_strerror)(SNDFILE *);

SNDFILE *(*sf_open_fd)(int, int, SF_INFO *, int);

// From: https://linux.die.net/man/3/dlsym
/* double (*cosine)(double);
 *
 * Writing: cosine = (double (*)(double)) dlsym(handle, "cos");
 *  would seem more natural, but the C99 standard leaves
 *  casting from "void *" to a function pointer undefined.
 *  The assignment used below is the POSIX.1-2003 (Technical
 *  Corrigendum 1) workaround; see the Rationale for the
 *  POSIX specification of dlsym().
 */
#define DL_SYM(var, handle, string_name) \
        do {                                                                  \
          dlerror();                                                          \
          *(void **) (&var) = dlsym(handle, string_name);                     \
          auto err = dlerror();                                               \
          DALI_ENFORCE(!err, make_string("Failed to load symbol: ", err));    \
        } while(false)


void init_snd() {
  auto handle = dlopen("/usr/local/lib/libsndfile.so", RTLD_NOW);
  DALI_ENFORCE(handle, "Failed to load libsnd");
  DL_SYM(snd::sf_open, handle, "sf_open");
  DL_SYM(snd::sf_readf_short, handle, "sf_readf_short");
  DL_SYM(snd::sf_open_virtual, handle, "sf_open_virtual");
  DL_SYM(snd::sf_strerror, handle, "sf_strerror");
  DL_SYM(snd::sf_open_fd, handle, "sf_open_fd");
}

}}





//typedef decltype(&sf_open)               tsf_open;
//typedef decltype(&sf_readf_short)        tsf_readf_short;
//typedef decltype(&sf_open_virtual)       tsf_open_virtual;
//typedef decltype(&sf_strerror)           tsf_strerror;
//typedef decltype(&sf_open_fd)            tsf_open_fd;
//
//extern tsf_open                          ptr_sf_open;
//extern tsf_readf_short                   ptr_sf_readf_short;
//extern tsf_open_virtual                  ptr_sf_open_virtual;
//extern tsf_strerror                      ptr_sf_strerror;
//extern tsf_open_fd                       ptr_sf_open_fd;
//
///**********************************************************************************************/
//extern bool LibsndInitChecked();


#endif //DALI_DYNLINK_SND_H

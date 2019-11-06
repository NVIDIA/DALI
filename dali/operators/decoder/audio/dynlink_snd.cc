//// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
////
//// Licensed under the Apache License, Version 2.0 (the "License");
//// you may not use this file except in compliance with the License.
//// You may obtain a copy of the License at
////
////     http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//// See the License for the specific language governing permissions and
//// limitations under the License.
//
//#include <mutex>
//#include "dynlink_snd.h"
//
////namespace dali {
////namespace snd {
//
//
//tsf_open           ptr_sf_open;
//tsf_readf_short    ptr_sf_readf_short;
//tsf_open_virtual   ptr_sf_open_virtual;
//tsf_strerror       ptr_sf_strerror;
//tsf_open_fd        ptr_sf_open_fd;
//
//#define STRINGIFY(X) #X
//
//#include <dlfcn.h>
//
//static char __DriverLibName[] = "libsndfile.so";
//static char __DriverLibName1[] = "libsndfile.so.1";
//
//typedef void *DLLDRIVER;
//
//
//static bool LOAD_LIBRARY(DLLDRIVER *pInstance) {
//  *pInstance = dlopen(__DriverLibName, RTLD_NOW);
//
//  if (*pInstance == NULL) {
//    *pInstance = dlopen(__DriverLibName1, RTLD_NOW);
//
//    if (*pInstance == NULL) {
//      printf("dlopen \"%s\" failed!\n", __DriverLibName);
//      return false;
//    }
//  }
//
//  return true;
//}
//
//
//#define GET_PROC_EX(name, alias, required)                              \
//    ptr_##alias = (t##name )dlsym(DriverLib, #name);                    \
//    if (ptr_##alias == NULL && required) {                              \
//        printf("Failed to find required function \"%s\" in %s\n",       \
//               #name, __DriverLibName);                                 \
//        return false             ;                                      \
//    }
//
//#define GET_PROC_EX_V2(name, alias, required)                           \
//    alias = (t##name *)dlsym(DriverLib, STRINGIFY(name##_v2));          \
//    if (alias == NULL && required) {                                    \
//        printf("Failed to find required function \"%s\" in %s\n",       \
//               STRINGIFY(name##_v2), __DriverLibName);                  \
//        return false             ;                                      \
//    }
//
//#define GET_PROC_REQUIRED(name) GET_PROC_EX(name,name,1)
//#define GET_PROC_OPTIONAL(name) GET_PROC_EX(name,name,0)
//#define GET_PROC(name)          GET_PROC_REQUIRED(name)
//#define GET_PROC_V2(name)       GET_PROC_EX_V2(name,name,1)
//
//
//bool LibsndInit() {
//  DLLDRIVER DriverLib;
//
//  auto res = LOAD_LIBRARY(&DriverLib);
//  if (!res) return false;
//
//  // fetch all function pointers
//  GET_PROC(sf_open);
//  GET_PROC(sf_readf_short);
//  GET_PROC(sf_open_virtual);
//  GET_PROC(sf_strerror);
//  GET_PROC(sf_open_fd);
//
//  return true;
//}
//
//
//bool LibsndInitChecked() {
//  static std::mutex m;
//  static bool initialized = false;
//
//  if (initialized)
//    return true;
//
//  std::lock_guard<std::mutex> lock(m);
//  initialized = LibsndInit();
//  return initialized;
//}
////}}
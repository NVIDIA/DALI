/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef HEADER_NVC_H
#define HEADER_NVC_H

#include <sys/types.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define NVC_MAJOR   1
#define NVC_MINOR   0
#define NVC_PATCH   2
#define NVC_VERSION "1.0.2"

#define NVC_ARG_MAX 256

struct nvc_context;
struct nvc_container;

struct nvc_version {
        unsigned int major;
        unsigned int minor;
        unsigned int patch;
        const char *string;
};

struct nvc_config {
        char *root;
        char *ldcache;
        uid_t uid;
        gid_t gid;
};

struct nvc_device_node {
        char *path;
        dev_t id;
};

struct nvc_driver_info {
        char *nvrm_version;
        char *cuda_version;
        char **bins;
        size_t nbins;
        char **libs;
        size_t nlibs;
        char **libs32;
        size_t nlibs32;
        char **ipcs;
        size_t nipcs;
        struct nvc_device_node *devs;
        size_t ndevs;
};

struct nvc_device {
        char *model;
        char *uuid;
        char *busid;
        char *arch;
        char *brand;
        struct nvc_device_node node;
};

struct nvc_device_info {
        struct nvc_device *gpus;
        size_t ngpus;
};

struct nvc_container_config {
        pid_t pid;
        char *rootfs;
        char *bins_dir;
        char *libs_dir;
        char *libs32_dir;
        char *cudart_dir;
        char *ldconfig;
};

const struct nvc_version *nvc_version(void);

struct nvc_context *nvc_context_new(void);
void nvc_context_free(struct nvc_context *);

struct nvc_config *nvc_config_new(void);
void nvc_config_free(struct nvc_config *);

int nvc_init(struct nvc_context *, const struct nvc_config *, const char *);
int nvc_shutdown(struct nvc_context *);

struct nvc_container_config *nvc_container_config_new(pid_t, const char *);
void nvc_container_config_free(struct nvc_container_config *);

struct nvc_container *nvc_container_new(struct nvc_context *, const struct nvc_container_config *, const char *);
void nvc_container_free(struct nvc_container *);

struct nvc_driver_info *nvc_driver_info_new(struct nvc_context *, const char *);
void nvc_driver_info_free(struct nvc_driver_info *);

struct nvc_device_info *nvc_device_info_new(struct nvc_context *, const char *);
void nvc_device_info_free(struct nvc_device_info *);

int nvc_driver_mount(struct nvc_context *, const struct nvc_container *, const struct nvc_driver_info *);

int nvc_device_mount(struct nvc_context *, const struct nvc_container *, const struct nvc_device *);

int nvc_ldcache_update(struct nvc_context *, const struct nvc_container *);

const char *nvc_error(struct nvc_context *);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* HEADER_NVC_H */

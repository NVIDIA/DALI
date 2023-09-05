variable "ARCH" {
  default = "x86_64"
}

variable "CUDA_VERSION" {
  default = "122"
}

target "cuda_toolkit" {
  dockerfile = "docker/Dockerfile.cuda${CUDA_VERSION}.${ARCH}.deps"
  tags  = [
    "nvidia/dali:cuda${CUDA_VERSION}_${ARCH}.toolkit"
  ]
}

target "deps_ubuntu" {
  args = {
    FROM_IMAGE_NAME = "ubuntu:22.04"
  }
  dockerfile = "docker/Dockerfile.deps.ubuntu"
  tags  = [
    "nvidia/dali:${ARCH}.deps"
  ]
}

target "deps_with_cuda" {
  contexts = {
    "nvidia/dali:${ARCH}.deps" = "target:deps_ubuntu",
    "nvidia/dali:cuda${CUDA_VERSION}_${ARCH}.toolkit" = "target:cuda_toolkit"
  }
  args = {
    FROM_IMAGE_NAME = "nvidia/dali:${ARCH}.deps",
    CUDA_IMAGE = "nvidia/dali:cuda${CUDA_VERSION}_${ARCH}.toolkit"
  }
  dockerfile = "docker/Dockerfile.cuda.deps"
  tags  = [
    "nvidia/dali:cu${CUDA_VERSION}_${ARCH}.deps"
  ]
}

target "builder_image" {
  contexts = {
    "nvidia/dali:cu${CUDA_VERSION}_${ARCH}.deps" = "target:deps_with_cuda"
  }
  args = {
    DEPS_IMAGE_NAME = "nvidia/dali:cu${CUDA_VERSION}_${ARCH}.deps"
  }
  dockerfile = "docker/Dockerfile.ubuntu"
  tags  = [
    "nvidia/dali:cu${CUDA_VERSION}_${ARCH}.build"
  ]
  target = "builder"
}

group "default" {
  targets = ["builder_image"]
}
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp


def hue_mat(hue):
    h_rad = hue * jnp.pi / 180
    ret = jnp.eye(3)
    ret = ret.at[1, 1].set(jnp.cos(h_rad))
    ret = ret.at[2, 2].set(jnp.cos(h_rad))
    ret = ret.at[1, 2].set(jnp.sin(h_rad))
    ret = ret.at[2, 1].set(-jnp.sin(h_rad))
    return ret


def sat_mat(sat):
    ret = jnp.eye(3)
    ret = ret.at[1, 1].set(sat)
    ret = ret.at[2, 2].set(sat)
    return ret


def eye3(val):
    return jnp.diag(jnp.full(3, val, dtype=jnp.float32))


def color_twist_mat(brightness, contrast, saturation, hue, value):
    rgb2yiq = jnp.array(
        [
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.321],
            [0.211, -0.523, 0.311],
        ]
    )
    yiq2rgb = jnp.linalg.inv(rgb2yiq)
    return jnp.transpose(
        eye3(brightness)
        @ eye3(contrast)
        @ yiq2rgb
        @ hue_mat(hue)
        @ sat_mat(saturation)
        @ eye3(value)
        @ rgb2yiq
    )


def jax_color_twist(image, bcs, hue):
    brightness, contrast, saturation = bcs
    mat = color_twist_mat(brightness, contrast, saturation, hue, 1)
    dtype = image.dtype
    if dtype != jnp.float32:
        half_range = 128  # in DALI it's 128 even for bigger integers
        offset = (half_range - half_range * contrast) * brightness
        dtype_info = jnp.iinfo(dtype)
        image = jnp.clip(image @ mat + offset, dtype_info.min, dtype_info.max)
    else:
        half_range = 0.5
        offset = (half_range - half_range * contrast) * brightness
        image = image @ mat + offset
    return jnp.asarray(image, dtype=dtype)

#!/usr/bin/env python
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================

from __future__ import print_function
from builtins import range
import nvutils
import tensorflow as tf
import argparse

nvutils.init()

default_args = {
    'image_width' : 224,
    'image_height' : 224,
    'image_format' : 'channels_first',
    'distort_color' : False,
    'batch_size' : 256,
    'data_dir' : None,
    'log_dir' : None,
    'precision' : 'fp16',
    'momentum' : 0.9,
    'learning_rate_init' : 2.0,
    'learning_rate_power' : 2.0,
    'weight_decay' : 1e-4,
    'loss_scale' : 128.0,
    'larc_eta' : 0.003,
    'larc_mode' : 'clip',
    'num_iter' : 90,
    'iter_unit' : 'epoch',
    'checkpoint_secs' : None,
    'display_every' : 10,
}

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('--layers', default=50, type=int, required=True,
                    choices=[18, 34, 50, 101, 152],
                    help="""Number of resnet layers.""")
args, flags = nvutils.parse_cmdline(default_args, parser)

def resnet_bottleneck_v1(builder, inputs, depth, depth_bottleneck, stride,
                         basic=False):
    if builder.data_format == 'channels_first':
        num_inputs = inputs.get_shape().as_list()[1]
    else:
        num_inputs = inputs.get_shape().as_list()[-1]
    x  = inputs
    with tf.name_scope('resnet_v1'):
        if depth == num_inputs:
            if stride == 1:
                shortcut = x
            else:
                shortcut = builder.max_pooling2d(x, 1, stride)
        else:
            shortcut_depth = depth_bottleneck if basic else depth
            shortcut = builder.conv2d_linear(x, shortcut_depth, 1, stride, 'SAME')
        if basic:
            x = builder.pad2d(x, 1)
            x = builder.conv2d(       x, depth_bottleneck, 3, stride, 'VALID')
            x = builder.conv2d_linear(x, depth_bottleneck, 3, 1,      'SAME')
        else:
            x = builder.conv2d(       x, depth_bottleneck, 1, stride, 'SAME')
            x = builder.conv2d(       x, depth_bottleneck, 3, 1,      'SAME')
            x = builder.conv2d_linear(x, depth,            1, 1,      'SAME')
        x = tf.nn.relu(x + shortcut)
        return x

def inference_resnet_v1_impl(builder, inputs, layer_counts, basic=False):
    x = inputs
    x = builder.pad2d(x, 3)
    x = builder.conv2d(       x, 64, 7, 2, 'VALID')
    x = builder.max_pooling2d(x,     3, 2, 'SAME')
    for i in range(layer_counts[0]):
        x = resnet_bottleneck_v1(builder, x,  256,  64, 1, basic)
    for i in range(layer_counts[1]):
        x = resnet_bottleneck_v1(builder, x,  512, 128, 2 if i==0 else 1, basic)
    for i in range(layer_counts[2]):
        x = resnet_bottleneck_v1(builder, x, 1024, 256, 2 if i==0 else 1, basic)
    for i in range(layer_counts[3]):
        x = resnet_bottleneck_v1(builder, x, 2048, 512, 2 if i==0 else 1, basic)
    return builder.spatial_average2d(x)

def resnet_v1(inputs, training=False):
    """Deep Residual Networks family of models
    https://arxiv.org/abs/1512.03385
    """
    builder = nvutils.LayerBuilder(tf.nn.relu, args['image_format'], training, use_batch_norm=True)
    if   flags.layers ==  18: return inference_resnet_v1_impl(builder, inputs, [2,2, 2,2], basic=True)
    elif flags.layers ==  34: return inference_resnet_v1_impl(builder, inputs, [3,4, 6,3], basic=True)
    elif flags.layers ==  50: return inference_resnet_v1_impl(builder, inputs, [3,4, 6,3])
    elif flags.layers == 101: return inference_resnet_v1_impl(builder, inputs, [3,4,23,3])
    elif flags.layers == 152: return inference_resnet_v1_impl(builder, inputs, [3,8,36,3])
    else: raise ValueError("Invalid layer count (%i); must be one of: 18,34,50,101,152" %
                           flags.layers)

nvutils.train(resnet_v1, args)

if args['log_dir'] is not None and args['data_dir'] is not None:
    nvutils.validate(resnet_v1, args)

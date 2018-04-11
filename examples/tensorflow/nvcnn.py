#!/usr/bin/env python
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""
Changelog:
1.4
  - Fixed minor bug in model name exception
  - Added fallback to support PNG images in input dataset

1.3
  - Added support for synthetic data when no --data_dir is specified
  - Added support for overfeat model
  - Disabled color distortions by default
  - Disabled XLA by default
  - Disabled zero_debias_moving_mean in batch norm
  - Converted is_training from placeholder to constant
  - Simplified dropout by avoiding in-graph is_training conditional
  - Merge all gradients into a contiguous buffer and use a single nccl.all_sum

1.2
  - Added support for half-precision training, enabled via the --fp16 flag
  - Forced trainable weights to always be stored as float32
  - Added loss-scaling to improve numerical stability
  - Fixed trivial and lenet models by disabling batchnorm
  - Tweaked ranges of aspect ratio and area distortions

1.1
  - Disable all image distortions when in evaluation mode
  - Fixed bug in final FC layer of evaluation mode
  - Apply FLAGS.display_every in evaluation mode too
  - Changed keep_checkpoint_every_n_hours from 1 to 3
  - Remove reference to tf.python module
  - Tabs --> spaces
"""

from __future__ import print_function
from builtins import range

__version__ = "1.4"

import ndll.plugin.tf as ndll_tf
from ndll.pipeline import Pipeline
import ndll.ops as ops
import ndll.types as types
import ndll.tfrecord as tfrec

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import init_ops
try:
    from tensorflow.contrib import nccl
    have_nccl = True
except ImportError:
    have_nccl = False
    print("WARNING: NCCL support not available")

import sys
import os
import time
import math
from collections import defaultdict
import argparse

def tensorflow_version_tuple():
    v = tf.__version__
    major, minor, patch = v.split('.')
    return (int(major), int(minor), patch)
def tensorflow_version():
    vt = tensorflow_version_tuple()
    return vt[0]*100 + vt[1]

class DummyScope(object):
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable

class GPUNetworkBuilder(object):
    """This class provides convenient methods for constructing feed-forward
    networks with internal data layout of 'NCHW'.
    """
    def __init__(self,
                 is_training,
                 dtype=tf.float32,
                 activation='RELU',
                 use_batch_norm=True,
                 batch_norm_config = {'decay':   0.9,
                                      'epsilon': 1e-4,
                                      'scale':   True,
                                      'zero_debias_moving_mean': False},
                 use_xla=False):
        self.dtype             = dtype
        self.activation_func   = activation
        self.is_training       = is_training
        self.use_batch_norm    = use_batch_norm
        self.batch_norm_config = batch_norm_config
        self._layer_counts     = defaultdict(lambda: 0)
        if use_xla:
            self.jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
        else:
            self.jit_scope = DummyScope
    def _count_layer(self, layer_type):
        idx  = self._layer_counts[layer_type]
        name = layer_type + str(idx)
        self._layer_counts[layer_type] += 1
        return name
    def _get_variable(self, name, shape, dtype=None,
                      initializer=None, seed=None):
        if dtype is None:
            dtype = self.dtype
        if initializer is None:
            initializer = init_ops.glorot_uniform_initializer(seed=seed)
        elif (isinstance(initializer, float) or
              isinstance(initializer, int)):
            initializer = tf.constant_initializer(float(initializer))
        return tf.get_variable(name, shape, dtype, initializer)
    def _to_nhwc(self, x):
        return tf.transpose(x, [0,2,3,1])
    def _from_nhwc(self, x):
        return tf.transpose(x, [0,3,1,2])
    def _bias(self, input_layer):
        num_outputs = input_layer.get_shape().as_list()[1]
        biases = self._get_variable('biases', [num_outputs], input_layer.dtype,
                                    initializer=0)
        if len(input_layer.get_shape()) == 4:
            return tf.nn.bias_add(input_layer, biases,
                                  data_format='NCHW')
        else:
            return input_layer + biases
    def _batch_norm(self, input_layer, scope):
        return tf.contrib.layers.batch_norm(input_layer,
                                            is_training=self.is_training,
                                            scope=scope,
                                            data_format='NCHW',
                                            fused=True,
                                            **self.batch_norm_config)
    def _bias_or_batch_norm(self, input_layer, scope, use_batch_norm):
        if use_batch_norm is None:
            use_batch_norm = self.use_batch_norm
        if use_batch_norm:
            return self._batch_norm(input_layer, scope)
        else:
            return self._bias(input_layer)
    def input_layer(self, input_layer):
        """Converts input data into the internal format"""
        with self.jit_scope():
            # for ndll
            #x = self._from_nhwc(input_layer)
            x = input_layer
            x = tf.cast(x, self.dtype)
            # Rescale and shift to [-1,1]
            x = x * (1./127.5) - 1
        print("input layer")
        print(x.get_shape())
        return x
    def conv(self, input_layer, num_filters, filter_size,
             filter_strides=(1,1), padding='SAME',
             activation=None, use_batch_norm=None):
        """Applies a 2D convolution layer that includes bias or batch-norm
        and an activation function.
        """
        print(input_layer.get_shape())
        num_inputs = input_layer.get_shape().as_list()[1]
        kernel_shape = [filter_size[0], filter_size[1],
                        num_inputs, num_filters]
        strides = [1, 1, filter_strides[0], filter_strides[1]]
        with tf.variable_scope(self._count_layer('conv')) as scope:
            kernel = self._get_variable('weights', kernel_shape,
                                        input_layer.dtype)
            if padding == 'SAME_RESNET': # ResNet models require custom padding
                kh, kw = filter_size
                rate = 1
                kernel_size_effective = kh + (kw - 1) * (rate - 1)
                pad_total = kernel_size_effective - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                padding = [[0, 0], [0, 0],
                           [pad_beg, pad_end], [pad_beg, pad_end]]
                input_layer = tf.pad(input_layer, padding)
                padding = 'VALID'
            x = tf.nn.conv2d(input_layer, kernel, strides,
                             padding=padding, data_format='NCHW')
            x = self._bias_or_batch_norm(x, scope, use_batch_norm)
            x = self.activate(x, activation)
            return x
    def deconv(self, input_layer, num_filters, filter_size,
               filter_strides=(2,2), padding='SAME',
               activation=None, use_batch_norm=None):
        """Applies a 'transposed convolution' layer that includes bias or
        batch-norm and an activation function.
        """
        num_inputs  = input_layer.get_shape().as_list()[1]
        ih, iw      = input_layer.get_shape().as_list()[2:]
        output_shape = [-1, num_filters,
                        ih*filter_strides[0], iw*filter_strides[1]]
        kernel_shape = [filter_size[0], filter_size[1],
                        num_filters, num_inputs]
        strides = [1, 1, filter_strides[0], filter_strides[1]]
        with tf.variable_scope(self._count_layer('deconv')) as scope:
            kernel = self._get_variable('weights', kernel_shape,
                                        input_layer.dtype)
            x = tf.nn.conv2d_transpose(input_layer, kernel, output_shape,
                                       strides, padding=padding,
                                       data_format='NCHW')
            x = self._bias_or_batch_norm(x, scope, use_batch_norm)
            x = self.activate(x, activation)
            return x
    def activate(self, input_layer, funcname=None):
        """Applies an activation function"""
        if isinstance(funcname, tuple):
            funcname = funcname[0]
            params = funcname[1:]
        if funcname is None:
            funcname = self.activation_func
        if funcname == 'LINEAR':
            return input_layer
        activation_map = {
            'RELU':    tf.nn.relu,
            'RELU6':   tf.nn.relu6,
            'ELU':     tf.nn.elu,
            'SIGMOID': tf.nn.sigmoid,
            'TANH':    tf.nn.tanh,
            'LRELU':   lambda x, name: tf.maximum(params[0]*x, x, name=name)
        }
        return activation_map[funcname](input_layer, name=funcname.lower())
    def pool(self, input_layer, funcname, window_size,
                 window_strides=(2,2),
                 padding='VALID'):
        """Applies spatial pooling"""
        pool_map = {
            'MAX': tf.nn.max_pool,
            'AVG': tf.nn.avg_pool
        }
        kernel_size    = [1, 1, window_size[0], window_size[1]]
        kernel_strides = [1, 1, window_strides[0], window_strides[1]]
        return pool_map[funcname](input_layer, kernel_size, kernel_strides,
                                  padding, data_format='NCHW',
                                  name=funcname.lower())
    def project(self, input_layer, num_outputs, height, width,
                activation=None):
        """Linearly projects to an image-like tensor"""
        with tf.variable_scope(self._count_layer('project')):
            x = self.fully_connected(input_layer, num_outputs*height*width,
                                     activation=activation)
            x = tf.reshape(x, [-1, num_outputs, height, width])
            return x
    def flatten(self, input_layer):
        """Flattens the spatial and channel dims into a single dim (4D->2D)"""
        # Note: This ensures the output order matches that of NHWC networks
        input_layer = self._to_nhwc(input_layer)
        input_shape = input_layer.get_shape().as_list()
        num_inputs  = input_shape[1]*input_shape[2]*input_shape[3]
        return tf.reshape(input_layer, [-1, num_inputs], name='flatten')
    def spatial_avg(self, input_layer):
        """Averages over spatial dimensions (4D->2D)"""
        return tf.reduce_mean(input_layer, [2, 3], name='spatial_avg')
    def fully_connected(self, input_layer, num_outputs, activation=None):
        """Applies a fully-connected set of weights"""
        num_inputs = input_layer.get_shape().as_list()[1]
        kernel_size = [num_inputs, num_outputs]
        with tf.variable_scope(self._count_layer('fully_connected')):
            kernel = self._get_variable('weights', kernel_size,
                                        input_layer.dtype)
            x = tf.matmul(input_layer, kernel)
            x = self._bias(x)
            x = self.activate(x, activation)
            return x
    def inception_module(self, input_layer, name, cols):
        """Applies an inception module with a given form"""
        with tf.name_scope(name):
            col_layers      = []
            col_layer_sizes = []
            for c, col in enumerate(cols):
                col_layers.append([])
                col_layer_sizes.append([])
                x = input_layer
                for l, layer in enumerate(col):
                    ltype, args = layer[0], layer[1:]
                    if   ltype == 'conv': x = self.conv(x, *args)
                    elif ltype == 'pool': x = self.pool(x, *args)
                    elif ltype == 'share':
                        # Share matching layer from previous column
                        x = col_layers[c-1][l]
                    else: raise KeyError("Invalid layer type for " +
                                         "inception module: '%s'" % ltype)
                    col_layers[c].append(x)
            catdim  = 1
            catvals = [layers[-1] for layers in col_layers]
            x = tf.concat(catvals, catdim)
            return x
    def residual(self, input_layer, net, scale=1.0, activation='RELU'):
        """Applies a residual layer"""
        input_size     = input_layer.get_shape().as_list()
        num_inputs     = input_size[1]
        output_layer   = scale*net(self, input_layer)
        output_size    = output_layer.get_shape().as_list()
        num_outputs    = output_size[1]
        kernel_strides = (input_size[2]//output_size[2],
                          input_size[3]//output_size[3])
        with tf.name_scope('residual'):
            if (num_outputs != num_inputs or
                kernel_strides[0] != 1 or
                kernel_strides[1] != 1):
                input_layer = self.conv(input_layer, num_outputs, [1, 1],
                                        kernel_strides, activation='LINEAR')
            with self.jit_scope():
                x = self.activate(input_layer + output_layer, activation)
            return x
    def dropout(self, input_layer, keep_prob=0.5):
        """Applies a dropout layer if is_training"""
        if self.is_training:
            dtype = input_layer.dtype
            with tf.variable_scope(self._count_layer('dropout')):
                keep_prob_tensor = tf.constant(keep_prob, dtype=dtype)
                return tf.nn.dropout(input_layer, keep_prob_tensor)
        else:
            return input_layer

def deserialize_image_record(record):
    feature_map = {
        'image/encoded':          tf.FixedLenFeature([ ], tf.string, ''),
        'image/class/label':      tf.FixedLenFeature([1], tf.int64,  -1),
        'image/class/text':       tf.FixedLenFeature([ ], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label   = tf.cast(obj['image/class/label'], tf.int32)
        bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                            for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
        text    = obj['image/class/text']
        return imgdata, label, bbox, text

def decode_jpeg(imgdata, channels=3):
    return tf.image.decode_jpeg(imgdata, channels=channels,
                                fancy_upscaling=False,
                                dct_method='INTEGER_FAST')

def decode_png(imgdata, channels=3):
    return tf.image.decode_png(imgdata, channels=channels)

def random_crop_and_resize_image(image, bbox, height, width):
    with tf.name_scope('random_crop_and_resize'):
        if not FLAGS.eval:
            bbox_begin, bbox_size, distorted_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=bbox,
                min_object_covered=0.1,
                aspect_ratio_range=[0.8, 1.25],
                area_range=[0.1, 1.0],
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)
            # Crop the image to the distorted bounding box
            image = tf.slice(image, bbox_begin, bbox_size)
        # Resize to the desired output size
        image = tf.image.resize_images(
            image,
            [height, width],
            tf.image.ResizeMethod.BILINEAR,
            align_corners=False)
        image.set_shape([height, width, 3])
        return image

def distort_image_color(image, order):
    with tf.name_scope('distort_color'):
        image /= 255.
        brightness = lambda img: tf.image.random_brightness(img, max_delta=32. / 255.)
        saturation = lambda img: tf.image.random_saturation(img, lower=0.5, upper=1.5)
        hue        = lambda img: tf.image.random_hue(img, max_delta=0.2)
        contrast   = lambda img: tf.image.random_contrast(img, lower=0.5, upper=1.5)
        if order == 0: ops = [brightness, saturation, hue, contrast]
        else:          ops = [brightness, contrast, saturation, hue]
        for op in ops:
            image = op(image)
        # The random_* ops do not necessarily clamp the output range
        image = tf.clip_by_value(image, 0.0, 1.0)
        # Restore the original scaling
        image *= 255
        return image

class DummyPreprocessor(object):
    def __init__(self, height, width, batch, nclass):
        self.height = height
        self.width  = width
        self.batch = batch
        self.nclass = nclass

class ImagePreprocessor(object):
    def __init__(self, height, width, subset='train', dtype=tf.uint8):
        self.height = height
        self.width  = width
        self.num_devices = FLAGS.num_gpus
        self.subset = subset
        self.dtype = dtype
        self.nsummary = 10 # Max no. images to generate summaries for
    def preprocess(self, imgdata, bbox, thread_id):
        with tf.name_scope('preprocess_image'):
            try:
                image = decode_jpeg(imgdata)
            except:
                image = decode_png(imgdata)
            if thread_id < self.nsummary:
                image_with_bbox = tf.image.draw_bounding_boxes(
                    tf.expand_dims(tf.to_float(image), 0), bbox)
                tf.summary.image('original_image_and_bbox', image_with_bbox)
            image = random_crop_and_resize_image(image, bbox,
                                                 self.height, self.width)
            if thread_id < self.nsummary:
                tf.summary.image('cropped_resized_image',
                                 tf.expand_dims(image, 0))
            if not FLAGS.eval:
                image = tf.image.random_flip_left_right(image)
            if thread_id < self.nsummary:
                tf.summary.image('flipped_image',
                                 tf.expand_dims(image, 0))
            if FLAGS.distort_color and not FLAGS.eval:
                image = distort_image_color(image, order=thread_id%2)
                if thread_id < self.nsummary:
                    tf.summary.image('distorted_color_image',
                                     tf.expand_dims(image, 0))
        return image
    def device_minibatches(self, total_batch_size):
        record_input = data_flow_ops.RecordInput(
            file_pattern=os.path.join(FLAGS.data_dir, '%s-*' % self.subset),
            parallelism=64,
            # Note: This causes deadlock during init if larger than dataset
            buffer_size=FLAGS.input_buffer_size,
            batch_size=total_batch_size)
        records = record_input.get_yield_op()
        # Split batch into individual images
        records = tf.split(records, total_batch_size, 0)
        records = [tf.reshape(record, []) for record in records]
        # Deserialize and preprocess images into batches for each device
        images = defaultdict(list)
        labels = defaultdict(list)
        with tf.name_scope('input_pipeline'):
            for i, record in enumerate(records):
                imgdata, label, bbox, text = deserialize_image_record(record)
                image = self.preprocess(imgdata, bbox, thread_id=i)
                label -= 1 # Change to 0-based (don't use background class)
                device_num = i % self.num_devices
                images[device_num].append(image)
                labels[device_num].append(label)
            # Stack images back into a sub-batch for each device
            for device_num in range(self.num_devices):
                images[device_num] = tf.parallel_stack(images[device_num])
                labels[device_num] = tf.concat(labels[device_num], 0)
                images[device_num] = tf.reshape(images[device_num],
                                                [-1, self.height, self.width, 3])
                images[device_num] = tf.clip_by_value(images[device_num], 0., 255.)
                images[device_num] = tf.cast(images[device_num], self.dtype)
        return images, labels

class HybridPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, pipelined = True, async = True):
        super(HybridPipe, self).__init__(batch_size,
                                         num_threads,
                                         device_id,
                                         exec_pipelined=pipelined,
                                         exec_async=async)
        base = '/opt/ndll/examples/recordio/'
        idx_files = [base + "val.idx"]
        rec_files = [base + "val.rec"]
        self.input = ops.MXNetReader(path = rec_files, index_path = idx_files, shard_id = device_id, num_shards = num_gpus)

        self.huffman = ops.HuffmanDecoder()
        self.idct = ops.DCTQuantInv(device = "gpu", output_type = types.RGB)
        self.resize = ops.Resize(device = "gpu", random_resize = True,
                                 resize_a = 256, resize_b = 480,
                                 image_type = types.RGB,
                                 interp_type = types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                            output_type = types.FLOAT,
                                            random_crop = True,
                                            crop = (224, 224),
                                            image_type = types.RGB,
                                            mean = [128., 128., 128.],
                                            std = [1., 1., 1.])
        self.iter = 0

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        dct_coeff, jpeg_meta = self.huffman(inputs)
        images = self.idct(dct_coeff.gpu(), jpeg_meta)
        images = self.resize(images)
        output = self.cmn(images)
        return (output, labels.gpu())

    def iter_setup(self):
        pass

class NdllPreprocessor(object):
    def __init__(self, height, width, batch_size, dtype=tf.uint8):
        self.height = height
        self.width  = width
        self.batch = batch_size
        pipe = HybridPipe(batch_size=self.batch, num_threads=2, device_id = 0, num_gpus = 1, pipelined = True, async = True)
        serialized_pipe = pipe.serialize()
        ndllop = ndll_tf.NDLLIterator()
        self.images, self.labels = ndllop(serialized_pipeline = serialized_pipe,
                batch_size = batch_size,
                height = 224,
                width = 224)
    def get_device_minibatch(self):
        return self.images, self.labels

def stage(tensors):
    """Stages the given tensors in a StagingArea for asynchronous put/get.
    """
    stage_area = data_flow_ops.StagingArea(
        dtypes=[tensor.dtype       for tensor in tensors],
        shapes=[tensor.get_shape() for tensor in tensors])
    put_op      = stage_area.put(tensors)
    get_tensors = stage_area.get()

    get_tensors = [tf.reshape(gt, t.get_shape())
                   for (gt,t) in zip(get_tensors, tensors)]
    return put_op, get_tensors

def all_sync_params(tower_params, devices):
    """Assigns the params from the first tower to all others"""
    if len(devices) == 1:
        return tf.no_op()
    sync_ops = []
    if have_nccl and FLAGS.nccl:
        for param_on_devices in zip(*tower_params):
            # Note: param_on_devices is [paramX_gpu0, paramX_gpu1, ...]
            param0 = param_on_devices[0]
            send_op, received_tensors = nccl.broadcast(param0, devices[1:])
            sync_ops.append(send_op)
            for device, param, received in zip(devices[1:],
                                               param_on_devices[1:],
                                               received_tensors):
                with tf.device(device):
                    sync_op = param.assign(received)
                    sync_ops.append(sync_op)
    else:
        params0 = tower_params[0]
        for device, params in zip(devices, tower_params):
            with tf.device(device):
                for param, param0 in zip(params, params0):
                    sync_op = param.assign(param0.read_value())
                    sync_ops.append(sync_op)
    return tf.group(*sync_ops)

def all_avg_gradients(tower_gradvars, devices, param_server_device='/gpu:0'):
    if len(devices) == 1:
        return tower_gradvars

    if have_nccl and FLAGS.nccl:
        new_tower_grads = []
        contig_list = []
        for d, grad_list in zip(devices, tower_gradvars):
            with tf.device(d):
                flat_grads = [tf.reshape(g, [-1]) for (g, _) in grad_list]
                contig_grads = tf.concat(flat_grads, 0)
                contig_list.append(contig_grads)

        summed_grads = nccl.all_sum(contig_list)
        for d, s, grad_list in zip(devices, summed_grads, tower_gradvars):
            with tf.device(d):
                new_grad_list = [];
                sizes = [tf.size(g) for (g, _) in grad_list]
                flat_grads = tf.split(s, sizes)
                for newg, (oldg, v) in zip(flat_grads, grad_list):
                    newg = tf.reshape(newg, tf.shape(oldg))
                    newg *= 1. / len(devices)
                    new_grad_list.append((newg, v))
                new_tower_grads.append(new_grad_list)
        return new_tower_grads
    else:
        num_devices = len(tower_gradvars)
        avg_gradvars = []
        for layer in zip(*tower_gradvars):
            grads_on_devices, vars_on_devices = zip(*layer)
            with tf.device(param_server_device):
                avg_grad = tf.reduce_mean(tf.stack(grads_on_devices), 0)
            avg_grads_on_devices = [avg_grad]*num_devices
            avg_gradvars_on_devices = zip(*(avg_grads_on_devices, vars_on_devices))
            avg_gradvars.append(avg_gradvars_on_devices)
        return list(zip(*avg_gradvars))

class FeedForwardTrainer(object):
    def __init__(self, preprocessor, loss_func, nstep_per_epoch=None):
        self.image_preprocessor = preprocessor
        self.loss_func          = loss_func
        with tf.device('/cpu:0'):
            self.global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0),
                dtype=tf.int64,
                trainable=False)
        if FLAGS.lr_decay_policy == 'poly':
            self.learning_rate = tf.train.polynomial_decay(
                FLAGS.learning_rate,
                self.global_step,
                decay_steps=FLAGS.num_epochs*nstep_per_epoch,
                end_learning_rate=0.,
                power=FLAGS.lr_poly_power,
                cycle=False)
        else:
            self.learning_rate = tf.train.exponential_decay(
                FLAGS.learning_rate,
                self.global_step,
                decay_steps=FLAGS.lr_decay_epochs*nstep_per_epoch,
                decay_rate=FLAGS.lr_decay_rate,
                staircase=True)
    def make_optimizer(self):
        opt = tf.train.MomentumOptimizer(self.learning_rate, FLAGS.momentum,
                                         use_nesterov=True)
        return opt
    def training_step(self, total_batch_size, devices):
        preload_ops = [] # CPU pre-load
        gpucopy_ops = [] # H2D transfer
        self.tower_params = []
        tower_losses   = []
        tower_gradvars = []
        tower_top1s    = []
        tower_top5s    = []
        if type(self.image_preprocessor) is not DummyPreprocessor:
            if type(self.image_preprocessor) is NdllPreprocessor:
                dev_images, dev_labels = self.image_preprocessor.get_device_minibatch()
            else:
                with tf.device('/cpu:0'):
                    dev_images, dev_labels = self.image_preprocessor.device_minibatches(
                        total_batch_size)
        # Each device has its own copy of the model, referred to as a tower
        for device_num, device in enumerate(devices):
            if type(self.image_preprocessor) is not DummyPreprocessor:
                if type(self.image_preprocessor) is NdllPreprocessor:
                    images, labels = dev_images, dev_labels
                    #labels.set_shape((FLAGS.batch_size));
                else:
                    images, labels = dev_images[device_num], dev_labels[device_num]
                    with tf.device('/cpu:0'):
                        # Stage images on the host
                        preload_op, (images, labels) = stage([images, labels])
                        preload_ops.append(preload_op)
            with tf.device(device):
                if type(self.image_preprocessor) is ImagePreprocessor:
                    # Copy images from host to device
                    gpucopy_op, (images, labels) = stage([images, labels])
                    gpucopy_ops.append(gpucopy_op)
                elif type(self.image_preprocessor) is DummyPreprocessor:
                    input_shape = [self.image_preprocessor.batch,
                                   self.image_preprocessor.height,
                                   self.image_preprocessor.width,
                                   3]
                    images = tf.truncated_normal(
                        input_shape,
                        dtype=tf.float32,
                        stddev=1.e-1,
                        name='synthetic_images')
                    labels = tf.random_uniform(
                        [self.image_preprocessor.batch],
                        minval=0,
                        maxval=self.image_preprocessor.nclass-1,
                        dtype=tf.int32,
                        name='synthetic_labels')
                # Evaluate the loss and compute the gradients
                with tf.variable_scope(
                        'GPU_%i' % device_num,
                        # Force all variables to be stored as float32
                        custom_getter=float32_variable_storage_getter) \
                        as var_scope, \
                     tf.name_scope('tower_%i' % device_num):
                    loss, logits = self.loss_func(images, labels, var_scope)
                    tower_losses.append(loss)
                    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=var_scope.name)
                    self.tower_params.append(params)
                    # Apply loss scaling to improve numerical stability
                    if FLAGS.loss_scale != 1:
                        scale = FLAGS.loss_scale
                        grads  = [grad*(1./scale)
                                  for grad in tf.gradients(loss*scale, params)]
                    else:
                        grads = tf.gradients(loss, params)
                    gradvars = list(zip(grads, params))
                    tower_gradvars.append(gradvars)
                    with tf.device('/cpu:0'): # No in_top_k implem on GPU
                        labels = tf.cast(labels, tf.int32)
                        top1 = tf.reduce_mean(
                            tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
                        top5 = tf.reduce_mean(
                            tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))
                    tower_top1s.append(top1)
                    tower_top5s.append(top5)
        # Average the losses and gradients from each tower
        with tf.device('/cpu:0'):
            total_loss = tf.reduce_mean(tower_losses)
            total_top1 = tf.reduce_mean(tower_top1s)
            total_top5 = tf.reduce_mean(tower_top5s)

            averager = tf.train.ExponentialMovingAverage(0.90, name='loss_avg',
                                                         zero_debias=True)
            avg_op = averager.apply([total_loss])
            total_loss_avg = averager.average(total_loss)
            # Note: This must be done _after_ the averager.average() call
            #         because it changes total_loss into a new object.
            with tf.control_dependencies([avg_op]):
                total_loss     = tf.identity(total_loss)
                total_loss_avg = tf.identity(total_loss_avg)
            tf.summary.scalar('total loss raw', total_loss)
            tf.summary.scalar('total loss avg', total_loss_avg)
            tf.summary.scalar('train accuracy top-1 %', 100.*total_top1)
            tf.summary.scalar('train accuracy top-5 %', 100.*total_top5)
            tf.summary.scalar('learning rate', self.learning_rate)
        tower_gradvars = all_avg_gradients(tower_gradvars, devices)

        for grad, var in tower_gradvars[0]:
            tf.summary.histogram(var.op.name + '/values', var)
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Apply the gradients to optimize the loss function
        train_ops = []
        for device_num, device in enumerate(devices):
            with tf.device(device):
                gradvars = tower_gradvars[device_num]
                opt = self.make_optimizer()
                train_op = opt.apply_gradients(gradvars)
                train_ops.append(train_op)
        # Combine all of the ops required for a training step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []
        with tf.device('/cpu:0'):
            increment_global_step_op = tf.assign_add(self.global_step, 1)
        update_ops.append(increment_global_step_op)
        self.enqueue_ops = []
        self.enqueue_ops.append(tf.group(*preload_ops))
        self.enqueue_ops.append(tf.group(*gpucopy_ops))
        train_and_update_ops = tf.group(*(train_ops + update_ops))
        all_training_ops = (self.enqueue_ops + [train_and_update_ops])
        return total_loss_avg, self.learning_rate, all_training_ops
    def init(self, sess, devices):
        init_op = tf.global_variables_initializer()
        sync_op = all_sync_params(self.tower_params, devices)
        sess.run(init_op)
        sess.run(sync_op)
    def prefill_pipeline(self, sess):
        # Pre-fill the input pipeline with data
        for i in range(len(self.enqueue_ops)):
            sess.run(self.enqueue_ops[:i+1])

class FeedForwardEvaluator(object):
    def __init__(self, preprocessor, eval_func):
        self.eval_func          = eval_func
        self.image_preprocessor = preprocessor
    def evaluation_step(self, total_batch_size, devices):
        preload_ops = [] # CPU pre-load
        gpucopy_ops = [] # H2D transfer
        tower_top1s = []
        tower_top5s = []
        with tf.device('/cpu:0'):
            dev_images, dev_labels = self.image_preprocessor.device_minibatches(
                total_batch_size)
        # Each device has its own copy of the model, referred to as a tower
        for device_num, device in enumerate(devices):
            images, labels = dev_images[device_num], dev_labels[device_num]
            with tf.device('/cpu:0'):
                # Stage images on the host
                preload_op, (images, labels) = stage([images, labels])
                preload_ops.append(preload_op)
            with tf.device(device):
                # Copy images from host to device
                gpucopy_op, (images, labels) = stage([images, labels])
                gpucopy_ops.append(gpucopy_op)
                # Evaluate the loss and compute the gradients
                with tf.variable_scope('GPU_%i' % device_num) as var_scope, \
                     tf.name_scope('tower_%i' % device_num):
                    top1, top5 = self.eval_func(images, labels, var_scope)
                    tower_top1s.append(top1)
                    tower_top5s.append(top5)
        # Average the topN from each tower
        with tf.device('/cpu:0'):
            total_top1 = tf.reduce_mean(tower_top1s)
            total_top5 = tf.reduce_mean(tower_top5s)
        self.enqueue_ops = [tf.group(*preload_ops),
                            tf.group(*gpucopy_ops)]
        return total_top1, total_top5, self.enqueue_ops
    def prefill_pipeline(self, sess):
        # Pre-fill the input pipeline with data
        for i in range(len(self.enqueue_ops)):
            sess.run(self.enqueue_ops[:i+1])

def inference_trivial(net, input_layer):
    """A trivial model for benchmarking input pipeline performance"""
    net.use_batch_norm = False
    x = net.input_layer(input_layer)
    x = net.flatten(x)
    x = net.fully_connected(x, 1)
    return x

def inference_lenet5(net, input_layer):
    """Tiny network matching TF's MNIST tutorial model"""
    net.use_batch_norm = False
    x = net.input_layer(input_layer)
    x = net.conv(x, 32,    (5,5))
    x = net.pool(x, 'MAX', (2,2))
    x = net.conv(x, 64,    (5,5))
    x = net.pool(x, 'MAX', (2,2))
    x = net.flatten(x)
    x = net.fully_connected(x, 512)
    return x

def inference_overfeat(net, input_layer):
    net.use_batch_norm = False
    x = net.input_layer(input_layer)
    x = net.conv(x, 96,   (11,11), (4,4), 'VALID')
    x = net.pool(x, 'MAX', (2,2))
    x = net.conv(x, 256,   (5,5), (1,1), 'VALID')
    x = net.pool(x, 'MAX', (2,2))
    x = net.conv(x, 512,   (3,3))
    x = net.conv(x, 1024,  (3,3))
    x = net.conv(x, 1024,  (3,3))
    x = net.pool(x, 'MAX', (2,2))
    x = net.flatten(x)
    x = net.fully_connected(x, 3072)
    x = net.fully_connected(x, 4096)
    return x

def inference_alexnet_owt(net, input_layer):
    """Alexnet One Weird Trick model
    https://arxiv.org/abs/1404.5997
    """
    net.use_batch_norm = False
    x = net.input_layer(input_layer)
    # Note: VALID requires padding the images by 3 in width and height
    x = net.conv(x, 64, (11,11), (4,4), 'VALID')
    x = net.pool(x, 'MAX', (3,3))
    x = net.conv(x, 192,   (5,5))
    x = net.pool(x, 'MAX', (3,3))
    x = net.conv(x, 384,   (3,3))
    x = net.conv(x, 256,   (3,3))
    x = net.conv(x, 256,   (3,3))
    x = net.pool(x, 'MAX', (3,3))
    x = net.flatten(x)
    x = net.fully_connected(x, 4096)
    x = net.dropout(x)
    x = net.fully_connected(x, 4096)
    x = net.dropout(x)
    return x

def inference_vgg_impl(net, input_layer, layer_counts):
    net.use_batch_norm = False
    x = net.input_layer(input_layer)
    for _ in range(layer_counts[0]): x = net.conv(x,  64, (3,3))
    x = net.pool(x, 'MAX', (2,2))
    for _ in range(layer_counts[1]): x = net.conv(x, 128, (3,3))
    x = net.pool(x, 'MAX', (2,2))
    for _ in range(layer_counts[2]): x = net.conv(x, 256, (3,3))
    x = net.pool(x, 'MAX', (2,2))
    for _ in range(layer_counts[3]): x = net.conv(x, 512, (3,3))
    x = net.pool(x, 'MAX', (2,2))
    for _ in range(layer_counts[4]): x = net.conv(x, 512, (3,3))
    x = net.pool(x, 'MAX', (2,2))
    x = net.flatten(x)
    x = net.fully_connected(x, 4096)
    x = net.fully_connected(x, 4096)
    return x
def inference_vgg(net, input_layer, nlayer):
    """Visual Geometry Group's family of models
    https://arxiv.org/abs/1409.1556
    """
    if   nlayer == 11: return inference_vgg_impl(net, input_layer, [1,1,2,2,2]) # A
    elif nlayer == 13: return inference_vgg_impl(net, input_layer, [2,2,2,2,2]) # B
    elif nlayer == 16: return inference_vgg_impl(net, input_layer, [2,2,3,3,3]) # D
    elif nlayer == 19: return inference_vgg_impl(net, input_layer, [2,2,4,4,4]) # E
    else: raise ValueError("Invalid nlayer (%i); must be one of: 11,13,16,19" %
                           nlayer)

def inference_googlenet(net, input_layer):
    """GoogLeNet model
    https://arxiv.org/abs/1409.4842
    """
    net.use_batch_norm = False
    def inception_v1(net, x, k, l, m, n, p, q):
        cols = [[('conv', k, (1,1))],
                [('conv', l, (1,1)), ('conv', m, (3,3))],
                [('conv', n, (1,1)), ('conv', p, (5,5))],
                [('pool', 'MAX', (3,3), (1,1), 'SAME'), ('conv', q, (1,1))]]
        return net.inception_module(x, 'incept_v1', cols)
    x = net.input_layer(input_layer)
    x = net.conv(x,    64, (7,7), (2,2))
    x = net.pool(x, 'MAX', (3,3), padding='SAME')
    x = net.conv(x,    64, (1,1))
    x = net.conv(x,   192, (3,3))
    x = net.pool(x, 'MAX', (3,3), padding='SAME')
    x = inception_v1(net, x,  64,  96, 128, 16,  32,  32)
    x = inception_v1(net, x, 128, 128, 192, 32,  96,  64)
    x = net.pool(x, 'MAX', (3,3), padding='SAME')
    x = inception_v1(net, x, 192,  96, 208, 16,  48,  64)
    x = inception_v1(net, x, 160, 112, 224, 24,  64,  64)
    x = inception_v1(net, x, 128, 128, 256, 24,  64,  64)
    x = inception_v1(net, x, 112, 144, 288, 32,  64,  64)
    x = inception_v1(net, x, 256, 160, 320, 32, 128, 128)
    x = net.pool(x, 'MAX', (3,3), padding='SAME')
    x = inception_v1(net, x, 256, 160, 320, 32, 128, 128)
    x = inception_v1(net, x, 384, 192, 384, 48, 128, 128)
    x = net.spatial_avg(x)
    return x

def inference_inception_v3(net, input_layer):
    """Google's Inception v3 model
    https://arxiv.org/abs/1512.00567
    """
    def inception_v3_a(net, x, n):
        cols = [[('conv',  64, (1,1))],
                [('conv',  48, (1,1)), ('conv',  64, (5,5))],
                [('conv',  64, (1,1)), ('conv',  96, (3,3)), ('conv',  96, (3,3))],
                [('pool', 'AVG', (3,3), (1,1), 'SAME'), ('conv',   n, (1,1))]]
        return net.inception_module(x, 'incept_v3_a', cols)
    def inception_v3_b(net, x):
        cols = [[('conv',  64, (1,1)), ('conv',  96, (3,3)), ('conv',  96, (3,3), (2,2), 'VALID')],
                [('conv', 384, (3,3), (2,2), 'VALID')],
                [('pool', 'MAX', (3,3), (2,2), 'VALID')]]
        return net.inception_module(x, 'incept_v3_b', cols)
    def inception_v3_c(net, x, n):
        cols = [[('conv', 192, (1,1))],
                [('conv',   n, (1,1)), ('conv',   n, (1,7)), ('conv', 192, (7,1))],
                [('conv',   n, (1,1)), ('conv',   n, (7,1)), ('conv',   n, (1,7)), ('conv',   n, (7,1)), ('conv', 192, (1,7))],
                [('pool', 'AVG', (3,3), (1,1), 'SAME'), ('conv', 192, (1,1))]]
        return net.inception_module(x, 'incept_v3_c', cols)
    def inception_v3_d(net, x):
        cols = [[('conv', 192, (1,1)), ('conv', 320, (3,3), (2,2), 'VALID')],
                [('conv', 192, (1,1)), ('conv', 192, (1,7)), ('conv', 192, (7,1)), ('conv', 192, (3,3), (2,2), 'VALID')],
                [('pool', 'MAX', (3,3), (2,2), 'VALID')]]
        return net.inception_module(x, 'incept_v3_d',cols)
    def inception_v3_e(net, x, pooltype):
        cols = [[('conv', 320, (1,1))],
                [('conv', 384, (1,1)), ('conv', 384, (1,3))],
                [('share',),           ('conv', 384, (3,1))],
                [('conv', 448, (1,1)), ('conv', 384, (3,3)), ('conv', 384, (1,3))],
                [('share',),          ('share',),            ('conv', 384, (3,1))],
                [('pool', pooltype, (3,3), (1,1), 'SAME'),   ('conv', 192, (1,1))]]
        return net.inception_module(x, 'incept_v3_e', cols)

    # TODO: This does not include the extra 'arm' that forks off
    #         from before the 3rd-last module (the arm is designed
    #         to speed up training in the early stages).
    net.use_batch_norm = True
    x = net.input_layer(input_layer)
    x = net.conv(x,    32, (3,3), (2,2), padding='VALID')
    x = net.conv(x,    32, (3,3), (1,1), padding='VALID')
    x = net.conv(x,    64, (3,3), (1,1), padding='SAME')
    x = net.pool(x, 'MAX', (3,3))
    x = net.conv(x,    80, (1,1), (1,1), padding='VALID')
    x = net.conv(x,   192, (3,3), (1,1), padding='VALID')
    x = net.pool(x, 'MAX', (3,3))
    x = inception_v3_a(net, x, 32)
    x = inception_v3_a(net, x, 64)
    x = inception_v3_a(net, x, 64)
    x = inception_v3_b(net, x)
    x = inception_v3_c(net, x, 128)
    x = inception_v3_c(net, x, 160)
    x = inception_v3_c(net, x, 160)
    x = inception_v3_c(net, x, 192)
    x = inception_v3_d(net, x)
    x = inception_v3_e(net, x, 'AVG')
    x = inception_v3_e(net, x, 'MAX')
    x = net.spatial_avg(x)
    return x

def resnet_bottleneck_v1(net, input_layer, depth, depth_bottleneck, stride,
                         basic=False):
    num_inputs = input_layer.get_shape().as_list()[1]
    x = input_layer
    s = stride
    with tf.name_scope('resnet_v1'):
        if depth == num_inputs:
            if stride == 1:
                shortcut = input_layer
            else:
                shortcut = net.pool(x, 'MAX', (1,1), (s,s))
        else:
            shortcut = net.conv(x, depth, (1,1), (s,s), activation='LINEAR')
        if basic:
            x = net.conv(x, depth_bottleneck, (3,3), (s,s), padding='SAME_RESNET')
            x = net.conv(x, depth,            (3,3), activation='LINEAR')
        else:
            x = net.conv(x, depth_bottleneck, (1,1), (s,s))
            x = net.conv(x, depth_bottleneck, (3,3), padding='SAME')
            x = net.conv(x, depth,            (1,1), activation='LINEAR')
        with net.jit_scope():
            x = net.activate(x + shortcut)
        return x

def resnext_split_branch(net, input_layer, stride):
    x = input_layer
    with tf.name_scope('resnext_split_branch'):
        x = net.conv(x, net.bottleneck_width, (1, 1), (stride, stride), activation='RELU', use_batch_norm=True)
        x = net.conv(x, net.bottleneck_width, (3, 3), (1, 1), activation='RELU', use_batch_norm=True)
    return x

def resnext_shortcut(net, input_layer, stride, input_size, output_size):
    x = input_layer
    useConv = net.shortcut_type == 'C' or (net.shortcut_type == 'B' and input_size != output_size)
    with tf.name_scope('resnext_shortcut'):
        if useConv:
            x = net.conv(x, output_size, (1,1), (stride, stride), use_batch_norm=True)
        elif output_size == input_size:
            if stride == 1:
                x = input_layer
            else:
                x = net.pool(x, 'MAX', (1,1), (stride, stride))
        else:
            x = input_layer
    return x

def resnext_bottleneck_v1(net, input_layer, depth, depth_bottleneck, stride):
    num_inputs = input_layer.get_shape().as_list()[1]
    x = input_layer
    with tf.name_scope('resnext_bottleneck_v1'):
        shortcut = resnext_shortcut(net, x, stride, num_inputs, depth)
        branches_list = []
        for i in range(net.cardinality):
            branch = resnext_split_branch(net, x, stride)
            branches_list.append(branch)
        concatenated_branches = tf.concat(values=branches_list, axis=1, name='concat')
        bottleneck_depth = concatenated_branches.get_shape().as_list()[1]
        x = net.conv(concatenated_branches, depth, (1, 1), (1, 1), activation=None)
        x = net.activate(x + shortcut, 'RELU')
    return x

def inference_residual(net, input_layer, layer_counts, bottleneck_callback):
    net.use_batch_norm = True
    x = net.input_layer(input_layer)
    x = net.conv(x, 64,    (7,7), (2,2), padding='SAME_RESNET')
    x = net.pool(x, 'MAX', (3,3), (2,2), padding='SAME')
    for i in range(layer_counts[0]):
        x = bottleneck_callback(net, x,  256,  64, 1)
    for i in range(layer_counts[1]):
        x = bottleneck_callback(net, x, 512, 128, 2 if i==0 else 1)
    for i in range(layer_counts[2]):
        x = bottleneck_callback(net, x, 1024, 256, 2 if i==0 else 1)
    for i in range(layer_counts[3]):
        x = bottleneck_callback(net, x, 2048, 512, 2 if i==0 else 1)
    x = net.spatial_avg(x)
    return x

def inference_resnet_v1_basic_impl(net, input_layer, layer_counts):
    basic_resnet_bottleneck_callback = partial(resnet_bottleneck_v1, basic=True)
    return inference_residual(net, input_layer, layer_counts, basic_resnet_bottleneck_callback)

def inference_resnet_v1_impl(net, input_layer, layer_counts):
    return inference_residual(net, input_layer, layer_counts, resnet_bottleneck_v1)

def inference_resnext_v1_impl(net, input_layer, layer_counts):
    return inference_residual(net, input_layer, layer_counts, resnext_bottleneck_v1)

def inference_resnet_v1(net, input_layer, nlayer):
    """Deep Residual Networks family of models
    https://arxiv.org/abs/1512.03385
    """
    if   nlayer ==  18: return inference_resnet_v1_basic_impl(net, input_layer, [2,2, 2,2])
    elif nlayer ==  34: return inference_resnet_v1_basic_impl(net, input_layer, [3,4, 6,3])
    elif nlayer ==  50: return inference_resnet_v1_impl(net, input_layer, [3,4, 6,3])
    elif nlayer == 101: return inference_resnet_v1_impl(net, input_layer, [3,4,23,3])
    elif nlayer == 152: return inference_resnet_v1_impl(net, input_layer, [3,8,36,3])
    else: raise ValueError("Invalid nlayer (%i); must be one of: 18,34,50,101,152" %
                           nlayer)

def inference_resnext_v1(net, input_layer, nlayer):
    """Aggregated  Residual Networks family of models
    https://arxiv.org/abs/1611.05431
    """
    cardinality_to_bottleneck_width = { 1:64, 2:40, 4:24, 8:14, 32:4 }
    net.cardinality = 32
    net.shortcut_type = 'B'
    assert net.cardinality in cardinality_to_bottleneck_width.keys(), \
    "Invalid  cardinality (%i); must be one of: 1,2,4,8,32" % net.cardinality
    net.bottleneck_width = cardinality_to_bottleneck_width[net.cardinality]
    if nlayer ==  50: return inference_resnext_v1_impl(net, input_layer, [3,4, 6,3])
    elif nlayer == 101: return inference_resnext_v1_impl(net, input_layer, [3,4,23,3])
    elif nlayer == 152: return inference_resnext_v1_impl(net, input_layer, [3,8,36,3])
    else: raise ValueError("Invalid nlayer (%i); must be one of: 50,101,152" %
                           nlayer)

# Stem functions
def inception_v4_sa(net, x):
    cols = [[('pool', 'MAX', (3,3))],
            [('conv',  96, (3,3), (2,2), 'VALID')]]
    return net.inception_module(x, 'incept_v4_sa', cols)
def inception_v4_sb(net, x):
    cols = [[('conv',  64, (1,1)), ('conv',  96, (3,3), (1,1), 'VALID')],
            [('conv',  64, (1,1)), ('conv',  64, (7,1)), ('conv',  64, (1,7)), ('conv',  96, (3,3), (1,1), 'VALID')]]
    return net.inception_module(x, 'incept_v4_sb', cols)
def inception_v4_sc(net, x):
    cols = [[('conv', 192, (3,3), (2,2), 'VALID')],
            [('pool', 'MAX', (3,3))]]
    return net.inception_module(x, 'incept_v4_sc', cols)
# Reduction functions
def inception_v4_ra(net, x, k, l, m, n):
    cols = [[('pool', 'MAX', (3,3))],
            [('conv',   n, (3,3), (2,2), 'VALID')],
            [('conv',   k, (1,1)), ('conv',   l, (3,3)), ('conv',   m, (3,3), (2,2), 'VALID')]]
    return net.inception_module(x, 'incept_v4_ra', cols)
def inception_v4_rb(net, x):
    cols = [[('pool', 'MAX', (3,3))],
            [('conv', 192, (1,1)), ('conv', 192, (3,3), (2,2), 'VALID')],
            [('conv', 256, (1,1)), ('conv', 256, (1,7)), ('conv', 320, (7,1)), ('conv', 320, (3,3), (2,2), 'VALID')]]
    return net.inception_module(x, 'incept_v4_rb', cols)
def inception_resnet_v2_rb(net, x):
    cols = [[('pool', 'MAX', (3,3))],
            # Note: These match Facebook's Torch implem
            [('conv', 256, (1,1)), ('conv', 384, (3,3), (2,2), 'VALID')],
            [('conv', 256, (1,1)), ('conv', 256, (3,3), (2,2), 'VALID')],
            [('conv', 256, (1,1)), ('conv', 256, (3,3)), ('conv', 256, (3,3), (2,2), 'VALID')]]
    return net.inception_module(x, 'incept_resnet_v2_rb', cols)

def inference_inception_v4(net, input_layer):
    """Google's Inception v4 model
    https://arxiv.org/abs/1602.07261
    """
    def inception_v4_a(net, x):
        cols = [[('pool', 'AVG', (3,3), (1,1), 'SAME'), ('conv',  96, (1,1))],
                [('conv',  96, (1,1))],
                [('conv',  64, (1,1)), ('conv',  96, (3,3))],
                [('conv',  64, (1,1)), ('conv',  96, (3,3)), ('conv',  96, (3,3))]]
        return net.inception_module(x, 'incept_v4_a', cols)
    def inception_v4_b(net, x):
        cols = [[('pool', 'AVG', (3,3), (1,1), 'SAME'), ('conv', 128, (1,1))],
                [('conv', 384, (1,1))],
                [('conv', 192, (1,1)), ('conv', 224, (1,7)), ('conv', 256, (7,1))],
                [('conv', 192, (1,1)), ('conv', 192, (1,7)), ('conv', 224, (7,1)), ('conv', 224, (1,7)), ('conv', 256, (7,1))]]
        return net.inception_module(x, 'incept_v4_b', cols)
    def inception_v4_c(net, x):
        cols = [[('pool', 'AVG', (3,3), (1,1), 'SAME'), ('conv', 256, (1,1))],
                [('conv', 256, (1,1))],
                [('conv', 384, (1,1)), ('conv', 256, (1,3))],
                [('share',),           ('conv', 256, (3,1))],
                [('conv', 384, (1,1)), ('conv', 448, (1,3)), ('conv', 512, (3,1)), ('conv', 256, (3,1))],
                [('share',),           ('share',),           ('share',),           ('conv', 256, (1,3))]]
        return net.inception_module(x, 'incept_v4_c', cols)

    net.use_batch_norm = True
    x = net.input_layer(input_layer)
    x = net.conv(x, 32, (3,3), (2,2), padding='VALID')
    x = net.conv(x, 32, (3,3), (1,1), padding='VALID')
    x = net.conv(x, 64, (3,3))
    x = inception_v4_sa(net, x)
    x = inception_v4_sb(net, x)
    x = inception_v4_sc(net, x)
    for _ in range(4):
        x = inception_v4_a(net, x)
    x = inception_v4_ra(net, x, 192, 224, 256, 384)
    for _ in range(7):
        x = inception_v4_b(net, x)
    x = inception_v4_rb(net, x)
    for _ in range(3):
        x = inception_v4_c(net, x)
    x = net.spatial_avg(x)
    x = net.dropout(x, 0.8)
    return x

def inference_inception_resnet_v2(net, input_layer):
    """Google's Inception-Resnet v2 model
    https://arxiv.org/abs/1602.07261
    """
    def inception_resnet_v2_a(net, x):
        cols = [[('conv',  32, (1,1))],
                [('conv',  32, (1,1)), ('conv',  32, (3,3))],
                [('conv',  32, (1,1)), ('conv',  48, (3,3)), ('conv',  64, (3,3))]]
        x = net.inception_module(x, 'incept_resnet_v2_a', cols)
        x = net.conv(x, 384, (1,1), activation='LINEAR')
        return x
    def inception_resnet_v2_b(net, x):
        cols = [[('conv', 192, (1,1))],
                [('conv', 128, (1,1)), ('conv', 160, (1,7)), ('conv', 192, (7,1))]]
        x = net.inception_module(x, 'incept_resnet_v2_b', cols)
        x = net.conv(x, 1152, (1,1), activation='LINEAR')
        return x
    def inception_resnet_v2_c(net, x):
        cols = [[('conv', 192, (1,1))],
                [('conv', 192, (1,1)), ('conv', 224, (1,3)), ('conv', 256, (3,1))]]
        x = net.inception_module(x, 'incept_resnet_v2_c', cols)
        x = net.conv(x, 2048, (1,1), activation='LINEAR')
        return x

    net.use_batch_norm = True
    residual_scale = 0.2
    x = net.input_layer(input_layer)
    x = net.conv(x, 32, (3,3), (2,2), padding='VALID')
    x = net.conv(x, 32, (3,3), (1,1), padding='VALID')
    x = net.conv(x, 64, (3,3))
    x = inception_v4_sa(net, x)
    x = inception_v4_sb(net, x)
    x = inception_v4_sc(net, x)
    for _ in range(5):
        x = net.residual(x, inception_resnet_v2_a, scale=residual_scale)
    x = inception_v4_ra(net, x, 256, 256, 384, 384)
    for _ in range(10):
        x = net.residual(x, inception_resnet_v2_b, scale=residual_scale)
    x = inception_resnet_v2_rb(net, x)
    for _ in range(5):
        x = net.residual(x, inception_resnet_v2_c, scale=residual_scale)
    x = net.spatial_avg(x)
    x = net.dropout(x, 0.8)
    return x

def run_evaluation(nstep, sess, top1_op, top5_op, enqueue_ops):
    print("Evaluating")
    top1s = []
    top5s = []
    print("  Step  Top-1  Top-5")
    for step in range(nstep):
        try:
            top1, top5 = sess.run([top1_op, top5_op, enqueue_ops])[:2]
            if step == 0 or (step+1) % FLAGS.display_every == 0:
                print("% 6i %5.1f%% %5.1f%%" % (step+1, top1*100, top5*100))
            top1s.append(top1)
            top5s.append(top5)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
    nstep = len(top1s)
    if nstep == 0:
        return
    top1s = np.asarray(top1s) * 100.
    top5s = np.asarray(top5s) * 100.
    top1_mean = np.mean(top1s)
    top5_mean = np.mean(top5s)
    if nstep > 2:
        top1_uncertainty = np.std(top1s, ddof=1) / np.sqrt(float(nstep))
        top5_uncertainty = np.std(top5s, ddof=1) / np.sqrt(float(nstep))
    else:
        top1_uncertainty = float('nan')
        top5_uncertainty = float('nan')
    top1_madstd = 1.4826*np.median(np.abs(top1s - np.median(top1s)))
    top5_madstd = 1.4826*np.median(np.abs(top5s - np.median(top5s)))
    print('-' * 64)
    print('Validation Top-1: %.3f %% +/- %.2f (jitter = %.1f)' % (
        top1_mean, top1_uncertainty, top1_madstd))
    print('Validation Top-5: %.3f %% +/- %.2f (jitter = %.1f)' % (
        top5_mean, top5_uncertainty, top5_madstd))
    print('-' * 64)

def get_num_records(tf_record_pattern):
    def count_records(tf_record_filename):
        count = 0
        for _ in tf.python_io.tf_record_iterator(tf_record_filename):
            count += 1
        return count
    filenames = sorted(tf.gfile.Glob(tf_record_pattern))
    nfile = len(filenames)
    return (count_records(filenames[0])*(nfile-1) +
            count_records(filenames[-1]))

def add_bool_argument(cmdline, shortname, longname=None, default=False, help=None):
    if longname is None:
        shortname, longname = None, shortname
    elif default == True:
        raise ValueError("""Boolean arguments that are True by default should not have short names.""")
    name = longname[2:]
    feature_parser = cmdline.add_mutually_exclusive_group(required=False)
    if shortname is not None:
        feature_parser.add_argument(shortname, '--'+name, dest=name, action='store_true', help=help, default=default)
    else:
        feature_parser.add_argument(           '--'+name, dest=name, action='store_true', help=help, default=default)
    feature_parser.add_argument('--no'+name, dest=name, action='store_false')
    return cmdline

def main():
    tf.set_random_seed(1234)
    np.random.seed(4321)
    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Basic options
    cmdline.add_argument('-m', '--model', required=True,
                         help="""Name of model to run:
                         trivial, lenet,
                         alexnet, googlenet, vgg[11,13,16,19],
                         inception[3,4], resnet[18,34,50,101,152],
                         resnext[50,101,152], inception-resnet2.""")
    cmdline.add_argument('--data_dir', default=None,
                         help="""Path to dataset in TFRecord format
                         (aka Example protobufs). Files should be
                         named 'train-*' and 'validation-*'.""")
    cmdline.add_argument('-b', '--batch_size', default=64, type=int,
                         help="""Size of each minibatch.""")
    cmdline.add_argument('--num_batches', default=50, type=int,
                         help="""Number of batches to run.""")
    cmdline.add_argument('--num_epochs', default=None, type=int,
                         help="""Number of epochs to run
                         (overrides --num_batches).""")
    cmdline.add_argument('-g', '--num_gpus', default=1, type=int,
                         help="""Number of GPUs to run on.""")
    cmdline.add_argument('--log_dir', default="",
                         help="""Directory in which to write training
                         summaries and checkpoints.""")
    cmdline.add_argument('--display_every', default=1, type=int,
                         help="""How often (in iterations) to print out
                         running information.""")
    add_bool_argument(cmdline, '--eval',
                      help="""Evaluate the top-1 and top-5 accuracy of
                      a checkpointed model.""")
    add_bool_argument(cmdline, '--fp16',
                      help="""Train using float16 (half) precision instead
                      of float32.""")
    add_bool_argument(cmdline, '--ndll',
                      help="""Use NDLL for image loading.""")
    global FLAGS
    FLAGS, unknown_args = cmdline.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    FLAGS.strong_scaling = False
    FLAGS.nccl           = True
    FLAGS.xla            = False

    nclass = 1000
    total_batch_size = FLAGS.batch_size
    if not FLAGS.strong_scaling:
        total_batch_size *= FLAGS.num_gpus
    devices = ['/gpu:%i' % i for i in range(FLAGS.num_gpus)]
    subset = 'validation' if FLAGS.eval else 'train'

    tfversion = tensorflow_version_tuple()
    print("TensorFlow:  %i.%i.%s" % tfversion)
    print("This script:", __file__, "v%s" % __version__)
    print("Cmd line args:")
    print('\n'.join(['  '+arg for arg in sys.argv[1:]]))

    if FLAGS.data_dir is not None and FLAGS.data_dir != '':
        nrecord = get_num_records(os.path.join(FLAGS.data_dir, '%s-*' % subset))
    else:
        nrecord = FLAGS.num_batches * total_batch_size

    # Training hyperparameters
    FLAGS.learning_rate         = 0.001 # Model-specific values are set below
    FLAGS.momentum              = 0.9
    FLAGS.lr_decay_policy       = 'step'
    FLAGS.lr_decay_epochs       = 30
    FLAGS.lr_decay_rate         = 0.1
    FLAGS.lr_poly_power         = 2.
    FLAGS.weight_decay          = 1e-4
    FLAGS.input_buffer_size     = min(10000, nrecord)
    FLAGS.distort_color         = False
    FLAGS.nstep_burnin          = 20
    FLAGS.summary_interval_secs = 600
    FLAGS.save_interval_secs    = 600
    # Scaling to avoid fp16 underflow
    #*FLAGS.loss_scale            = (1 << 7) if FLAGS.fp16 else 1
    FLAGS.loss_scale            = 1 # TODO: May need to decide this based on model

    model_dtype = tf.float16 if FLAGS.fp16 else tf.float32

    print("Num images: ", nrecord if FLAGS.data_dir is not None else 'Synthetic')
    print("Model:      ", FLAGS.model)
    print("Batch size: ", total_batch_size, 'global')
    print("            ", total_batch_size/len(devices), 'per device')
    print("Devices:    ", devices)
    print("Data format:", 'NCHW')
    print("Data type:  ", 'fp16' if model_dtype == tf.float16 else 'fp32')
    print("Have NCCL:  ", have_nccl)
    print("Using NCCL: ", FLAGS.nccl)
    print("Using XLA:  ", FLAGS.xla)

    if FLAGS.num_epochs is not None:
        if FLAGS.data_dir is None:
            raise ValueError("num_epochs requires data_dir to be specified")
        nstep = nrecord * FLAGS.num_epochs // total_batch_size
    else:
        nstep = FLAGS.num_batches
        FLAGS.num_epochs = max(nstep * total_batch_size // nrecord, 1)

    model_name = FLAGS.model
    if   model_name == 'trivial':
        height, width = 224, 224
        model_func = inference_trivial
    elif model_name == 'lenet':
        height, width = 28, 28
        model_func = inference_lenet5
    elif model_name == 'alexnet':
        height, width = 227, 227
        model_func = inference_alexnet_owt
        FLAGS.learning_rate = 0.03
    elif model_name == 'overfeat':
        height, width = 231, 231
        model_func = inference_overfeat
    elif model_name.startswith('vgg'):
        height, width = 224, 224
        nlayer = int(model_name[len('vgg'):])
        model_func = lambda net, images: inference_vgg(net, images, nlayer)
        FLAGS.learning_rate = 0.02
    elif model_name == 'googlenet':
        height, width = 224, 224
        model_func = inference_googlenet
        FLAGS.learning_rate = 0.04
    elif model_name == 'inception3':
        height, width = 299, 299
        model_func = inference_inception_v3
        FLAGS.learning_rate = 0.2
    elif model_name.startswith('resnet'):
        height, width = 224, 224
        nlayer = int(model_name[len('resnet'):])
        model_func = lambda net, images: inference_resnet_v1(net, images, nlayer)
        FLAGS.learning_rate = 0.1 if nlayer > 18 else 0.5
    elif model_name.startswith('resnext'):
        height, width = 224, 224
        nlayer = int(model_name[len('resnext'):])
        model_func = lambda net, images: inference_resnext_v1(net, images, nlayer)
        FLAGS.learning_rate = 0.1
    elif model_name == 'inception4':
        height, width = 299, 299
        model_func = inference_inception_v4
        FLAGS.learning_rate = 0.045
    elif model_name == 'inception-resnet2':
        height, width = 299, 299
        model_func = inference_inception_resnet_v2
        FLAGS.learning_rate = 0.045
    else:
        raise ValueError("Invalid model type: %s" % model_name)

    if FLAGS.ndll:
        preprocessor = NdllPreprocessor(height, width, FLAGS.batch_size)
    elif FLAGS.data_dir is None:
        preprocessor = DummyPreprocessor(height, width, total_batch_size//len(devices), nclass)
    else:
        preprocessor = ImagePreprocessor(height, width, subset)

    def loss_func(images, labels, var_scope):
        # Build the forward model
        net = GPUNetworkBuilder(
            True, dtype=model_dtype, use_xla=FLAGS.xla)
        # Tmp: need to implem shape in the NdllOp
        #images.set_shape((FLAGS.batch_size, 3, height, width))
        print(images.shape)
        output = model_func(net, images)
        # Add final FC layer to produce nclass outputs
        logits = net.fully_connected(output, nclass, activation='LINEAR')
        if logits.dtype != tf.float32:
            logits = tf.cast(logits, tf.float32)
        if labels.dtype != tf.int32:
            labels = tf.cast(labels, tf.int32)
        print("LABELS")
        print(labels)
        loss = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)
        # Add weight decay
        if FLAGS.weight_decay is not None and FLAGS.weight_decay != 0.:
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=var_scope.name)
            with net.jit_scope():
                l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in params])
                if l2_loss.dtype != tf.float32:
                    l2_loss = tf.cast(l2_loss, tf.float32)
                loss += FLAGS.weight_decay * l2_loss
        return loss, logits
    def eval_func(images, labels, var_scope):
        net = GPUNetworkBuilder(
            False, dtype=model_dtype, use_xla=FLAGS.xla)
        output = model_func(net, images)
        logits = net.fully_connected(output, nclass, activation='LINEAR')
        if logits.dtype != tf.float32:
            logits = tf.cast(logits, tf.float32)
        with tf.device('/cpu:0'):
            top1 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
            top5 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))
        return top1, top5

    if FLAGS.eval:
        if FLAGS.data_dir is None:
            raise ValueError("eval requires data_dir to be specified")
        if FLAGS.fp16:
            raise ValueError("eval cannot be run with in fp16")
        evaluator = FeedForwardEvaluator(preprocessor, eval_func)
        print("Building evaluation graph")
        top1_op, top5_op, enqueue_ops = evaluator.evaluation_step(
            total_batch_size, devices)
    else:
        nstep_per_epoch = nrecord // total_batch_size
        trainer = FeedForwardTrainer(preprocessor, loss_func, nstep_per_epoch)
        print("Building training graph")
        total_loss, learning_rate, train_ops = trainer.training_step(
            total_batch_size, devices)

    print("Creating session")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.86)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.intra_op_parallelism_threads = 1

    sess = tf.Session(config=config)

    train_writer = None
    saver = None
    summary_ops = None
    if len(FLAGS.log_dir):
        log_dir = FLAGS.log_dir
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        summary_ops = tf.summary.merge_all()
        last_summary_time = time.time()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=3)
        last_save_time = time.time()

    restored = False
    if saver is not None:
        ckpt = tf.train.get_checkpoint_state(log_dir)
        checkpoint_file = os.path.join(log_dir, "checkpoint")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            restored = True
            print("Restored session from checkpoint " + ckpt.model_checkpoint_path)
        else:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

    if FLAGS.eval:
        if not restored:
            raise ValueError("No checkpoint found for evaluation")
        else:
            print("Pre-filling input pipeline")
            evaluator.prefill_pipeline(sess)
            nstep = nrecord // total_batch_size
            run_evaluation(nstep, sess, top1_op, top5_op, enqueue_ops)
            return

    if not restored:
        print("Initializing variables")
        trainer.init(sess, devices)
        if saver is not None:
            save_path = saver.save(sess, checkpoint_file, global_step=0)
            print("Checkpoint written to", save_path)

    print("Pre-filling input pipeline")
    trainer.prefill_pipeline(sess)

    print("Training")
    print("  Step Epoch Img/sec   Loss   LR")
    batch_times = []
    oom = False
    step0 = int(sess.run(trainer.global_step))
    for step in range(step0, nstep):
        ops_to_run = [total_loss, learning_rate] + train_ops
        try:
            start_time = time.time()
            if (summary_ops is not None and
                (step == 0 or
                 time.time() - last_summary_time > FLAGS.summary_interval_secs)):
                if step != 0:
                    last_summary_time += FLAGS.summary_interval_secs
                print("Writing summaries to ", log_dir)
                summary, loss, lr = sess.run([summary_ops] + ops_to_run)[:3]
                train_writer.add_summary(summary, step)
            else:
                loss, lr = sess.run(ops_to_run)[:2]
            elapsed = time.time() - start_time
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
        except tf.errors.ResourceExhaustedError:
            elapsed = -1.
            loss    = 0.
            lr      = -1
            oom = True

        if (saver is not None and
            time.time() - last_save_time > FLAGS.save_interval_secs):
            last_save_time += FLAGS.save_interval_secs
            save_path = saver.save(sess, checkpoint_file,
                                   global_step=trainer.global_step)
            print("Checkpoint written to", save_path)

        if step >= FLAGS.nstep_burnin:
            batch_times.append(elapsed)
        img_per_sec = total_batch_size / elapsed
        effective_accuracy = 100. / math.exp(min(loss,20.))
        if step == 0 or (step+1) % FLAGS.display_every == 0:
            epoch = step*total_batch_size // nrecord
            print("%6i %5i %7.1f %7.3f %7.5f" % (
                step+1, epoch+1, img_per_sec, loss, lr))
        if oom:
            break
    nstep = len(batch_times)
    if nstep > 0:
        batch_times = np.array(batch_times)
        speeds = total_batch_size / batch_times
        speed_mean = np.mean(speeds)
        if nstep > 2:
            speed_uncertainty = np.std(speeds, ddof=1) / np.sqrt(float(nstep))
        else:
            speed_uncertainty = float('nan')
        speed_madstd = 1.4826*np.median(np.abs(speeds - np.median(speeds)))
        speed_jitter = speed_madstd
        print('-' * 64)
        print('Images/sec: %.1f +/- %.1f (jitter = %.1f)' % (
            speed_mean, speed_uncertainty, speed_jitter))
        print('-' * 64)
    else:
        print("No results, did not get past burn-in phase (%i steps)" %
              FLAGS.nstep_burnin)

    if train_writer is not None:
        train_writer.close()

    if oom:
        print("Out of memory error detected, exiting")
        sys.exit(-2)

if __name__ == '__main__':
    main()

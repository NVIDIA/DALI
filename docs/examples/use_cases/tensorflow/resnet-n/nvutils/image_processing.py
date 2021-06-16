# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf
import sys
import os
import numpy as np
from subprocess import call
import horovod.tensorflow.keras as hvd

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
try:
  import nvidia.dali.plugin.tf as dali_tf
except:
  pass

NUM_CLASSES = 1000

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

def _deserialize_image_record(record):
  feature_map = {
      'image/encoded':          tf.io.FixedLenFeature([ ], tf.string, ''),
      'image/class/label':      tf.io.FixedLenFeature([1], tf.int64,  -1),
      'image/class/text':       tf.io.FixedLenFeature([ ], tf.string, ''),
      'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
  }
  with tf.name_scope('deserialize_image_record'):
    obj = tf.io.parse_single_example(record, feature_map)
    imgdata = obj['image/encoded']
    label   = tf.cast(obj['image/class/label'], tf.int32)
    bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                        for x in ['ymin', 'xmin', 'ymax', 'xmax']])
    bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
    text    = obj['image/class/text']
    return imgdata, label, bbox, text

def _decode_jpeg(imgdata, channels=3):
  return tf.image.decode_jpeg(imgdata, channels=channels,
                              fancy_upscaling=False,
                              dct_method='INTEGER_FAST')

def _crop_and_resize_image(image, original_bbox, height, width, deterministic=False, random_crop=False):
  with tf.name_scope('random_crop_and_resize'):
    eval_crop_ratio = 0.8
    if random_crop:
      bbox_begin, bbox_size, bbox = \
          tf.image.sample_distorted_bounding_box(
              tf.shape(image),
              bounding_boxes=tf.zeros(shape=[1,0,4]), # No bounding boxes
              min_object_covered=0.1,
              aspect_ratio_range=[0.8, 1.25],
              area_range=[0.1, 1.0],
              max_attempts=100,
              seed=7 * (1+hvd.rank()) if deterministic else 0,
              use_image_if_no_bounding_boxes=True)
      image = tf.slice(image, bbox_begin, bbox_size)
    else:
      # Central crop
      image = tf.image.central_crop(image, eval_crop_ratio)
    image = tf.compat.v1.image.resize_images(
        image,
        [height, width],
        tf.image.ResizeMethod.BILINEAR,
        align_corners=False)
    image.set_shape([height, width, 3])
    return image

def _distort_image_color(image, order=0):
  with tf.name_scope('distort_color'):
    image = tf.math.multiply(image, 1. / 255.)
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
    image = tf.multiply(image, 255.)
    return image

def _parse_and_preprocess_image_record(record, height, width,
                                       deterministic=False, random_crop=False,
                                       distort_color=False):
  imgdata, label, bbox, text = _deserialize_image_record(record)
  label -= 1 # Change to 0-based (don't use background class)
  with tf.name_scope('preprocess_train'):
    try:    image = _decode_jpeg(imgdata, channels=3)
    except: image = tf.image.decode_png(imgdata, channels=3)

    image = _crop_and_resize_image(image, bbox, height, width, deterministic, random_crop)

    # image comes out of crop as float32, which is what distort_color expects
    if distort_color:
      image = _distort_image_color(image)
    image = tf.cast(image, tf.float32)
    if random_crop:
      image = tf.image.random_flip_left_right(image,
                    seed=11 * (1 + hvd.rank()) if deterministic else None)
    return image, label

# Synthetic images are generated once, and the same batch is repeated again and
# again. The H2D copy is also repeated.
def fake_image_set(batch_size, height, width, with_label=True):
  data_shape = [batch_size, height, width, 3] # 3 channels
  images = tf.random.truncated_normal(
               data_shape, dtype=tf.float32, mean=112, stddev=70,
               name='fake_images')
  images = tf.clip_by_value(images, 0.0, 255.0)
  images = tf.cast(images, tf.float32)
  if with_label:
    labels = tf.random.uniform(
                 [batch_size], minval=0, maxval=1000-1, dtype=tf.int32,
                 name='fake_labels')
    ds = tf.data.Dataset.from_tensor_slices(([images], [labels]))
  else:
    ds = tf.data.Dataset.from_tensor_slices(([images]))
  ds = ds.repeat()
  return ds


@pipeline_def
def get_dali_pipeline(
            tfrec_filenames,
            tfrec_idx_filenames,
            height, width,
            shard_id,
            num_gpus,
            dali_cpu=True,
            training=True):

    inputs = fn.readers.tfrecord(
                    path=tfrec_filenames,
                    index_path=tfrec_idx_filenames,
                    random_shuffle=training,
                    shard_id=shard_id,
                    num_shards=num_gpus,
                    initial_fill=10000,
                    features={
                        'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                        'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                        'image/class/text': tfrec.FixedLenFeature([ ], tfrec.string, ''),
                        'image/object/bbox/xmin': tfrec.VarLenFeature(tfrec.float32, 0.0),
                        'image/object/bbox/ymin': tfrec.VarLenFeature(tfrec.float32, 0.0),
                        'image/object/bbox/xmax': tfrec.VarLenFeature(tfrec.float32, 0.0),
                        'image/object/bbox/ymax': tfrec.VarLenFeature(tfrec.float32, 0.0)})

    decode_device = "cpu" if dali_cpu else "mixed"
    resize_device = "cpu" if dali_cpu else "gpu"
    if training:
        images = fn.decoders.image_random_crop(
            inputs["image/encoded"],
            device=decode_device,
            output_type=types.RGB,
            random_aspect_ratio=[0.75, 1.25],
            random_area=[0.05, 1.0],
            num_attempts=100,
            # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
            preallocate_width_hint=5980 if decode_device == 'mixed' else 0,
            preallocate_height_hint=6430 if decode_device == 'mixed' else 0)
        images = fn.resize(images, device=resize_device, resize_x=width, resize_y=height)
    else:
        images = fn.decoders.image(
            inputs["image/encoded"],
            device=decode_device,
            output_type=types.RGB)
        # Make sure that every image > 224 for CropMirrorNormalize
        images = fn.resize(images, device=resize_device, resize_shorter=256)

    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        crop=(height, width),
        mean=[123.68, 116.78, 103.94],
        std=[58.4, 57.12, 57.3],
        output_layout="HWC",
        mirror = fn.random.coin_flip())
    labels = inputs["image/class/label"].gpu()

    labels -= 1 # Change to 0-based (don't use background class)
    return images, labels


class DALIPreprocessor(object):
  def __init__(self,
               filenames,
               idx_filenames,
               height, width,
               batch_size,
               num_threads,
               dtype=tf.uint8,
               dali_cpu=True,
               deterministic=False,
               training=False):
    device_id = hvd.local_rank()
    shard_id = hvd.rank()
    num_gpus = hvd.size()
    self.pipe = get_dali_pipeline(
        tfrec_filenames=filenames,
        tfrec_idx_filenames=idx_filenames,
        height=height,
        width=width,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        shard_id=shard_id,
        num_gpus=num_gpus,
        dali_cpu=dali_cpu,
        training=training,
        seed=7 * (1 + hvd.rank()) if deterministic else None)

    self.daliop = dali_tf.DALIIterator()

    self.batch_size = batch_size
    self.height = height
    self.width = width
    self.device_id = device_id

    self.dalidataset = dali_tf.DALIDataset(
        pipeline=self.pipe,
        output_shapes=((batch_size, height, width, 3), (batch_size)),
        batch_size=batch_size,
        output_dtypes=(tf.float32, tf.int64),
        device_id=device_id)

  def get_device_minibatches(self):
    with tf.device("/gpu:0"):
      images, labels = self.daliop(
          pipeline=self.pipe,
          shapes=[(self.batch_size, self.height, self.width, 3), ()],
          dtypes=[tf.float32, tf.int64],
          device_id=self.device_id)
    return images, labels

  def get_device_dataset(self):
    return self.dalidataset

def image_set(filenames, batch_size, height, width, training=False,
              distort_color=False, num_threads=10, nsummary=10,
              deterministic=False, use_dali=None, idx_filenames=None):
  if use_dali:
    if idx_filenames is None:
      raise ValueError("Must provide idx_filenames if Dali is enabled")

    preprocessor = DALIPreprocessor(
        filenames,
        idx_filenames,
        height, width,
        batch_size,
        num_threads,
        dali_cpu=True if use_dali == 'CPU' else False,
        deterministic=deterministic, training=training)
    return preprocessor
  else:
    shuffle_buffer_size = 10000
    num_readers = 10
    ds = tf.data.Dataset.from_tensor_slices(filenames)

    # AUTOTUNE can give better perf for non-horovod cases
    thread_config = num_threads

    # shard should be before any randomizing operations
    if training:
      ds = ds.shard(hvd.size(), hvd.rank())

    # read up to num_readers files and interleave their records
    ds = ds.interleave(
        tf.data.TFRecordDataset, cycle_length=num_readers)

    if training:
      # Improve training performance when training data is in remote storage and
      # can fit into worker memory.
      ds = ds.cache()

    if training:
      # shuffle data before repeating to respect epoch boundaries
      ds = ds.shuffle(shuffle_buffer_size)
      ds = ds.repeat()

    preproc_func = (lambda record:
        _parse_and_preprocess_image_record(record, height, width,
            deterministic=deterministic, random_crop=training,
            distort_color=distort_color))
    ds = ds.map(preproc_func,
                num_parallel_calls=thread_config)

    ds = ds.batch(batch_size, drop_remainder=True)

    # prefetching
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_slack = True
    ds = ds.with_options(options)

    return ds


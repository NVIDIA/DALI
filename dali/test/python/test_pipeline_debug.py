# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np

rn50_pipeline_base_debug_values = {}


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def rn50_pipeline_base(debug=False):
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    if debug:
        rn50_pipeline_base_debug_values['rng'] = rng.get()
    jpegs, labels = fn.readers.file(
        file_root='/home/ksztenderski/DALI_extra/db/single/jpeg', shard_id=0, num_shards=2)
    if debug:
        rn50_pipeline_base_debug_values['jpegs'] = jpegs.get()
        rn50_pipeline_base_debug_values['labels'] = labels.get()
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
    if debug:
        rn50_pipeline_base_debug_values['images'] = images.get()
    resized_images = fn.random_resized_crop(images, device="gpu", size=(224, 224), seed=27)
    if debug:
        rn50_pipeline_base_debug_values['resized_images'] = resized_images.get()
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(resized_images.gpu(), mirror=rng, device="gpu", dtype=out_type, crop=(
        224, 224), mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    if debug:
        rn50_pipeline_base_debug_values['output'] = output.get()
    return rng, jpegs, labels, images, resized_images, output


@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def rn50_pipeline():
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    print(f'rng: {rng.get().as_array()}')
    tmp = rng ^ 1
    print(f'rng xor: {tmp.get().as_array()}')
    jpegs, labels = fn.readers.file(
        file_root='/home/ksztenderski/DALI_extra/db/single/jpeg', shard_id=0, num_shards=2)
    if jpegs.get().is_dense_tensor():
        print(f'jpegs: {jpegs.get().as_array()}')
    else:
        print('jpegs shapes:')
        for j in jpegs.get():
            print(j.shape())
    print(f'labels: {labels.get().as_array()}')
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
    for i in images.get().as_cpu():
        print(i)
    for i in images.get():
        print(i.shape())
    images = fn.random_resized_crop(images, device="gpu", size=(224, 224), seed=27)
    for i in images.get():
        print(i.shape())
    print(np.array(images.get().as_cpu()[0]))
    images += 1
    print(np.array(images.get().as_cpu()[0]))
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(images.gpu(), mirror=rng, device="gpu", dtype=out_type, crop=(
        224, 224), mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return (output, labels.gpu())


@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def load_images_pipeline():
    jpegs, labels = fn.readers.file(
        file_root='/home/ksztenderski/DALI_extra/db/single/jpeg', shard_id=0, num_shards=2)
    images = fn.decoders.image(jpegs, output_type=types.RGB)
    return images, labels



@pipeline_def(batch_size=8, num_threads=3, device_id=0, debug=True)
def numpy_array_injection_pipeline(images):
    rng = fn.random.coin_flip(probability=0.5, seed=47)
    images = fn.random_resized_crop(images, size=(224, 224), seed=27)
    print(images.get().layout())
    out_type = types.FLOAT16

    output = fn.crop_mirror_normalize(images.gpu(), mirror=rng, device="gpu", dtype=out_type, crop=(
        224, 224), mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return (output)


def run_pipeline(func, *args, **kwargs):
    pipe = func(*args, **kwargs)
    pipe.build()
    return pipe.run()


def test_debug_pipeline_base():
    rng, jpegs, labels, images, resized_images, output = run_pipeline(rn50_pipeline_base)
    rng_, jpegs_, labels_, images_, resized_images_, output_ = run_pipeline(
        rn50_pipeline_base, debug=True)

    np.testing.assert_array_equal(rng.as_array(), rng_.as_array())
    np.testing.assert_array_equal(rng.as_array(), rn50_pipeline_base_debug_values['rng'].as_array())
    np.testing.assert_array_equal(labels.as_array(), labels_.as_array())
    np.testing.assert_array_equal(labels.as_array(), labels_.as_array(),
                                  rn50_pipeline_base_debug_values['labels'])
    for j, j_, jd in zip(jpegs, jpegs_, rn50_pipeline_base_debug_values['jpegs']):
        np.testing.assert_array_equal(j, j_)
        np.testing.assert_array_equal(j, jd)
    for i, i_, id in zip(images, images_, rn50_pipeline_base_debug_values['images']):
        np.testing.assert_array_equal(i.as_cpu(), i_.as_cpu())
        np.testing.assert_array_equal(i.as_cpu(), id.as_cpu())
    for i, i_, id in zip(resized_images, resized_images_, rn50_pipeline_base_debug_values['resized_images']):
        np.testing.assert_array_equal(i.as_cpu(), i_.as_cpu())
        np.testing.assert_array_equal(i.as_cpu(), id.as_cpu())
    for o, o_, od in zip(output, output_, rn50_pipeline_base_debug_values['output']):
        np.testing.assert_array_equal(o.as_cpu(), o_.as_cpu())
        np.testing.assert_array_equal(o.as_cpu(), od.as_cpu())


def test_operations_on_debug_pipeline():
    run_pipeline(rn50_pipeline)


def test_numpy_injection():
    load_pipeline = load_images_pipeline()
    load_pipeline.build()
    images_o, _ = load_pipeline.run()
    images_o = [np.array(i) for i in images_o]
    run_pipeline(numpy_array_injection_pipeline, images_o)

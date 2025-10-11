import nvidia.dali.experimental.dali2 as D
import nvidia.dali.experimental.dali2.ops as ops
import nvidia.dali as dali
import nvidia.dali.fn as fn
import os
import numpy as np
from test_utils import get_dali_extra_path


def get_rn50_pipeline_fn(max_batch_size):
    uniform = ops.random.Uniform(max_batch_size=max_batch_size, seed=42)
    resize_uniform = ops.random.Uniform(max_batch_size=max_batch_size, seed=43)
    mirror = ops.random.CoinFlip(max_batch_size=max_batch_size, seed=44)

    def f(jpegs):
        batch_size = jpegs.batch_size
        images = D.decoders.image(jpegs, device="mixed")
        xy = uniform(batch_size=batch_size, range=[0.0, 1.0], shape=2)
        do_mirror = mirror(batch_size=batch_size, probability=0.5)
        size = resize_uniform(batch_size=batch_size, range=[256.0, 480.0])
        resized_images = D.fast_resize_crop_mirror(
            images,
            crop=[224.0, 224.0],
            crop_pos_x=xy.slice[0],
            crop_pos_y=xy.slice[1],
            mirror=do_mirror,
            resize_shorter=size,
            interp_type=dali.types.INTERP_LANCZOS3,
        )
        output = D.crop_mirror_normalize(
            resized_images,
            device="gpu",
            dtype=D.float16,
            mean=[128.0, 128.0, 128.0],
            std=[1.0, 1.0, 1.0],
        )
        return output

    return f


@dali.pipeline_def(num_threads=4, device_id=0)
def rn50_pipeline():
    jpegs, labels = fn.readers.file(
        name="reader", file_root=file_root, file_list=img_list, random_shuffle=False
    )
    uniform = fn.random.uniform(range=[0.0, 1.0], shape=2, seed=42)
    resize_uniform = fn.random.uniform(range=[256.0, 480.0], seed=43)
    mirror = fn.random.coin_flip(probability=0.5, seed=44)
    images = fn.decoders.image(jpegs, device="mixed")
    xy = uniform
    do_mirror = mirror
    size = resize_uniform
    resized_images = fn.fast_resize_crop_mirror(
        images,
        crop=[224.0, 224.0],
        crop_pos_x=xy[0],
        crop_pos_y=xy[1],
        mirror=do_mirror,
        resize_shorter=size,
        interp_type=dali.types.DALIInterpType.INTERP_LANCZOS3,
    )
    output = fn.crop_mirror_normalize(
        resized_images,
        device="gpu",
        dtype=dali.types.DALIDataType.FLOAT16,
        mean=[128.0, 128.0, 128.0],
        std=[1.0, 1.0, 1.0],
    )
    return output, labels


file_root = os.path.join(get_dali_extra_path(), "db/single/jpeg")
img_list = os.path.join(file_root, "image_list.txt")


def test_rn50_pipeline():
    batch_size = 16
    f = get_rn50_pipeline_fn(max_batch_size=batch_size)
    r = D.readers.File(file_root=file_root, file_list=img_list, random_shuffle=False)
    iterations = 0
    p = rn50_pipeline(batch_size=batch_size)
    with D.EvalContext(num_threads=4, device_id=0):
        for epoch in range(10):
            for jpegs, lbl_dynamic in r.next_epoch(batch_size=batch_size):
                iterations += 1
                imgs = f(jpegs)
                out_dynamic = imgs.cpu().evaluate()
                assert out_dynamic.batch_size == batch_size
                out_pipeline, lbl_pipeline = p.run()
                out_pipeline = out_pipeline.as_cpu()
                for i in range(batch_size):
                    assert np.array_equal(lbl_dynamic.tensors[i], np.array(lbl_pipeline[i]))
                    assert np.array_equal(out_dynamic.tensors[i], np.array(out_pipeline[i]))

    assert iterations >= 10, "Empty test - no iterations were run"

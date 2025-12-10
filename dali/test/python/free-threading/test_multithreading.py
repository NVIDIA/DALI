import os
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn

test_data_root = os.environ["DALI_EXTRA_PATH"]
image_file = os.path.join(test_data_root, "db", "single", "jpeg", "100", "swan-3584559_640.jpg")

batch_size = 12
prefetch_queue_depth = 3
num_dali_threads = 8
num_workers = 100
test_input = [np.fromfile(image_file, dtype=np.uint8)] * batch_size


@pipeline_def(
    batch_size=batch_size,
    num_threads=num_dali_threads,
    device_id=0,
    prefetch_queue_depth=prefetch_queue_depth,
)
def dali_pipeline():
    enc = fn.external_source(name="INPUT")
    img = fn.decoders.image(enc, device="mixed")
    img = fn.resize(img, size=(224, 224))
    img = fn.crop_mirror_normalize(
        img,
        crop=(224, 224),
        dtype=types.FLOAT,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return img


def run_thread(dali_pipeline, barrier, thread_id, results):
    results[thread_id] = []
    barrier.wait()
    dali_pipeline.build()
    for _ in range(prefetch_queue_depth * 2):  # Feed twice as many as the queue depth
        dali_pipeline.feed_input("INPUT", test_input)
    barrier.wait()
    for _ in range(prefetch_queue_depth):
        outputs = dali_pipeline.run()
        results[thread_id].append(outputs[0].as_cpu().as_array())


def test_parallel_pipelines(num_workers=num_workers):
    """Test running two separate DALI pipelines in different threads."""
    results = {}
    barrier = threading.Barrier(num_workers)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(run_thread, dali_pipeline(), barrier, i, results)
            for i in range(num_workers)
        ]
        for future in futures:
            future.result()

    assert len(results) == num_workers
    ref = results[0][0]
    for worker_id in range(num_workers):
        assert len(results[worker_id]) == prefetch_queue_depth
        for i in range(prefetch_queue_depth):
            assert np.allclose(results[worker_id][i], ref)

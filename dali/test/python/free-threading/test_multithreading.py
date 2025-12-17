import os
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from typing import Any, Dict, List

test_data_root = os.environ["DALI_EXTRA_PATH"]
image_file = os.path.join(test_data_root, "db", "single", "jpeg", "100", "swan-3584559_640.jpg")

batch_size = 12
prefetch_queue_depth = 3
num_dali_threads = 8
num_workers = 100
test_input = [np.fromfile(image_file, dtype=np.uint8)] * batch_size


class Result:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result_by_thread: Dict[Any, List[Any]] = {}

    def set(self, thread_id: Any, value: Any) -> None:
        """Record one result for thread_id (append)."""
        with self._lock:
            self._result_by_thread.setdefault(thread_id, []).append(value)

    def all_equal(self) -> bool:
        with self._lock:
            lists = [list(vs) for vs in self._result_by_thread.values()]

        # find a reference element
        ref = None
        for vs in lists:
            if vs:
                ref = vs[0]
                break
        assert ref is not None

        for vs in lists:
            for v in vs:
                if not bool(np.all(np.equal(v, ref))):
                    return False
        return True


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
    barrier.wait()
    dali_pipeline.build()
    for _ in range(prefetch_queue_depth * 2):  # Feed twice as many as the queue depth
        dali_pipeline.feed_input("INPUT", test_input)
    barrier.wait()
    for _ in range(prefetch_queue_depth):
        outputs = dali_pipeline.run()
        results.set(thread_id, outputs[0].as_cpu().as_array())


def test_parallel_pipelines():
    """Test running multiple separate DALI pipelines in different threads."""
    print(f"Sanity check: PYTHON_GIL={os.environ.get('PYTHON_GIL')}")
    results = Result()
    barrier = threading.Barrier(num_workers)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(run_thread, dali_pipeline(), barrier, i, results)
            for i in range(num_workers)
        ]
        for future in futures:
            future.result()

    assert results.all_equal()

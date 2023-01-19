import nvidia.dali as dali
import nvidia.dali.fn as fn
from test_utils import get_dali_extra_path
import os

path = os.path.join(get_dali_extra_path, "db", "single", "jpeg")

@dali.pipeline_def(batch_size=1, num_threads=4, device_id=0)
def median_pipe():
    file, _ = fn.readers.file(path)
    img = fn.decoders.image(file, device="mixed")
    blurred = fn.experimental.median_blur(img, window_size=[3,3])
    return img, blurred

def main():
    pass

if __name__ == "__main__":
    main()

from nvidia.dali import pipeline_def, fn, types
import numpy as np
import os
from nose2.tools import params
from nose_utils import attr, assert_raises


def read_file(path):
    return np.fromfile(path, dtype=np.uint8)


def read_filepath(path):
    return np.frombuffer(path.encode(), dtype=np.int8)


def read_image(path):
    from PIL import Image

    return np.array(Image.open(path))


dali_extra = os.environ["DALI_EXTRA_PATH"]
jpeg = os.path.join(dali_extra, "db", "single", "jpeg")
jpeg_113 = os.path.join(jpeg, "113")
test_files = [
    os.path.join(jpeg_113, f)
    for f in ["snail-4291306_1280.jpg", "snail-4345504_1280.jpg", "snail-4368154_1280.jpg"]
]
test_input_filenames = [read_filepath(fname) for fname in test_files]


@pipeline_def
def image_pipe(dali_device="gpu", random_pipe=True):
    filepaths = fn.external_source(name="images", no_copy=True)
    jpegs = fn.io.file.read(filepaths)
    decoder_device = "mixed" if dali_device == "gpu" else "cpu"

    if random_pipe:
        images = fn.decoders.image_random_crop(
            jpegs,
            device=decoder_device,
            output_type=types.RGB,
            random_aspect_ratio=[0.75, 4.0 / 3.0],
            random_area=[0.08, 1.0],
        )
    else:
        images = fn.decoders.image(
            jpegs,
            device=decoder_device,
            output_type=types.RGB,
        )

    images = fn.resize(
        images,
        size=[224, 224],
        interp_type=types.INTERP_LINEAR,
        antialias=False,
    )
    mirror = fn.random.coin_flip(probability=0.5) if random_pipe else False
    output = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=mirror,
    )
    return output


@attr("pytorch")
@params(("cpu",), ("gpu",))
def test_dali_proxy_torch_data_loader(device, debug=False):
    # Shows how DALI proxy is used in practice with a PyTorch data loader

    from nvidia.dali.plugin.pytorch.experimental import proxy as dali_proxy
    import torchvision.datasets as datasets
    from torch.utils import data as torchdata
    from nvidia.dali.tensors import TensorCPU, TensorGPU

    batch_size = 4
    num_threads = 3
    device_id = 0
    nworkers = 4
    pipe = image_pipe(
        dali_device=device,
        random_pipe=False,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=2 + nworkers,
    )

    pipe_ref = image_pipe(
        dali_device=device,
        random_pipe=False,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=1,
    )
    pipe_ref.build()

    # Run the server (it also cleans up on scope exit)
    with dali_proxy.DALIServer(pipe) as dali_server:

        dataset = datasets.ImageFolder(jpeg, transform=dali_server.proxy, loader=read_filepath)
        dataset_ref = datasets.ImageFolder(jpeg, transform=lambda x: x.copy(), loader=read_filepath)

        loader = dali_proxy.DataLoader(
            dali_server,
            dataset,
            batch_size=batch_size,
            num_workers=nworkers,
            drop_last=True,
        )

        def ref_collate_fn(batch):
            filepaths, labels = zip(*batch)  # Separate the inputs and labels
            # Just return the batch as they are, a list of individual tensors
            return filepaths, labels

        loader_ref = torchdata.dataloader.DataLoader(
            dataset_ref,
            batch_size=batch_size,
            num_workers=1,
            collate_fn=ref_collate_fn,
            shuffle=False,
        )

        for _, ((data, target), (ref_filepaths, ref_target)) in enumerate(zip(loader, loader_ref)):
            np.testing.assert_equal([batch_size, 3, 224, 224], data.shape)
            np.testing.assert_equal(
                [
                    batch_size,
                ],
                target.shape,
            )
            np.testing.assert_array_equal(target, ref_target)
            ref_filepaths_tensors = [TensorCPU(obj) for obj in ref_filepaths]
            pipe_ref.feed_input("images", ref_filepaths_tensors)
            (ref_data,) = pipe_ref.run()
            for sample_idx in range(batch_size):
                ref_tensor = ref_data[sample_idx]
                if isinstance(ref_tensor, TensorGPU):
                    ref_tensor = ref_tensor.as_cpu()
                np.testing.assert_array_equal(ref_tensor, data[sample_idx].cpu())


@attr("pytorch")
def test_dali_proxy_torch_data_loader_manual_integration(device="gpu", debug=False):
    # Shows how to integrate with DALI proxy manually with an existing data loader

    from nvidia.dali.plugin.pytorch.experimental import proxy as dali_proxy
    import torch
    from torch.utils import data as torchdata
    from PIL import Image

    batch_size = 4
    num_threads = 3
    device_id = 0
    nworkers = 4

    class CustomDatasetOnlyDecoding(torchdata.Dataset):
        def __init__(self, folder_path):
            self.folder_path = folder_path
            self.image_files = self._find_images_in_folder(folder_path)

        def _find_images_in_folder(self, folder_path):
            """
            Recursively find all image files in the folder and its subdirectories.
            """
            image_files = []

            # Walk through all directories and subdirectories
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                        image_files.append(os.path.join(root, file))

            return image_files

        def __len__(self):
            """Returns the number of images in the folder."""
            return len(self.image_files)

        def __getitem__(self, idx):
            img_name = self.image_files[idx]
            img_path = os.path.join(self.folder_path, img_name)
            img = Image.open(img_path).convert("RGB")  # Convert image to RGB (3 channels)
            other = 1
            return np.array(img), other

    @pipeline_def
    def processing_pipe(dali_device="gpu"):
        images = fn.external_source(name="images", no_copy=True)
        rng = fn.random.coin_flip(probability=0.5)
        if dali_device == "gpu":
            images = images.gpu()
        images = fn.resize(
            images,
            device=dali_device,
            size=[224, 224],
            interp_type=types.INTERP_LINEAR,
            antialias=False,
        )
        images = fn.flip(images, horizontal=rng)
        output = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        return output

    plain_dataset = CustomDatasetOnlyDecoding(jpeg)
    pipe = processing_pipe(
        device,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=2 + nworkers,
    )

    class CustomDatasetDALI(torchdata.Dataset):
        def __init__(self, orig_dataset, dali_proxy):
            self.dataset = orig_dataset
            self.dali_proxy = dali_proxy

        def __len__(self):
            return self.dataset.__len__()

        def __getitem__(self, idx):
            img, other = self.dataset.__getitem__(idx)
            img2 = self.dali_proxy(img)
            return img2, other

    # This is just for educational purposes. It is recommended to rely
    # default_collate_fn_map, which is updated to handle DALIProcessedSampleRef
    def custom_collate_fn(batch):
        images, labels = zip(*batch)
        return dali_proxy._collate_dali_processed_sample_ref_fn(images), torch.tensor(
            labels, dtype=torch.long
        )

    # Run the server (it also cleans up on scope exit)
    with dali_proxy.DALIServer(pipe) as dali_server:

        dataset = CustomDatasetDALI(plain_dataset, dali_server.proxy)

        loader = torchdata.dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=nworkers,
            drop_last=True,
            collate_fn=custom_collate_fn,
        )

        assert len(loader) > 0

        for next_input, next_target in loader:
            assert isinstance(next_input, dali_proxy.DALIPipelineRunRef)
            next_input = dali_server.produce_data(next_input)
            assert isinstance(next_input, torch.Tensor)
            np.testing.assert_equal([batch_size, 3, 224, 224], next_input.shape)
            np.testing.assert_equal(
                [
                    batch_size,
                ],
                next_target.shape,
            )


@attr("pytorch")
@params((False,), (True,))
def test_dali_proxy_deterministic(deterministic, debug=False):
    # Shows how DALI proxy can be configured for deterministic results
    from nvidia.dali.plugin.pytorch.experimental import proxy as dali_proxy
    import torchvision.datasets as datasets
    import torch

    # Use a high number of iterations for non-deterministic tests, even though
    # we stop the test once we get different results (usually in the first iteration).
    # For deterministic tests, we check that all runs produce the same results.
    niterations = 1 if deterministic else 10
    num_workers = 8
    seed0 = 123456
    seed1 = 5555464
    seed2 = 775653

    outputs = []
    for i in range(niterations):
        pipe = image_pipe(
            random_pipe=True,
            dali_device="gpu",
            batch_size=1,
            num_threads=1,
            device_id=0,
            seed=seed0,
            prefetch_queue_depth=1,
        )
        outputs_i = []
        torch.manual_seed(seed2)
        with dali_proxy.DALIServer(pipe, deterministic=deterministic) as dali_server:
            dataset = datasets.ImageFolder(jpeg, transform=dali_server.proxy, loader=read_filepath)
            # many workers so that we introduce a lot of variability in the order of arrival
            loader = dali_proxy.DataLoader(
                dali_server,
                dataset,
                batch_size=1,
                num_workers=num_workers,
                shuffle=True,
                worker_init_fn=lambda worker_id: np.random.seed(seed1 + worker_id),
            )
            outputs_i = []
            for _ in range(num_workers):
                for data, _ in loader:
                    outputs_i.append(data.cpu())
                    break
        outputs.append(outputs_i)

        if i > 0:
            if deterministic:
                for k in range(num_workers):
                    assert np.array_equal(outputs[i][k], outputs[0][k])
            else:
                for k in range(num_workers):
                    if not np.array_equal(outputs[i][k], outputs[0][k]):
                        return  # OK

        pipe._shutdown()
        del pipe

    if not deterministic:
        assert False, "we got exactly the same results in all runs"


@attr("pytorch")
def test_dali_proxy_error_propagation():
    from nvidia.dali.plugin.pytorch.experimental import proxy as dali_proxy
    import torchvision.datasets as datasets

    batch_size = 4
    num_threads = 3
    device_id = 0
    nworkers = 2

    @pipeline_def
    def pipe_with_error():
        images = fn.external_source(name="images", no_copy=True)
        error_anchor = types.Constant(np.array([-10], dtype=np.float32))
        return fn.crop(
            images, crop=(224, 224), crop_pos_x=error_anchor, out_of_bounds_policy="error"
        )

    pipe = pipe_with_error(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=3,
    )
    with dali_proxy.DALIServer(pipe) as dali_server:

        dataset = datasets.ImageFolder(jpeg, transform=dali_server.proxy, loader=read_image)
        loader = dali_proxy.DataLoader(
            dali_server,
            dataset,
            batch_size=batch_size,
            num_workers=nworkers,
        )

        err_msg = "Critical error in pipeline:*Anchor for dimension 1*is out of range*"
        with assert_raises(RuntimeError, glob=err_msg):
            next(iter(loader))

    # For some reason if we don't do this in this test, we see some ignored exception
    # messages in the next test
    pipe._shutdown()
    del pipe
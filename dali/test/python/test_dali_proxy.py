from nvidia.dali import pipeline_def, fn, types
import numpy as np
import os
import threading
from nose2.tools import params
from nose_utils import attr


def read_file(path):
    return np.fromfile(path, dtype=np.uint8)


def read_filepath(path):
    return np.frombuffer(path.encode(), dtype=np.int8)


dali_extra = os.environ["DALI_EXTRA_PATH"]
jpeg = os.path.join(dali_extra, "db", "single", "jpeg")
jpeg_113 = os.path.join(jpeg, "113")
test_files = [
    os.path.join(jpeg_113, f)
    for f in ["snail-4291306_1280.jpg", "snail-4345504_1280.jpg", "snail-4368154_1280.jpg"]
]
test_input_filenames = [read_filepath(fname) for fname in test_files]


@pipeline_def(exec_dynamic=True)
def pipe_decoder(device):
    filepaths = fn.external_source(name="images", no_copy=True, blocking=True)
    images = fn.io.file.read(filepaths)
    decoder_device = "mixed" if device == "gpu" else "cpu"
    images = fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
    images = fn.crop(images, crop=(224, 224))
    return images


@attr("pytorch")
@params(("cpu",), ("gpu",))
def test_dali_proxy_demo_basic_communication(device, debug=False):
    # This is a test that is meant to illustrate how the inter-process or inter-thread communication
    # works when using DALI proxy. The code here is not really meant to be run like this by a user.
    # A better example for user API is `test_dali_proxy_torch_data_loader`

    import torch
    from nvidia.dali.plugin.pytorch import proxy as dali_proxy

    threads = []
    batch_size = 4
    num_threads = 3
    device_id = 0
    nworkers = 3
    niter = 5
    pipe = pipe_decoder(
        device,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=2 + nworkers,
    )

    # Runs the server (and clean up on exit)
    with dali_proxy.DALIServer(pipe) as dali_server:

        # Creating a bunch of worker threads that call the proxy callabe sample by sample
        # and call the collate function directly, which will trigger a pipeline run on the server
        for _ in range(nworkers):

            def thread_fn(proxy_pipe_call):
                for _ in range(niter):
                    # The proxy call is run per sample
                    pipe_run_refs = [
                        proxy_pipe_call(test_input_filenames[i % len(test_input_filenames)])
                        for i in range(batch_size)
                    ]
                    # this forms a batch and sends it to DALI
                    dali_proxy._collate_pipeline_run_ref_fn(pipe_run_refs)

            thread = threading.Thread(target=thread_fn, args=(dali_server.proxy,))
            threads.append(thread)
            thread.start()

        collected_data_info = {}
        for thread in threads:
            collected_data_info[thread.ident] = [None for _ in range(niter)]

        # On the main thread, we can query the server for new outputs
        for _ in range(nworkers * niter):
            info, outputs = dali_server.next_outputs()
            worker_id = info[0]
            data_idx = info[1]
            if debug:
                print(f"worker_id={worker_id}, data_idx={data_idx}, data_shape={outputs[0].shape}")
            assert worker_id in collected_data_info
            collected_data_info[worker_id][data_idx] = outputs
            assert len(outputs) == 1
            np.testing.assert_equal([batch_size, 224, 224, 3], outputs[0].shape)

        for thread in threads:
            thread.join()

        # Make sure we received all the data we expected
        for thread in threads:
            for data_idx in range(niter):
                (data,) = collected_data_info[thread.ident][data_idx]
                assert data is not None
                expected_device = (
                    torch.device(type="cuda", index=device_id)
                    if device == "gpu"
                    else torch.device("cpu")
                )
                np.testing.assert_equal(expected_device, data.device)


@pipeline_def
def rn50_train_pipe(dali_device="gpu"):
    rng = fn.random.coin_flip(probability=0.5)

    filepaths = fn.external_source(name="images", no_copy=True, blocking=True)
    jpegs = fn.io.file.read(filepaths)
    if dali_device == "gpu":
        decoder_device = "mixed"
        resize_device = "gpu"
    else:
        decoder_device = "cpu"
        resize_device = "cpu"

    images = fn.decoders.image_random_crop(
        jpegs,
        device=decoder_device,
        output_type=types.RGB,
        random_aspect_ratio=[0.75, 4.0 / 3.0],
        random_area=[0.08, 1.0],
    )

    images = fn.resize(
        images,
        device=resize_device,
        size=[224, 224],
        interp_type=types.INTERP_LINEAR,
        antialias=False,
    )

    # Make sure that from this point we are processing on GPU regardless of dali_device parameter
    images = images.gpu()

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


@attr("pytorch")
@params(("cpu",), ("gpu",))
def test_dali_proxy_torch_data_loader(device, debug=False):
    # Shows how DALI proxy is used in practice with a PyTorch data loader

    from nvidia.dali.plugin.pytorch import proxy as dali_proxy
    import torchvision.datasets as datasets

    batch_size = 4
    num_threads = 3
    device_id = 0
    nworkers = 4
    pipe = rn50_train_pipe(
        device,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=2 + nworkers,
    )

    # Run the server (it also cleans up on scope exit)
    with dali_proxy.DALIServer(pipe) as dali_server:

        dataset = datasets.ImageFolder(jpeg, transform=dali_server.proxy, loader=read_filepath)

        loader = dali_proxy.DataLoader(
            dali_server,
            dataset,
            batch_size=batch_size,
            num_workers=nworkers,
            drop_last=True,
        )

        for next_input, next_target in loader:
            np.testing.assert_equal([batch_size, 3, 224, 224], next_input.shape)
            np.testing.assert_equal(
                [
                    batch_size,
                ],
                next_target.shape,
            )


@attr("pytorch")
@params(("cpu",), ("gpu",))
def test_dali_proxy_torch_data_loader_manual_integration(device, debug=False):
    # Shows how to integrate with DALI proxy manually with an existing data loader

    from nvidia.dali.plugin.pytorch import proxy as dali_proxy
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
        images = fn.external_source(name="images", no_copy=True, blocking=True)
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
    # default_collate_fn_map, which is updated to handle PipelineRunRef
    def custom_collate_fn(batch):
        images, labels = zip(*batch)
        return dali_proxy._collate_pipeline_run_ref_fn(images), torch.tensor(
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
            assert isinstance(next_input, dali_proxy.DALIPipelineOutputRef)
            next_input = dali_server.get_outputs(next_input)
            assert isinstance(next_input, torch.Tensor)
            np.testing.assert_equal([batch_size, 3, 224, 224], next_input.shape)
            np.testing.assert_equal(
                [
                    batch_size,
                ],
                next_target.shape,
            )

# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os

os.environ["KMP_AFFINITY"] = (
    "disabled"  # We need to do this before importing anything else as a workaround for this bug: https://github.com/pytorch/pytorch/issues/28389
)

import argparse
import random
from contextlib import nullcontext

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import image_classification.logger as log

from image_classification.smoothing import LabelSmoothing
from image_classification.mixup import NLLMultiLabelSmooth, MixUpWrapper
from image_classification.dataloaders import *
from image_classification.training import *
from image_classification.utils import *
from image_classification.models import efficientnet_b0
from image_classification.optimizers import (
    get_optimizer,
    lr_cosine_policy,
    lr_linear_policy,
    lr_step_policy,
)
from image_classification.gpu_affinity import set_affinity, AffinityMode
import dllogger


def available_models():
    models = {m.name: m for m in [efficientnet_b0]}
    return models


def add_parser_arguments(parser, skip_arch=False):
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--data-backend",
        metavar="BACKEND",
        default="dali",
        choices=DATA_BACKEND_CHOICES,
        help="data backend: "
        + " | ".join(DATA_BACKEND_CHOICES)
        + " (default: dali)",
    )
    parser.add_argument(
        "--interpolation",
        metavar="INTERPOLATION",
        default="bicubic",
        help="interpolation type for resizing images: bilinear, bicubic or triangular (DALI only)",
    )
    if not skip_arch:
        model_names = available_models().keys()
        parser.add_argument(
            "--arch",
            "-a",
            metavar="ARCH",
            default="efficientnet-b0",
            choices=model_names,
            help="model architecture: "
            + " | ".join(model_names)
            + " (default: efficientnet-b0)",
        )

    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help=(
            "number of data loading workers (default: 4)."
            " The number of workers for PyTorch loader is doubled."
        ),
    )
    parser.add_argument(
        "--prefetch",
        default=4,
        type=int,
        metavar="N",
        help="number of samples prefetched by each loader (PyTorch only)",
    )
    parser.add_argument(
        "--dali-device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu"],
        help=(
            "The placement of DALI decode and random resized crop operations (default: gpu)"
        ),
    )
    parser.add_argument(
        "--epochs",
        default=400,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--run-epochs",
        default=-1,
        type=int,
        metavar="N",
        help="run only N epochs, used for checkpointing runs",
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=-1,
        type=int,
        metavar="N",
        help="early stopping after N epochs without validation accuracy improving",
    )
    parser.add_argument(
        "--image-size", default=224, type=int, help="resolution of image"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        metavar="N",
        help="mini-batch size (default: 64) per gpu",
    )

    parser.add_argument(
        "--optimizer-batch-size",
        default=4096,
        type=int,
        metavar="N",
        help="size of a total batch size, for simulating bigger batches using gradient accumulation",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.08,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr-schedule",
        default="cosine",
        type=str,
        metavar="SCHEDULE",
        choices=["step", "linear", "cosine"],
        help="Type of LR schedule: {}, {}, {}".format(
            "step", "linear", "cosine"
        ),
    )

    parser.add_argument("--end-lr", default=0, type=float)

    parser.add_argument(
        "--warmup",
        default=16,
        type=int,
        metavar="E",
        help="number of warmup epochs",
    )

    parser.add_argument(
        "--label-smoothing",
        default=0.1,
        type=float,
        metavar="S",
        help="label smoothing",
    )
    parser.add_argument(
        "--mixup", default=0.2, type=float, metavar="ALPHA", help="mixup alpha"
    )
    parser.add_argument(
        "--optimizer", default="rmsprop", type=str, choices=("sgd", "rmsprop")
    )

    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-05,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-5)",
    )
    parser.add_argument(
        "--bn-weight-decay",
        action="store_true",
        help="use weight_decay on batch normalization learnable parameters, (default: false)",
    )
    parser.add_argument(
        "--rmsprop-alpha",
        default=0.9,
        type=float,
        help="value of alpha parameter in rmsprop optimizer (default: 0.9)",
    )
    parser.add_argument(
        "--rmsprop-eps",
        default=0.01,
        type=float,
        help="value of eps parameter in rmsprop optimizer (default: 0.01)",
    )

    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="use nesterov momentum, (default: false)",
    )

    parser.add_argument(
        "--print-freq",
        "-p",
        default=100,
        type=int,
        metavar="N",
        help="print frequency (default: 100)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--static-loss-scale",
        type=float,
        default=1,
        help="Static loss scale, positive power of 2 values can improve amp convergence.",
    )
    parser.add_argument(
        "--prof",
        type=int,
        default=-1,
        metavar="N",
        help="Run only N iterations",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run model AMP (automatic mixed precision) mode.",
    )

    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="random seed used for numpy and pytorch",
    )

    parser.add_argument(
        "--gather-checkpoints",
        default="0",
        type=int,
        help=(
            "Gather N last checkpoints throughout the training,"
            " without this flag only best and last checkpoints will be stored. "
            "Use -1 for all checkpoints"
        ),
    )

    parser.add_argument(
        "--report-file",
        default="experiment_report.json",
        type=str,
        help="file in which to store JSON experiment report",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate checkpoint/model"
    )
    parser.add_argument(
        "--training-only", action="store_true", help="do not evaluate"
    )

    parser.add_argument(
        "--no-checkpoints",
        action="store_false",
        dest="save_checkpoints",
        help="do not store any checkpoints, useful for benchmarking",
    )
    parser.add_argument(
        "--jit",
        type=str,
        default="no",
        choices=["no", "script"],
        help="no -> do not use torch.jit; script -> use torch.jit.script",
    )

    parser.add_argument(
        "--checkpoint-filename", default="checkpoint.pth.tar", type=str
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default="./",
        metavar="DIR",
        help="path to directory where checkpoints will be stored",
    )
    parser.add_argument(
        "--memory-format",
        type=str,
        default="nhwc",
        choices=["nchw", "nhwc"],
        help="memory layout, nchw or nhwc",
    )
    parser.add_argument("--use-ema", default=None, type=float, help="use EMA")
    parser.add_argument(
        "--automatic-augmentation",
        type=str,
        default="autoaugment",
        choices=["disabled", "autoaugment", "trivialaugment"],
        help="Automatic augmentation method, trivialaugment is supported only for DALI data backend",
    )

    parser.add_argument(
        "--gpu-affinity",
        type=str,
        default="socket_unique_contiguous",
        required=False,
        choices=[am.name for am in AffinityMode],
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        required=False,
    )


def prepare_for_training(args, model_args, model_arch):
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    affinity = set_affinity(args.gpu, mode=args.gpu_affinity)
    print(f"Training process {args.local_rank} affinity: {affinity}")

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            # Worker process should inherit its affinity from parent
            affinity = os.sched_getaffinity(0)
            print(
                f"Process {args.local_rank} Worker {id} set affinity to: {affinity}"
            )

            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)

    else:

        def _worker_init_fn(id):
            # Worker process should inherit its affinity from parent
            affinity = os.sched_getaffinity(0)
            print(
                f"Process {args.local_rank} Worker {id} set affinity to: {affinity}"
            )

    if args.static_loss_scale != 1.0:
        if not args.amp:
            print(
                "Warning: if --amp is not used, static_loss_scale will be ignored."
            )

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print(
                "Warning: simulated batch size {} is not divisible by actual batch size {}".format(
                    args.optimizer_batch_size, tbs
                )
            )
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print("BSM: {}".format(batch_size_multiplier))

    start_epoch = 0
    best_prec1 = 0
    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume,
                map_location=lambda storage, loc: storage.cuda(args.gpu),
            )
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model_state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer"]
            if "state_dict_ema" in checkpoint:
                model_state_ema = checkpoint["state_dict_ema"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            if start_epoch >= args.epochs:
                print(
                    f"Launched training for {args.epochs}, checkpoint already run {start_epoch}"
                )
                exit(1)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            model_state_ema = None
            optimizer_state = None
    else:
        model_state = None
        model_state_ema = None
        optimizer_state = None

    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)

    memory_format = (
        torch.channels_last
        if args.memory_format == "nhwc"
        else torch.contiguous_format
    )
    model = model_arch(
        **{
            k: (
                v
                if k != "pretrained"
                else v and (not args.distributed or dist.get_rank() == 0)
            )
            for k, v in model_args.__dict__.items()
        }
    )

    image_size = (
        args.image_size
        if args.image_size is not None
        else model.arch.default_image_size
    )

    scaler = torch.cuda.amp.GradScaler(
        init_scale=args.static_loss_scale,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=100,
        enabled=args.amp,
    )

    executor = Executor(
        model,
        loss(),
        cuda=True,
        memory_format=memory_format,
        amp=args.amp,
        scaler=scaler,
        divide_loss=batch_size_multiplier,
        ts_script=args.jit == "script",
    )

    # Create data loaders and optimizers as needed
    if args.data_backend == "pytorch":
        args.workers = args.workers * 2
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == "pytorch_optimized":
        args.workers = args.workers * 2
        get_train_loader = get_pytorch_optimized_train_loader
        get_val_loader = get_pytorch_optimize_val_loader
    elif args.data_backend == "dali":
        get_train_loader = get_dali_train_loader(dali_device=args.dali_device)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "dali_proxy":
        get_train_loader = get_dali_proxy_train_loader(
            dali_device=args.dali_device
        )
        get_val_loader = get_dali_proxy_val_loader()
    elif args.data_backend == "synthetic":
        get_val_loader = get_synthetic_loader
        get_train_loader = get_synthetic_loader
    else:
        print("Bad databackend picked")
        exit(1)

    train_loader, train_loader_len = get_train_loader(
        args.data,
        image_size,
        args.batch_size,
        model_args.num_classes,
        args.mixup > 0.0,
        interpolation=args.interpolation,
        augmentation=args.automatic_augmentation,
        start_epoch=start_epoch,
        workers=args.workers,
        _worker_init_fn=_worker_init_fn,
        memory_format=memory_format,
        prefetch_factor=args.prefetch,
    )
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)

    val_loader, val_loader_len = get_val_loader(
        args.data,
        image_size,
        args.batch_size,
        model_args.num_classes,
        False,
        interpolation=args.interpolation,
        workers=args.workers,
        _worker_init_fn=_worker_init_fn,
        memory_format=memory_format,
        prefetch_factor=args.prefetch,
    )

    if (
        not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    ):
        logger = log.Logger(
            args.print_freq,
            [
                dllogger.StdOutBackend(
                    dllogger.Verbosity.DEFAULT, step_format=log.format_step
                ),
                dllogger.JSONStreamBackend(
                    dllogger.Verbosity.VERBOSE,
                    os.path.join(args.workspace, args.report_file),
                ),
            ],
            start_epoch=start_epoch - 1,
        )

    else:
        logger = log.Logger(args.print_freq, [], start_epoch=start_epoch - 1)

    logger.log_parameter(args.__dict__, verbosity=dllogger.Verbosity.DEFAULT)
    logger.log_parameter(
        {f"model.{k}": v for k, v in model_args.__dict__.items()},
        verbosity=dllogger.Verbosity.DEFAULT,
    )

    optimizer = get_optimizer(
        list(executor.model.named_parameters()),
        args.lr,
        args=args,
        state=optimizer_state,
    )

    if args.lr_schedule == "step":
        lr_policy = lr_step_policy(args.lr, [30, 60, 80], 0.1, args.warmup)
    elif args.lr_schedule == "cosine":
        lr_policy = lr_cosine_policy(
            args.lr, args.warmup, args.epochs, end_lr=args.end_lr
        )
    elif args.lr_schedule == "linear":
        lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs)

    if args.distributed:
        executor.distributed(args.gpu)

    if model_state is not None:
        executor.model.load_state_dict(model_state)

    trainer = Trainer(
        executor,
        optimizer,
        grad_acc_steps=batch_size_multiplier,
        ema=args.use_ema,
    )

    if (args.use_ema is not None) and (model_state_ema is not None):
        trainer.ema_executor.model.load_state_dict(model_state_ema)

    return (
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger,
        start_epoch,
        best_prec1,
    )


def main(args, model_args, model_arch):
    exp_start_time = time.time()

    (
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger,
        start_epoch,
        best_prec1,
    ) = prepare_for_training(args, model_args, model_arch)

    def get_ctx(loader):
        """
        Get context from a dataloader object. This is a utility so that we can run with the
        same code for DALI iterators, PyTorch dataloader, or DALI proxy dataloader.
        """
        if isinstance(loader, dali_proxy.DataLoader):
            return loader.dali_server
        if hasattr(loader, "dataloader"):
            return get_ctx(loader.dataloader)
        return nullcontext()

    with get_ctx(train_loader), get_ctx(val_loader):
        train_loop(
            trainer,
            lr_policy,
            train_loader,
            train_loader_len,
            val_loader,
            logger,
            start_epoch=start_epoch,
            end_epoch=(
                min((start_epoch + args.run_epochs), args.epochs)
                if args.run_epochs != -1
                else args.epochs
            ),
            early_stopping_patience=args.early_stopping_patience,
            best_prec1=best_prec1,
            prof=args.prof,
            skip_training=args.evaluate,
            skip_validation=args.training_only,
            save_checkpoints=args.save_checkpoints and not args.evaluate,
            checkpoint_dir=args.workspace,
            checkpoint_filename=args.checkpoint_filename,
            keep_last_n_checkpoints=args.gather_checkpoints,
            topk=args.topk,
        )
    exp_duration = time.time() - exp_start_time
    if (
        not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    ):
        logger.end()
    print("Experiment ended")


if __name__ == "__main__":
    epilog = [
        "Based on the architecture picked by --arch flag, you may use the following options:\n"
    ]
    for model, ep in available_models().items():
        model_help = "\n".join(ep.parser().format_help().split("\n")[2:])
        epilog.append(model_help)
    parser = argparse.ArgumentParser(
        description="PyTorch EfficientNet Training",
        epilog="\n".join(epilog),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_parser_arguments(parser)

    args, rest = parser.parse_known_args()

    model_arch = available_models()[args.arch]
    model_args, rest = model_arch.parser().parse_known_args(rest)
    print(model_args)

    assert len(rest) == 0, f"Unknown args passed: {rest}"

    cudnn.benchmark = True

    main(args, model_args, model_arch)

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.utils import dboxes300_coco, COCODetection, SSDTransformer
from src.coco import COCO
from src.coco_pipeline import create_coco_pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


def set_seeds(args):
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda')

    if args.distributed:
        args.seed = broadcast_seeds(args.seed, device)
        local_seed = (args.seed + torch.distributed.get_rank()) % 2**32
        local_rank = torch.distributed.get_rank()
    else:
        local_seed = args.seed % 2**32
        local_rank = 0

    print("Rank", local_rank, "using seed = {}".format(local_seed))

    torch.manual_seed(local_seed)
    np.random.seed(seed=local_seed)

    return local_seed


def broadcast_seeds(seed, device):
    if torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor([seed]).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seed = seeds_tensor.item()
    return seed


def get_train_pytorch_loader(args, num_workers, default_boxes):
    dataset = COCODetection(
        args.train_coco_root,
        args.train_annotate,
        SSDTransformer(default_boxes, args, (300, 300), val=False))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers)

    return train_dataloader


def get_train_dali_loader(args, default_boxes, local_seed):
    train_pipe = create_coco_pipeline(
        default_boxes,
        args,
        batch_size=args.batch_size,
        num_threads=args.num_workers,
        device_id=args.local_rank,
        seed=local_seed)

    train_loader = DALIGenericIterator(
        train_pipe,
        ["images", "boxes", "labels"],
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.FILL)

    return train_loader


def get_train_loader(args, dboxes):
    args.train_annotate = os.path.join(
        args.data, "annotations/instances_train2017.json")
    args.train_coco_root = os.path.join(args.data, "train2017")

    local_seed = set_seeds(args)

    if args.data_pipeline == 'no_dali':
        return get_train_pytorch_loader(args, args.num_workers, dboxes)
    elif args.data_pipeline == 'dali':
        return get_train_dali_loader(args, dboxes, local_seed)


def get_val_dataset(args):
    dboxes = dboxes300_coco()
    val_trans = SSDTransformer(dboxes, args,(300, 300), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")

    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    return val_coco


def get_val_dataloader(args):
    dataset = get_val_dataset(args)
    inv_map = {v: k for k, v in dataset.label_map.items()}

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        val_sampler = None

    val_dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,  # Note: distributed sampler is shuffled :(
        sampler=val_sampler,
        num_workers=args.num_workers)

    return val_dataloader, inv_map


def get_coco_ground_truth(args):
    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    cocoGt = COCO(annotation_file=val_annotate)
    return cocoGt

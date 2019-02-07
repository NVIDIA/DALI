import os

import torch
from torch.utils.data import DataLoader

from src.utils import dboxes300_coco, COCODetection, SSDTransformer
from src.coco import COCO
from src.coco_pipeline import COCOPipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

def get_train_dataset(args, transform):
    train_coco = COCODetection(
        args.train_coco_root, 
        args.train_annotate, 
        transform)

    return train_coco

def get_train_loader(dataset, args, num_workers):
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
    train_pipe = COCOPipeline(
        default_boxes, 
        args,
        seed=local_seed)

    train_loader = DALIGenericIterator(
        train_pipe, 
        ["images", "boxes", "labels"], 
        118287 / args.N_gpu, 
        stop_at_epoch=False)

    return train_loader

def get_val_dataset(args):
    dboxes = dboxes300_coco()
    val_trans = SSDTransformer(dboxes, args,(300, 300), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")

    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    return val_coco

def get_val_dataloader(dataset, args):
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
        
    return val_dataloader

def get_coco_ground_truth(args):
    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    cocoGt = COCO(annotation_file=val_annotate)
    return cocoGt

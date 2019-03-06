import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from src.data import *
from src.evaluate import evaluate
from src.model import SSD300, Loss
from src.utils import Encoder, SSDTransformer, dboxes300_coco

# Apex imports
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError(
        "Please install APEX from https://github.com/nvidia/apex")


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


def warmup(optim, warmup_iters, iteration, base_lr):
    if iteration < warmup_iters:
        new_lr = 1. * base_lr / warmup_iters * iteration
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr


def tencent_trick(model):
    """
    Divide parameters into 2 groups.
    First group is BNs and all biases.
    Second group is the remaining model's parameters.
    Weight decay will be disabled in first group (aka tencent trick).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args):
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        args.N_gpu = torch.distributed.get_world_size()
    else:
        args.N_gpu = 1

    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)

    val_dataset = get_val_dataset(args)
    val_dataloader = get_val_dataloader(val_dataset, args)

    ssd300 = SSD300(len(cocoGt.cats) + 1)
    args.learning_rate = args.learning_rate * \
        args.N_gpu * (args.batch_size / 32)
    iteration = 0
    loss_func = Loss(dboxes)

    ssd300.cuda()
    loss_func.cuda()

    if args.fp16:
        ssd300 = network_to_half(ssd300)

    if args.distributed:
        ssd300 = DDP(ssd300)

    optimizer = torch.optim.SGD(
        tencent_trick(ssd300),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=args.multistep,
        gamma=0.1)

    if args.fp16:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.)

    inv_map = {v: k for k, v in val_dataset.label_map.items()}

    avg_loss = 0.0
    acc = 0
    batch_perf = AverageMeter()
    end = time.time()
    train_start = end

    args.train_annotate = os.path.join(
        args.data, "annotations/instances_train2017.json")
    args.train_coco_root = os.path.join(args.data, "train2017")
    local_seed = set_seeds(args)

    if args.data_pipeline == 'no_dali':
        train_trans = SSDTransformer(dboxes, args, (300, 300), val=False)
        train_dataset = get_train_dataset(args, train_trans)
        train_loader = get_train_loader(train_dataset, args, args.num_workers)
    elif args.data_pipeline == 'dali':
        train_loader = get_train_dali_loader(args, dboxes, local_seed)

    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        scheduler.step()

        epoch_loop(train_loader, args, ssd300, time.time(),
                   loss_func, optimizer, iteration, avg_loss, batch_perf, epoch)
        torch.cuda.synchronize()

        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_dataloader,
                           cocoGt, encoder, inv_map, args)

        try:
            train_loader.reset()
        except AttributeError:
            pass

    if args.local_rank == 0:
        print("Training end: Average speed: {:3f} img/sec, Total time: {:3f} sec, Final accuracy: {:3f} mAP"
          .format(
              args.N_gpu * args.batch_size / batch_perf.avg, 
              time.time() - train_start,
              acc))


def epoch_loop(train_loader, args, ssd300, end, loss_func, optimizer, iteration, avg_loss, batch_perf, epoch):
    for nbatch, data in enumerate(train_loader):
        if args.data_pipeline == 'no_dali':
            (img, _, img_size, bbox, label) = data
            img = img.cuda()
            bbox = bbox.cuda()
            label = label.cuda()
        else:
            img = data[0]["images"]
            bbox = data[0]["boxes"]
            label = data[0]["labels"]
            label = label.type(torch.cuda.LongTensor)

        boxes_in_batch = len(label.nonzero())

        if boxes_in_batch != 0:
            ploc, plabel = ssd300(img)
            ploc, plabel = ploc.float(), plabel.float()

            trans_bbox = bbox.transpose(1, 2).contiguous().cuda()

            label = label.cuda()
            gloc = Variable(trans_bbox, requires_grad=False)
            glabel = Variable(label, requires_grad=False)

            loss = loss_func(ploc, plabel, gloc, glabel)

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

        if args.warmup is not None:
            warmup(optimizer, args.warmup, iteration, args.learning_rate)

        if not np.isinf(loss.item()):
            avg_loss = 0.999*avg_loss + 0.001*loss.item()

        optimizer.step()
        optimizer.zero_grad()

        batch_perf.update(time.time() - end)
        if args.local_rank == 0:
            log_perf(epoch, args, iteration, loss, avg_loss, batch_perf)

        iteration += 1
        end = time.time()


def log_perf(epoch, args, iteration, loss, avg_loss, batch_perf):
    print("Epoch {:2d}, Iteration: {:3d}, Loss function: {:5.3f}, Average Loss: {:.3f}, Speed: {:4f} img/sec, Avg speed: {:4f} img/sec"
          .format(epoch, iteration, loss.item(), avg_loss, args.batch_size / batch_perf.val, args.batch_size / batch_perf.avg))

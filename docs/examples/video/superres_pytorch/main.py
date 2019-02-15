import argparse
import logging as log
import os
import time
import datetime
from functools import reduce

import numpy as np

from math import ceil, floor
from tensorboardX import SummaryWriter

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data.distributed
from torch.multiprocessing import Process
from torch.autograd import Variable

from dataloading.dataloaders import get_loader

from model.model import VSRNet
from model.clr import cyclic_learning_rate

from nvidia.fp16 import FP16_Optimizer
from nvidia.fp16util import network_to_half
from nvidia.distributed import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--root', type=str, default='.',
                    help='input data root folder')
parser.add_argument('--frames', type=int, default = 3,
                    help='num frames in input sequence')
parser.add_argument('--is_cropped', action='store_true',
                    help='crop input frames?')
parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                    help='[height, width] for input crop')
parser.add_argument('--batchsize', type=int, default=1,
                    help='per rank batch size')
parser.add_argument('--loader', type=str, default='DALI',
                    help='dataloader: pytorch or DALI')
parser.add_argument('--rank', type=int, default=0,
                    help='pytorch distributed rank')
parser.add_argument('--world_size', default=1, type=int, metavar='N',
                    help='num processes for pytorch distributed')
parser.add_argument('--ip', default='localhost', type=str,
                    help='IP address for distributed init.')
parser.add_argument('--max_iter', type=int, default=1000,
                    help='num training iters')
parser.add_argument('--fp16', action='store_true',
                    help='train in fp16?')
parser.add_argument('--checkpoint_dir', type=str, default='.',
                    help='where to save checkpoints')
parser.add_argument('--min_lr', type=float, default=0.000001,
                    help='min learning rate for cyclic learning rate')
parser.add_argument('--max_lr', type=float, default=0.00001,
                    help='max learning rate for cyclic learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0004,
                    help='ADAM weight decay')
parser.add_argument('--flownet_path', type=str,
                    default='flownet2-pytorch/networks/FlowNet2-SD_checkpoint.pth.tar',
                    help='FlowNetSD weights path')
parser.add_argument('--image_freq', type=int, default=100,
                    help='num iterations between image dumps to Tensorboard ')
parser.add_argument('--timing', action='store_true',
                    help="Time data loading and model training (default: False)")

def main(args):

    if args.rank == 0:
        log.basicConfig(level=log.INFO)
        writer = SummaryWriter()
        writer.add_text('config', str(args))
    else:
        log.basicConfig(level=log.WARNING)
        writer = None

    torch.cuda.set_device(args.rank % args.world_size)
    torch.manual_seed(args.seed + args.rank)
    torch.cuda.manual_seed(args.seed + args.rank)
    torch.backends.cudnn.benchmark = True

    if args.world_size > 1:
        log.info('Initializing process group')
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://' + args.ip + ':3567',
            world_size=args.world_size,
            rank=args.rank)
        log.info('Process group initialized')

    log.info('Initializing ' + args.loader + ' training dataloader...')
    train_loader, train_batches, sampler = get_loader(args, 'train')
    samples_per_epoch = train_batches * args.batchsize
    log.info('Dataloader initialized')

    model = VSRNet(args.frames, args.flownet_path, args.fp16)
    if args.fp16:
        network_to_half(model)
    model.cuda()
    model.train()
    for param in model.FlowNetSD_network.parameters():
        param.requires_grad = False

    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(model_params, lr=1, weight_decay=args.weight_decay)
    stepsize = 2 * train_batches
    clr_lambda = cyclic_learning_rate(args.min_lr, args.max_lr, stepsize)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[clr_lambda])
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

    if args.world_size > 1:
        model = DistributedDataParallel(model)

    # TRAINING
    total_iter = 0
    while total_iter * args.world_size < args.max_iter:

        epoch = floor(total_iter / train_batches)

        # only if we are using DistributedSampler
        if args.world_size > 1 and args.loader == 'pytorch':
            sampler.set_epoch(epoch)

        model.train()
        total_epoch_loss = 0.0

        sample_timer = 0.0
        data_timer = 0.0
        compute_timer = 0.0

        iter_start = time.perf_counter()

        training_data_times = []
        training_start = datetime.datetime.now()

        # TRAINING EPOCH LOOP
        for i, inputs in enumerate(train_loader):
            training_stop = datetime.datetime.now()
            dataloading_time = training_stop - training_start
            training_data_times.append(dataloading_time.total_seconds() * 1000.0)

            if args.loader == 'DALI':
                inputs = inputs[0]["data"]
                # Needed? It is already gpu
                inputs = inputs.cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
                if args.fp16:
                    inputs = inputs.half()

            if args.timing:
                torch.cuda.synchronize()
                data_end = time.perf_counter()


            optimizer.zero_grad()

            im_out = total_iter % args.image_freq == 0
            # writer.add_graph(model, inputs)
            loss = model(Variable(inputs), i, writer, im_out)

            total_epoch_loss += loss.item()

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            optimizer.step()
            scheduler.step()

            if args.rank == 0:
                if args.timing:
                    torch.cuda.synchronize()
                    iter_end = time.perf_counter()
                    sample_timer += (iter_end - iter_start)
                    data_duration = data_end - iter_start
                    data_timer += data_duration
                    compute_timer += (iter_end - data_end)
                    torch.cuda.synchronize()
                    iter_start = time.perf_counter()
                writer.add_scalar('learning_rate', scheduler.get_lr()[0], total_iter)
                writer.add_scalar('train_loss', loss.item(), total_iter)

            log.info('Rank %d, Epoch %d, Iteration %d of %d, loss %.5f' %
                    (args.rank, epoch, i+1, train_batches, loss.item()))

            if total_iter % 100 == 0:
                print("Avg dataloading time: " + str(reduce(lambda x, y: x + y, training_data_times) / len(training_data_times)) + "ms")

            total_iter += 1
            if total_iter > args.max_iter:
                break

            training_start = datetime.datetime.now()

        if args.rank == 0:
            if args.timing:
                sample_timer_avg = sample_timer / samples_per_epoch
                writer.add_scalar('sample_time', sample_timer_avg, total_iter)
                data_timer_avg = data_timer / samples_per_epoch
                writer.add_scalar('sample_data_time', data_timer_avg, total_iter)
                compute_timer_avg = compute_timer / samples_per_epoch
                writer.add_scalar('sample_compute_time', compute_timer_avg, total_iter)
            epoch_loss_avg = total_epoch_loss / train_batches
            log.info('Rank %d, epoch %d: %.5f' % (args.rank, epoch, epoch_loss_avg))



        ### VALIDATION
        log.info('Initializing ' + args.loader + ' validation dataloader...')
        val_loader, val_batches, sampler = get_loader(args, 'val')
        model.eval()
        total_loss = 0
        total_psnr = 0
        for i, inputs in enumerate(val_loader):
            if args.loader == 'DALI':
                inputs = inputs[0]["data"]
                # Needed? It is already gpu
                inputs = inputs.cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
                if args.fp16:
                    inputs = inputs.half()

            log.info('Validation it %d of %d' % (i + 1, val_batches))
            loss, psnr = model(Variable(inputs), i, None)
            total_loss += loss.item()
            total_psnr += psnr.item()

        loss = total_loss / i
        psnr = total_psnr / i

        if args.rank == 0:
            writer.add_scalar('val_loss', loss, total_iter)
            writer.add_scalar('val_psnr', psnr, total_iter)
        log.info('Rank %d validation loss %.5f' % (args.rank, loss))
        log.info('Rank %d validation psnr %.5f' % (args.rank, psnr))

if __name__=='__main__':
    main(parser.parse_args())


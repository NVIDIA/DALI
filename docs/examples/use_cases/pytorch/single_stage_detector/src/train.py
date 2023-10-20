from torch.autograd import Variable
import torch
import time


def train_loop(model, loss_func, scaler, epoch, optim, train_loader, iteration, logger, args):
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
            with torch.cuda.amp.autocast(enabled=args.fp16_mode):
                ploc, plabel = model(img)
                ploc, plabel = ploc.float(), plabel.float()

                trans_bbox = bbox.transpose(1, 2).contiguous().cuda()

                label = label.cuda()
                gloc = Variable(trans_bbox, requires_grad=False)
                glabel = Variable(label, requires_grad=False)

                loss = loss_func(ploc, plabel, gloc, glabel)

            scaler.scale(loss).backward()
            logger.update_iter(epoch, iteration, loss.item())


        if args.warmup is not None:
            warmup(optim, args.warmup, iteration, args.learning_rate)


        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
        iteration += 1

    return iteration


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

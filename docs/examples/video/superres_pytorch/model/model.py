import time
import scipy.misc
import numpy as np
from math import floor, log

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn.functional import upsample

import sys
sys.path.append('flownet2-pytorch/networks')
try:
    from submodules import *
except ModuleNotFoundError:
    raise ModuleNotFoundError("flownet2-pytorch not found, did you update the git submodule?")

def lp_error(img1, img2, lp=2):
    return torch.mean((img1 - img2)**lp)


# https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
def psnr(img1, img2):
    mse = lp_error(img1, img2, 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    # getting the noise in dB
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def rgb2ycbcr(input_tensor):
    # Conversion from RGB to YCbCr according to
    # https://en.wikipedia.org/wiki/YCbCr?section=6#JPEG_conversion
    # Expecting batch of RGB images with values in [0, 255]

    kr = 0.299
    kg = 0.587
    kb = 0.114

    # Expecting batch of image sequence inputs with values in [0, 255]
    r = input_tensor[:, 0, :, :, :]
    g = input_tensor[:, 1, :, :, :]
    b = input_tensor[:, 2, :, :, :]

    y = torch.unsqueeze(kr * r + kg * g + kb * b, 1)
    cb = torch.unsqueeze(128 - (0.1687346 * r) - (0.331264 * g) + (0.5 * b), 1)
    cr = torch.unsqueeze(128 + (0.5 * r) - (0.418688 * g) - (0.081312 * b), 1)

    return y, cb, cr


def ycbcr2rgb(input_tensor):
    # Conversion from YCbCr to RGB according to
    # https://en.wikipedia.org/wiki/YCbCr/16?section=6#JPEG_conversion
    # Expecting batch of YCbCr images with values in [0, 255]

    y = input_tensor[:, 0, :, :]
    cb = input_tensor[:, 1, :, :]
    cr = input_tensor[:, 2, :, :]

    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    r = torch.unsqueeze(r, 1)
    g = torch.unsqueeze(g, 1)
    b = torch.unsqueeze(b, 1)

    return torch.clamp(torch.cat((r, g, b), 1), 0, 255)


def get_grid(batchsize, rows, cols, fp16):

    # Input is a tensor with shape [batchsize, channels, rows, cols]
    # Output is tensor with shape [batchsize, 2, rows, cols]
    # where each col in [:, 1, :, :] and each row in [:, 0, :, :]
    # is an evenly spaced arithmetic progression from -1.0 to 1.0

    hor = torch.linspace(-1.0, 1.0, cols)
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([hor, ver], 1)

    if fp16:
        return Variable(t_grid.half().cuda())
    else:
        return Variable(t_grid.cuda())


def tensorboard_image(name, image, iteration, writer):
    # tensorboardX expects CHW images
    out_im = image.data.cpu().numpy().astype('uint8')
    writer.add_image(name, out_im, iteration)

class VSRNet(nn.Module):
    def __init__(self, frames=3, flownet_path='', fp16=False):
        super(VSRNet, self).__init__()

        self.frames = frames
        self.fp16 = fp16
        self.mi = floor(self.frames / 2)

        self.pooling = nn.AvgPool2d(4, ceil_mode=False)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

        if fp16:
            #from FlowNetSD16 import FlowNetSD
            from FlowNetSD import FlowNetSD
        else:
            from FlowNetSD import FlowNetSD

        FlowNetSD_network = FlowNetSD(args=[], batchNorm=False)
        try:
            FlowNetSD_weights = torch.load(flownet_path)['state_dict']
        except:
            raise IOError('FlowNet weights could not be loaded from %s' % flownet_path)
        FlowNetSD_network.load_state_dict(FlowNetSD_weights)
        self.FlowNetSD_network = FlowNetSD_network

        self.train_grid = None
        self.val_grid = None

        self.batchNorm = True
        self.conv1 = conv(self.batchNorm, 1, 64, kernel_size=9)
        self.conv2 = conv(self.batchNorm, 64 * self.frames, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv3.weight = torch.nn.init.normal(self.conv3.weight, 0, 0.1)

    def forward(self, inputs, iteration, writer, im_out=False):
        batchsize, channels, frames, rows, cols = inputs.size()

        # inputs are normalized
        y, cb, cr = rgb2ycbcr(inputs)
        y /= 255
        target = y[:, :, self.mi, :, :]

        if writer is not None and im_out:
            out_im = inputs[0, :, self.mi, :, :] # / 255.0 will we need this?
            tensorboard_image('target', out_im, iteration, writer)
            out_im = self.pooling(out_im)
            tensorboard_image('downsampled', out_im, iteration, writer)
            out_im = self.upsample(out_im.unsqueeze(0)).squeeze(0)
            tensorboard_image('upsampled', out_im, iteration, writer)

        # Compute per RGB channel mean across pixels for each image in input batch
        rgb_mean = inputs.view((batchsize, channels) + (-1, )).float().mean(dim=-1)
        rgb_mean = rgb_mean.view((batchsize, channels) + (1, 1, 1, ))
        if self.fp16:
            rgb_mean = rgb_mean.half()

        inputs = (inputs - rgb_mean) / 255

        if self.training:
            if self.train_grid is None:
                self.train_grid = get_grid(batchsize, rows, cols, self.fp16)
            grid = self.train_grid
        else:
            if self.val_grid is None:
                self.val_grid = get_grid(batchsize, rows, cols, self.fp16)
            grid = self.val_grid
        grid.requires_grad = False

        downsampled_input = self.pooling(cb[:, :, self.mi, :, :])
        cb[:, :, self.mi, :, :] = self.upsample(downsampled_input)
        downsampled_input = self.pooling(cr[:, :, self.mi, :, :])
        cr[:, :, self.mi, :, :] = self.upsample(downsampled_input)

        conv1_out = []
        for fr in range(self.frames):

            downsampled_input = self.pooling(y[:, :, fr, :, :])
            y[:, :, fr, :, :] = self.upsample(downsampled_input)

            if fr == self.mi:
                conv1_out.append(self.conv1(y[:, :, self.mi, :, :]))
            else:
                im1 = inputs[:, :, fr, :, :]
                im2 = inputs[:, :, self.mi, :, :]
                im_pair = torch.cat((im2, im1), 1)

                to_warp = y[:, :, fr, :, :]

                flow = self.upsample(self.FlowNetSD_network(im_pair)[0]) / 16

                flow = torch.cat([flow[:, 0:1, :, :] / ((cols - 1.0) / 2.0),
                                  flow[:, 1:2, :, :] / ((rows - 1.0) / 2.0)], 1)

                warped = torch.nn.functional.grid_sample(
                    input=to_warp,
                    grid=(grid + flow).permute(0, 2, 3, 1),
                    mode='bilinear',
                    padding_mode='border')

                conv1_out.append(self.conv1(warped))

        conv1_out = torch.cat(conv1_out, 1)

        conv2_out = self.conv2(conv1_out)

        # Loss must be computed for pixel values in [0, 255] to prevent
        # divergence in fp16
        prediction = torch.nn.functional.sigmoid(self.conv3(conv2_out).float())

        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())

        if not self.training:
            # Following [1], remove 12 pixels around border to prevent
            # convolution edge effects affecting PSNR
            psnr_metric = psnr(prediction[:, :, 12:, :-12].float() * 255,
                               target[:, :, 12:, :-12].float() * 255)

        prediction = ycbcr2rgb(torch.cat((prediction * 255, cb[:, :, self.mi, :, :],
                               cr[:, :, self.mi, :, :]), 1))

        if writer is not None and im_out:
            out_im = prediction[0, :, :, :]
            tensorboard_image('prediction', out_im, iteration, writer)

        if self.training:
            return loss
        else:
            return loss, psnr_metric


# [1] Osama Makansi, Eddy Ilg, Thomas Brox, "End-to-End Learning of Video Super-Resolution with Motion Compensation", https://arxiv.org/abs/1707.00471

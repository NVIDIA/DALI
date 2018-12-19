import copy
import sys, time, argparse, os, subprocess, shutil
import math, numbers, random, bisect

from random import Random

from skimage import io, transform
from os import listdir
from os.path import join
from glob import glob

import numpy as np

import torch
import torch.utils.data as data


class imageDataset():
    def __init__(self, frames, is_cropped, crop_size,
                 root, batch_size, world_size):
        self.root = root
        self.frames = frames
        self.is_cropped = is_cropped
        self.crop_size = crop_size

        self.files = glob(os.path.join(self.root, '*/*.png'))

        if len(self.files) < 1:
            print(("[Error] No image files in %s" % (self.root)))
            raise LookupError

        self.files = sorted(self.files)

        self.total_frames = 0
        # Find start_indices for different folders
        self.start_index = [0]
        prev_folder = self.files[0].split('/')[-2]
        for (i, f) in enumerate(self.files):
            folder = f.split('/')[-2]
            if i > 0 and folder != prev_folder:
                self.start_index.append(i)
                prev_folder = folder
                self.total_frames -= (self.frames + 1)
            else:
                self.total_frames += 1
        self.total_frames -= (self.frames + 1)
        self.start_index.append(i)

        if self.is_cropped:
            self.image_shape = self.crop_size
        else:
            self.image_shape = list(io.imread(self.files[0]).shape[:2])

        # Frames are enforced to be mod64 in each dimension
        # as required by FlowNetSD convolutions
        self.frame_size = self.image_shape
        self.frame_size[0] = int(math.floor(self.image_shape[0]/64.)*64)
        self.frame_size[1] = int(math.floor(self.image_shape[1]/64.)*64)

        self.frame_buffer = np.zeros((3, self.frames,
                                      self.frame_size[0], self.frame_size[1]),
                                      dtype = np.float32)

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):

        index = index % self.total_frames
        # we want bisect_right here so that the first frame in a file gets the
        # file, not the previous file
        next_file_index = bisect.bisect_right(self.start_index, index)
        if self.start_index[next_file_index] < index + self.frames:
            index = self.start_index[next_file_index] - self.frames - 1

        for (i, file_idx) in enumerate(range(index, index + self.frames)):

            image = io.imread(self.files[file_idx])

            #TODO(jbarker): Tidy this up and remove redundant computation
            if i == 0 and self.is_cropped:
                crop_x = random.randint(0, self.image_shape[1] - self.frame_size[1])
                crop_y = random.randint(0, self.image_shape[0] - self.frame_size[0])
            elif self.is_cropped == False:
                crop_x = math.floor((self.image_shape[1] - self.frame_size[1]) / 2)
                crop_y = math.floor((self.image_shape[0] - self.frame_size[0]) / 2)
                self.crop_size = self.frame_size

            image = image[crop_y:crop_y + self.crop_size[0],
                          crop_x:crop_x + self.crop_size[1],
                          :]

            self.frame_buffer[:, i, :, :] = np.rollaxis(image, 2, 0)

        return torch.from_numpy(self.frame_buffer)

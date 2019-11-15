# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable

__all__ = ['ResNet']


class ResNet(object):
    def __init__(self, depth=50, num_classes=1000):
        super(ResNet, self).__init__()

        assert depth in [18, 34, 50, 101, 152], \
            "depth {} not in [18, 34, 50, 101, 152]"

        self.depth = depth
        self.num_classes = num_classes
        self.stage_filters = [64, 128, 256, 512]
        self.stages, self.block_func = {
            18: ([2, 2, 2, 2], self.basicblock),
            34: ([3, 4, 6, 3], self.basicblock),
            50: ([3, 4, 6, 3], self.bottleneck),
            101: ([3, 4, 23, 3], self.bottleneck),
            152: ([3, 8, 36, 3], self.bottleneck)
        }[depth]

    def _conv_norm(self,
                   input,
                   num_filters,
                   filter_size,
                   stride=1,
                   act=None,
                   name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            name=name + '.conv2d.output.1')

        if 'conv1' in name:
            bn_name = "bn_" + name
        else:
            bn_name = name.replace("res", "bn")
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=bn_name + '.output.1',
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(name=bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance', )

    def _shortcut(self, input, ch_out, stride, is_first, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1 or (self.depth < 50 and is_first):
            return self._conv_norm(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck(self, input, num_filters, stride, is_first, name):
        stride1, stride2 = 1, stride

        conv_def = [[num_filters, 1, stride1, 'relu', name + "_branch2a"],
                    [num_filters, 3, stride2, 'relu', name + "_branch2b"],
                    [num_filters * 4, 1, 1, None, name + "_branch2c"]]

        residual = input
        for (c, k, s, act, _name) in conv_def:
            residual = self._conv_norm(
                input=residual,
                num_filters=c,
                filter_size=k,
                stride=s,
                act=act,
                name=_name)
        short = self._shortcut(
            input,
            num_filters * 4,
            stride,
            is_first=is_first,
            name=name + "_branch1")

        return fluid.layers.elementwise_add(
            x=short, y=residual, act='relu', name=name + ".add.output.5")

    def basicblock(self, input, num_filters, stride, is_first, name):
        conv0 = self._conv_norm(
            input=input,
            num_filters=num_filters,
            filter_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a")
        conv1 = self._conv_norm(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b")
        short = self._shortcut(
            input, num_filters, stride, is_first, name=name + "_branch1")
        return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')

    def layer_warp(self, input, stage_num):
        assert stage_num in [2, 3, 4, 5]

        stages, block_func = self.stages, self.block_func
        count = stages[stage_num - 2]

        ch_out = self.stage_filters[stage_num - 2]
        is_first = False if stage_num != 2 else True

        conv = input
        for i in range(count):
            if self.depth in [101, 152] and stage_num == 2:
                if i == 0:
                    conv_name = "res" + str(stage_num) + "a"
                else:
                    conv_name = "res" + str(stage_num) + "b" + str(i)
            else:
                conv_name = "res" + str(stage_num) + chr(97 + i)
            if self.depth < 50:
                is_first = True if i == 0 and stage_num == 2 else False
            conv = block_func(
                input=conv,
                num_filters=ch_out,
                stride=2 if i == 0 and stage_num != 2 else 1,
                is_first=is_first,
                name=conv_name)
        return conv

    def c1_stage(self, input):
        input = self._conv_norm(
            input=input,
            num_filters=self.stage_filters[0],
            filter_size=7,
            stride=2,
            act='relu',
            name='conv1')

        output = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        return output

    def __call__(self, input):
        assert isinstance(input, Variable)
        res = self.c1_stage(input)

        for i in range(2, 6):
            res = self.layer_warp(res, i)

        pool = fluid.layers.pool2d(res, pool_size=7, pool_type='avg',
                                   global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        return fluid.layers.fc(pool,
                               size=self.num_classes,
                               param_attr=ParamAttr(
                                   initializer=fluid.initializer.Uniform(
                                       -stdv, stdv)))

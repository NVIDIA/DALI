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
import paddle
from paddle import static


class TSM():
    def __init__(self, training=False):
        self.training = training
        self.num_segs = 8
        self.num_classes = 400
        self.depth = 50
        self.layers = [3, 4, 6, 3]
        self.num_filters = [64, 128, 256, 512]

    def shift_module(self, input):
        output = paddle.nn.functional.temporal_shift(input,
                                    self.num_segs,
                                    1.0 / self.num_segs)
        return output

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        conv = paddle.static.nn.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            param_attr=paddle.ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]

        return paddle.static.nn.batch_norm(
            input=conv,
            act=act,
            is_test=(not self.training),
            param_attr=paddle.ParamAttr(name=bn_name + "_scale"),
            bias_attr=paddle.ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + '_variance')

    def shortcut(self, input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name):
        shifted = self.shift_module(input)
        conv0 = self.conv_bn_layer(input=shifted,
                                   num_filters=num_filters,
                                   filter_size=1,
                                   act='relu',
                                   name=name + "_branch2a")
        conv1 = self.conv_bn_layer(input=conv0,
                                   num_filters=num_filters,
                                   filter_size=3,
                                   stride=stride,
                                   act='relu',
                                   name=name + "_branch2b")
        conv2 = self.conv_bn_layer(input=conv1,
                                   num_filters=num_filters * 4,
                                   filter_size=1,
                                   act=None,
                                   name=name + "_branch2c")

        short = self.shortcut(input,
                              num_filters * 4,
                              stride,
                              name=name + "_branch1")

        add = paddle.add(x=short, y=conv2)
        return paddle.nn.functional.relu(add)

    def __call__(self, input):
        channels = input.shape[2]
        short_size = input.shape[3]
        input = paddle.reshape(
            x=input, shape=[-1, channels, short_size, short_size])

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name='conv1')
        conv = paddle.nn.functional.max_pool2d(x=conv,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1)

        for block in range(len(self.layers)):
            for i in range(self.layers[block]):
                conv_name = "res" + str(block + 2) + chr(97 + i)

                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=self.num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    name=conv_name)

        pool = paddle.nn.functional.avg_pool2d(x=conv, kernel_size=7)

        dropout = paddle.nn.functional.dropout(x=pool, p=0.5, training=(not self.training))

        feature = paddle.reshape(x=dropout, shape=[-1, self.num_segs, pool.shape[1]])
        out = paddle.mean(x=feature, axis=1)

        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = static.nn.fc(
            x=out,
            size=self.num_classes,
            activation='softmax',
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(low=-stdv, high=stdv)),
            bias_attr=paddle.ParamAttr(
                                  learning_rate=2.0,
                                  regularizer=paddle.regularizer.L2Decay(0.)))
        return out

# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn

from rnn import rnn
from rnn import StackTime


class RNNT(torch.nn.Module):
    """A Recurrent Neural Network Transducer (RNN-T).

    Args:
        in_features: Number of input features per step per batch.
        vocab_size: Number of output symbols (inc blank).
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
        batch_norm: Use batch normalization in encoder and prediction network
            if true.
        encoder_n_hidden: Internal hidden unit size of the encoder.
        encoder_rnn_layers: Encoder number of layers.
        pred_n_hidden:  Internal hidden unit size of the prediction network.
        pred_rnn_layers: Prediction network number of layers.
        joint_n_hidden: Internal hidden unit size of the joint network.
        rnn_type: string. Type of rnn in SUPPORTED_RNNS.
    """
    def __init__(self, rnnt=None, num_classes=1, **kwargs):
        super().__init__()
        if kwargs.get("no_featurizer", False):
            in_features = kwargs.get("in_features")
        else:
            feat_config = kwargs.get("feature_config")
            in_features = feat_config['features'] * feat_config.get("frame_splicing", 1)

        self._pred_n_hidden = rnnt['pred_n_hidden']

        self.encoder_n_hidden = rnnt["encoder_n_hidden"]
        self.encoder_pre_rnn_layers = rnnt["encoder_pre_rnn_layers"]
        self.encoder_post_rnn_layers = rnnt["encoder_post_rnn_layers"]

        self.pred_n_hidden = rnnt["pred_n_hidden"]
        self.pred_rnn_layers = rnnt["pred_rnn_layers"]

        self.encoder = self._encoder(
            in_features,
            rnnt["encoder_n_hidden"],
            rnnt["encoder_pre_rnn_layers"],
            rnnt["encoder_post_rnn_layers"],
            rnnt["forget_gate_bias"],
            None if "norm" not in rnnt else rnnt["norm"],
            rnnt["rnn_type"],
            rnnt["encoder_stack_time_factor"],
            rnnt["dropout"],
        )

        self.prediction = self._predict(
            num_classes,
            rnnt["pred_n_hidden"],
            rnnt["pred_rnn_layers"],
            rnnt["forget_gate_bias"],
            None if "norm" not in "rnnt" else rnnt["norm"],
            rnnt["rnn_type"],
            rnnt["dropout"],
        )

        self.joint_net = self._joint_net(
            num_classes,
            rnnt["pred_n_hidden"],
            rnnt["encoder_n_hidden"],
            rnnt["joint_n_hidden"],
            rnnt["dropout"],
        )

    def _encoder(self, in_features, encoder_n_hidden,
                 encoder_pre_rnn_layers, encoder_post_rnn_layers,
                 forget_gate_bias, norm, rnn_type, encoder_stack_time_factor,
                 dropout):
        layers = torch.nn.ModuleDict({
            "pre_rnn": rnn(
                rnn=rnn_type,
                input_size=in_features,
                hidden_size=encoder_n_hidden,
                num_layers=encoder_pre_rnn_layers,
                norm=norm,
                forget_gate_bias=forget_gate_bias,
                dropout=dropout,
            ),
            "stack_time": StackTime(factor=encoder_stack_time_factor),
            "post_rnn": rnn(
                rnn=rnn_type,
                input_size=encoder_stack_time_factor*encoder_n_hidden,
                hidden_size=encoder_n_hidden,
                num_layers=encoder_post_rnn_layers,
                norm=norm,
                forget_gate_bias=forget_gate_bias,
                norm_first_rnn=True,
                dropout=dropout,
            ),
        })
        return layers

    def _predict(self, vocab_size, pred_n_hidden, pred_rnn_layers,
                 forget_gate_bias, norm, rnn_type, dropout):
        layers = torch.nn.ModuleDict({
            "embed": torch.nn.Embedding(vocab_size - 1, pred_n_hidden),
            "dec_rnn": rnn(
                rnn=rnn_type,
                input_size=pred_n_hidden,
                hidden_size=pred_n_hidden,
                num_layers=pred_rnn_layers,
                norm=norm,
                forget_gate_bias=forget_gate_bias,
                dropout=dropout,
            ),
        })
        return layers

    def _joint_net(self, vocab_size, pred_n_hidden, enc_n_hidden,
                   joint_n_hidden, dropout):
        layers = [
            torch.nn.Linear(pred_n_hidden + enc_n_hidden, joint_n_hidden),
            torch.nn.ReLU(),
        ] + ([ torch.nn.Dropout(p=dropout), ] if dropout else [ ]) + [
            torch.nn.Linear(joint_n_hidden, vocab_size)
        ]
        return torch.nn.Sequential(
                *layers
        )

    def forward(self, batch, state=None):
        # batch: ((x, y), (x_lens, y_lens))

        # x: (B, channels, features, seq_len)
        (x, y), (x_lens, y_lens) = batch
        y = label_collate(y)

        f, x_lens = self.encode((x, x_lens))

        g, _ = self.predict(y, state)
        out = self.joint(f, g)

        return out, (x_lens, y_lens)

    def encode(self, x):
        """
        Args:
            x: tuple of ``(input, input_lens)``. ``input`` has shape (T, B, I),
                ``input_lens`` has shape ``(B,)``.

        Returns:
            f: tuple of ``(output, output_lens)``. ``output`` has shape
                (B, T, H), ``output_lens``
        """
        x, x_lens = x
        x, _ = self.encoder["pre_rnn"](x, None)
        x, x_lens = self.encoder["stack_time"]((x, x_lens))
        x, _ = self.encoder["post_rnn"](x, None)

        return x.transpose(0, 1), x_lens

    def predict(self, y, state=None, add_sos=True):
        """
        B - batch size
        U - label length
        H - Hidden dimension size
        L - Number of decoder layers = 2

        Args:
            y: (B, U)

        Returns:
            Tuple (g, hid) where:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is
                    the final cell state:
                        h (tensor), shape (L, B, H)
                        c (tensor), shape (L, B, H)
        """
        if y is not None:
            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            B = 1 if state is None else state[0].size(1)
            y = torch.zeros((B, 1, self.pred_n_hidden)).to(
                device=self.joint_net[0].weight.device,
                dtype=self.joint_net[0].weight.dtype
            )

        # preprend blank "start of sequence" symbol
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H)).to(device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()   # (B, U + 1, H)
        else:
            start = None   # makes del call later easier

        #if state is None:
        #    batch = y.size(0)
        #    state = [
        #        (torch.zeros(batch, self.pred_n_hidden, dtype=y.dtype, device=y.device),
        #         torch.zeros(batch, self.pred_n_hidden, dtype=y.dtype, device=y.device))
        #        for _ in range(self.pred_rnn_layers)
        #    ]

        y = y.transpose(0, 1)#.contiguous()   # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)#.contiguous()   # (B, U + 1, H)
        del y, start, state
        return g, hid

    def joint(self, f, g):
        """
        f should be shape (B, T, H)
        g should be shape (B, U + 1, H)

        returns:
            logits of shape (B, T, U, K + 1)
        """
        # Combine the input states and the output states
        B, T, H = f.shape
        B, U_, H2 = g.shape

        f = f.unsqueeze(dim=2)   # (B, T, 1, H)
        f = f.expand((B, T, U_, H))

        g = g.unsqueeze(dim=1)   # (B, 1, U + 1, H)
        g = g.expand((B, T, U_, H2))

        inp = torch.cat([f, g], dim=3)   # (B, T, U, 2H)
        res = self.joint_net(inp)
        del f, g, inp
        return res


def label_collate(labels):
    """Collates the label inputs for the rnn-t prediction network.

    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(
            f"`labels` should be a list or tensor not {type(labels)}"
        )

    batch_size = len(labels)
    max_len = max(len(l) for l in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, :len(l)] = l
    labels = torch.LongTensor(cat_labels)

    return labels

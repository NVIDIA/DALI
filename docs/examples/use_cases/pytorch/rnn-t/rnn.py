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

import math

from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch.nn import Parameter


def rnn(rnn, input_size, hidden_size, num_layers, norm=None,
        forget_gate_bias=1.0, dropout=0.0, **kwargs):
    """TODO"""
    if rnn != "lstm":
        raise ValueError(f"Unknown rnn={rnn}")
    if norm not in [None, "batch_norm", "layer_norm"]:
        raise ValueError(f"unknown norm={norm}")

    if rnn == "lstm":
        if norm is None:
            return LstmDrop(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                forget_gate_bias=forget_gate_bias,
                **kwargs
            )

        if norm == "batch_norm":
            return BNRNNSum(
                input_size=input_size,
                hidden_size=hidden_size,
                rnn_layers=num_layers,
                batch_norm=True,
                dropout=dropout,
                forget_gate_bias=forget_gate_bias,
                **kwargs
            )

        if norm == "layer_norm":
            return torch.jit.script(lnlstm(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                forget_gate_bias=forget_gate_bias,
                **kwargs
            ))


class OverLastDim(torch.nn.Module):
    """Collapses a tensor to 2D, applies a module, and (re-)expands the tensor.

    An n-dimensional tensor of shape (s_1, s_2, ..., s_n) is first collapsed to
    a tensor with shape (s_1*s_2*...*s_n-1, s_n). The module is called with
    this as input producing (s_1*s_2*...*s_n-1, s_n') --- note that the final
    dimension can change. This is expanded to (s_1, s_2, ..., s_n-1, s_n') and
    returned.

    Args:
        module (torch.nn.Module): Module to apply. Must accept a 2D tensor as
            input and produce a 2D tensor as output, optionally changing the
            size of the last dimension.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        *dims, input_size = x.size()

        reduced_dims = 1
        for dim in dims:
            reduced_dims *= dim

        x = x.view(reduced_dims, -1)
        x = self.module(x)
        x = x.view(*dims, -1)
        return x


class LstmDrop(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, forget_gate_bias,
             **kwargs):
        """Returns an LSTM with forget gate bias init to `forget_gate_bias`.

        Args:
            input_size: See `torch.nn.LSTM`.
            hidden_size: See `torch.nn.LSTM`.
            num_layers: See `torch.nn.LSTM`.
            dropout: See `torch.nn.LSTM`.
            forget_gate_bias: For each layer and each direction, the total value of
                to initialise the forget gate bias to.

        Returns:
            A `torch.nn.LSTM`.
        """
        super(LstmDrop, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        if forget_gate_bias is not None:
            for name, v in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size:2*hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size:2*hidden_size].fill_(0)

        self.dropout = torch.nn.Dropout(dropout) if dropout else None

    def forward(self, x, h=None):

        x, h = self.lstm(x, h)

        if self.dropout:
            x = self.dropout(x)

        return x, h



class RNNLayer(torch.nn.Module):
    """A single RNNLayer with optional batch norm."""
    def __init__(self, input_size, hidden_size, rnn_type=torch.nn.LSTM,
                 batch_norm=True, forget_gate_bias=1.0):
        super().__init__()

        if batch_norm:
            self.bn = OverLastDim(torch.nn.BatchNorm1d(input_size))

        if isinstance(rnn_type, torch.nn.LSTM) and not batch_norm:
            # batch_norm will apply bias, no need to add a second to LSTM
            self.rnn = lstm(input_size=input_size,
                            hidden_size=hidden_size,
                            forget_gate_bias=forget_gate_bias)
        else:
            self.rnn = rnn_type(input_size=input_size,
                                hidden_size=hidden_size,
                                bias=not batch_norm)

    def forward(self, x, hx=None):
        if hasattr(self, 'bn'):
            x = x.contiguous()
            x = self.bn(x)
        x, h = self.rnn(x, hx=hx)
        return x, h

    def _flatten_parameters(self):
        self.rnn.flatten_parameters()


class BNRNNSum(torch.nn.Module):
    """RNN wrapper with optional batch norm.

    Instantiates an RNN. If it is an LSTM it initialises the forget gate
    bias =`lstm_gate_bias`. Optionally applies a batch normalisation layer to
    the input with the statistics computed over all time steps.  If dropout > 0
    then it is applied to all layer outputs except the last.
    """
    def __init__(self, input_size, hidden_size, rnn_type=torch.nn.LSTM,
                 rnn_layers=1, batch_norm=True, dropout=0.0,
                 forget_gate_bias=1.0, norm_first_rnn=False, **kwargs):
        super().__init__()
        self.rnn_layers = rnn_layers

        self.layers = torch.nn.ModuleList()
        for i in range(rnn_layers):
            final_layer = (rnn_layers - 1) == i

            self.layers.append(
                RNNLayer(
                    input_size,
                    hidden_size,
                    rnn_type=rnn_type,
                    batch_norm=batch_norm and (norm_first_rnn or i > 0),
                    forget_gate_bias=forget_gate_bias,
                )
            )

            if dropout > 0.0 and not final_layer:
                self.layers.append(torch.nn.Dropout(dropout))

            input_size = hidden_size

    def forward(self, x, hx=None):
        hx = self._parse_hidden_state(hx)

        hs = []
        cs = []
        rnn_idx = 0
        for layer in self.layers:
            if isinstance(layer, torch.nn.Dropout):
                x = layer(x)
            else:
                x, h_out = layer(x, hx=hx[rnn_idx])
                hs.append(h_out[0])
                cs.append(h_out[1])
                rnn_idx += 1
                del h_out

        h_0 = torch.stack(hs, dim=0)
        c_0 = torch.stack(cs, dim=0)
        return x, (h_0, c_0)

    def _parse_hidden_state(self, hx):
        """
        Dealing w. hidden state:
        Typically in pytorch: (h_0, c_0)
            h_0 = ``[num_layers * num_directions, batch, hidden_size]``
            c_0 = ``[num_layers * num_directions, batch, hidden_size]``
        """
        if hx is None:
            return [None] * self.rnn_layers
        else:
            h_0, c_0 = hx
            assert h_0.shape[0] == self.rnn_layers
            return [(h_0[i], c_0[i]) for i in range(h_0.shape[0])]

    def _flatten_parameters(self):
        for layer in self.layers:
            if isinstance(layer, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
                layer._flatten_parameters()


class StackTime(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x):
        # T, B, U
        x, x_lens = x
        seq = [x]
        for i in range(1, self.factor):
            tmp = torch.zeros_like(x)
            tmp[:-i, :, :] = x[i:, :, :]
            seq.append(tmp)
        x_lens = torch.ceil(x_lens.float() / self.factor).int()
        return torch.cat(seq, dim=2)[::self.factor, :, :], x_lens


def lnlstm(input_size, hidden_size, num_layers, dropout, forget_gate_bias,
           **kwargs):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""
    # The following are not implemented.
    assert dropout == 0.0

    return StackedLSTM(
        num_layers,
        LSTMLayer,
        first_layer_args=[
            LayerNormLSTMCell,
            input_size,
            hidden_size,
            forget_gate_bias,
        ],
        other_layer_args=[
            LayerNormLSTMCell,
            hidden_size,
            hidden_size,
            forget_gate_bias,
        ]
    )


class LSTMLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(
        self,
        input: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class LayerNormLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, forget_gate_bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))

        # layernorms provide learnable biases
        self.layernorm_i = torch.nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = torch.nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = torch.nn.LayerNorm(hidden_size)

        self.reset_parameters()

        self.layernorm_i.bias.data[hidden_size:2*hidden_size].fill_(0.0)
        self.layernorm_h.bias.data[hidden_size:2*hidden_size].fill_(
            forget_gate_bias
        )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self,
        input: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return torch.nn.ModuleList(layers)


class StackedLSTM(torch.nn.Module):
    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM, self).__init__()
        self.layers: Final[torch.nn.ModuleList] = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    def forward(
        self,
        input: torch.Tensor,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        if states is None:
            states: List[Tuple[torch.Tensor, torch.Tensor]] = []
            batch = input.size(1)
            for layer in self.layers:
                states.append(
                    (torch.zeros(
                        batch,
                        layer.cell.hidden_size,
                        dtype=input.dtype,
                        device=input.device
                     ),
                     torch.zeros(
                         batch,
                         layer.cell.hidden_size,
                         dtype=input.dtype,
                         device=input.device
                     )
                    )
                )

        output_states: List[Tuple[Tensor, Tensor]] = []
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states

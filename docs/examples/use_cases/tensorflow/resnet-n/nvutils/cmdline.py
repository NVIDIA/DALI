#!/usr/bin/env python
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================

import horovod.tensorflow as hvd
import argparse

def _add_bool_argument(cmdline, shortname, longname=None, default=False,
                       required=False, help=None):
    if longname is None:
        shortname, longname = None, shortname
    elif default == True:
        raise ValueError("""Boolean arguments that are True by default should not have short names.""")
    name = longname[2:]
    feature_parser = cmdline.add_mutually_exclusive_group(required=required)
    if shortname is not None:
        feature_parser.add_argument(shortname, '--'+name, dest=name, action='store_true', help=help, default=default)
    else:
        feature_parser.add_argument(           '--'+name, dest=name, action='store_true', help=help, default=default)
    feature_parser.add_argument('--no'+name, dest=name, action='store_false')
    return cmdline


class RequireInCmdline(object):
    pass

def _default(vals, key):
    v = vals.get(key)
    return None if v is RequireInCmdline else v

def _required(vals, key):
    return vals.get(key) is RequireInCmdline

def parse_cmdline(init_vals, custom_parser=None):
    if custom_parser is None:
        f = argparse.ArgumentDefaultsHelpFormatter
        p = argparse.ArgumentParser(formatter_class=f)
    else:
        p = custom_parser

    p.add_argument('--data_dir',
                   default=_default(init_vals, 'data_dir'),
                   required=_required(init_vals, 'data_dir'),
                   help="""Path to dataset in TFRecord format
                   (aka Example protobufs). Files should be
                   named 'train-*' and 'validation-*'.""")
    p.add_argument('--data_idx_dir',
                   default=_default(init_vals, 'data_idx_dir'),
                   required=_required(init_vals, 'data_idx_dir'),
                   help="""Path to index files of TFRecord dataset
                   Files should be named 'train-*.idx' and
                   'validation-*.idx'.""")
    p.add_argument('-b', '--batch_size', type=int,
                   default=_default(init_vals, 'batch_size'),
                   required=_required(init_vals, 'batch_size'),
                   help="""Size of each minibatch.""")
    p.add_argument('-i', '--num_iter', type=int,
                   default=_default(init_vals, 'num_iter'),
                   required=_required(init_vals, 'num_iter'),
                   help="""Number of batches or epochs to run.""")
    p.add_argument('-u', '--iter_unit', choices=['epoch', 'batch'],
                   default=_default(init_vals, 'iter_unit'),
                   required=_required(init_vals, 'iter_unit'),
                  help="""Select whether 'num_iter' is interpreted
                  in terms of batches or epochs.""")
    p.add_argument('--log_dir',
                   default=_default(init_vals, 'log_dir'),
                   required=_required(init_vals, 'log_dir'),
                   help="""Directory in which to write training
                   summaries and checkpoints.""")
    p.add_argument('--display_every', type=int,
                   default=_default(init_vals, 'display_every'),
                   required=_required(init_vals, 'display_every'),
                   help="""How often (in batches) to print out
                   running information.""")
    p.add_argument('--precision', choices=['fp32', 'fp16'],
                   default=_default(init_vals, 'precision'),
                   required=_required(init_vals, 'precision'),
                   help="""Select single or half precision arithmetic.""")
    p.add_argument('--dali_cpu', action='store_true',
                   default=False,
                   help="""Use CPU backend for DALI for input pipeline.""")
    p.add_argument('--epoch_evaluation', action='store_true',
                   default=False,
                   help="""Additionally runs the evaluation after every epoch.""")

    FLAGS, unknown_args = p.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    if hvd.rank() == 0:
        print("Script arguments:")
        for flag, val in vars(FLAGS).items():
            if val is not None:
                print("  --{} {}".format(flag, val))

    vals = init_vals
    vals['data_dir'] = FLAGS.data_dir
    del FLAGS.data_dir
    vals['data_idx_dir'] = FLAGS.data_idx_dir
    del FLAGS.data_idx_dir
    vals['batch_size'] = FLAGS.batch_size
    del FLAGS.batch_size
    vals['num_iter'] = FLAGS.num_iter
    del FLAGS.num_iter
    vals['iter_unit'] = FLAGS.iter_unit
    del FLAGS.iter_unit
    vals['log_dir'] = FLAGS.log_dir
    del FLAGS.log_dir
    vals['display_every'] = FLAGS.display_every
    del FLAGS.display_every
    vals['precision'] = FLAGS.precision
    del FLAGS.precision
    vals['dali_cpu'] = FLAGS.dali_cpu
    del FLAGS.dali_cpu
    vals['epoch_evaluation'] = FLAGS.epoch_evaluation
    del FLAGS.epoch_evaluation

    return vals, FLAGS


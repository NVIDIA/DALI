#!/usr/bin/env python
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import horovod.tensorflow as hvd

def parse_cmdline(init_vals):
  f = argparse.ArgumentDefaultsHelpFormatter
  p = argparse.ArgumentParser(formatter_class=f)

  p.add_argument('--image_format', choices=['channels_last', 'channels_first'],
                 default=init_vals.get('image_format'),
                 required=False,
                 nargs='?', const='GPU',
                 help="""Set the input format, available values are
                 [channels_first|channels_last]. Default is channels_last.""")
  p.add_argument('--data_dir',
                 default=init_vals.get('data_dir'),
                 required=False,
                 help="""Path to dataset in TFRecord format (aka Example
                 protobufs). Files should be named 'train-*' and
                 'validation-*'.""")
  p.add_argument('--data_idx_dir',
                 default=init_vals.get('data_idx_dir'),
                 required=False,
                 help="""Path to index files of TFRecord dataset Files should
                 be named 'train-*.idx' and 'validation-*.idx'.""")
  p.add_argument('-b', '--batch_size', type=int,
                 default=init_vals.get('batch_size'),
                 required=False,
                 help="""Size of each minibatch.""")
  p.add_argument('-i', '--num_iter', type=int,
                 default=init_vals.get('num_iter'),
                 required=False,
                 help="""Number of batches or epochs to run.""")
  p.add_argument('-u', '--iter_unit', choices=['epoch', 'batch'],
                 default=init_vals.get('iter_unit'),
                 required=False,
                 help="""Select whether 'num_iter' is interpreted in terms of
                 batches or epochs.""")
  p.add_argument('--log_dir',
                 default=init_vals.get('log_dir'),
                 required=False,
                 help="""Directory in which to write training summaries and
                 checkpoints.""")
  p.add_argument('--export_dir',
                 default=init_vals.get('export_dir'),
                 required=False,
                 help="""Directory in which to write the saved model.""")
  p.add_argument('--tensorboard_dir',
                 default=init_vals.get('tensorboard_dir'),
                 required=False,
                 help="""Directory in which to write tensorboard logs.""")
  p.add_argument('--display_every', type=int,
                 default=init_vals.get('display_every'),
                 required=False,
                 help="""How often (in batches) to print out running
                 information.""")
  p.add_argument('--precision', choices=['fp32', 'fp16'],
                 default=init_vals.get('precision'),
                 required=False,
                 help="""Select single or half precision arithmetic.""")
  p.add_argument('--dali_mode', choices=['CPU', 'GPU'],
                 default=init_vals.get('dali_mode'),
                 required=False,
                 nargs='?', const='GPU',
                 help="""Use DALI for input pipeline, available values are
                 [CPU|GPU] which tell which version of the pipeline run.
                 Default is GPU""")
  p.add_argument('--dali_threads', type=int,
                 default=4,
                 required=False,
                 help="""Number of threads used by DALI.""")
  p.add_argument('--use_xla', action='store_true',
                 help="""Whether to enable xla execution.""")
  p.add_argument('--predict', action='store_true',
                 help="""Whether to conduct prediction""")


  FLAGS, unknown_args = p.parse_known_args()
  if len(unknown_args) > 0:
    for bad_arg in unknown_args:
      print("ERROR: Unknown command line arg: %s" % bad_arg)
    raise ValueError("Invalid command line arg(s)")

  vals = init_vals
  vals['image_format'] = FLAGS.image_format
  vals['data_dir'] = FLAGS.data_dir
  vals['data_idx_dir'] = FLAGS.data_idx_dir
  vals['batch_size'] = FLAGS.batch_size
  vals['num_iter'] = FLAGS.num_iter
  vals['iter_unit'] = FLAGS.iter_unit
  vals['log_dir'] = FLAGS.log_dir
  vals['export_dir'] = FLAGS.export_dir
  vals['tensorboard_dir'] = FLAGS.tensorboard_dir
  vals['display_every'] = FLAGS.display_every
  vals['precision'] = FLAGS.precision
  vals['dali_mode'] = FLAGS.dali_mode
  vals['dali_threads'] = FLAGS.dali_threads
  vals['use_xla'] = FLAGS.use_xla or vals['use_xla']
  vals['predict'] = FLAGS.predict or vals['predict']

  if hvd.rank() == 0:
    print("Script arguments:")
    for flag, val in vals.items():
      print(f"  --{flag}={val}")

  return vals

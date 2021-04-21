# Copyright 2021 Jagoda Kamińska. All Rights Reserved.
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
r"""Generate TFRecord index files necessary when using DALI preprocessing.

Example usage:
    python create_tfrecord_indexes.py  --dali_dir=~/DALI  \
        --tfrecord_file_pattern=tfrecord/pascal*.tfrecord
"""
from absl import app
from absl import flags
from absl import logging

from glob import glob
from subprocess import call
import os.path

flags.DEFINE_string(
    'tfrecord_file_pattern', None,
    'Glob for tfrecord files.')
flags.DEFINE_string('dali_dir', None, 'Absolute path to root directory of DALI library installation.')
FLAGS = flags.FLAGS


def main(_):
    if FLAGS.tfrecord_file_pattern is None:
        raise RuntimeError('Must specify --tfrecord_file_pattern.')
    if FLAGS.dali_dir is None:
        raise RuntimeError('Must specify --dali_dir.')

    tfrecord_files = glob(FLAGS.tfrecord_file_pattern)
    tfrecord_idxs = [filename + "_idx" for filename in tfrecord_files]
    tfrecord2idx_script = os.path.join(FLAGS.dali_dir, 'tools', 'tfrecord2idx')
    if not os.path.isfile(tfrecord2idx_script):
        raise ValueError('{FLAGS.dali_dir} does not lead to valid DALI installation.')

    for tfrecord, tfrecord_idx in zip(tfrecord_files, tfrecord_idxs):
        if not os.path.isfile(tfrecord_idx):
            logging.info(f"Generating index file for {tfrecord}")
            call([tfrecord2idx_script, tfrecord, tfrecord_idx])


if __name__ == '__main__':
  app.run(main)

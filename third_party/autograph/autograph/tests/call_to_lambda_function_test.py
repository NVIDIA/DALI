# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Simple call to lambda functions."""

import tensorflow.compat.v1 as tf

from tensorflow.python.autograph.tests import reference_test_base


def inline_lambda(x):
  l = lambda x: x * x if x > 0 else -x
  return l(x)


def external_lambda(x, l):
  return l(x)


class ReferenceTest(reference_test_base.TestCase):

  def test_inline(self):
    self.assertFunctionMatchesEager(inline_lambda, 1)
    self.assertFunctionMatchesEager(inline_lambda, tf.constant(1))

  def test_external(self):
    self.assertFunctionMatchesEager(external_lambda, 1, lambda x: x == 0)
    self.assertFunctionMatchesEager(
        external_lambda, tf.constant(1), lambda x: x == 0)

if __name__ == '__main__':
  tf.test.main()

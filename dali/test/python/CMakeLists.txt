# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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


set(DALI_PYTHON_NOSETESTS
  test_backend_impl.py
  test_pipeline.py
  test_operator_crop_mirror_normalize.py
  test_operator_crop.py
  test_operator_decoder.py
  test_operator_flip.py
  test_operator_python_function.py
  test_operator_slice.py
PARENT_SCOPE)

set(DALI_PYTHON_TESTS
  "python test_detection_pipeline.py -i 300"
  "python test_RN50_data_pipeline.py -i 10"
  "python test_coco_tfrecord.py -i 64"
PARENT_SCOPE)
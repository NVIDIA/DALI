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

from setuptools import setup, find_packages

setup(name='nvidia-dali',
      description='NVIDIA DALI',
      url='https://github.com/NVIDIA/dali',
      version='0.1.2',
      author='NVIDIA Corporation',
      license='Apache License 2.0',
      packages=find_packages(),
      zip_safe=False,
      package_data={
          '': ['*.so','*.bin','Acknowledgements.txt','LICENSE','COPYRIGHT'],
          },
      py_modules = [
          'rec2idx',
          ],
      scripts = [
          'tfrecord2idx',
          ],
      entry_points = {
          'console_scripts': [
              'rec2idx = rec2idx:main',
              ],
          },
      install_requires = [
          'future',
          ],
     )


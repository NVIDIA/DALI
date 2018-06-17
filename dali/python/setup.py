# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
from setuptools import setup, find_packages

setup(name='nvidia-dali',
      description='NVIDIA DALI',
      url='https://github.com/NVIDIA/dali',
      version='0.1.0',
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


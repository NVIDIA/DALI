from setuptools import setup

setup(name='nvidia-dali',
      description='NVIDIA DALI',
      url='https://github.com/NVIDIA/dali',
      version='0.1.0',
      author='NVIDIA Corporation',
      license='Apache License 2.0',
      packages=[
          'dali',
          'dali.plugin'
          ],
      package_data={
          '': ['*.so','Acknowledgements.txt','LICENSE','COPYRIGHT']
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
     )

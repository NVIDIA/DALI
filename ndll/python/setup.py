from setuptools import setup

setup(name='ndll',
      version='0.1.0',
      packages=[
          'ndll',
          'ndll.plugin',
          ],
      py_modules = [
          'rec2idx',
          ],
      package_data={
          '': ['*.so']},
      scripts = [
          'tfrecord2idx',
          ],
      entry_points = {
          'console_scripts': [
              'rec2idx = rec2idx:main',
              ],
          },
     )

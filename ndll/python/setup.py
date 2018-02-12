from setuptools import setup

setup(name='ndll',
      version='0.0.1',
      packages=[
          'ndll',
          'ndll.plugin'
          ],
      package_data={
          '': ['*ndll_backend.*']}
     )

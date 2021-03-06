#from sys import 
from setuptools import setup, find_packages
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(name = 'pydecomp',
      version = "0.1",
      author = ['Lucas Lestandi','Diego Britez'],
      author_email = 'lucas.lestandi@gmail.com',
      maintainer = 'Lucas Lestandi',
      maintainer_email = 'lucas.lestandi@gmail.com',
      keywords = 'tensor decomposition package Python POD',
      url="https://git.notus-cfd.org/llestandi/python_decomposition_library",
      # classifiers = ['Topic :: Tensor Decomposition'],
      #packages_dir = {'core':'core/', 
      #                'utils':'utils/',
      #                'interfaces':'interfaces/',
      #                'analysis':'analysis/'},
      packages = find_packages(where="."),
      #install_requires=required,
      description = 'Tensor decomposition library',
      long_description = open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
      license = 'GPL V3',
      platforms = 'ALL',
     )

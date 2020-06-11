from setuptools import setup, find_packages
import os

setup(name = 'pydecomp',
      version = "0.1",
      author = ['Lucas Lestandi','Diego Britez'],
      author_email = 'lucas.lestandi@gmail.com',
      maintainer = 'Lucas Lestandi',
      maintainer_email = 'lucas.lestandi@gmail.com',
      keywords = 'tensor decomposition package Python POD',
      classifiers = ['Topic :: Tensor Decomposition'],
      # packages_dir = {'core':'core/', 'utils':'utils/',
      #                 'deprecated':'deprecated/', 'interfaces':'interfaces/',
      #                 'analysis':'analysis/'},
      packages = ['core', 'utils', 'deprecated', 'interfaces', 'analysis'],
      description = 'Tensor decomposition library',
      long_description = open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
      license = 'GPL V3',
      platforms = 'ALL',
     )

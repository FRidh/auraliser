import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand
import numpy as np

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setup(
      name='auraliser',
      version='0.0',
      description="Auralisation module.",
      author='Frederik Rietdijk',
      author_email='freddyrietdijk@fridh.nl',
      license='LICENSE.txt',
      packages=['auraliser'],
      zip_safe=False,
      include_dirs = [np.get_include()],
      install_requires=[
          'cytoolz',
          'geometry',
          'ism',
          'matplotlib',
          'numba',
          'numpy',
          'scintillations',
          'scipy',
          'streaming',
          'turbulence',
          ],
      tests_require = [ 'pytest' ],
      cmdclass = {'test': PyTest},
      )

import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand
#from Cython.Build import cythonize
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
      #long_description=open('README.txt').read(),
      author='Frederik Rietdijk',
      author_email='frederik.rietdijk@empa.ch',
      license='LICENSE.txt',
      packages=['auraliser'],
      #scripts=['bin/beams.py'],
      zip_safe=False,
      include_dirs = [np.get_include()],
      install_requires=[
          'numpy',
          'matplotlib'],
      #ext_modules = cythonize('auraliser/*.pyx')
      tests_require = [ 'pytest' ],
      cmdclass = {'test': PyTest},
      )

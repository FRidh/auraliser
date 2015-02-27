from setuptools import setup
#from Cython.Build import cythonize
import numpy as np

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
      )

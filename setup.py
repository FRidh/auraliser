from setuptools import setup

setup(
      name='auraliser',
      version='0.0',
      description="Auralisation module.",
      long_description=open('README.txt').read(),
      author='Frederik Rietdijk',
      author_email='frederik.rietdijk@empa.ch',
      license='LICENSE.txt',
      packages=['auraliser'],
      #scripts=['bin/beams.py'],
      zip_safe=False,
      install_requires=[
          'numpy',
          'matplotlib'
          ],
      )
from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
  name = 'pyrulelearn',
  packages = ['pyrulelearn'],
  version = 'v1.0.5',
  license='MIT',
  description = 'This library can be used to generate interpretable classification rules expressed as CNF/DNF and relaxed-CNF',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Bishwamittra Ghosh',
  author_email = 'bishwamittra.ghosh@gmail.com',
  url = 'https://github.com/meelgroup/MLIC',
  download_url = 'https://github.com/meelgroup/MLIC/archive/v1.0.4.tar.gz',
  keywords = ['Classification Rules', 'Interpretable Rules', 'CNF Classification Rules', 'DNF Classification Rules','MaxSAT-based Rule Learning'],   # Keywords that define your package best
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
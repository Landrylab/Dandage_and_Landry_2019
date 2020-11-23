#!/usr/bin/env python

"""
========
setup.py
========

installs rohan

USAGE :
python setup.py install

Or for local installation:

python setup.py install --prefix=/your/local/dir

"""
import sys
try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup, find_packages, Extension
if (sys.version_info[0], sys.version_info[1],sys.version_info[2]) != (3, 6 ,5):
    raise RuntimeError('Python 3.6.5 required ')

with open('requirements.txt') as f:
    required = f.read().splitlines()

# main setup
setup(
name='human_paralogs',
author='Rohan Dandage',
author_email='rohanadandage@gmail.com',
version='0.0.0',
url='https://github.com/rraadd88/human_paralogs',
download_url='https://github.com/rraadd88/human_paralogs/archive/main.zip',
description='human_paralogs project',
long_description='https://github.com/rraadd88/human_paralogs/README.md',
license='General Public License v. 3',
install_requires=required,
platforms='Tested on Ubuntu 16.04 64bit',
packages=find_packages(exclude=['test*', 'deps*', 'data*', 'data']),
)

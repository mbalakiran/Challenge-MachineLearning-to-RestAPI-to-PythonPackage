# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 18:45:14 2021

@author: makn0023
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os, sys
from pathlib import Path

from setuptools import find_packages, setup


# Package meta-data.
NAME = 'Challenge'
DESCRIPTION = 'Train and deploy the challenge.'
URL = 'https://github.com/mbalakiran'
EMAIL = 'balakiranmanthri1995@email.com'
AUTHOR = 'Bala Kiran'
REQUIRES_PYTHON = '>=3.6.0'


# What packages are required for this module to be executed?
def list_reqs(fname='package\\requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the
# Trove Classifier for that!

# Load the package's __version__.py module as a dictionary.
ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))
#PACKAGE_DIR = ROOT_DIR // NAME
#about = {}
#with open(PACKAGE_DIR / 'VERSION') as f:
#    _version = f.read().strip()
#    about['__version__'] = _version


# Where the magic happens:
setup(
    name=NAME,
    version='0.0.1',
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    #install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
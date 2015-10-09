#!/usr/bin/env python
'''
Installation script for pyvolplot

Note:
To make a source distribution:
python setup.py sdist

To make an RPM distribution:
python setup.py bdist_rpm

To Install:
python setup.py install --prefix=/usr/local

See also:
python setup.py bdist --help-formats
'''

from distutils.core import setup

setup(name='pyvolplot',
      version='0.2',
      description='Volume Plotting Utilities',
      author='Gregory Lee',
      author_email='grlee77@gmail.com',
      url='https://github.com/grlee77/pyvolplot',
      packages=['pyvolplot'],
     )

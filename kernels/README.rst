kernels
============
kernels is a Python module with utility functions for data mining to support 
the Data Sciences group at NASA Ames


Dependencies
============

The required dependencies to build the software are 
Python >= 3.7, setuptools, 
Numpy >= 1.15.4, 
SciPy >= 1.2.0, 
scikit-learn >= 0.20.3 
and a working C/C++ compiler.


Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --user

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install



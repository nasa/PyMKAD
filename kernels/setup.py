from distutils.core import setup, Extension
from os.path import join
import os
import numpy as np


nlcs_sources = [join('src', 'nlcs', 'nlcs_wrapper.cpp'),\
                    join('src', 'nlcs', 'lcs.cpp')]  

setup(name = 'nlcs', version = '1.0',  \
   ext_modules = [Extension('nlcs', nlcs_sources, include_dirs=[join(os.path.split(np.__file__)[0],'core','include')])])

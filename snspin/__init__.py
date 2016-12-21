#!/usr/bin/env python

"""
SuperNova SPectral INdicators: Spectral indicators measurement for supernovae

.. moduleauthor:: N. Chotard <nchotard@in2p3.fr>

"""

import os
import glob

# Automatically import all modules (python files)
__all__ = [os.path.basename(m).replace('.py', '') for m in glob.glob("snspin/*.py")
           if '__init__' not in m]

# Set to True if you want to import all previous modules directly
if True:
    for pkg in __all__:
        __import__(__name__ + '.' + pkg)

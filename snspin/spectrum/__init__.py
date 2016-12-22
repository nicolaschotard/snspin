#!/usr/bin/env python

"""
Spectrum utilities.
"""

import os
import glob

# Automatically import all modules (python files)
__all__ = [os.path.basename(m).replace('.py', '')
           for m in glob.glob("snspin/spectrum/*.py")
           if '__init__' not in m]

# Set to True if you want to import all previous modules directly
if True:
    for pkg in __all__:
        __import__(__name__ + '.' + pkg)

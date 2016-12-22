#!/usr/bin/env python

"""
Spectral covariance matrix (variance and correlation) study
"""


import cPickle
import yaml


def loaddata(filename):
    """
    Load data from a yaml (.yaml or .yml) or a pickle file (.pkl).
    """
    if filename.endswith('yaml') or filename.endswith('yml'):
        return yaml.load(open(filename))
    elif filename.endswith('pkl'):
        return cPickle.load(open(filename))
    else:
        raise 'Wrong extension: %s (extension needed: .yaml, .yml or .pkl).' % filename

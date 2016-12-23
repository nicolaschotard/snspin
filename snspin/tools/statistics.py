#!/usr/bin/env python

"""
Statistics tools
"""

import warnings
import numpy as N


# Histogram utilities ==============================


def get_range(x, range=None, log=False, percentiles=False):
    """
    Get range from *x* and *range*=(*min*, *max*) or `None`. If *min*
    (resp. *max*) is `None`, *xmin* is set to min of *x*, or of strictly
    positive *x* if *log* (resp. *xmax* is set to the max of *x*). If
    *percentiles*, *range* is actually expressed in percentiles (in percents).

    >>> import numpy as N
    >>> get_range(N.linspace(0, 10, 101), range=(5, 95), percentiles=True)
    (0.5, 9.5)
    """

    if range is not None:               # Specified range
        if percentiles:                 # Range in percentiles
            vmin, vmax = range
            if vmin is None:            # Take the min
                vmin = 0
            if vmax is None:            # Take the max
                vmax = 100
            xmin, xmax = N.percentile(x, (vmin, vmax))  # Range in values
        else:
            xmin, xmax = range
    else:                               # Full range
        xmin, xmax = None, None

    xx = N.ravel(x)
    if xmin is None:                    # Automatic xmin = min(x)
        if log:                         # xmin = min(x>0)
            xmin = xx[xx > 0].min()     # Might raise ValueError
        else:
            xmin = xx.min()
    if xmax is None:                    # Automatic xmax = max(x)
        xmax = xx.max()

    return xmin, xmax


def hist_binwidth(x, choice='FD', range=None, percentiles=False):
    """
    Optimal histogram binwidth. Choices are:

    - 'S': Scott's choice
    - 'FD': Freedman & Diaconis (1981), fast, fair if single-peaked [default]
    - 'SS': Shimazaki and Shinomoto (2007), slow, best choice if double-peaked
    - 'BR': Birge and Rozenholc (2006), slow

    Analysis is restricted to *range*=(*min*, *max*) if not `None`
    (full range by default).

    References:

    - `Histogram <http://en.wikipedia.org/wiki/Histogram>`_
    - `Histogram Bin-width Optimization
      <http://176.32.89.45/~hideaki/res/histogram.html>`_
    """

    xx = N.ravel(x)
    xmin, xmax = get_range(xx, range, percentiles=percentiles)
    xx = xx[(xx >= xmin) & (xx <= xmax)]

    if choice == 'FD':                     # Freedman and Diaconis (1981)
        l, h = N.percentile(xx, [25., 75.])
        h = 2 * (h - l) / len(xx)**(1./3.)
    elif choice == 'S':                    # Scott's choice
        h = 3.49 * N.std(xx, ddof=1) / len(xx)**(1./3.)
    elif choice == 'BR':                   # Birge and Rozenholc (2006)
        def penalty(nbin):
            return nbin - 1 + N.log(nbin)**2.5

        def likelihood(nbin):
            hist, bins = N.histogram(xx, bins=nbin)
            return (hist *
                    N.log(nbin * N.maximum(hist, 1) / float(len(xx)))).sum()
        nbins = N.arange(2, round(len(xx) / N.log(len(xx))) + 1, dtype='i')
        nbin = nbins[N.argmax([likelihood(n) - penalty(n) for n in nbins])]
        h = (xmax - xmin) / nbin
    elif choice == 'SS':                   # Shimazaki and Shinomoto (2007)
        # http://web.mit.edu/hshimaza/www//res/histogram.html
        def objf(nbin):
            hist, bins = N.histogram(xx, bins=nbin)
            delta = bins[1] - bins[0]
            k = hist.mean()
            v = hist.var(ddof=0)
            # print "nbin", nbin, delta, k, v, (2*k - v)/delta**2
            return (2 * k - v) / delta**2
        nbins = N.arange(2, round(len(xx) / N.log(len(xx))) + 1, dtype='i')
        nbin = nbins[N.argmin([objf(n) for n in nbins])]
        h = (xmax - xmin) / nbin
    else:
        raise ValueError("Unknow histogram binwidth's choice '%s'" % choice)

    if not h:
        warnings.warn("Cannot compute binwidth for Dirac-like distribution")
        h = 1                   # Stupid default value

    return h


def hist_bins(x, choice='FD', range=None, percentiles=False, log=False):
    """Optimal binning. See :func:`hist_binwidth` for details."""

    xmin, xmax = get_range(x, range=range, percentiles=percentiles, log=log)
    if log:
        from math import log10
        lxmin, lxmax = log10(xmin), log10(xmax)
        xx = N.ravel(x)
        xx = xx[xx >= xmin]
        return N.logspace(lxmin, lxmax,
                          hist_nbin(N.log10(xx), choice=choice,
                                    range=(lxmin, lxmax)))
    else:
        return N.linspace(xmin, xmax,
                          hist_nbin(x, choice=choice,
                                    range=range, percentiles=percentiles))


# Robust statistics ==============================

    
def hist_nbin(x, choice='FD', range=None, percentiles=False):
    """Optimal number of bins. See :func:`hist_binwidth` for details."""

    xmin, xmax = get_range(x, range=range, percentiles=percentiles)

    return int(N.ceil((xmax - xmin) /
                      hist_binwidth(x, choice=choice,
                                    range=range, percentiles=percentiles)))


def wpercentile(a, q, weights=None):
    """Compute weighted percentiles *q* [%] of input 1D-array *a*."""

    a = N.asarray(a)
    if a.ndim > 1:
        raise NotImplementedError("implemented on 1D-arrays only")

    if weights is None:
        weights = N.ones_like(a)
    else:
        assert len(weights) == len(a), "incompatible weight and input arrays"
        assert (weights > 0).all(), "weights are not always strictly positive"

    isorted = N.argsort(a)
    sa = a[isorted]
    sw = weights[isorted]
    sumw = N.cumsum(sw)                        # Strictly increasing
    # 0-100 score at center of bins
    scores = 1e2 * (sumw - 0.5 * sw) / sumw[-1]

    def interpolate(q):
        i = scores.searchsorted(q)
        if i == 0:                        # Below 1st score
            val = sa[0]
        elif i == len(a):                 # Above last score
            val = sa[-1]
        else:                           # Linear score interpolation
            val = (sa[i-1] * (scores[i] - q) + sa[i] * (q - scores[i-1])) / \
                  (scores[i] - scores[i-1])
        return val

    out = N.array([interpolate(qq) for qq in N.atleast_1d(q)])

    return out.reshape(N.shape(q))      # Same shape as input q


def median_stats(a, weights=None, axis=None, scale=1.4826, corrected=True):
    """
    Compute [weighted] median and :func:`nMAD` of array *a* along
    *axis*. Weighted computation is implemented for *axis* = None only. If
    *corrected*, apply finite-sample correction from Croux & Rousseeuw (1992).
    """

    if weights is not None:
        if axis is not None:
            raise NotImplementedError("implemented on 1D-arrays only")
        else:
            med = wpercentile(a, 50., weights=weights)
            nmad = wpercentile(N.abs(a - med), 50., weights=weights) * scale
    else:
        med = N.median(a, axis=axis)
        if axis is None:
            umed = med                       # Scalar
        else:
            umed = N.expand_dims(med, axis)  # Same ndim as a
        nmad = N.median(N.absolute(a - umed), axis=axis) * scale

    if corrected:
        # Finite-sample correction on nMAD (Croux & Rousseeuw, 1992)
        if axis is None:
            n = N.size(a)
        else:
            n = N.shape(a)[axis]
        if n <= 9:
            c = [0, 0, 1.196, 1.495, 1.363, 1.206, 1.200, 1.140, 1.129, 1.107][n]
        else:
            c = n / (n - 0.8)
        nmad *= c

    return med, nmad


def nMAD(a, weights=None, axis=None, scale=1.4826, corrected=True):
    """
    Normalized `Median Absolute Deviation
    <http://en.wikipedia.org/wiki/Median_absolute_deviation>`_ (nMAD) along
    given *axis* of array *a*::

      median(abs(a - median(a))) * scale

    For normally distributed data, *scale* should be set to::

      1/scipy.stats.norm.ppf(0.75) = 1.4826022185056018...

    If *corrected*, a finite-sample correction is applied (Croux &
    Rousseeuw, 1992), equals to n/(n-0.8) for n>=10.
    """

    med, nmad = median_stats(a, axis=axis, weights=weights,
                             scale=scale, corrected=corrected)
    return nmad

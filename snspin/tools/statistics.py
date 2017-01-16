#!/usr/bin/env python

"""Statistics tools/"""

import warnings
import numpy as N
import scipy.stats as stats

# Histogram utilities ==============================


def get_range(x, arange=None, log=False, percentiles=False):
    """
    Get arange from *x* and *arange*=(*min*, *max*) or `None`. If *min*
    (resp. *max*) is `None`, *xmin* is set to min of *x*, or of strictly
    positive *x* if *log* (resp. *xmax* is set to the max of *x*). If
    *percentiles*, *arange* is actually expressed in percentiles (in percents).

    >>> import numpy as N
    >>> get_range(N.linspace(0, 10, 101), arange=(5, 95), percentiles=True)
    (0.5, 9.5)
    """

    if arange is not None:              # Specified arange
        if percentiles:                 # Arange in percentiles
            vmin, vmax = arange
            if vmin is None:            # Take the min
                vmin = 0
            if vmax is None:            # Take the max
                vmax = 100
            xmin, xmax = N.percentile(x, (vmin, vmax))  # Arange in values
        else:
            xmin, xmax = arange
    else:                               # Full arange
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


def hist_binwidth(x, choice='FD', arange=None, percentiles=False):
    """
    Optimal histogram binwidth. Choices are:

    - 'S': Scott's choice
    - 'FD': Freedman & Diaconis (1981), fast, fair if single-peaked [default]
    - 'SS': Shimazaki and Shinomoto (2007), slow, best choice if double-peaked
    - 'BR': Birge and Rozenholc (2006), slow

    Analysis is restricted to *arange*=(*min*, *max*) if not `None`
    (full arange by default).

    References:

    - `Histogram <http://en.wikipedia.org/wiki/Histogram>`_
    - `Histogram Bin-width Optimization
      <http://176.32.89.45/~hideaki/res/histogram.html>`_
    """

    xx = N.ravel(x)
    xmin, xmax = get_range(xx, arange, percentiles=percentiles)
    xx = xx[(xx >= xmin) & (xx <= xmax)]

    if choice == 'FD':                     # Freedman and Diaconis (1981)
        l, h = N.percentile(xx, [25., 75.])
        h = 2 * (h - l) / len(xx)**(1. / 3.)
    elif choice == 'S':                    # Scott's choice
        h = 3.49 * N.std(xx, ddof=1) / len(xx)**(1. / 3.)
    elif choice == 'BR':                   # Birge and Rozenholc (2006)
        def penalty(nbin):
            return nbin - 1 + N.log(nbin)**2.5

        def likelihood(nbin):
            hist, bins = N.histogram(xx, bins=nbin)
            return (hist * N.log(nbin * N.maximum(hist, 1) / float(len(xx)))).sum()
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


def hist_bins(x, choice='FD', arange=None, percentiles=False, log=False):
    """Optimal binning. See :func:`hist_binwidth` for details."""

    xmin, xmax = get_range(x, arange=arange, percentiles=percentiles, log=log)
    if log:
        from math import log10
        lxmin, lxmax = log10(xmin), log10(xmax)
        xx = N.ravel(x)
        xx = xx[xx >= xmin]
        return N.logspace(lxmin, lxmax,
                          hist_nbin(N.log10(xx), choice=choice,
                                    arange=(lxmin, lxmax)))
    else:
        return N.linspace(xmin, xmax,
                          hist_nbin(x, choice=choice,
                                    arange=arange, percentiles=percentiles))


# Robust statistics ==============================


def hist_nbin(x, choice='FD', arange=None, percentiles=False):
    """Optimal number of bins. See :func:`hist_binwidth` for details."""

    xmin, xmax = get_range(x, arange=arange, percentiles=percentiles)

    return int(N.ceil((xmax - xmin) /
                      hist_binwidth(x, choice=choice,
                                    arange=arange, percentiles=percentiles)))


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


# Correlation coefficients ==============================


def correlation_CI(rho, n, cl=0.95):
    """
    Compute Pearson's correlation coefficient confidence interval, at
    (fractional) confidence level *cl*, for an observed correlation coefficient
    *rho* obtained on a sample of *n* (effective) points.

    cl=0.6827 corresponds to a 1-sigma error, 0.9973 for a 3-sigma
    error (`2*scipy.stats.norm.cdf(n)-1` or `1-2*sigma2pvalue(n)` for
    a n-sigma error).

    Sources: `Confidence Interval of rho
    <http://vassarstats.net/rho.html>`_, `Correlation CI
    <http://onlinestatbook.com/chapter8/correlation_ci.html>`_
    """

    assert -1 < rho < 1, "Correlation coefficient should be in ]-1,1["
    assert n >= 6, "Insufficient sample size"
    assert 0 < cl < 1, "Confidence level should be in ]0,1["

    z = N.arctanh(rho)                  # Fisher's transformation
    # z is normally distributed with std error = 1/sqrt(N-3)
    zsig = stats.distributions.norm.ppf(0.5 * (cl + 1)) / N.sqrt(n - 3)
    # Confidence interval on z is [z-zsig, z+zsig]

    return N.tanh([z - zsig, z + zsig])      # Confidence interval on rho


def correlation(x, y, method='pearson',
                error=False, confidence=0.6827, symmetric=False):
    """
    Compute Pearson/Spearman (unweighted) coefficient correlation rho between
    *x* and *y*.

    If `error=True`, returns asymmetric/symmetrized errors on *rho*,
    for a given confidence, see :func:`correlation_CI` (only
    implemented for Pearson).
    """

    assert len(x) == len(y), "Incompatible input arrays x and y"

    if method.lower() == 'pearson':
        rho, p = stats.pearsonr(x, y)
    elif method.lower() == 'spearman':
        rho, p = stats.spearmanr(x, y)
    else:
        raise ValueError("Unknown correlation method '%s'" % method)

    if not error:
        return rho

    # Compute error on correlation coefficient
    if method.lower() != 'pearson':
        raise NotImplementedError("Error on correlation coefficient is "
                                  "implemented for Pearson's correlation only.")

    rho_dn, rho_up = correlation_CI(rho, n=len(x), cl=confidence)
    drm = rho - rho_dn
    drp = rho_up - rho

    if symmetric:
        return rho, N.hypot(drm, drp) / 1.4142  # Symmetrized error
    else:
        return rho, drm, drp              # Assymmetric errors


# Statistical tests ==============================


def pvalue2sigma(p):
    """
    Express the input one-sided *p*-value as a sigma equivalent significance
    from a normal distribution (the so-called *z*-value).

    =====  =======  =================
    sigma  p-value  terminology
    =====  =======  =================
    1      0.1587
    1.64   0.05     significant
    2      0.0228
    2.33   0.01     highly significant
    3      0.0013   evidence
    3.09   0.001
    5      2.9e-7   discovery
    =====  =======  =================

    >>> pvalue2sigma(1e-3) # p=0.1% corresponds to a ~3-sigma significance
    3.0902323061678132
    """

    return stats.distributions.norm.isf(p)  # isf = ppf(1 - p)

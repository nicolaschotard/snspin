#/bin/env python


"""Merge two spectrum of the same object.

The spec object used as a spectrum should contain :
spec.x as the wavelength (an array of size N) as the cernter of bin
spec.y as the flux (an array of size N)
spec.v as the variance of the flux (an array of size N)
spec.step : either an array (of size N) containing the individual bin width or a single 
            integer in case fo constant binning.
"""


import numpy as N


def bin(spec, x):
    """
    Return the bin in which x is in the spectrum.

    In case x is not in the spectrum, the return value is an empty array
    """
    # comput the true / false array
    ubound = spec.x + spec.step / 2
    lbound = N.concatenate(([(spec.x - spec.step / 2)[0]], ubound[: - 1]))
    cond = (x < ubound) & (x >= lbound)
    return N.nonzero(cond)


def mean(spec, x1, x2):
    """Compute a mean value bewteen x1 and x1.

    Compute the integral of the flux over the wavelength range defined as [x1,x2]
    divided by the wavelength range in order to get a flux / wavelength.
    the variance of this quantity is returned as a 2nd parameter
    Raises ValueError if the spec range soesn't cover the intended bin width"""
    # determine first, middle and last bins : upper bound belongs to upper bin
    bin1 = bin(spec, x1)
    bin2 = bin(spec, x2)
    if bin1 == bin2 or x2 == (spec.x + spec.step / 2)[-1]:
        return spec.y[bin1], spec.v[bin1]
    binbetween = range(bin1[0][0] + 1, bin2[0][0])
    # compute flux integral
    flux1 = spec.y[bin1] * ((spec.x + spec.step / 2)[bin1] - x1)
    flux2 = spec.y[bin2] * (x2 + (- spec.x + spec.step / 2)[bin2])
    fluxbetween = sum((spec.y * spec.step)[binbetween])
    retflux = (flux1 + flux2 + fluxbetween) / (x2 - x1)
    # compute variance of the previous quantity
    var1 = spec.v[bin1] * ((spec.x + spec.step / 2)[bin1] - x1)**2
    var2 = spec.v[bin2] * (x2 + (- spec.x + spec.step / 2)[bin2])**2
    varbetween = sum((spec.v * spec.step**2)[binbetween])
    retvar = (var1 + var2 + varbetween) / (x2 - x1)**2
    if len(retflux) == 0:
        raise ValueError("Bound error %f %f"%(x1, x2))
    return retflux, retvar

def rebin(spec, xarray):
    """xarray is the array of bin edges (1 more than number of bins)."""
    # this implementation was chosen instead of providing 2 arrays of length n
    outx = (xarray[1:] + xarray[: - 1]) / 2
    outflux = N.zeros(len(outx))
    outvar = N.zeros(len(outx))
    for i in xrange(len(outx)):
        outflux[i], outvar[i] = mean(spec, xarray[i], xarray[i + 1])
    return outx, outflux, outvar

class MergedSpectrum(object):

    """Merge two spectra."""

    def __init__(self, specb, specr):
        """
        Takes 2 SNIFS spectra and returns a merged output.

        The binning in the interregion is aligned to R binning to avoid oversampling.
        """
        if specb.x[0] > specr.x[0]:
            raise ValueError("specb has to be bluer than specr")

        # compute the range of pure original spectra and of overlap
        condb = specb.x - specb.step / 2 < specr.x[0] - specr.step / 2
        condrbin = specr.x - specr.step / 2 < specb.x[-1] + specb.step / 2
        condr = specr.x + specr.step / 2 < specb.x[-1] + specb.step / 2
        # compute the rebinning wavelength array and adjuste the first bin
        # to match the end of pure b spectrum
        rebinr = specr.x[condrbin] - specr.step / 2
        rebinr[0] = specb.x[condb][-1] + specb.step / 2

        # rebin b to r in the overlap region and prepare new r spectrum (sr)
        # the ' + 0' is here to force a copy.
        rbx, rbflux, rbvar = rebin(specb, rebinr)
        srx = specr.x + 0
        srflux = specr.y + 0
        srvar = specr.v + 0

        # replace new r spectrum in the overlap range width :
        # the weighted (optimal) average of b rebinned and original r is taken.
        srx[condr] = rbx
        srflux[condr] = (rbflux  /  rbvar  +  specr.y[condr]  /  specr.v[condr])  /  \
                        (1. / rbvar  +  1. / specr.v[condr])
        srvar[condr] = 1. / (1. / rbvar + 1. / specr.v[condr])
        srstep = N.ones(len(srx)) * specr.step
        srstep[0] = rebinr[1] - rebinr[0]

        # put together untouched b spectrum and the new r spectrum
        self.x = N.concatenate((specb.x[condb], srx))
        self.y = N.concatenate((specb.y[condb], srflux))
        self.v = N.concatenate((specb.v[condb], srvar))
        self.step = N.concatenate((specb.step * N.ones(len(specb.x[condb])), srstep))
        self.name = specb.name




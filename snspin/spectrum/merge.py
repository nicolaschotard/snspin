#/bin/env python

"""
the spec object used as a spectrum should contain :
spec.x as the wavelength (an array of size N) as the cernter of bin
spec.y as the flux (an array of size N)
spec.v as the variance of the flux (an array of size N)
spec.step : either an array (of size N) containing the individual bin width or a single 
            integer in case fo constant binning.
"""

import numpy as N

def bin(spec, x):
    """returns the bin in which x is in the spectrum
    in case x is not in the spectrum, the return value is an empty array"""
    # comput the true / false array
    ubound = spec.x + spec.step / 2
    lbound = N.concatenate(([(spec.x - spec.step / 2)[0]], ubound[: - 1]))
    cond = (x < ubound) & (x >= lbound)
    # check abiguities : this case should never (any more) happen
    # if len(cond[cond])==2:
    #    print "Warning : 2 solutions found"
    #    cond[cond]=N.array((True,False))

    # return as an integer array
    return N.nonzero(cond)

def mean(spec, x1, x2):
    """returns the integral of the flux over the wavelength range defined as [x1,x2]
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
    """xarray is the array of bin edges (1 more than number of bins)"""
    # this implementation was chosen instead of providing 2 arrays of length n
    outx = (xarray[1:] + xarray[: - 1]) / 2
    outflux = N.zeros(len(outx))
    outvar = N.zeros(len(outx))
    for i in xrange(len(outx)):
        outflux[i], outvar[i] = mean(spec, xarray[i], xarray[i + 1])
    return outx, outflux, outvar

class MergedSpectrum:
    def __init__(self, specB, specR):
        """
        takes 2 SNIFS spectra and returns a merged output.

        The binning in the interregion is aligned to R binning
        to avaoid oversampling."""

        if specB.x[0] > specR.x[0]:
            raise ValueError("specB has to be bluer than specR")

        # compute the range of pure original spectra and of overlap
        condB = specB.x - specB.step / 2 < specR.x[0] - specR.step / 2
        condRbin = specR.x - specR.step / 2 < specB.x[-1] + specB.step / 2
        condR = specR.x + specR.step / 2 < specB.x[-1] + specB.step / 2
        # compute the rebinning wavelength array and adjuste the first bin
        # to match the end of pure B spectrum
        rebinR = specR.x[condRbin] - specR.step / 2
        rebinR[0] = specB.x[condB][-1] + specB.step / 2

        # rebin B to R in the overlap region and prepare new R spectrum (sR)
        # the ' + 0' is here to force a copy.
        rBx, rBflux, rBvar = rebin(specB, rebinR)
        sRx = specR.x + 0
        sRflux = specR.y + 0
        sRvar = specR.v + 0

        # replace new R spectrum in the overlap range width :
        # the weighted (optimal) average of B rebinned and original R is taken.
        sRx[condR] = rBx
        sRflux[condR] = (rBflux  /  rBvar  +  specR.y[condR]  /  specR.v[condR])  /  \
                        (1. / rBvar  +  1. / specR.v[condR])
        sRvar[condR] = 1. / (1. / rBvar + 1. / specR.v[condR])
        sRstep = N.ones(len(sRx)) * specR.step
        sRstep[0] = rebinR[1] - rebinR[0]

        # put together untouched B spectrum and the new R spectrum
        self.x = N.concatenate((specB.x[condB], sRx))
        self.y = N.concatenate((specB.y[condB], sRflux))
        self.v = N.concatenate((specB.v[condB], sRvar))
        self.step = N.concatenate((specB.step * N.ones(len(specB.x[condB])), sRstep))
        self.name = specB.name




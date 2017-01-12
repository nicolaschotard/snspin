######################################################################
# Filename:      pySnurp.py
# Version:       $Revision: 1.137 $
# Description:   Simple module providing few FITS-based I/O classes.
# Author:        Yannick Copin <yannick@ipnl.in2p3.fr>
# Author:        $Author: ycopin $
# Created at:    Tue Oct 18 11:42:37 2005
# Modified at:   Tue Apr 27 18:04:54 2010
# Modified by:   Yannick Copin <ycopin@ipnl.in2p3.fr>
# $Id: pySnurp.py,v 1.137 2015/05/18 08:23:53 ycopin Exp $
######################################################################

"""
Simple module providing few FITS-based I/O classes and other facilities
(mostly historical name parsing).

* ReStructuredText facilities: now in ToolBox.ReST
* Plotting utilities (e.g. get_backend): now in ToolBox.MPL
* Extinction routines: now in ToolBox.Astro.Extinction
* Flux and date conversion routines: now in ToolBox.Astro.Coords
"""

__author__ = "Yannick Copin <ycopin@ipnl.in2p3.fr>"
__version__ = "$Id: pySnurp.py,v 1.137 2015/05/18 08:23:53 ycopin Exp $"

import os
import re

import numpy
import pyfits

import warnings
# Ignore warnings from pyfits.writeto
# (https://github.com/spacetelescope/PyFITS/issues/43)
warnings.filterwarnings("ignore", "Overwriting existing file")

# Spectrum class ##############################


class Spectrum:
    """
    Class to read and manage a spectrum from a FITS file (NAXIS=1),
    including the associated [co]variance from an extension or an
    external file.
    """

    def __init__(self, name, varname=None, keepFits=True):
        """
        Spectrum initialization.

        Note: use helper function [Spectrum.]read_spectrum method for
        a transparent use.
        """

        self.name = name        # Generic name
        if name is None:        # Blank instance
            return
        self._readFits(name,    # Read signal [and variance if any]
                       mode='update' if keepFits else 'readonly')
        if not keepFits:
            self.close()
        if varname:             # Override variance extension if any
            if self.varname:    # Set by _readFits from var. extension
                warnings.warn("%s: VARIANCE extension overriden by %s" %
                              (name, varname), RuntimeWarning)
            self.varname = varname
            V = Spectrum(varname, varname=None, keepFits=keepFits)
            assert (V.npts, V.start, V.step) == (self.npts, self.start, self.step), \
                "Incompatible variance spectrum '%s' wrt. to spectrum '%s'" % \
                (varname, name)
            self.v = V.y.copy()
            # All other attributes and header keywords should be
            # essentially the same as for signal spectrum, no need to
            # keep them

        if self.hasCov and self.hasVar:  # Test variance vs. cov. coherence
            assert numpy.allclose(self.v, self.cov.diagonal()), \
                "%s: VARIANCE and COVARiance diagonal are incompatible"

        # Channel
        self.X = self.readKey('CHANNEL', 'X')[0].upper()  # 'B' or 'R' (or 'X')

    @property
    def hasVar(self):

        return hasattr(self, 'v') and self.v is not None

    @property
    def hasCov(self):

        return hasattr(self, 'cov') and self.cov is not None

    def close(self):
        """Close FITS file (if any) and forget about it."""

        if self._fits is not None:
            self._fits.close()
            self._fits = None

    def __str__(self):

        s = "Spectrum %s%s: %d px [%.2f-%.2f A] @%.2f A/px" % \
            (self.name, ' [%c]' % self.X if self.X != 'X' else '',
             self.npts, self.start, self.end, self.step)
        if self.hasCov:
            s += " with covariance"
        elif self.hasVar:
            s += " with variance"
        else:
            s += " (no [co]variance)"
        if self._fits is None:
            s += " (closed)"
        if hasattr(self, 'ebmv'):       # Dereddened spectrum
            s += "\n   Dereddened: E(B-V)=%.3f, Rv=%.2f, ExtLaw=%s" % \
                 (self.ebmv, self.rv, self.law)
        if hasattr(self, 'zorig'):      # Deredshifted spectrum
            s += "\n   Deredshifted: z=%.5f, exp=%d" % (self.zorig, self.zexp)

        return s

    def _readFits(self, name, mode='readonly'):
        """
        Initialize a Spectrum from FITS spectrum name. 'name' can be
        'name[ext]', in which case only extension 'ext' is
        considered.
        """

        # Decipher name and extension from name[EXT]
        self.filename, self.ext = get_extension(name)

        self._fits = pyfits.open(self.filename,
                                 mode=mode, ignore_missing_end=True)
        extnames = [h.name for h in self._fits]  # "PRIMARY", etc.

        try:
            spec = self._fits[self.ext]      # Spectrum extension
        except (IndexError, KeyError,):
            raise IOError("Cannot read extension %s in %s:%s" %
                          (self.ext, self.filename, extnames))

        self._hdr = spec.header.copy()       # Spectrum header

        self._hdr['CRPIX1'] = self._hdr.get('CRPIX1', 1)  # Make it mandatory

        self.npts = self._hdr['NAXIS1']
        self.step = self._hdr['CDELT1']
        self.start = self._hdr['CRVAL1'] - \
            (self._hdr['CRPIX1'] - 1) * self.step
        self.end = self.start + (self.npts - 1) * self.step
        self.x = numpy.linspace(self.start, self.end, self.npts)  # Wavelength
        self.y = spec.data.copy()                                 # Signal

        if 'VARIANCE' in extnames:      # Read VARIANCE extension
            vhdr = self._fits['VARIANCE'].header
            vhdr['CRPIX1'] = vhdr.get('CRPIX1', 1)  # Make it mandatory
            try:
                assert vhdr['NAXIS1'] == self.npts
                assert vhdr['CDELT1'] == self.step
                assert vhdr['CRVAL1'] == self._hdr['CRVAL1']
                assert vhdr['CRPIX1'] == self._hdr['CRPIX1']
            except AssertionError:
                warnings.warn(
                    "%s[VARIANCE]: header incompatible with primary header" %
                    self.filename, RuntimeWarning)
            self.varname = "%s[VARIANCE]" % (self.filename)
            self.v = self._fits['VARIANCE'].data.copy()  # Variance
        else:
            self.varname = None
            self.v = None

        if 'COVAR' in extnames:         # Read COVAR extension
            vhdr = self._fits['COVAR'].header
            vhdr['CRPIX1'] = vhdr.get('CRPIX1', 1)  # Make it mandatory
            vhdr['CRPIX2'] = vhdr.get('CRPIX2', 1)
            try:
                assert vhdr['NAXIS1'] == vhdr['NAXIS2'] == self.npts
                assert vhdr['CDELT1'] == vhdr['CDELT2'] == self.step
                assert vhdr['CRVAL1'] == vhdr['CRVAL2'] == self._hdr['CRVAL1']
                assert vhdr['CRPIX1'] == vhdr['CRPIX2'] == self._hdr['CRPIX1']
            except AssertionError:
                warnings.warn(
                    "%s[VARIANCE]: header incompatible with primary header" %
                    self.filename, RuntimeWarning)
            self.covname = "%s[COVAR]" % (self.filename)

            self.cov = self._fits['COVAR'].data.copy()  # Lower-tri. covariance
            self.cov += numpy.triu(self.cov.T, 1)       # Reconstruct full cov.
        else:
            self.covname = None
            self.cov = None

    def readKey(self, keyword, default=None):
        """Read a single keyword, defaulting to *default* if any."""

        if default is None:
            return self._hdr[keyword]
        else:
            return self._hdr.get(keyword, default)

    def setKey(self, keywords=(), **kwargs):
        """
        Set keywords from *keywords*=((key, val[, comment]),) or kwargs
        'key=val' or 'key=(val, comment)'.
        """

        for key in keywords:
            name, val = key[0], key[1:]   # name, (value, [comment])
            self._hdr[name.upper()] = val
        for key in kwargs:
            self._hdr[key.upper()] = kwargs[key]

    def resetHeader(self):
        """Delete all non-standard keywords."""

        # Delete all reference keywords
        for k in self._hdr.items():
            del self._hdr[k[0]]

        # Add mandatory keywords
        self._hdr['SIMPLE'] = True
        self._hdr['BITPIX'] = -64
        self._hdr['NAXIS'] = 1
        self._hdr['NAXIS1'] = self.npts
        self._hdr['CDELT1'] = self.step
        self._hdr['CRPIX1'] = 1
        self._hdr['CRVAL1'] = self.start

    def writeto(self, outName, force=False, hdrOnly=False,
                keywords=(), **kwargs):
        """Save Spectrum to new FITS-file."""

        if self._fits is None:          # FITS file has been closed
            raise IOError("Cannot write to disk to closed FITS file")
        else:
            spec = self._fits[self.ext]

        self._hdr['CRPIX1'] = self._hdr.get('CRPIX1', 1)  # Make it mandatory

        if not hdrOnly:                 # Update FITS-data
            spec.data = numpy.array(self.y)

            # Update FITS-header
            self._hdr['NAXIS1'] = self.npts
            self._hdr['CDELT1'] = self.step
            self._hdr['CRVAL1'] = self.start + \
                (self._hdr['CRPIX1'] - 1) * self.step

            # Remove any prior VARIANCE/COVARiance extensions if any:
            # they will then be re-added as needed
            extnames = [ext.name for ext in self._fits]
            for extname in ("VARIANCE", "COVAR"):
                if extname in extnames:
                    i = self._fits.index_of(extname)
                    self._fits.remove(self._fits[i])

            if self.hasVar and kwargs.pop('varext', True):
                # Add variance spectrum as extension VARIANCE
                assert len(self.v) == self.npts, \
                    "Variance extension (%d px) " \
                    "is not coherent with signal (%d px)" % \
                    (len(self.v), self.npts)
                var = pyfits.ImageHDU(self.v, name='VARIANCE')
                var.header['CRVAL1'] = (
                    self.start + (self._hdr['CRPIX1'] - 1) * self.step)
                var.header['CDELT1'] = self.step
                var.header['CRPIX1'] = self._hdr['CRPIX1']
                self._fits.append(var)

            if self.hasCov and kwargs.pop('covext', True):
                # Add covariance array as extension COVAR
                assert self.cov.shape == (self.npts, self.npts), \
                    "Covariance extension %s " \
                    "is not coherent with signal (%d px)" % \
                    (self.cov.shape, self.npts)
                # Add lower-tri COVARiance matrix as an image extension
                # cov = pyfits.CompImageHDU(numpy.tril(self.cov), name='COVAR')
                cov = pyfits.ImageHDU(numpy.tril(self.cov), name='COVAR')
                cov.header['CRVAL1'] = cov.header['CRVAL2'] = (
                    self.start + (self._hdr['CRPIX1'] - 1) * self.step)
                cov.header['CDELT1'] = cov.header['CDELT2'] = \
                    self.step
                cov.header['CRPIX1'] = cov.header['CRPIX2'] = \
                    self._hdr['CRPIX1']
                self._fits.append(cov)

        # Update required keywords
        if keywords or kwargs:
            self.setKey(keywords=keywords, **kwargs)

        # Test output file presence
        if force:
            clobber = True              # Overwrite existing file
        else:
            clobber = False             # DO NOT overwrite existing file...
            if os.path.exists(outName):
                ans = raw_input("Overwrite output file '%s'? [N/y] " % outName)
                if ans and ans[0].lower() == 'y':
                    clobber = True      # ...except if confirmed
                else:
                    warnings.warn("Output file %s not overwritten" % outName)
                    return

        # Reset header from local copy self._hdr
        spec.header = self._hdr
        # Fix missing keywords (but should be OK)
        self._fits.writeto(outName, clobber=clobber, output_verify='silentfix')
        self.name = outName
        self.filename = outName

    def gaussianFilter(self, sigma, excl=None, inplace=True):
        """
        Apply a gaussian smoothing to a Spectrum, w/ possible
        ExlcDomain. Sigma in A. If not inplace, the smoothed signal is
        returned and the spectrum is not modified.

        TODO: - linearly interpolate excluded bands before smoothing
        """

        from scipy.ndimage import filters

        # Linear interpolation over excluded domains
        if excl:
            # print "WARNING: No interpolation over excluded domains."
            pass

        # Conversion to Float64 is required...
        f = filters.gaussian_filter1d(
            self.y.astype('d'), sigma / self.step, mode='nearest')

        if inplace:
            self.y = f
            if self.hasVar or self.hasCov:
                warnings.warn("%s: [co]variance extension left unfiltered" %
                              self.name)
            self.setKey(FILTGAUS=(sigma / self.step,
                                  "Gaussian filtering sigma [px]"))

        return f

    def sgFilter(self, hsize=11, order=4, excl=None, inplace=True):
        """
        Apply a Savitzky-Golay smoothing to a Spectrum, w/ possible
        ExlcDomain. If not inplace, the smoothed signal is returned
        and the spectrum is not modified.
        """

        from snspin.tools.smoothing import savitzky_golay

        # Linear interpolation over excluded domains
        if excl:
            # print "WARNING: No interpolation over excluded domains."
            pass

        f = savitzky_golay(self.y, hsize, order, derivative=0)
        if inplace:
            self.y = f
            if self.hasVar or self.hasCov:
                warnings.warn("%s: [co]variance extension left unfiltered" %
                              self.name)
            self.setKey(FILTSG=(','.join(map(str, (hsize, order))),
                                "Savitzky-Golay filter size, order"))

        return f

    def findRange(self, range=(None, None)):
        """Find pixel indices corresponding to world-coord range."""

        rmin, rmax = range                 # Requested range
        lmin, lmax = self.start, self.end  # Actual range

        # Some consistency tests
        if rmin is not None and rmax is not None and rmin >= rmax:
            raise ValueError("Requested range %s is not ordered." % range)
        if (rmin is not None and rmin >= lmax) or \
           (rmax is not None and rmax <= lmin):
            raise ValueError("Requested range %s incompatible "
                             "with actual range %s." % (range, (lmin, lmax)))

        # Find new limit indices
        if rmin is not None:
            r = numpy.round((rmin - lmin) / self.step, 6)
            imin = max(int(numpy.ceil(r)), 0)
        else:
            imin = 0
        if rmax is not None:
            r = numpy.round((rmax - lmin) / self.step, 6)
            imax = min(int(numpy.floor(r)), self.npts - 1)
        else:
            imax = self.npts - 1

        return imin, imax + 1              # imax pixel should be included

    def truncate(self, range=(None, None), verbose=False):
        """Truncate spectrum to world-coord range."""

        rmin, rmax = range                 # Requested range
        lmin, lmax = self.start, self.end  # Actual range

        imin, imax = self.findRange(range=range)

        if imin > 0 or imax < self.npts:   # Apply truncation
            if verbose:
                print "%s: truncation to [%.2f-%.2f] " \
                      "gives new range [%.2f-%.2f] (%d px)" % \
                      (self.name, rmin, rmax,
                       self.x[imin], self.x[imax - 1], len(self.x[imin:imax]))

            # Truncation and parameter update
            self.x = self.x[imin:imax]
            self.y = self.y[imin:imax]
            if self.hasVar:
                self.v = self.v[imin:imax]
            if self.hasCov:
                self.cov = self.cov[imin:imax, imin:imax]
            self.npts = len(self.x)
            self.start, self.end = self.x[0], self.x[-1]


    def deredden(self, ebmv, law='OD94', Rv=3.1):
        """Deredden spectrum using E(B-V) and a nextinction law:

        :param float ebmv: E(B-V) value.
        :param int law: Extinction law. Could be CCM89, OD94, FM98 or G08
        :param float Rv: Value for Rv. Default is 3.1"""
        from ToolBox.Astro.Extinction import extinctionFactor

        if hasattr(self, 'zorig'):      # Spectrum has been deredshifted
            raise ValueError, \
                "Dereddening should be done prior to deredshifting."

        # Extinction factor (<1)
        ext = extinctionFactor(self.x, ebmv, Rv=Rv, law=law)
        self.y /= ext
        if self.hasVar:
            self.v /= ext**2
        if self.hasCov:
            self.cov /= ext**2

        self.ebmv = ebmv        # Mark spectrum as unreddened
        self.rv = Rv
        self.law = law

        self.setKey(MWEBMV=(ebmv, "MW E(B-V) correction applied"),
                    MWRV=(Rv, "R_V used for MW E(B-V) correction"),
                    MWLAW=(law, "Extinction law used for MW correction"))

    def deredshift(self, z, exp=3):
        """
        Deredshift spectrum from z to 0, and apply a (1+z)**exp flux-correction.

        exp=3 is for erg/s/cm2/A spectra to be latter corrected using proper
        (comoving) distance but *not* luminosity distance.
        """

        zp1 = 1. + z
        self.x /= zp1           # Wavelength correction
        self.step /= zp1
        self.start, self.end = self.x[0], self.x[-1]
        zp1exp = zp1 ** exp
        self.y *= zp1exp        # Flux correction
        if self.hasVar:
            self.v *= zp1exp**2
        if self.hasCov:
            self.cov *= zp1exp**2

        self.zorig = z          # Mark spectrum as deredshifted
        self.zexp = exp

        self.setKey(ZORIG=(z, "Redshift correction applied"),
                    ZEXP=(exp, "Flux correction applied is (1+z)**zexp"))

    @classmethod
    def read_spectrum(cls, arg, keepFits=True):
        """
        Return an initiated Spectrum from arg=name[, var_name], including
        proper deciphering of arg.
        """

        innames = arg.split(',')     # Check for spectrum, var_spectrum
        specname = innames[0]
        if len(innames) == 2:        # Explicit specName, var_specName
            varname = innames[1]
        else:                        # Get variance name and test existence
            varname = cls.get_varname(specname, exists=True)

        # Set variance if any
        return cls(specname, varname=varname, keepFits=keepFits)

    @staticmethod
    def get_varname(specname, exists=False):
        """
        Return variance spectrum name associated to spectrum 'specname',
        for historical or DB filenaming. Assumes variance spectrum is
        located in same directory as spectrum. If exists, test if
        variance file exists or return None.
        """

        path, bname = os.path.split(specname)

        tokens = match_DBname(bname)
        if tokens is None:            # Historical filenaming
            varname = 'var_' + bname  # Prefix bname with 'var_'
        else:                         # DB filenaming
            # Increase XFclass by one
            tokens[7] = '%03d' % (int(tokens[7]) + 1)
            varname = tokens[0] + '_'.join(tokens[1:-1]) + tokens[-1]

        outname = os.path.join(path, varname)        # Add path to varname
        if exists and not os.path.isfile(outname):
            outname = None

        return outname

# Utilities ##############################

read_spectrum = Spectrum.read_spectrum  # Helper function


def match_DBname(name):
    """A DB-name is PYY_DOY_RRR_SSS_C_FFF_XXX_VV-vv_NNNS."""

    dbpattern = '(.*)' + '_'.join([r'(\d{2})'] +          # Prefix, YY
                                  [r'(\d{3})'] * 3 +        # DOY, RRR, SSS
                                  [r'(\d{1})'] +          # C
                                  [r'(\d{3})'] * 2 +        # FFF, XXX
                                  [r'(\d{2}-\d{2})'] +    # VV-vv
                                  [r'(\d{3})']) + '(.*)'  # NNN, Suffix
    search = re.search(dbpattern, name)
    if search is None:          # No match
        return None
    else:                       # Splitted match
        return list(search.groups())


def get_extension(name, default=0):
    """Return name, EXT from name[ext], using default ext if unspecified."""

    # Decipher name and extension from name[EXT]
    search = re.search(r'(.*)\[(.*)\]', name)
    if search:
        bname, ext = search.groups()
    else:
        bname, ext = name, default

    try:
        ext = int(ext)          # ext is an integer
    except ValueError:
        ext = ext.upper()       # ext is a string

    return bname, ext

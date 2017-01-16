#!/usr/bin/env python

"""
Simple module providing few FITS-based I/O classes and other facilities.
"""

import re
import warnings
import numpy
import pyfits


# Spectrum class ##############################


class Spectrum(object):

    """Class to read and manage a spectrum from a FITS file (NAXIS=1)."""

    def __init__(self, name, varname=None, keepfits=True):
        """
        Spectrum initialization.

        Class to read and manage a spectrum from a FITS file (NAXIS=1)
        including the associated [co]variance from an extension or an
        external file.

        Note: use helper function [Spectrum.]read_spectrum method for
        a transparent use.
        """

        self.name = name        # Generic name
        if name is None:        # Blank instance
            return
        self._readFits(name,    # Read signal [and variance if any]
                       mode='update' if keepfits else 'readonly')
        if not keepfits:
            self.close()
        if varname:             # Override variance extension if any
            if self.varname:    # Set by _readFits from var. extension
                warnings.warn("%s: VARIANCE extension overriden by %s" %
                              (name, varname), RuntimeWarning)
            self.varname = varname
            V = Spectrum(varname, varname=None, keepfits=keepfits)
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
        """Check if variance exists."""
        return hasattr(self, 'v') and self.v is not None

    @property
    def hasCov(self):
        """Check if covariance exists."""
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
        Initialize a Spectrum from FITS spectrum name.

        'name' can be 'name[ext]', in which case only extension 'ext' is considered.
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
        Set keywords.

        Set keywords from *keywords*=((key, val[, comment]),) or kwargs
        'key=val' or 'key=(val, comment)'.
        """
        for key in keywords:
            name, val = key[0], key[1:]   # name, (value, [comment])
            self._hdr[name.upper()] = val
        for key in kwargs:
            self._hdr[key.upper()] = kwargs[key]

    def deredden(self, ebmv, law='OD94', Rv=3.1):
        """
        Deredden spectrum using E(B-V) and a nextinction law.

        :param float ebmv: E(B-V) value.
        :param int law: Extinction law. Could be CCM89, OD94, FM98 or G08
        :param float Rv: Value for Rv. Default is 3.1
        """
        from Extinction.extinction import extinction_factor

        if hasattr(self, 'zorig'):      # Spectrum has been deredshifted
            raise ValueError, \
                "Dereddening should be done prior to deredshifting."

        # Extinction factor (<1)
        ext = extinction_factor(self.x, ebmv, rv=Rv, law=law)
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

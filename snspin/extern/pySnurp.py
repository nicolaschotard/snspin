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

        from ToolBox.Signal import savitzky_golay

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

    def dateFromJD(self, modified=False):
        """Returns datetime.datetime from JD keyword."""

        from ToolBox.Astro.Coords import jd2date

        return jd2date(self.readKey('JD'), modified=modified)

    def dateFromDateUT(self):
        """Returns datetime.datetime from DATE-OBS + UTC keywords."""

        import datetime

        date = self.readKey('DATE-OBS')  # 'YYYY-MM-DD'
        if 'T' in date:
            warnings.warn("%s: non-standard DATE-OBS '%s'" % (self.name, date),
                          SyntaxWarning)
            # Some DATE-OBS are YYYY-MM-DDTHH:MM:SS
            date = date.split('T')[0]
        YY, MM, DD = [int(x) for x in date.split('-')]
        if YY < 1900:                    # Some dates have year~100
            YY += 1900

        ut = self.readKey('UTC')         # 'HH:MM:SS'
        if 'T' in ut:
            warnings.warn("%s: non-standard UTC '%s'" % (self.name, ut),
                          SyntaxWarning)
            ut = ut.split('T')[1]
        hh, mm, ss = [int(x) for x in ut.split(':')]

        return datetime.datetime(YY, MM, DD, hh, mm, ss)

    def isoDate(self):
        """
        Return ISO8601 date 'YYYY-MM-DDTHH:MMZ' from JD keyword (see
        https://projects.lbl.gov/mantis/view.php?id=289 for the choice
        of JD keyword). See http://en.wikipedia.org/wiki/ISO_8601 for
        details.
        """

        try:
            d = self.dateFromJD()
        except KeyError:
            warnings.warn("%s: no JD keyword, trying DATE-OBS+UTC keywords" %
                          self.name, SyntaxWarning)
            d = self.dateFromDateUT()

        return d.strftime("%Y-%m-%dT%H:%M")  # Don't keep seconds

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

    def init_skycalc(self, verbose=False):
        """Instantiate a skycalc observation from spectrum."""

        try:
            from ToolBox.SkyCalc import SkyCalc
        except ImportError as err:
            warnings.warn("Cannot import SkyCalc (%s)" % err, ImportWarning)
            self.skycalc = None
            return

        ra = self.readKey('RA')         # 'HH:MM:SS.SSS'
        dec = self.readKey('DEC')        # 'DD:mm:ss.sss'
        jd = self.readKey('JD')         # Julian date
        if verbose:
            print "Skycalc object: RA=%s, Dec=%s, JD=%f" % (ra, dec, jd)
        self.skycalc = SkyCalc(ra, dec, jd=jd)

    def get_skycalc(self, attr):
        """
        Get attribute from skycalc if available.

        baryvcor: barycentric velocity correction [km/s]. The amount to add to
                  an observed radial velocity to correct it to the solar
                  system barycenter.
        """

        if not hasattr(self, 'skycalc'):
            self.init_skycalc()
        if self.skycalc is None:        # Could not instantiate
            return None
        else:
            return getattr(self.skycalc, attr)

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

    dbpattern = '(.*)' + '_'.join(['(\d{2})'] +          # Prefix, YY
                                  ['(\d{3})'] * 3 +        # DOY, RRR, SSS
                                  ['(\d{1})'] +          # C
                                  ['(\d{3})'] * 2 +        # FFF, XXX
                                  ['(\d{2}-\d{2})'] +    # VV-vv
                                  ['(\d{3})']) + '(.*)'  # NNN, Suffix
    search = re.search(dbpattern, name)
    if search is None:          # No match
        return None
    else:                       # Splitted match
        return list(search.groups())


def get_channel(name):
    """Get channel 'B' or 'R' fron historical or DB-style filename."""

    path, bname = os.path.split(name)

    tokens = match_DBname(bname)
    if tokens is None:           # Historical filenaming YY_DDD_RRR_SSS_C
        X = os.path.splitext(bname)[0][-1].upper()
        if X not in ('B', 'R'):
            X = None
    else:                        # DB filenaming
        try:
            X = {'2': 'R', '4': 'B'}[tokens[5]]
        except KeyError:
            X = None

    return X


def get_extension(name, default=0):
    """Return name, EXT from name[ext], using default ext if unspecified."""

    # Decipher name and extension from name[EXT]
    search = re.search('(.*)\[(.*)\]', name)
    if search:
        bname, ext = search.groups()
    else:
        bname, ext = name, default

    try:
        ext = int(ext)          # ext is an integer
    except ValueError:
        ext = ext.upper()       # ext is a string

    return bname, ext


def group_spectra(args):
    """Return a obsId dictionary { 'YY_DDD_RRR_SSS':[specnames] }."""

    odict = {}
    for arg in args:
        fname = arg.split(',')[0]       # arg could be spec, var_spec
        # Look for 'YY_DDD_RRR_SSS' in filename or files ending by _[B, R]
        search = lambda x: \
            re.search('(\d{2}_\d{3}_\d{3}_\d{3})', os.path.basename(x)) \
            or re.search('.+(?=_[B,R])', os.path.basename(x))
        if search(fname) is None:              # Failed search returns None
            # Maybe in the header?
            hname = pyfits.getheader(fname).get('FILENAME', 'None')
            if search(hname) is not None:
                name = os.path.basename(search(hname).group())
            else:
                warnings.warn("group_spectra: non-standard filename '%s'" %
                              fname, SyntaxWarning)
                name = os.path.splitext(os.path.basename(fname))[0]
        else:
            name = search(fname).group()        # YY_DDD_RRR_SSS
        odict.setdefault(name, []).append(arg)  # Or collections.defaultdict

    return odict


def isSorted(x, increasing=None, strictly=False):
    """
    Check if x in [strictly] sorted (increasing=None), increasing
    (True) or decreasing (False).
    """

    xx = numpy.ravel(x)
    if increasing is None or increasing:
        if strictly:
            inc = (xx[1:] > xx[:-1])
        else:
            inc = (xx[1:] >= xx[:-1])
    if increasing is None or not increasing:
        if strictly:
            dec = (xx[1:] < xx[:-1])
        else:
            dec = (xx[1:] <= xx[:-1])
    if increasing is None:
        return (inc.all() or dec.all())
    elif increasing:
        return inc.all()
    else:
        return dec.all()


# RefTable class ##############################

class RefTable:

    def __init__(self, name, colX, colY, ext=1, keepHdr=False):
        """
        colY can be a single column name (self.ny=1) or a tuple of column
        names (self.ny=len(colY)). If self.ny>1, self.y is a
        (self.ny, self.npts) 2D-array. ext specifies the extension
        number or name, which can also be specified in name[ext].
        """

        if name.endswith(']'):  # Decipher name and extension from name[EXT]
            name, ext = get_extension(name, default=ext)

        self.name = name
        self.basename = os.path.basename(self.name)
        self.colX = colX
        self.colY = colY
        self.ext = ext
        if isinstance(colY, (tuple, list)):
            self.ny = len(colY)
        else:
            self.ny = 1

        ffile = pyfits.open(self.name, ignore_missing_end=True)
        try:                    # Check existence of requested extension
            ffile.index_of(ext)
        except KeyError:
            raise KeyError("Cannot find extension %s among %s" %
                           (str(ext), ["%d:%s" % (i, ffile[i].name)
                                       for i in xrange(len(ffile))]))
        if keepHdr:             # Make a copy of requested header
            self.hdr = ffile[ext].header.copy()
        try:                    # Make a copy of requested columns
            self.x = ffile[ext].data.field(self.colX).copy()
            if self.ny == 1:
                self.y = ffile[ext].data.field(self.colY).copy()
            else:
                self.y = numpy.array([ffile[ext].data.field(col).copy()
                                      for col in self.colY])
        except NameError:
            raise NameError("Cannot find requested columns %s, %s among %s" %
                            (self.colX, self.colY, ffile[ext].data.names))
        ffile.close()

        self.npts = len(self.x)

    def __str__(self):

        return "Table '%s[%s]': col. %s, %s, %d rows" % \
               (self.basename, str(self.ext), self.colX, self.colY, self.npts)

    def sort(self):

        if not isSorted(self.x, increasing=True, strictly=True):
            isort = self.x.argsort()
            self.x = self.x[isort]
            self.y = self.y[isort]

    def interpolate(self, x, cols=None, monotonic=False):
        """
        Interpolate table over input array x. For self.ny>1, cols specifies
        the column indices to be interpolated (all by default).
        """

        if not isSorted(self.x, increasing=True, strictly=True):
            isort = self.x.argsort()
            xsort = self.x[isort]
            ysort = self.y[isort]
        else:
            isort = None
            xsort = self.x
            ysort = self.y

        if monotonic:
            from scipy.interpolate import pchip as interpolator
        else:
            from scipy.interpolate import UnivariateSpline
            interpolator = lambda x, y: UnivariateSpline(x, y, s=0)

        if self.ny == 1:
            return interpolator(xsort, ysort)(x)
        else:
            if cols is None:      # Interpolate all columns
                cols = xrange(self.ny)
            xx = numpy.asarray(x)
            res = numpy.array([interpolator(xsort, ysort[i])(xx)
                               for i in cols])
            return res.squeeze()  # Keep it simple


################################
# quick_magn magnitude dataset #
################################

class QMagn:

    def __init__(self, filename=None):

        from matplotlib.dates import datestr2num, num2date
        self.datestr2num = datestr2num
        self.num2date = num2date

        self.filename = os.path.basename(filename)
        self.read(filename)

    def read(self, filename, defaultPhoto=-1):
        """Read magnitude file as generated by quick_magn."""

        # Decipher header to read filter names (should be completed w/
        # other header info)
        self.filters = []
        self.zp = []            # filter zero points
        self.mode = 'MAG'       # mode: FLX or MAG, MAG by default
        self.units = None       # flux units: flambda, clambda, etc.
        for line in open(filename):
            if line.startswith("# KEYWORD FILTER"):
                self.filters.append('_'.join(line.split()[3:]))
            elif line.startswith("# KEYWORD FILTZP"):
                self.zp.append(float(line.strip().split()[-1]))
            elif line.startswith("# KEYWORD MODE"):
                self.mode = line.strip().split()[-1].upper()
                assert self.mode in ('FLX', 'MAG'), \
                    "ERROR: Unknown quick_magn mode '%s'" % self.mode
            elif line.startswith("# KEYWORD ABMAG0"):
                self.mag0 = float(line.strip().split()[-1])
            elif line.startswith("# KEYWORD FLXUNITS"):
                self.units = line.strip().split()[-1].lower()
            elif line.startswith("# KEYWORD EXTLAW"):
                self.extinction_law = line.strip().split()[-1]
            elif line.startswith("# KEYWORD MWEBV"):
                self.MWEBV = float(line.strip().split()[-1])
            elif line.startswith("# KEYWORD MWEBV_CORRECTED"):
                self.ebmv_corrected = bool(int(line.strip().split()[-1]))
        self.nfilters = len(self.filters)

        desc = [('obsid', 'S34'),  # 'YY_DDD_RRR_SSS'
                ('object', 'S15'),
                ('date', 'S16'),   # 'YYYY-MM-DDThh:mm'
                ('photo', 'i'),    # 1:photometric, 0:non-photo, -1:unknown
                ('airmass', 'f')]
        for f in self.filters:
            desc.extend([(f, 'f'), ('d' + f, 'f')])  # magn, dmagn

        # Read magnitudes: name object date airmass X1 dX1 X2 dX2...
        try:
            self.data = numpy.loadtxt(filename, comments='#',
                                      dtype=numpy.dtype(desc))
        except TypeError:
            warnings.warn("QMagn.read: "
                          "QMagn file without photometricity column, "
                          "default to %+d" % defaultPhoto, SyntaxWarning)
            tmpDesc = desc[:]
            del tmpDesc[3]  # Read without photo column (to be added latter)
            tmpData = numpy.loadtxt(filename, comments='#',
                                    dtype=numpy.dtype(tmpDesc))
            # Add a 'photo' column
            self.data = numpy.empty(tmpData.shape, dtype=desc)
            for field in tmpData.dtype.fields:       # Copy already present fields
                self.data[field] = tmpData[field]
            self.data['photo'] = defaultPhoto

        self.fulldata = numpy.atleast_1d(self.data)  # For single epoch magfile

        self.selection = []
        self.data = self.fulldata

    @property
    def nepochs(self):
        """Number of (selected) epochs."""

        return len(self.data)

    @property
    def numdates(self):
        """Numdates of (selected) epochs."""

        return self.datestr2num(self.data['date'])  # Dates in days

    def select(self, selection, name=''):
        """Apply a selection on self.data (leave self.fulldata intact)."""

        if not name:
            name = "selection #%d" % (len(self.selection) + 1)

        self.selection.append((name, selection))
        self.data = self.data[selection]

    def unselect(self):
        """
        Remove all selections on self.data (i.e. revert to
        self.fulldata).
        """

        self.selection = []
        self.data = self.fulldata

    def get_jdates(self, modified=False):
        """
        Return [Modified] Julian dates from internal (matplotlib-ready)
        self.numdates.

        See lightCurve.py
        """

        if modified:
            return self.numdates - 678576           # Modified Julian dates
        else:
            return self.numdates + 1721424.5        # Julian dates

    def get_magn(self, filtername=None, error=True):
        """
        If error, returns [m, dm].T (nepoch, 2) for filter filtername, or
        dictionary {filtername:[m, dm].T}.
        """

        if filtername is None:
            return dict(zip(self.filters,
                            [self.get_magn(f, error=error)
                             for f in self.filters]))  # {filter:magns}
        else:
            assert filtername in self.filters, "Unknown filter %s" % filtername
            m = self.data[filtername]         # (nepochs,)
            if not error:
                return m
            else:
                dm = self.data['d' + filtername]
                # [m, dm].T (nepochs, 2)
                return numpy.column_stack((m, dm))

    def __str__(self):

        s = ""
        if self.filename is not None:
            numdates = self.numdates
            s = "%s dataset '%s' [%s]:\n" \
                "  %d epochs [%s to %s]\n" \
                "  %d filters %s" % \
                ("Flux" if self.mode == 'FLX' else "Magnitude",
                 self.filename, self.units,
                 self.nepochs,
                 self.num2date(numdates[0]).strftime('%Y-%m-%d'),
                 self.num2date(numdates[-1]).strftime('%Y-%m-%d'),
                 self.nfilters, self.filters)
            if self.selection:
                s += "\nSelections: %s" % \
                    (' + '.join([name for name, sel in self.selection]))
        return s

    def discard_ref(self, maxdt=180):
        """
        Discard reference observations, ie. separated by more than maxdt
        days. Assumes the dates are sorted.
        """

        numdates = self.numdates
        dt = numpy.diff(numdates)  # Time difference in days

        assert (dt >= 0).all(), "Dates are not sorted."

        largedt, = numpy.where(dt > maxdt)    # More than maxdt apart
        if len(largedt):                    # Existence of ref. frames
            ref = largedt[0] + 1            # Index of 1st ref. frame
            print "Discarding %d reference points from %s" % \
                  (len(numdates[ref:]),
                   self.num2date(numdates[ref]).strftime('%y_%j'))
            self.data = self.data[:ref]

    def has_missingMagn(self):

        missing = numpy.zeros(self.nepochs, dtype='bool')
        for filter in self.filters:
            missing |= numpy.isnan(self.data[filter])

        numdates = self.numdates
        if len(numdates[missing]):
            print "%d epochs with missing magnitudes" % \
                (len(numdates[missing]))

        return missing

    def sort_dates(self):
        """
        Sort by numdates. Beware that the selections in self.selection are
        no more properly ordered!
        """

        self.data = self.data[numpy.argsort(self.numdates)]

    def merge(self, other, conflict='append'):
        """
        Merge 2 QMagn instances

        conflict: in case of conflict (multiple observations from the
                  same date), "append", keep "first" or the "last"
                  one.
        """

        assert conflict in ('append', 'first', 'last'), \
            'conflict must be one of ("append", "first", "last")'
        self.data = numpy.concatenate((self.data, other.data))
        if conflict != 'append':
            choice = {'first': 0, 'last': -1}[conflict]
            self.data = self.data[[(self.numdates == d).nonzero()[0][choice]
                                   for d in set(self.numdates)]]

    def dump(self, filename):
        """Dump to filename"""

        out = open(filename, 'w')

        out.write("# KEYWORD MODE %s\n" % self.mode)
        if hasattr(self, 'mag0'):
            out.write("# KEYWORD ABMAG0 %f\n" % self.mag0)
        out.write("# KEYWORD FLXUNITS %s\n" % self.units)
        for i, f in enumerate(self.filters):
            out.write("# KEYWORD FILTER%02d %s\n" % (i, f))
            out.write("# KEYWORD FILTZP%02d %f\n" % (i, self.zp[i]))

        if self.mode == 'FLX':
            fmt = "  %+10.4g  %7.2g"
        else:
            fmt = "  %+10.3f  %7.2g"

        out.write("# Spectrum                                   Object"
                  "               Date  Ph  Airm")
        for f in self.filters:
            out.write("  %10s  %7s" % (f, 'd' + f))
        out.write("\n")
        out.write('  '.join(('#', '=' * 31, '=' * 15, '=' * 17, '=' * 2, '=' * 4)))
        for _ in self.filters:
            out.write("  ==========  =======")
        out.write("\n")

        for i in xrange(len(self.data)):
            out.write("%34s" % self.data['obsid'][i])
            out.write("  %15s" % self.data['object'][i])
            out.write("  %17s" % self.data['date'][i])
            out.write("  %+d" % self.data['photo'][i])
            out.write("  %.2f" % self.data['airmass'][i])
            for f in self.filters:
                out.write(fmt % (self.data[f][i], self.data['d' + f][i]))
            out.write("\n")

        out.close()

    def metadata(self, prefix=''):
        """Dump as SnfMetaData compatible dict"""

        metadata = {}
        for obj in set(self.data['object']):
            metadata[obj] = {'target.name': obj,
                             'spectra': {},
                             'idr.saltprefix': 'qmagn',
                             'mag.filters': self.filters}
            w = self.data['object'] == obj  # (nepochs,)
            mags, dmags = numpy.array([
                self.get_magn(f)[w]
                for f in self.filters]).T  # 2 x (nepochs, nfilters)
            for exp, jd, mag, dmag, phot in zip(
                    self.data['obsid'][w], self.get_jdates()[w],
                    mags, dmags,
                    self.data['photo'][w]):
                expdata = [['.'.join(['obs', 'mjd']), jd - 2400000.5],
                           ['.'.join(['obs', 'photo']), phot],
                           # dummy phase - days from 1 January 2004
                           ['.'.join(['qmagn', 'phase']), jd - 2453005.5]]
                for m, dm, f in zip(mag, dmag, self.filters):
                    expdata.extend([
                        ['.'.join([prefix, f]) if prefix else f, m],
                        ['.'.join([prefix, f, 'err'])
                         if prefix else '.'.join([f, 'err']), dm]
                    ])

                metadata[obj]['spectra'][exp] = dict(expdata)

        return metadata

#!/usr/bin/env python

"""
Spectral indicator definition and measurements.
"""

import sys
import numpy as N
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline

from snspin.tools import smoothing
from snspin.spectrum import covariance


class Craniometer(object):

    """
    Initialization function
    """

    def __init__(self, wavelength, flux, variance):
        """
        Spectral indicator measurements.

        Class to feel bumps on SN spectra and conclude about how they
        work internaly
        How to use it:
        Create the craniometer:
        cranio = SNfPhrenology.Craniometer(wavelength, flux, variance)
        Smooth the craniometer:
        cranio.smooth()
        Generate simulated spectra:
        cranio.cranio_generator()
        Find all extrema:
        cranio.find_extrema()
        and compute spectral indicators:
        EWSi4000 = cranio.EW(3830, 3963, 4034, 4150, 'SiII4000')
        values will be saved in cranio.EWvalues for EWs computing

        Spectrum initialization.
        self.x = wavelength
        self.y = flux
        self.v = variance
        self.s = []
        self.smoother = None
        self.maxima = None
        self.minima = None
        self.ewvalues = {}
        """
        self.x = wavelength
        self.y = flux
        self.v = variance
        self.s = []
        self.maxima = None
        self.minima = None
        self.ewvalues = {}
        self.velocityvalues = {}
        self.p3590 = [3504, 3687]
        self.p3930 = [3887, 3990]
        self.p4075 = [4034, 4140]
        self.p4510 = [4452, 4573]
        self.p5165 = [5085, 5250]
        self.p5603 = [5550, 5681]
        self.p5930 = [5850, 6015]
        self.p6312 = [6250, 6365]
        self.init_only = False

#=========================================================================
# Analyse the spectrum
# Functions to smooth the spectrum, to find extrema of the spectrum
#=========================================================================

    def smooth(self, smoother="sgfilter", rho=0.482, s=None,
               hsize=None, order=2, lim=False, verbose=False):
        """
        Creates the smoother function and makes a smooth array out of spec.y.

        Mode = 0 : number and position are fixed (smoother='spline_fix_knot')
        interpolate.LSQUnivariateSpline(self.x,
                                        self.y,
                                        t=(self.x[::12])[1:],
                                        w = 1/(N.sqrt(self.v)))

        Mode = 1 : number and position of knots are not fixed
        (smoother='spline_free_knot')
        interpolate.UnivariateSpline(self.x, self.y, w = 1/(N.sqrt(self.v)), s=s)

        Mode = 2: used a savitzky_golay filter to smooth the spectrum
        (smoother='sgfilter')

        For mode = 0 or mode = 1: The spline function is put in self.smoother
        The smoothed array is in self.s
        The smoother type is in self.smoother_type
        """
        self.smoother_type = smoother
        self.lim = lim  # Limit for savitzky parameter
        if smoother == "spline_free_knot":
            if verbose:
                mess = "<SNfPhrenology.Craniometer> using spline with free"
                mess += "knots to smooth ", smoother
                print >> sys.stderr, mess
            self.spline_spec(mode=1, s=s, rho=rho, verbose=verbose)
        elif smoother == "spline_fix_knot":
            if verbose:
                mess = "<SNfPhrenology.Craniometer> using spline with fixed"
                mess += " knots to smooth ", smoother
                print >> sys.stderr, mess
            self.spline_spec(mode=0, s=s, rho=rho, verbose=verbose)
        elif smoother == 'sgfilter':
            if verbose:
                mess = "<SNfPhrenology.Craniometer> using savitzky_golay filter"
                print >> sys.stderr, mess
            self.sg_filter(hsize=hsize, order=order, rho=rho, verbose=verbose)
        else:
            warn = "<SNfPhrenology.Craniometer> WARNING: smoother not"
            warn += "implemented yet. Smoother asked for:", smoother
            print >> sys.stderr, warn

    def spline_spec(self, mode=1, s=None, rho=0.482, verbose=True):
        """
        Create a spline with interpolate.

        Mode = 0 : number and position are fixed
        interpolate.LSQUnivariateSpline(self.x,
                                        self.y,
                                        t=(self.x[::12])[1:],
                                        w = 1/(N.sqrt(self.v)))

        Mode = 1 : number and position of knots are not fixed
        interpolate.UnivariateSpline(self.x, self.y, w = 1/(N.sqrt(self.v)), s=s)
        """
        rc = 1. - 2. * (rho**2)
        if s is None:
            try:
                s = smoothing.spline_find_s(self.x,
                                            self.y,
                                            self.v * rc,
                                            corr=(rho**2) / rc)
            except TypeError:
                s = 0.492 * len(self.x)
        try:
            s = s[0]
        except TypeError:
            s = s
        if verbose:
            print >> sys.stderr, 'best_s=%i' % s
        if s <= 1:
            s *= len(self.x)
        self.smooth_parameter = s
        if len(self.x) == len(self.y):
            if len(self.v) > 0:
                if mode == 0:
                    self.spline = LSQUnivariateSpline(self.x,
                                                      self.y,
                                                      t=(self.x[::12])[1:],
                                                      w=1 / (N.sqrt(self.v)))
                if mode == 1:
                    self.spline = UnivariateSpline(self.x,
                                                   self.y,
                                                   w=1 / (N.sqrt(self.v)),
                                                   s=s)

                # Compute chi square for each point
                self.spline.chi2i = []
                for j in range(len(self.x)):
                    self.spline.chi2i.append(((self.y[j]
                                               - self.spline(self.x[j])[0])**2)
                                             / (self.v[j]))
                self.spline.chi2 = self.spline.get_residual()\
                    / (len(self.x)
                       - (len(self.spline.get_coeffs()) - 4))
                # Save smooth function
                self.s = self.spline(self.x)

            else:
                if verbose:
                    print >> sys.stderr, "No variance informations"
                self.spline = None

        else:
            if verbose:
                print >> sys.stderr, "ERROR. len(Wavelenght) != len(flux)"

    def sg_filter(self, hsize=None, order=2, rho=0.0, verbose=False):
        """
        Use savitzky_golay() to apply a savitzky golay filter on the spectrum.

        Input:
        - hsize : half size of the window (default:15)
        - order : order of the polynome used to smooth the spectrum (default:2)
        Output:
        - self.s : smoothing spectrum
        """
        rc = 1. - 2. * (rho**2)
        if hsize is None:
            try:
                hsize = int(smoothing.sg_find_num_points(self.x,
                                                         self.y,
                                                         self.v * rc,
                                                         corr=(rho**2) / rc))
            except TypeError:
                if verbose:
                    print >> sys.stderr, 'ERROR in computing of best hsize'
                hsize = 15
        if (hsize * 2) + 1 < (order + 2):
            hsize = 10  # order/2.+1
        if self.lim and hsize < self.lim:
            hsize = self.lim   # for oxygen zone only
        if verbose:
            print >> sys.stderr, 'best_w=%i' % hsize
        self.s = smoothing.savitzky_golay(self.y,
                                          kernel=(int(hsize) * 2) + 1,
                                          order=order,
                                          derivative=0)
        self.s_deriv = smoothing.savitzky_golay(self.y,
                                                kernel=(int(hsize) * 2) + 1,
                                                order=order,
                                                derivative=1)
        self.hsize = hsize
        self.order = order

    def smoother(self, lbd, verbose=False):
        """."""
        if len(self.s) != len(self.x):
            # If no smoothing function
            if verbose:
                print >> sys.stderr, 'ERROR: len(self.s) != len(self.x)'
            return None
        elif self.smoother_type != 'sgfilter':
            # If smoothing function is a spline
            return self.spline(lbd)
        else:
            # If smoothing function is a sgfilter
            if N.isscalar(lbd):
                if verbose:
                    print >> sys.stderr, 'lbd is a scalar'
                if lbd < self.x[0] or lbd > self.x[-1]:
                    if verbose:
                        mess = 'ERROR: lbd is not in the range, %.2f<lbd<%.2f' %\
                               (self.x[0], self.x[-1])
                        print >> sys.stderr, mess
                    return None
                else:
                    flux = self.s[(self.x > lbd - 2) & (self.x < lbd + 2)]
                    return flux[0]
            else:
                if verbose:
                    print >> sys.stderr, 'lbd is an array'
                for i in lbd:
                    if i < self.x[0] or i > self.x[-1]:
                        if verbose:
                            mess = 'ERROR: %.2f<lbd<%.2f' % (
                                self.x[0], self.x[-1])
                            print >> sys.stderr, mess
                        return None
                flux = N.array([float(self.s[(self.x > (l - 1)) &
                                             (self.x < (l + 1))])
                                for l in lbd])
                return flux

    def find_extrema(self, verbose=True):
        """
        Function to find all extrema in a smoothed spectrum.

        Return two arrays : maxima = {'x':maxima_x,
                                      'y':maxima_y,
                                      's':maxima_s,
                                      'v':maxima_v}
                            minima = {'x':minima_x,
                                      'y':minima_y,
                                      's':minima_s,
                                      'v':minima_v}
        and save it in self.maxima and self.minima
        """

        if not len(self.s):
            self.maxima = None
            self.minima = None
            if verbose:
                mess = "ERROR! Incompatible or non existent smoothing spectrum"
                print >> sys.stderr, mess
                print >> sys.stderr, "[try spec.smooth()]"
        else:
            # find extrema for the real spectrum
            maxima, minima = self._extrema(self.x, self.y, self.v, self.s)
            self.maxima = maxima
            self.minima = minima

    def cranio_generator(self, nsimu=1000, s=0.492, rho=0.482,
                         correl=True, hsize=15, order=2, factor=1,
                         simus=None, verbose=True):
        """
        Simulation generator.

        Generate 'nsimu' simulated spectra from smooth fuction applied on
        the spectra, make a smoother and find extrema on each of them,
        and save it in a new craniometer object in self.simulations
        Default: smoother='sgfilter'
        Default parameters for spline fitting:
        s=0.492, rho=0.23 (mean values)
        Default parameters for savitzky golay filter:
        hsize=15, order=2 (mean value)
        """
        self.v *= factor
        self.rho = rho
        # try:
        self.simulations = []
        if simus is not None:
            simulated_spectra = simus
            nsimu = len(simus)
        else:
            # Create gaussian distribution and simulated spectra
            if correl:
                if verbose:
                    print >> sys.stderr, 'Simulations with correlated pixels'
                simulated_spectra = self._correl_simulated_spectra(nsimu,
                                                                   rho=rho,
                                                                   verbose=verbose)
            else:
                normal_distribution = N.random.randn(nsimu, len(self.x))
                simulated_spectra = normal_distribution * (N.sqrt((self.v))) \
                    + self.s
        # Smooth and save simulated spectra
        for simulated_spectrum, number in zip(simulated_spectra, range(nsimu)):
            self.simulations.append(Craniometer(self.x,
                                                simulated_spectrum,
                                                self.v))
            if self.smoother_type == 'spline_fix_knot':
                self.simulations[number].smooth(smoother=self.smoother_type,
                                                rho=rho,
                                                verbose=False)
            elif self.smoother_type == 'spline_free_knot':
                self.simulations[number].smooth(smoother=self.smoother_type,
                                                rho=rho,
                                                s=self.smooth_parameter,
                                                verbose=False)
            elif self.smoother_type == 'sgfilter':
                self.simulations[number].smooth(smoother=self.smoother_type,
                                                hsize=self.hsize,
                                                order=self.order,
                                                verbose=False)
            self.simulations[number].find_extrema(verbose=False)

        try:
            self.systematic_error(verbose=verbose)
        except TypeError:
            self.syst = None
            print >> sys.stderr, "ERROR in systematic_error (cranio_generator)!"

    def _correl_simulated_spectra(self, nsimu, rho=0.482, verbose=True):
        """."""
        def comp_alpha(rho):
            """."""
            return 0.5 * (1 + N.sqrt(1 - 4 * (rho**2)))

        def comp_beta(rho):
            """."""
            return 0.5 * (1 - N.sqrt(1 - 4 * (rho**2)))

        alpha = comp_alpha(rho)
        beta = comp_beta(rho)

        normal_distribution = N.random.randn(nsimu, len(self.x) + 1)
        normal_distribution_correl = N.zeros((nsimu, len(self.x)))
        for i in range(normal_distribution.shape[0]):
            normal_distribution_correl[i] = alpha * normal_distribution[i][:-1] \
                + beta * normal_distribution[i][1:]
        simulated_spectra = normal_distribution_correl * (N.sqrt((self.v))) \
            + self.s
        return simulated_spectra

    def systematic_error(self, verbose=True):
        """
        Comput ethe systematic error.

        15% on hsize and 10% on s
        create an un-pre-defined number of craniometer with the variation
        of hsize or s and save it.
        in each spectral indicators functions, compute this one for each new
        craniometer and compute the standard deviation between those new ones
        and the initial one
        """
        self.syst = []
        if self.smoother_type == 'sgfilter':
            #- For cases where hsize is large, one has to make sure that the
            #- window explored doesn't include cases where hsize is larger than
            #- the size of the data
            hsizes = N.arange(self.hsize * 0.85,
                              min(len(self.x), self.hsize * 1.15),
                              dtype=int)
            for hsize, number in zip(hsizes, range(len(hsizes))):
                self.syst.append(Craniometer(self.x, self.y, self.v))
                self.syst[number].smooth(smoother=self.smoother_type,
                                         hsize=hsize,
                                         order=self.order,
                                         verbose=False)
                self.syst[number].find_extrema(verbose=False)
        else:
            sparams = N.linspace(self.smooth_parameter * 0.9,
                                 self.smooth_parameter * 1.1, 6)
            for s, number in zip(sparams, range(len(sparams))):
                self.syst.append(Craniometer(self.x, self.y, self.v))
                self.syst[number].smooth(smoother=self.smoother_type,
                                         rho=self.rho,
                                         s=s,
                                         verbose=False)
                self.syst[number].find_extrema(verbose=False)

#=========================================================================
# Utilities to compute spectral indicators
# Functions to intergate, compute lines ratio, variances...
#=========================================================================

    def _extrema(self, x, y, v, s, w=12, StoN=0.80):
        """
        Find all signicative extrema of a spectrum or spectrum's zone.

        Output:
        maxima = {'x':N.array(maxima_x),
                  'y':N.array(maxima_y),
                  's':N.array(maxima_s),
                  'v':N.array(maxima_v)}
        minima = {'x':N.array(minima_x),
                  'y':N.array(minima_y),
                  's':N.array(minima_s),
                  'v':N.array(minima_v)}

        """
        # parameters for the window (13*step on each side of the extrema)
        # 2 functions to keep only one extrema in a window

        def minima(i, w):
            """."""
            if (i > w) and (i < (len(x) - w)):
                window = (x >= x[i - w]) & (x <= x[i + w])
                lbdmin = (x[window])[N.argmin(s[window])]
                return x[i] == lbdmin
            else:
                return False

        def maxima(i, w):
            """."""
            if (i > w) and (i < (len(x) - w)):
                window = (x >= x[i - w]) & (x <= x[i + w])
                lbdmax = (x[window])[N.argmax(s[window])]
                return x[i] == lbdmax
            else:
                return False

        def signaltonoise(i):
            """."""
            good = (x > (x[i] - 20)) & (x < (x[i] + 20))
            return (y[i] / N.sqrt(v[i])) \
                / N.mean(y[good] / N.sqrt(v[good])) >= StoN

        # Define arrays: lambda, flux, smooth values and variance
        # for maxima and minima
        minima_x, minima_y, minima_s, minima_v = [], [], [], []
        maxima_x, maxima_y, maxima_s, maxima_v = [], [], [], []

        # parameter initialization
        p = not (s[0] < s[1])

        # Find extrema
        for i in range(len(x) - 1):
            if not p:
                if s[i] <= s[i + 1]:
                    continue
                elif not maxima(i, w):
                    p = 1
                    continue
                elif not signaltonoise(i):
                    p = 1
                    continue
                else:
                    maxima_x.append(x[i])
                    maxima_y.append(y[i])
                    maxima_s.append(s[i])
                    maxima_v.append(v[i])
                    p = 1
            else:
                if s[i] >= s[i + 1]:
                    continue
                elif not minima(i, w):
                    p = 0
                    continue
                else:
                    minima_x.append(x[i])
                    minima_y.append(y[i])
                    minima_s.append(s[i])
                    minima_v.append(v[i])
                    p = 0

        # Create output
        maxima = {'x': N.array(maxima_x),
                  'y': N.array(maxima_y),
                  's': N.array(maxima_s),
                  'v': N.array(maxima_v)}
        minima = {'x': N.array(minima_x),
                  'y': N.array(minima_y),
                  's': N.array(minima_s),
                  'v': N.array(minima_v)}

        return maxima, minima

    def _integration(self, x, y, min=None, max=None, verbose=True):
        """."""
        if min is None \
                or max is None \
                or (min >= max) \
                or (min <= 0) \
                or (max <= 0):
            if verbose:
                print >> sys.stderr, "ERROR in the definition of extrema"
            return N.nan
        elif x[0] > min or x[-1] < max:
            if verbose:
                print >> sys.stderr, "ERROR. Extrema are not in the interval"
            return N.nan
        else:
            return float(y[(x >= min) & (x <= max)].sum())

    def _var_integration(self, x, v, min=None, max=None, verbose=True):
        """Compute variance of an intergration."""
        if len(v):
            var_int = v[(x > min) & (x < max)].sum()
        else:
            if verbose:
                print >> sys.stderr, "No variance for this spectrum"
            var_int = N.nan
        return float(var_int)

    def _var_rapport(self, a, b, var_a, var_b, verbose=True):
        """Compute variance for a/b"""
        """."""
        if a and b and var_a and var_b:
            var = (1 / b**2) * var_a + (a**2) * (var_b / b**4)
        else:
            if verbose:
                mess = "Incompatible values to compute the ratio variance"
                print >> sys.stderr, mess
            var = N.nan

        return float(var)

    def _equivalentdepth(self, lbd1=None, lbd2=None, lbd3=None, flux1=None,
                         flux2=None, flux3=None, verbose=True):
        """."""
        if lbd1 >= lbd2 or lbd2 >= lbd3 or lbd1 >= lbd3:
            if verbose:
                print >> sys.stderr, 'ERROR in the definition of wavelenght '\
                    'to compute equivalent depth'
            return N.nan
        else:
            p = N.polyfit([lbd1, lbd3], [flux1, flux3], 1)  # y=p[0]*x+p[1]
            return float(N.polyval(p, lbd2) - flux2)

    def _equivalentwidth(self, x, y, lbd1=None, lbd2=None, flux1=None,
                         flux2=None, verbose=True):
        """."""
        if lbd1 >= lbd2:
            if verbose:
                print >> sys.stderr, 'ERROR in the definition of '\
                    'wavelenght to compute equivalent width'
            return N.nan
        else:
            step = x[1] - x[0]
            p = N.polyfit([lbd1, lbd2], [flux1, flux2], 1)  # y=p[0]*x+p[1]

            x_new = x[(x >= lbd1) & (x <= lbd2)]
            y_new = y[(x >= lbd1) & (x <= lbd2)]
            integration = N.sum((N.polyval(p, x_new) - y_new) /
                                N.polyval(p, x_new)) * step

            return float(integration)

    def _extrema_value_in_interval(self, min, max, lbd, var, smooth,
                                   extrema=None, verbose=True, right=False,
                                   left=False):
        """
        Find extrema.

        Function to find extrema values (lambda, flux and variance) in a given
        interval. Values are searched in self.minima and self.maxima.
        Use extrema='minima' to find minima, and extrema='maxima' to find maxima
        Lambda min < Lambda_max
        """
        try:
            if extrema == 'maxima':
                if right:
                    arg = N.argmax(lbd[(lbd > min) & (lbd < max)])
                elif left:
                    arg = N.argmin(lbd[(lbd > min) & (lbd < max)])
                else:
                    arg = N.argmax(smooth[(lbd > min) & (lbd < max)])

            elif extrema == 'minima':
                if right:
                    arg = N.argmax(lbd[(lbd > min) & (lbd < max)])
                elif left:
                    arg = N.argmin(lbd[(lbd > min) & (lbd < max)])
                else:
                    arg = N.argmin(smooth[(lbd > min) & (lbd < max)])

            wavelength = (lbd[(lbd >= min) & (lbd <= max)])[arg]
            flux = (smooth[(lbd >= min) & (lbd <= max)])[arg]
            variance = (var[(lbd >= min) & (lbd <= max)])[arg]
            return wavelength, flux, variance

        except TypeError:
            return [None, None, None]

    def _find_special_peak(self, min, max, maxima=False,
                           minima=False, right=False, left=False):
        """."""
        if maxima is False and minima is False:
            return None, None, None

        limit = (self.x >= min) & (self.x <= max)
        maxi, mini = self._extrema(self.x[limit], self.y[limit],
                                   self.v[limit], self.s[limit], w=1)

        if right:
            if maxima:
                arg = N.argmax(maxi['x'])
                return maxi['x'][arg], maxi['s'][arg], maxi['v'][arg]
            elif minima:
                arg = N.argmax(mini['x'])
                return mini['x'][arg], mini['s'][arg], mini['v'][arg]
            else:
                return None, None, None
        elif left:
            if maxima:
                arg = N.argmin(maxi['x'])
                return maxi['x'][arg], maxi['s'][arg], maxi['v'][arg]
            elif minima:
                arg = N.argmin(mini['x'])
                return mini['x'][arg], mini['s'][arg], mini['v'][arg]
            else:
                return None, None, None
        else:
            if maxima:
                arg = N.argmax(maxi['s'])
                return maxi['x'][arg], maxi['s'][arg], maxi['v'][arg]
            elif minima:
                arg = N.argmin(mini['s'])
                return mini['x'][arg], mini['s'][arg], mini['v'][arg]

            else:
                return None, None, None

    def max_of_interval(self, min, max, verbose=False):
        """."""
        la = self.x[(self.x > min) & (self.x < max)]
        fa = self.smoother(la)
        arg = N.argmax(fa)
        l, f = la[arg], fa[arg]
        v = 0.0
        return l, f, v

    def std2(self, x, x0):
        """
        conpute the standard deviation of the distribution x compared with x0.

        x0 can be the mean or an other value
        """
        return N.sqrt(N.mean(N.absolute(x - x0)**2))

    def _get_min(self, lbd, plot=False):
        """
        Get the minimum the flux around a given bin.

        It uses the derivative of the smoothed function. 
        A linear interpolation is made using the given
        bin and the left and right bins.
        """
        # now take the minimum and the bins around it
        bin_c = N.argmin(N.abs(self.x - lbd))
        xx = self.x[bin_c - 1:bin_c + 1]
        yy = self.s_deriv[bin_c - 1:bin_c + 1]

        # make a linear fit
        pol = N.polyfit(xx, yy, 1)

        # check if this is constistant with a ~1 bin shift
        if N.abs(lbd + pol[1] / pol[0]) < 1.5 * (self.x[1] - self.x[0]):
            return -(pol[1] / pol[0])
        else:
            return lbd

#=========================================================================
# Compute spectral indicators on the spectrum
# Functions to compute several spectral indicators
#=========================================================================

    def rca(self, verbose=True, simu=True, syst=True):
        """
        Return the value and the error of rca.

        [rca, rca_sigma]
        """
        # Initialisation
        self.rcavalues = {'rca': N.nan, 'rca.err': N.nan, 'rca.stat': N.nan,
                          'rca.syst': N.nan, 'rca.mean': N.nan,
                          'rca_lbd': [N.nan, N.nan], 'rca_flux': [N.nan, N.nan]}
        if self.init_only:
            return

        lbd1, flux1, var1 = self._extrema_value_in_interval(self.p3590[0],
                                                            self.p3590[1],
                                                            self.maxima['x'],
                                                            self.maxima['v'],
                                                            self.maxima['s'],
                                                            extrema='maxima',
                                                            right=True,
                                                            verbose=verbose)
        if simu and lbd1 is None:
            lbd1, flux1, var1 = self.max_of_interval(
                self.p3590[0], self.p3590[1])

        lbd2, flux2, var2 = self._extrema_value_in_interval(self.p3930[0],
                                                            self.p3930[1],
                                                            self.maxima['x'],
                                                            self.maxima['v'],
                                                            self.maxima['s'],
                                                            extrema='maxima',
                                                            verbose=verbose)
        if simu and lbd2 is None:
            lbd2, flux2, var2 = self.max_of_interval(
                self.p3930[0], self.p3930[1])
        try:
            rca_value = flux2 / flux1
        except TypeError:
            if verbose:
                print >> sys.stderr, "ERROR in computing rca"
            rca_value = N.nan

        if simu:

            if not N.isfinite(rca_value):
                return [float(N.nan), float(N.nan)]

            rca_simu = []
            for simu in self.simulations:
                try:
                    rca_simu.append(simu.rca(simu=False, syst=False,
                                             verbose=False))
                except TypeError:
                    continue
            rca_sigma = self.std2(N.array(rca_simu)[N.isfinite(rca_simu)],
                                  rca_value)
            rca_mean = N.mean(N.array(rca_simu)[N.isfinite(rca_simu)])

            if N.isfinite(rca_value):
                self.rcavalues = {'rca': float(rca_value),
                                  'rca.err': float(rca_sigma),
                                  'rca.stat': float(rca_sigma),
                                  'rca.mean': float(rca_mean),
                                  'rca_lbd': [float(lbd1), float(lbd2)],
                                  'rca_flux': [float(flux1), float(flux2)]}

        if syst:

            rca_syst = []
            for system in self.syst:
                try:
                    rca_syst.append(system.rca(syst=False, simu=False,
                                               verbose=False))
                except TypeError:
                    continue
            rca_sigma_syst = self.std2(N.array(rca_syst)[N.isfinite(rca_syst)],
                                       rca_value)

            if N.isfinite(rca_sigma_syst):
                rca_sigma = float(N.sqrt(rca_sigma**2 + rca_sigma_syst**2))
            else:
                rca_sigma *= 2
            self.rcavalues['rca.syst'] = float(rca_sigma_syst)
            self.rcavalues['rca.err'] = float(rca_sigma)

            return [float(rca_value), float(rca_sigma)]

        if simu == False and syst == False:

            if N.isfinite(rca_value):
                self.rcavalues = {'rca': float(rca_value),
                                  'rca_lbd': [float(lbd1), float(lbd2)],
                                  'rca_flux': [float(flux1), float(flux2)],
                                  'rca.err': N.nan, 'rca.stat': N.nan,
                                  'rca.mean': N.nan}
            return rca_value

    def rcaS(self, verbose=True, simu=True, syst=True):
        """
        Return the value and the error of rcaS.

        [rcaS, rcaS_sigma]
        """
        # Initialisation
        self.rcaSvalues = {'rcaS': N.nan, 'rcaS.err': N.nan, 'rcaS.stat': N.nan,
                           'rcaS.syst': N.nan, 'rcaS.mean': N.nan,
                           'rcaS_lbd': [N.nan, N.nan, N.nan, N.nan]}
        if self.init_only:
            return

        min_1 = 3620
        max_1 = 3716
        min_2 = 3887
        max_2 = 4012

        try:
            rcaS_value = (self._integration(self.x, self.y, min=min_2, max=max_2,
                                            verbose=verbose)) / \
                (self._integration(self.x, self.y,
                                   min=min_1,
                                   max=max_1,
                                   verbose=verbose))
        except TypeError:
            if verbose:
                print >> sys.stderr, 'ERROR in compute of rcaS'
            rcaS_value = float(N.nan)

        if simu:
            if not N.isfinite(rcaS_value):
                return [float(N.nan), float(N.nan)]

            rcaS_simu = []
            for simu in self.simulations:
                try:
                    rcaS_simu.append(simu.rcaS(simu=False, syst=False,
                                               verbose=False))
                except TypeError:
                    continue
            rcaS_sigma = self.std2(N.array(rcaS_simu)[N.isfinite(rcaS_simu)],
                                   rcaS_value)
            rcaS_mean = N.mean(N.array(rcaS_simu)[N.isfinite(rcaS_simu)])

            if N.isfinite(rcaS_value):
                self.rcaSvalues = {'rcaS': float(rcaS_value),
                                   'rcaS.err': float(rcaS_sigma),
                                   'rcaS.stat': float(rcaS_sigma),
                                   'rcaS.mean': float(rcaS_mean),
                                   'rcaS_lbd': [float(min_1),
                                                float(max_1),
                                                float(min_2),
                                                float(max_2)]}

        if syst:

            rcaS_syst = []
            for system in self.syst:
                try:
                    rcaS_syst.append(system.rcaS(simu=False,
                                                 syst=False, verbose=False))
                except TypeError:
                    continue
            rcaS_sigma_syst = self.std2(
                N.array(rcaS_syst)[N.isfinite(rcaS_syst)], rcaS_value)

            if N.isfinite(rcaS_sigma_syst):
                rcaS_sigma = float(N.sqrt(rcaS_sigma**2 + rcaS_sigma_syst**2))
            else:
                rcaS_sigma *= 2
            self.rcaSvalues['rcaS.syst'] = float(rcaS_sigma_syst)
            self.rcaSvalues['rcaS.err'] = float(rcaS_sigma)

            return [float(rcaS_value), float(rcaS_sigma)]

        if simu == False and syst == False:

            if N.isfinite(rcaS_value):
                self.rcaSvalues = {'rcaS': float(rcaS_value),
                                   'rcaS_lbd': [min_1, max_1, min_2, max_2]}
            return rcaS_value

    def rcaS2(self, verbose=True, simu=True, syst=True):
        """
        New rcaS where peaks are following.

        Return the value and the error of rcaS
        [rcaS, rcaS_sigma]
        """
        interval_1 = 48
        interval_2 = 62.5

        # Initialisation
        self.rcaS2values = {'rcaS2': N.nan, 'rcaS2.err': N.nan,
                            'rcaS2.stat': N.nan,
                            'rcaS2.syst': N.nan,
                            'rcaS2.mean': N.nan,
                            'rcaS2_lbd': [N.nan, N.nan, N.nan, N.nan]}
        if self.init_only:
            return

        try:
            lbd1, flux1, var1 = self._extrema_value_in_interval(self.p3590[0],
                                                                self.p3590[1],
                                                                self.maxima[
                                                                    'x'],
                                                                self.maxima[
                                                                    'v'],
                                                                self.maxima[
                                                                    's'],
                                                                extrema='maxima',
                                                                verbose=verbose)
            if simu and lbd1 is None:
                lbd1, flux1, var1 = self.max_of_interval(self.p3590[0],
                                                         self.p3590[1])

            lbd2, flux2, var2 = self._extrema_value_in_interval(self.p3930[0],
                                                                self.p3930[1],
                                                                self.maxima[
                                                                    'x'],
                                                                self.maxima[
                                                                    'v'],
                                                                self.maxima[
                                                                    's'],
                                                                extrema='maxima',
                                                                verbose=verbose)
            if simu and lbd2 is None:
                lbd2, flux2, var2 = self.max_of_interval(self.p3930[0],
                                                         self.p3930[1])

            min_1 = lbd1 - interval_1
            max_1 = lbd1 + interval_1
            min_2 = lbd2 - interval_2
            max_2 = lbd2 + interval_2

            rcaS2_value = (self._integration(self.x, self.y, min=min_2,
                                             max=max_2, verbose=verbose)) / \
                self._integration(self.x, self.y,
                                  min=min_1,
                                  max=max_1,
                                  verbose=verbose)
        except TypeError:
            if verbose:
                print >> sys.stderr, 'ERROR in compute of rcaS2'
            rcaS2_value = float(N.nan)

        if simu:
            if not N.isfinite(rcaS2_value):
                return [float(N.nan), float(N.nan)]

            rcaS2_simu = []
            for simu in self.simulations:
                try:
                    rcaS2_simu.append(simu.rcaS2(simu=False,
                                                 syst=False,
                                                 verbose=False))
                except TypeError:
                    continue
            rcaS2_sigma = self.std2(N.array(rcaS2_simu)[N.isfinite(rcaS2_simu)],
                                    rcaS2_value)
            rcaS2_mean = N.mean(N.array(rcaS2_simu)[N.isfinite(rcaS2_simu)])

            if N.isfinite(rcaS2_value):
                self.rcaS2values = {'rcaS2': float(rcaS2_value),
                                    'rcaS2.err': float(rcaS2_sigma),
                                    'rcaS2.stat': float(rcaS2_sigma),
                                    'rcaS2.mean': float(rcaS2_mean),
                                    'rcaS2_lbd': [float(min_1), float(max_1),
                                                  float(min_2), float(max_2)]}

        if syst:
            rcaS2_syst = []
            for system in self.syst:
                try:
                    rcaS2_syst.append(system.rcaS2(simu=False,
                                                   syst=False,
                                                   verbose=False))
                except TypeError:
                    continue

            rcaS2_sigma_syst = self.std2(
                N.array(rcaS2_syst)[N.isfinite(rcaS2_syst)], rcaS2_value)

            if N.isfinite(rcaS2_sigma_syst):
                rcaS2_sigma = float(N.sqrt(rcaS2_sigma**2 +
                                           rcaS2_sigma_syst**2))
            else:
                rcaS2_sigma *= 2
            self.rcaS2values['rcaS2.syst'] = float(rcaS2_sigma_syst)
            self.rcaS2values['rcaS2.err'] = float(rcaS2_sigma)

            return [float(rcaS2_value), float(rcaS2_sigma)]

        if simu == False and syst == False:

            if N.isfinite(rcaS2_value):
                self.rcaS2values = {'rcaS2': float(rcaS2_value),
                                    'rcaS2_lbd': [float(min_1),
                                                  float(max_1),
                                                  float(min_2),
                                                  float(max_2)],
                                    'rcaS2.err': N.nan,
                                    'rcaS2.stat': N.nan,
                                    'rcaS2.mean': N.nan}

            return rcaS2_value

    def rcaSbis(self, verbose=True, simu=True):
        """
        Return the value and the error of rcaSbis.

        [rcaSbis, rcaSbis_sigma]
        """

        min_1 = 3620
        max_1 = 3716
        min_2 = 3887
        max_2 = 4012

        try:
            rcaS_value = self._integration(self.x, self.y, min=min_2,
                                           max=max_2, verbose=verbose) / \
                self._integration(self.x, self.y,
                                  min=min_1,
                                  max=max_1,
                                  verbose=verbose)
            a = self._integration(self.x, self.y, min=min_2,
                                  max=max_2, verbose=verbose)
            b = self._integration(self.x, self.y, min=min_1,
                                  max=max_1, verbose=verbose)
            var_a = self._var_integration(self.x, self.v, min=min_2,
                                          max=max_2, verbose=verbose)
            var_b = self._var_integration(self.x, self.v, min=min_1,
                                          max=max_1, verbose=verbose)
            rcaS_sigma = N.sqrt(self._var_rapport(a, b, var_a, var_b,
                                                  verbose=verbose))
        except TypeError:
            if verbose:
                print >> sys.stderr, 'ERROR in compute of rcaS'
            rcaS_value = float(N.nan)
            rcaS_sigma = float(N.nan)

        return [float(rcaS_value), float(rcaS_sigma)]

    def rcaS2bis(self, verbose=True, simu=True):
        """
        Return the value and the error of rcaS2bis
        [rcaS2bis, rcaS2bis_sigma]
        """
        interval_1 = 48
        interval_2 = 62.5

        lbd1, flux1, var1 = self._extrema_value_in_interval(self.p3590[0],
                                                            self.p3590[1],
                                                            self.maxima['x'],
                                                            self.maxima['v'],
                                                            self.maxima['s'],
                                                            extrema='maxima',
                                                            verbose=verbose)
        if simu and lbd1 is None:
            lbd1, flux1, var1 = self.max_of_interval(
                self.p3590[0], self.p3590[1])

        lbd2, flux2, var2 = self._extrema_value_in_interval(self.p3930[0],
                                                            self.p3930[1],
                                                            self.maxima['x'],
                                                            self.maxima['v'],
                                                            self.maxima['s'],
                                                            extrema='maxima',
                                                            verbose=verbose)
        if simu and lbd2 is None:
            lbd2, flux2, var2 = self.max_of_interval(
                self.p3930[0], self.p3930[1])

        min_1 = lbd1 - interval_1
        max_1 = lbd1 + interval_1
        min_2 = lbd2 - interval_2
        max_2 = lbd2 + interval_2

        try:
            rcaS_value = (self._integration(self.x, self.y, min=min_2,
                                            max=max_2, verbose=verbose)) / \
                (self._integration(self.x, self.y,
                                   min=min_1,
                                   max=max_1,
                                   verbose=verbose))
            a = self._integration(self.x, self.y, min=min_2, max=max_2,
                                  verbose=verbose)
            b = self._integration(self.x, self.y, min=min_1, max=max_1,
                                  verbose=verbose)
            var_a = self._var_integration(self.x, self.v, min=min_2, max=max_2,
                                          verbose=verbose)
            var_b = self._var_integration(self.x, self.v, min=min_1,
                                          max=max_1, verbose=verbose)
            rcaS_sigma = N.sqrt(self._var_rapport(a, b, var_a, var_b,
                                                  verbose=verbose))
        except TypeError:
            if verbose:
                print >> sys.stderr, 'ERROR in compute of rcaS'
            rcaS_value = float(N.nan)
            rcaS_sigma = float(N.nan)

        return [float(rcaS_value), float(rcaS_sigma)]

    def edca(self, verbose=True, simu=True, syst=True):
        """
        Return the value and the error of edca.

        [edca, edca_sigma]
        """
        self.edcavalues = {'edca': N.nan, 'edca.err': N.nan, 'edca.stat': N.nan,
                           'edca.syst': N.nan, 'edca.mean': N.nan,
                           'edca_lbd': [N.nan, N.nan]}
        if self.init_only:
            return

        try:
            lbd1, flux1, var1 = self._extrema_value_in_interval(self.p3590[0],
                                                                self.p3590[1],
                                                                self.maxima[
                                                                    'x'],
                                                                self.maxima[
                                                                    'v'],
                                                                self.maxima[
                                                                    's'],
                                                                extrema='maxima',
                                                                verbose=verbose)
            if simu and lbd1 is None:
                lbd1, flux1, var1 = self.max_of_interval(self.p3590[0],
                                                         self.p3590[1])

            lbd2, flux2, var2 = self._extrema_value_in_interval(self.p3930[0],
                                                                self.p3930[1],
                                                                self.maxima[
                                                                    'x'],
                                                                self.maxima[
                                                                    'v'],
                                                                self.maxima[
                                                                    's'],
                                                                extrema='maxima',
                                                                verbose=verbose)
            if simu and lbd2 is None:
                lbd2, flux2, var2 = self.max_of_interval(self.p3930[0],
                                                         self.p3930[1])

            edca_value = (self._equivalentwidth(self.x, self.y, lbd1=lbd1,
                                                lbd2=lbd2, flux1=flux1,
                                                flux2=flux2,
                                                verbose=verbose)) / \
                (lbd2 - lbd1)
        except TypeError:
            if verbose:
                print >> sys.stderr, 'ERROR, none extrema found, try '\
                    'self.find_extrema()'
            edca_value = N.nan

        if simu:
            if not N.isfinite(edca_value):
                return [float(N.nan), float(N.nan)]

            edca_simu = []
            for simu in self.simulations:
                try:
                    edca_simu.append(simu.edca(simu=False, syst=False,
                                               verbose=False))
                except TypeError:
                    continue
            edca_sigma = self.std2(N.array(edca_simu)[N.isfinite(edca_simu)],
                                   edca_value)
            edca_mean = N.mean(N.array(edca_simu)[N.isfinite(edca_simu)])

            if N.isfinite(edca_value):
                self.edcavalues = {'edca': float(edca_value),
                                   'edca.err': float(edca_sigma),
                                   'edca.stat': float(edca_sigma),
                                   'edca.mean': float(edca_mean),
                                   'edca_lbd': [lbd1, lbd2]}

        if syst:
            edca_syst = []
            for system in self.syst:
                try:
                    edca_syst.append(system.edca(simu=False,
                                                 syst=False,
                                                 verbose=False))
                except TypeError:
                    continue
            edca_sigma_syst = self.std2(
                N.array(edca_syst)[N.isfinite(edca_syst)], edca_value)

            if N.isfinite(edca_sigma_syst):
                edca_sigma = float(N.sqrt(edca_sigma**2 + edca_sigma_syst**2))
            else:
                edca_sigma *= 2
            self.edcavalues['edca.syst'] = float(edca_sigma_syst)
            self.edcavalues['edca.err'] = float(edca_sigma)

            return [float(edca_value), float(edca_sigma)]

        if not simu and not syst:
            if N.isfinite(edca_value):
                self.edcavalues = {'edca': float(edca_value),
                                   'edca_lbd': [lbd1, lbd2]}
            return edca_value

    def rsi(self, verbose=True, simu=True, syst=True):
        """
        Retun the value and the error of rsi.

        [rsi, rsi_sigma]
        """
        # initialisation
        self.rsivalues = {'rsi': N.nan, 'rsi.err': N.nan, 'rsi.stat': N.nan,
                          'rsi.syst': N.nan, 'rsi.mean': N.nan, 'rsi_lbd': N.nan}
        if self.init_only:
            return

        lbd1, flux1, var1 = self._extrema_value_in_interval(self.p5603[0],
                                                            self.p5603[1],
                                                            self.maxima['x'],
                                                            self.maxima['v'],
                                                            self.maxima['s'],
                                                            extrema='maxima',
                                                            verbose=verbose)
        if lbd1 is None:
            lbd1, flux1, var1 = self.max_of_interval(self.p5603[0],
                                                     self.p5603[1],
                                                     verbose=verbose)

        lbd2, flux2, var2 = self._extrema_value_in_interval(5700, 5849,
                                                            self.minima['x'],
                                                            self.minima['v'],
                                                            self.minima['s'],
                                                            extrema='minima',
                                                            verbose=verbose)
        if lbd2 is None:
            try:
                lbd2, flux2, var2 = self._find_special_peak(5700, 5849,
                                                            minima=True)
            except TypeError:
                lbd2, flux2, var2 = None, None, None
        if simu and lbd2 is None:
            lbd2, flux2, var2 = self.max_of_interval(5700, 5849)

        lbd3, flux3, var3 = self._extrema_value_in_interval(5850, 6050,
                                                            self.maxima['x'],
                                                            self.maxima['v'],
                                                            self.maxima['s'],
                                                            extrema='maxima',
                                                            verbose=verbose,
                                                            right=True)
        if lbd3 is None:
            try:
                lbd3, flux3, var3 = self._find_special_peak(self.p5930[0],
                                                            self.p5930[1],
                                                            maxima=True,
                                                            right=True)
            except TypeError:
                lbd3, flux3, var3 = None, None, None
        if simu and lbd3 is None:
            lbd3, flux3, var3 = self.max_of_interval(self.p5930[0],
                                                     self.p5930[1])
        lbd4, flux4, var4 = self._extrema_value_in_interval(6000, 6210,
                                                            self.minima['x'],
                                                            self.minima['v'],
                                                            self.minima['s'],
                                                            extrema='minima',
                                                            verbose=verbose)
        if simu and lbd4 is None:
            lbd4, flux4, var4 = self.max_of_interval(6000, 6210)
        lbd5, flux5, var5 = self._extrema_value_in_interval(self.p6312[0],
                                                            self.p6312[1],
                                                            self.maxima['x'],
                                                            self.maxima['v'],
                                                            self.maxima['s'],
                                                            extrema='maxima',
                                                            verbose=verbose)
        if simu and lbd5 is None:
            lbd5, flux5, var5 = self.max_of_interval(
                self.p6312[0], self.p6312[1])
        # Check if the straight line in under the smoothing function
        x = N.polyval(N.polyfit([lbd3, lbd5], [flux3, flux5], 1),
                      self.x[(self.x > lbd3) & (self.x < lbd5)]) - \
            self.s[(self.x > lbd3) & (self.x < lbd5)]
        while len(x[x < 0]):
            lbd3 = self.x[(self.x == lbd3).nonzero()[0][0] + 1]
            flux3 = self.smoother(lbd3)
            x = N.polyval(N.polyfit([lbd3, lbd5], [flux3, flux5], 1),
                          self.x[(self.x > lbd3) & (self.x < lbd5)]) - \
                self.s[(self.x > lbd3) & (self.x < lbd5)]

        if lbd2 is None and lbd1 is not None and lbd3 is not None:
            try:
                p = N.polyfit([lbd1, lbd3], [flux1, flux3], 1)
                interval = (self.x >= lbd1) & (self.x <= lbd3)
                lbd2 = (self.x[interval])[N.argmax(N.polyval(p,
                                                             self.x[interval])
                                                   - self.s[interval])]
                flux2 = (self.s[interval])[N.argmax(N.polyval(p,
                                                              self.x[interval])
                                                    - self.s[interval])]
            except TypeError:
                lbd2, flux2, var2 = None, None, None
        lbd = [float(lbd1), float(lbd2), float(lbd3), float(lbd4), float(lbd5)]
        flux = [flux1, flux2, flux3, flux4, flux5]
        try:
            d_blue = self._equivalentdepth(lbd1=lbd[0], lbd2=lbd[1],
                                           lbd3=lbd[2], flux1=flux[0],
                                           flux2=flux[1], flux3=flux[2],
                                           verbose=verbose)
            d_red = self._equivalentdepth(lbd1=lbd[2], lbd2=lbd[3],
                                          lbd3=lbd[4], flux1=flux[2],
                                          flux2=flux[3], flux3=flux[4],
                                          verbose=verbose)
            rsi_value = d_blue / d_red

        except TypeError:
            if verbose:
                print >> sys.stderr, 'ERROR in computing of rsi, no '\
                    'wavelenght to compute rsi or maybe none extrema found, '\
                    'try self.find_extrema()'
            rsi_value = N.nan

        if simu:
            if not N.isfinite(rsi_value):
                return [float(N.nan), float(N.nan)]

            rsi_simu = []
            for simu in self.simulations:
                try:
                    rsi_simu.append(simu.rsi(simu=False, syst=False,
                                             verbose=False))
                except TypeError:
                    continue
            rsi_sigma = self.std2(N.array(rsi_simu)[N.isfinite(rsi_simu)],
                                  rsi_value)
            rsi_mean = N.mean(N.array(rsi_simu)[N.isfinite(rsi_simu)])

            if N.isfinite(rsi_value):
                self.rsivalues = {'rsi': float(rsi_value),
                                  'rsi.err': float(rsi_sigma),
                                  'rsi.stat': float(rsi_sigma),
                                  'rsi.mean': float(rsi_mean),
                                  'rsi_lbd': lbd}

        if syst:
            rsi_syst = []
            for system in self.syst:
                try:
                    rsi_syst.append(system.rsi(simu=False, syst=False,
                                               verbose=False))
                except TypeError:
                    continue
            rsi_sigma_syst = self.std2(N.array(rsi_syst)[N.isfinite(rsi_syst)],
                                       rsi_value)
            if N.isfinite(rsi_sigma_syst):
                rsi_sigma = float(N.sqrt(rsi_sigma**2 + rsi_sigma_syst**2))
            else:
                rsi_sigma *= 2
            self.rsivalues['rsi.syst'] = float(rsi_sigma_syst)
            self.rsivalues['rsi.err'] = float(rsi_sigma)

            return [float(rsi_value), float(rsi_sigma)]

        if simu == False and syst == False:
            if N.isfinite(rsi_value):
                self.rsivalues = {'rsi': float(rsi_value),
                                  'rsi_lbd': lbd,
                                  'rsi.err': N.nan,
                                  'rsi.stat': N.nan,
                                  'rsi.syst': N.nan,
                                  'rsi.mean': N.nan}

            return rsi_value

    def rsiS(self, verbose=True, simu=True, syst=True):
        """
        Return the value and the error of rsiS
        [rsiS, rsiS_sigma]
        """
        # initialisation
        self.rsiSvalues = {'rsiS': N.nan, 'rsiS.err': N.nan, 'rsiS.stat': N.nan,
                           'rsiS.syst': N.nan, 'rsiS.mean': N.nan,
                           'rsiS_lbd': [N.nan, N.nan], 'rsiS_flux': [N.nan, N.nan]}
        if self.init_only:
            return

        lbd1, flux1, var1 = self._extrema_value_in_interval(self.p5603[0],
                                                            self.p5603[1],
                                                            self.maxima['x'],
                                                            self.maxima['v'],
                                                            self.maxima['s'],
                                                            extrema='maxima',
                                                            verbose=verbose)
        if simu and lbd1 is None:
            lbd1, flux1, var1 = self.max_of_interval(
                self.p5603[0], self.p5603[1])

        lbd2, flux2, var2 = self._extrema_value_in_interval(self.p6312[0],
                                                            self.p6312[1],
                                                            self.maxima['x'],
                                                            self.maxima['v'],
                                                            self.maxima['s'],
                                                            extrema='maxima',
                                                            verbose=verbose)
        if simu and lbd2 is None:
            lbd2, flux2, var2 = self.max_of_interval(
                self.p6312[0], self.p6312[1])

        try:
            rsiS_value = flux1 / flux2
        except TypeError:
            if verbose:
                print >> sys.stderr, "ERROR in computing rsiS"
            rsiS_value = N.nan

        if simu:
            if not N.isfinite(rsiS_value):
                return [float(N.nan), float(N.nan)]

            rsiS_simu = []
            for simu in self.simulations:
                try:
                    rsiS_simu.append(simu.rsiS(simu=False, syst=False,
                                               verbose=False))
                except TypeError:
                    continue

            rsiS_sigma = self.std2(N.array(rsiS_simu)[N.isfinite(rsiS_simu)],
                                   rsiS_value)
            rsiS_mean = N.mean(N.array(rsiS_simu)[N.isfinite(rsiS_simu)])

            if N.isfinite(rsiS_value):
                self.rsiSvalues = {'rsiS': float(rsiS_value),
                                   'rsiS.err': float(rsiS_sigma),
                                   'rsiS.stat': float(rsiS_sigma),
                                   'rsiS.mean': float(rsiS_mean),
                                   'rsiS_lbd': [float(lbd1), float(lbd2)],
                                   'rsiS_flux': [float(flux1), float(flux2)]}

        if syst:

            rsiS_syst = []
            for system in self.syst:
                try:
                    rsiS_syst.append(system.rsiS(simu=False, syst=False,
                                                 verbose=False))
                except TypeError:
                    continue

            rsiS_sigma_syst = self.std2(
                N.array(rsiS_syst)[N.isfinite(rsiS_syst)], rsiS_value)
            if N.isfinite(rsiS_sigma_syst):
                rsiS_sigma = float(N.sqrt(rsiS_sigma**2 + rsiS_sigma_syst**2))
            else:
                rsiS_sigma *= 2
            self.rsiSvalues['rsiS.syst'] = float(rsiS_sigma_syst)
            self.rsiSvalues['rsiS.err'] = float(rsiS_sigma)

            return [float(rsiS_value), float(rsiS_sigma)]

        if simu == False and syst == False:

            if N.isfinite(rsiS_value):
                self.rsiSvalues = {'rsiS': float(rsiS_value),
                                   'rsiS_lbd': [float(lbd1), float(lbd2)],
                                   'rsiS_flux': [float(flux1), float(flux2)],
                                   'rsiS.err': N.nan, 'rsiS.stat': N.nan,
                                   'rsiS.syst': N.nan, 'rsiS.mean': N.nan}
            return rsiS_value

    def rsiSS2(self, verbose=True, simu=True):
        """
        Return the value and the error of rsiSS
        [rsiSS, rsiSS_sigma]
        """
        # initialisation
        self.rsiSSvalues = {'rsiSS': N.nan, 'rsiSS.err': N.nan,
                            'rsiSS.stat': N.nan, 'rsiSS.syst': N.nan,
                            'rsiSS.mean': N.nan,
                            'rsiSS_lbd': [N.nan, N.nan, N.nan, N.nan]}
        if self.init_only:
            return

        min_1 = 5500
        max_1 = 5700
        min_2 = 6200
        max_2 = 6450
        try:
            rsiSS_value = (self._integration(self.x, self.y, min=min_1,
                                             max=max_1, verbose=verbose)) / \
                (self._integration(self.x, self.y,
                                   min=min_2,
                                   max=max_2,
                                   verbose=verbose))
        except TypeError:
            if verbose:
                print >> sys.stderr, 'ERROR in rsiSS computing'
            rsiSS_value = float(N.nan)

        if simu:
            if not N.isfinite(rsiSS_value):
                return [float(N.nan), float(N.nan)]

            rsiSS_simu = []
            for simu in self.simulations:
                try:
                    rsiSS_simu.append(simu.rsiSS(simu=False, verbose=False))
                except TypeError:
                    continue

            rsiSS_sigma = self.std2(rsiSS_simu, rsiSS_value)
            rsiSS_mean = N.mean(rsiSS_simu)

            if N.isfinite(rsiSS_value):
                self.rsiSSvalues = {'rsiSS': float(rsiSS_value),
                                    'rsiSS.err': float(rsiSS_sigma),
                                    'rsiSS.stat': float(rsiSS_sigma),
                                    'rsiSS.mean': float(rsiSS_mean),
                                    'rsiSS_lbd': [float(min_1),
                                                  float(max_1),
                                                  float(min_2),
                                                  float(max_2)]}

            return [float(rsiSS_value), float(rsiSS_sigma)]

        else:
            if N.isfinite(rsiSS_value):
                self.rsiSSvalues = {'rsiSS': float(rsiSS_value),
                                    'rsiSS_lbd': [float(min_1),
                                                  float(max_1),
                                                  float(min_2),
                                                  float(max_2)],
                                    'rsiSS.err': N.nan, 'rsiSS.stat': N.nan,
                                    'rsiSS.mean': N.nan, 'rsiSS.syst': N.nan}

            return rsiSS_value

    def rsiSS(self, verbose=True, simu=True):
        """
        Return the value and the error of rsiSS
        [rsiSS, rsiSS_sigma]
        """

        min_1 = 5500
        max_1 = 5700
        min_2 = 6200
        max_2 = 6450
        try:
            a = self._integration(self.x, self.y, min=min_1, max=max_1,
                                  verbose=verbose)
            b = self._integration(self.x, self.y, min=min_2, max=max_2,
                                  verbose=verbose)
            var_a = self._var_integration(self.x, self.v, min=min_1, max=max_1,
                                          verbose=verbose)
            var_b = self._var_integration(self.x, self.v, min=min_2, max=max_2,
                                          verbose=verbose)
            rsiSS_value = a / b
            rsiSS_sigma = N.sqrt(self._var_rapport(a, b, var_a, var_b,
                                                   verbose=verbose))
        except TypeError:
            if verbose:
                print >> sys.stderr, 'ERROR in compute of rsiSS'
            rsiSS_value = float(N.nan)
            rsiSS_sigma = float(N.nan)
        self.rsiSSvalues = {'rsiSS': float(rsiSS_value),
                            'rsiSS.err': float(rsiSS_sigma),
                            'rsiSS_lbd': [float(min_1),
                                          float(max_1),
                                          float(min_2),
                                          float(max_2)]}

        return [float(rsiSS_value), float(rsiSS_sigma)]

    def ew(self, lambda_min_blue, lambda_max_blue, lambda_min_red,
           lambda_max_red, sf, verbose=True, simu=True,
           right1=False, left1=False, right2=False, left2=False,
           sup=False, syst=True, check=True):
        """
        Return the value and the error of an Equivalent Width
        [lambda_min_blue, lambda_max_blue] and [lambda_min_red, lambda_max_red]
        are the interval where the two peaks are searching.
        'sf' is the name (a string) of the spectral feature associated to the ew
        [ew, ew_sigma]
        """
        # shoftcut
        ewV = self.ewvalues

        # Initialisation
        ewV['ew%s' % sf] = N.nan
        ewV['lbd_ew%s' % sf] = [N.nan, N.nan]
        ewV['flux_ew%s' % sf] = [N.nan, N.nan]
        ewV['R%s' % sf] = N.nan
        ewV['flux_sum_norm_ew%s' % sf] = N.nan
        ewV['depth_norm_ew%s' % sf] = N.nan
        ewV['depth_ew%s' % sf] = N.nan
        ewV['depth_ew%s.err' % sf] = N.nan
        ewV['depth_ew%s.stat' % sf] = N.nan
        ewV['depth_ew%s.syst' % sf] = N.nan
        ewV['depth_ew%s.mean' % sf] = N.nan
        ewV['surf_norm_ew%s' % sf] = N.nan
        ewV['surf_ew%s.err' % sf] = N.nan
        ewV['surf_ew%s.stat' % sf] = N.nan
        ewV['surf_ew%s.syst' % sf] = N.nan
        ewV['surf_ew%s.mean' % sf] = N.nan
        ewV['width_ew%s' % sf] = N.nan
        ewV['ew%s.err' % sf] = N.nan
        ewV['ew%s.stat' % sf] = N.nan
        ewV['ew%s.syst' % sf] = N.nan
        ewV['flux_sum_norm_ew%s.err' % sf] = N.nan
        ewV['flux_sum_norm_ew%s.stat' % sf] = N.nan
        ewV['flux_sum_norm_ew%s.syst' % sf] = N.nan
        ewV['depth_norm_ew%s.err' % sf] = N.nan
        ewV['depth_norm_ew%s.stat' % sf] = N.nan
        ewV['depth_norm_ew%s.syst' % sf] = N.nan
        ewV['width_ew%s.err' % sf] = N.nan
        ewV['width_ew%s.stat' % sf] = N.nan
        ewV['width_ew%s.syst' % sf] = N.nan
        ewV['ew%s.mean' % sf] = N.nan
        ewV['R%s.mean' % sf] = N.nan
        ewV['R%s.stat' % sf] = N.nan
        ewV['R%s.syst' % sf] = N.nan
        ewV['R%s.err' % sf] = N.nan
        ewV['flux_sum_norm_ew%s.mean' % sf] = N.nan
        ewV['depth_norm_ew%s.mean' % sf] = N.nan
        ewV['width_ew%s.mean' % sf] = N.nan
        ewV['ew%s.med' % sf] = N.nan
        ewV['fmean_ew%s' % sf] = N.nan
        ewV['fmean_ew%s.err' % sf] = N.nan
        ewV['fmean_ew%s.stat' % sf] = N.nan
        ewV['fmean_ew%s.syst' % sf] = N.nan
        ewV['fmean_ew%s.mean' % sf] = N.nan

        if self.init_only:
            return

        # Function to compute the ew value and find its parameters ============
        try:
            lbd1, flux1, var1 = self._extrema_value_in_interval(lambda_min_blue,
                                                                lambda_max_blue,
                                                                self.maxima[
                                                                    'x'],
                                                                self.maxima[
                                                                    'v'],
                                                                self.maxima[
                                                                    's'],
                                                                extrema='maxima',
                                                                verbose=verbose,
                                                                right=right1,
                                                                left=left1)
            if lbd1 is None:
                try:
                    lbd1, flux1, var1 = self._find_special_peak(lambda_min_blue,
                                                                lambda_max_blue,
                                                                maxima=True,
                                                                right=right1,
                                                                left=left1)
                except TypeError:
                    lbd1, flux1, var1 = None, None, None

            if simu and lbd1 is None:
                lbd1, flux1, var1 = self.max_of_interval(lambda_min_blue,
                                                         lambda_max_blue)
                check = False

            lbd2, flux2, var2 = self._extrema_value_in_interval(lambda_min_red,
                                                                lambda_max_red,
                                                                self.maxima[
                                                                    'x'],
                                                                self.maxima[
                                                                    'v'],
                                                                self.maxima[
                                                                    's'],
                                                                extrema='maxima',
                                                                verbose=verbose,
                                                                right=right2,
                                                                left=left2)

            if lbd2 is None:
                try:
                    lbd2, flux2, var2 = self._find_special_peak(lambda_min_red,
                                                                lambda_max_red,
                                                                maxima=True,
                                                                right=right2,
                                                                left=left2)
                except TypeError:
                    lbd2, flux2, var2 = None, None, None

            if simu and lbd2 is None:
                lbd2, flux2, var2 = self.max_of_interval(lambda_min_red,
                                                         lambda_max_red)
                check = False

            # Check if the straight line in under the smoothing function
            if check:
                if sup == True and lbd2 is not None and lbd1 is not None:
                    x = N.polyval(N.polyfit([lbd1, lbd2],
                                            [flux1, flux2],
                                            1),
                                  self.x[(self.x > lbd1)
                                         & (self.x < lbd2)]) \
                        - self.s[(self.x > lbd1)
                                 & (self.x < lbd2)]
                    step = self.x[1] - self.x[0]
                    lbd1_tmp, flux1_tmp = lbd1, flux1
                    while len(x[x < 0]) > 5 and lbd1_tmp <= lambda_max_blue:
                        lbd1_tmp = self.x[
                            (self.x == lbd1_tmp).nonzero()[0][0] + 1]
                        flux1_tmp = self.smoother(lbd1_tmp)
                        x = N.polyval(N.polyfit([lbd1_tmp, lbd2],
                                                [flux1_tmp, flux2],
                                                1),
                                      self.x[(self.x > lbd1_tmp)
                                             & (self.x < lbd2)]) \
                            - self.s[(self.x > lbd1_tmp)
                                     & (self.x < lbd2)]
                    if len(x[x < 0]) > 5:
                        while len(x[x < 0]) > 5 and lbd2 >= lambda_min_red:
                            lbd2 = self.x[(self.x == lbd2).nonzero()[0][0] - 1]
                            flux2 = self.smoother(lbd2)
                            x = N.polyval(N.polyfit([lbd1, lbd2],
                                                    [flux1, flux2],
                                                    1),
                                          self.x[(self.x > lbd1)
                                                 & (self.x < lbd2)]) \
                                - self.s[(self.x > lbd1)
                                         & (self.x < lbd2)]
                    else:
                        lbd1, flux1 = lbd1_tmp, flux1_tmp
            #if sup and len(x[x<0]) > 5: ew_value = N.nan
            # else: ew_value = self._equivalentwidth(self.x, self.y,
            # lbd1=lbd1, lbd2=lbd2, flux1=flux1, flux2=flux2, verbose=verbose)
            ew_value = self._equivalentwidth(self.x,
                                             self.y,
                                             lbd1=lbd1,
                                             lbd2=lbd2,
                                             flux1=flux1,
                                             flux2=flux2,
                                             verbose=verbose)
        except TypeError:
            if verbose:
                print >> sys.stderr, 'ERROR, no extrema found, '\
                    'try self.find_extrema()'
            ew_value = N.nan
        #======================================================================

        if N.isfinite(ew_value):  # Additional informations
            interval = (self.x > lbd1) & (self.x < lbd2)
            arg = N.argmin(self.s[interval])
            lbd3 = self.x[interval][arg]
            surf = N.sum(N.polyval(N.polyfit([lbd1, lbd2],
                                             [flux1, flux2], 1),
                                   self.x[interval])
                         - self.y[interval])
            depth = self._equivalentdepth(lbd1=lbd1,
                                          lbd2=lbd3,
                                          lbd3=lbd2,
                                          flux1=self.smoother(lbd1),
                                          flux2=self.smoother(lbd3),
                                          flux3=self.smoother(lbd2),
                                          verbose=True)
            p = N.polyfit([lbd1, lbd2], [flux1, flux2], 1)  # y=p[0]*x+p[1]
            flux_norm = N.divide(N.sum(N.polyval(p, self.x[interval])),
                                 float(N.mean(N.polyval(p, self.x[interval]))))
            depth_n = depth / N.polyval(p, lbd3)
            fmean = 2 * (flux2 - self.smoother(lbd3)) \
                / (flux2 + self.smoother(lbd3))

            ewV['ew%s' % sf] = float(ew_value)
            ewV['lbd_ew%s' % sf] = [float(lbd1), float(lbd2)]
            ewV['flux_ew%s' % sf] = [float(flux1), float(flux2)]
            ewV['R%s' % sf] = float(flux2 / flux1)
            ewV['flux_sum_norm_ew%s' % sf] = float(flux_norm)
            ewV['depth_norm_ew%s' % sf] = float(depth_n)
            ewV['width_ew%s' % sf] = float(lbd2 - lbd1)
            ewV['depth_ew%s' % sf] = float(depth)
            ewV['surf_ew%s' % sf] = float(surf)
            ewV['fmean_ew%s' % sf] = float(fmean)

        # Compute statistiaue error
        if simu:
            if not N.isfinite(ew_value):
                return [float(N.nan), float(N.nan)]

            ew_simu, R_simu, d_simu, w_simu, f_simu, fm_simu = [
            ], [], [], [], [], []
            dep_simu, sur_simu = [], []
            for simu in self.simulations:
                try:
                    ew_simu.append(simu.ew(lambda_min_blue,
                                           lambda_max_blue,
                                           lambda_min_red,
                                           lambda_max_red,
                                           sf,
                                           simu=False,
                                           syst=False,
                                           verbose=False))
                    R_simu.append(float(simu.ewvalues['R%s' % sf]))
                    f_simu.append(
                        float(simu.ewvalues['flux_sum_norm_ew%s' % sf]))
                    d_simu.append(float(simu.ewvalues['depth_norm_ew%s' % sf]))
                    w_simu.append(float(simu.ewvalues['width_ew%s' % sf]))
                    dep_simu.append(float(simu.ewvalues['depth_ew%s' % sf]))
                    sur_simu.append(float(simu.ewvalues['surf_ew%s' % sf]))
                    fm_simu.append(float(simu.ewvalues['fmean_ew%s' % sf]))
                except TypeError:
                    continue
            ew_sigma = self.std2(
                N.array(ew_simu)[N.isfinite(ew_simu)], ew_value)
            R_sigma = self.std2(N.array(R_simu)[N.isfinite(R_simu)],
                                float(flux2 / flux1))
            f_sigma = self.std2(N.array(f_simu)[N.isfinite(f_simu)], flux_norm)
            d_sigma = self.std2(N.array(d_simu)[N.isfinite(d_simu)], depth_n)
            w_sigma = self.std2(N.array(w_simu)[N.isfinite(w_simu)],
                                float(lbd2 - lbd1))
            dep_sigma = self.std2(N.array(dep_simu)[N.isfinite(dep_simu)],
                                  depth)
            sur_sigma = self.std2(N.array(sur_simu)[N.isfinite(sur_simu)],
                                  surf)
            fmean_sigma = self.std2(N.array(fm_simu)[N.isfinite(fm_simu)],
                                    fmean)

            ew_mean = N.mean(N.array(ew_simu)[N.isfinite(ew_simu)])
            R_mean = N.mean(N.array(R_simu)[N.isfinite(R_simu)])
            f_mean = N.mean(N.array(f_simu)[N.isfinite(f_simu)])
            d_mean = N.mean(N.array(d_simu)[N.isfinite(d_simu)])
            w_mean = N.mean(N.array(w_simu)[N.isfinite(w_simu)])
            dep_mean = N.mean(N.array(dep_simu)[N.isfinite(dep_simu)])
            sur_mean = N.mean(N.array(sur_simu)[N.isfinite(sur_simu)])
            fmean_mean = N.mean(N.array(fm_simu)[N.isfinite(fm_simu)])

            ew_med = N.median(N.array(ew_simu)[N.isfinite(ew_simu)])

            ewV['ew%s.err' % sf] = float(ew_sigma)
            ewV['ew%s.stat' % sf] = float(ew_sigma)
            ewV['R%s.err' % sf] = float(R_sigma)
            ewV['R%s.stat' % sf] = float(R_sigma)
            ewV['flux_sum_norm_ew%s.err' % sf] = float(f_sigma)
            ewV['flux_sum_norm_ew%s.stat' % sf] = float(f_sigma)
            ewV['depth_norm_ew%s.err' % sf] = float(d_sigma)
            ewV['depth_norm_ew%s.stat' % sf] = float(d_sigma)
            ewV['width_ew%s.err' % sf] = float(w_sigma)
            ewV['width_ew%s.stat' % sf] = float(w_sigma)
            ewV['depth_ew%s.err' % sf] = float(dep_sigma)
            ewV['depth_ew%s.stat' % sf] = float(dep_sigma)
            ewV['surf_ew%s.err' % sf] = float(sur_sigma)
            ewV['surf_ew%s.stat' % sf] = float(sur_sigma)
            ewV['fmean_ew%s.err' % sf] = float(fmean_sigma)
            ewV['fmean_ew%s.stat' % sf] = float(fmean_sigma)

            ewV['ew%s.mean' % sf] = float(ew_mean)
            ewV['R%s.mean' % sf] = float(R_mean)
            ewV['flux_sum_norm_ew%s.mean' % sf] = float(f_mean)
            ewV['depth_norm_ew%s.mean' % sf] = float(d_mean)
            ewV['width_ew%s.mean' % sf] = float(w_mean)
            ewV['depth_ew%s.mean' % sf] = float(dep_mean)
            ewV['surf_ew%s.mean' % sf] = float(sur_mean)
            ewV['fmean_ew%s.mean' % sf] = float(fmean_mean)

            ewV['ew%s.med' % sf] = float(ew_med)

        # Compute systematic error
        if syst:
            if not N.isfinite(ew_value):
                return [float(N.nan), float(N.nan)]

            ew_syst, R_syst, d_syst, w_syst, f_syst, fm_syst = [
            ], [], [], [], [], []
            dep_syst, sur_syst = [], []
            for system in self.syst:
                try:
                    ew_syst.append(system.ew(lambda_min_blue,
                                             lambda_max_blue,
                                             lambda_min_red,
                                             lambda_max_red,
                                             sf,
                                             simu=False,
                                             syst=False,
                                             verbose=False))
                    R_syst.append(float(system.ewvalues['R%s' % sf]))
                    f_syst.append(float(system.ewvalues['flux_sum_norm_ew%s' %
                                                        sf]))
                    d_syst.append(
                        float(system.ewvalues['depth_norm_ew%s' % sf]))
                    w_syst.append(float(system.ewvalues['width_ew%s' % sf]))
                    dep_syst.append(float(system.ewvalues['depth_ew%s' % sf]))
                    sur_syst.append(float(system.ewvalues['surf_ew%s' % sf]))
                    fm_syst.append(float(system.ewvalues['fmean_ew%s' % sf]))
                except TypeError:
                    continue

            ew_sigma_syst = self.std2(N.array(ew_syst)[N.isfinite(ew_syst)],
                                      ew_value)
            R_sigma_syst = self.std2(N.array(R_syst)[N.isfinite(R_syst)],
                                     float(flux2 / flux1))
            f_sigma_syst = self.std2(N.array(f_syst)[N.isfinite(f_syst)],
                                     flux_norm)
            d_sigma_syst = self.std2(N.array(d_syst)[N.isfinite(d_syst)],
                                     depth_n)
            w_sigma_syst = self.std2(N.array(w_syst)[N.isfinite(w_syst)],
                                     float(lbd2 - lbd1))
            dep_sigma_syst = self.std2(N.array(dep_syst)[N.isfinite(dep_syst)],
                                       depth)
            sur_sigma_syst = self.std2(N.array(sur_syst)[N.isfinite(sur_syst)],
                                       surf)
            fm_sigma_syst = self.std2(N.array(fm_syst)[N.isfinite(fm_syst)],
                                      fmean)

            if not N.isfinite(ew_sigma_syst):
                ew_sigma_syst = float(0.0)
            if not N.isfinite(R_sigma_syst):
                R_sigma_syst = float(0.0)
            if not N.isfinite(f_sigma_syst):
                f_sigma_syst = float(0.0)
            if not N.isfinite(d_sigma_syst):
                d_sigma_syst = float(0.0)
            if not N.isfinite(w_sigma_syst):
                w_sigma_syst = float(0.0)
            if not N.isfinite(dep_sigma_syst):
                dep_sigma_syst = float(0.0)
            if not N.isfinite(sur_sigma_syst):
                sur_sigma_syst = float(0.0)
            if not N.isfinite(fm_sigma_syst):
                fm_sigma_syst = float(0.0)

            ew_sigma = N.sqrt(ew_sigma**2 + ew_sigma_syst**2)
            ewV['ew%s.syst' % sf] = float(ew_sigma_syst)
            ewV['ew%s.err' % sf] = float(N.sqrt(ewV['ew%s.err' % sf]**2 +
                                                ew_sigma_syst**2))
            ewV['R%s.syst' % sf] = float(R_sigma_syst)
            ewV['R%s.err' % sf] = float(N.sqrt(ewV['R%s.err' % sf]**2 +
                                               R_sigma_syst**2))
            ewV['flux_sum_norm_ew%s.syst' % sf] = float(f_sigma_syst)
            ewV['flux_sum_norm_ew%s.err' % sf] = float(
                N.sqrt(ewV['flux_sum_norm_ew%s.err' % sf]**2 + f_sigma_syst**2))
            ewV['depth_norm_ew%s.syst' % sf] = float(d_sigma_syst)
            ewV['depth_norm_ew%s.err' % sf] = float(
                N.sqrt(ewV['depth_norm_ew%s.err' % sf]**2 + d_sigma_syst**2))
            ewV['width_ew%s.syst' % sf] = float(w_sigma_syst)
            ewV['width_ew%s.err' % sf] = float(N.sqrt(ewV['width_ew%s.err' % sf]**2
                                                      + w_sigma_syst**2))
            ewV['depth_ew%s.syst' % sf] = float(dep_sigma_syst)
            ewV['depth_ew%s.err' % sf] = float(N.sqrt(ewV['depth_ew%s.err' % sf]**2
                                                      + dep_sigma_syst**2))
            ewV['surf_ew%s.syst' % sf] = float(sur_sigma_syst)
            ewV['surf_ew%s.err' % sf] = float(N.sqrt(ewV['surf_ew%s.err' % sf]**2 +
                                                     sur_sigma_syst**2))
            ewV['fmean_ew%s.syst' % sf] = float(fm_sigma_syst)
            ewV['fmean_ew%s.err' % sf] = float(N.sqrt(ewV['fmean_ew%s.err' % sf]**2
                                                      + fm_sigma_syst**2))

            return [float(ew_value), float(ew_sigma)]

        if simu == False and syst == False:
            return ew_value

    def velocity(self, infodict, verbose=False, simu=True, syst=True,
                 left=False, right=False):
        """Value and error of a velocity of an absorption feature.

        infodict should have the following structure :
        {'lmin' : minimum lambda for searching the dip,
        'lmax' : max lambda for searching the dip,
        'lrest' : restframe wavelangth of absorption feature
        'name' : name of the feature}
        the error will be coded as ['name']+'.err'
        and the lambda as ['name']+'_lbd'"""
        # shortcut
        velo = self.velocityvalues

        c = 299792.458
        # Initialisation
        velo[infodict['name']] = N.nan
        velo[infodict['name'] + '.err'] = N.nan
        velo[infodict['name'] + '.stat'] = N.nan
        velo[infodict['name'] + '.syst'] = N.nan
        velo[infodict['name'] + '_lbd'] = N.nan
        velo[infodict['name'] + '_lbd.stat'] = N.nan
        velo[infodict['name'] + '_lbd.syst'] = N.nan
        velo[infodict['name'] + '_lbd.err'] = N.nan
        velo[infodict['name'] + '_lbd.mean'] = N.nan
        velo[infodict['name'] + '_flux'] = N.nan
        velo[infodict['name'] + '_flux.stat'] = N.nan
        velo[infodict['name'] + '_flux.syst'] = N.nan
        velo[infodict['name'] + '_flux.err'] = N.nan
        velo[infodict['name'] + '_flux.mean'] = N.nan
        velo[infodict['name'] + '.binsyst'] = N.nan
        velo[infodict['name'] + '.bin'] = N.nan
        velo[infodict['name'] + '.mean'] = N.nan
        velo[infodict['name'] + '.med'] = N.nan
        if self.init_only:
            return

        try:
            lbd, flux, var = self._extrema_value_in_interval(infodict['lmin'],
                                                             infodict['lmax'],
                                                             self.minima['x'],
                                                             self.minima['v'],
                                                             self.minima['s'],
                                                             extrema='minima',
                                                             verbose=verbose,
                                                             right=right,
                                                             left=left)
            if lbd is None:
                if verbose:
                    print 'find special peak'
                lbd, flux, var = self._find_special_peak(infodict['lmin'],
                                                         infodict['lmax'],
                                                         minima=True,
                                                         right=right, left=left)

            if simu and lbd is None:
                lbd, flux, var = self.max_of_interval(infodict['lmin'],
                                                      infodict['lmax'])
            if lbd is not None:
                lbd = self._get_min(lbd)
            velocity = (infodict['lrest'] - lbd) / infodict['lrest'] * c
        except TypeError:
            velocity = N.nan

        # check for the vSiII5972 velocity
        # if < 8000 km/s, check the curvature of the spectral area
        if N.isfinite(velocity) \
                and velocity <  8000 \
                and infodict['name'] == 'vSiII_5972':
            filt = (self.x > 5650) & (self.x < 5950)
            pol = N.polyfit(self.x[filt], self.s[filt], 2)
            if pol[0] * 1e6 <= 0.55:
                print "ERROR: Velocity is under 8000km/s, " \
                      "and curvature of the spectral zone is too small."
                velocity = N.nan

        if N.isfinite(velocity):
            velo[infodict['name']] = float(velocity)
            velo[infodict['name'] + '_lbd'] = float(lbd)
            velo[infodict['name'] + '_flux'] = float(flux)

        velocity_sigma = None
        if simu:
            # store only in case we are at top level.
            if not N.isfinite(velocity):
                return [float(N.nan), float(N.nan)]

            velocity_simu, lbd_simu, flux_simu = [], [], []
            for simul in self.simulations:
                try:
                    velocity_simu.append(simul.velocity(infodict, simu=False,
                                                        syst=False,
                                                        verbose=False))
                    lbd_simu.append(simul.velocityvalues[infodict['name'] +
                                                         '_lbd'])
                    flux_simu.append(simul.velocityvalues[infodict['name'] +
                                                          '_flux'])
                except TypeError:
                    continue

            velocity_sigma = self.std2(
                N.array(velocity_simu)[N.isfinite(velocity_simu)], velocity)
            lbd_sigma = self.std2(N.array(lbd_simu)[N.isfinite(lbd_simu)], lbd)
            flux_sigma = self.std2(
                N.array(flux_simu)[N.isfinite(flux_simu)], flux)
            velocity_mean = N.mean(
                N.array(velocity_simu)[N.isfinite(velocity_simu)])
            lbd_mean = N.mean(N.array(lbd_simu)[N.isfinite(lbd_simu)])
            flux_mean = N.mean(N.array(flux_simu)[N.isfinite(flux_simu)])
            velocity_med = N.median(
                N.array(velocity_simu)[N.isfinite(velocity_simu)])
            lbd_med = N.median(N.array(lbd_simu)[N.isfinite(lbd_simu)])
            flux_med = N.median(N.array(flux_simu)[N.isfinite(flux_simu)])

            velo[infodict['name'] + '.err'] = float(velocity_sigma)
            velo[infodict['name'] + '.stat'] = float(velocity_sigma)
            velo[infodict['name'] + '.mean'] = float(velocity_mean)
            velo[infodict['name'] + '.med'] = float(velocity_med)
            velo[infodict['name'] + '_lbd.err'] = float(lbd_sigma)
            velo[infodict['name'] + '_lbd.stat'] = float(lbd_sigma)
            velo[infodict['name'] + '_lbd.mean'] = float(lbd_mean)
            velo[infodict['name'] + '_lbd.med'] = float(lbd_med)
            velo[infodict['name'] + '_flux.err'] = float(flux_sigma)
            velo[infodict['name'] + '_flux.stat'] = float(flux_sigma)
            velo[infodict['name'] + '_flux.mean'] = float(flux_mean)
            velo[infodict['name'] + '_flux.med'] = float(flux_med)

        if syst:
            velocity_syst, lbd_syst, flux_syst = [], [], []
            for system in self.syst:
                try:
                    velocity_syst.append(system.velocity(infodict,
                                                         simu=False,
                                                         syst=False,
                                                         verbose=False))
                    lbd_syst.append(system.velocityvalues[infodict['name'] +
                                                          '_lbd'])
                    flux_syst.append(system.velocityvalues[infodict['name'] +
                                                           '_flux'])
                except TypeError:
                    continue

            binning = self.x[1] - self.x[0]
            velocity_syst_sigma = self.std2(
                N.array(velocity_syst)[N.isfinite(velocity_syst)], velocity)
            lbd_syst_sigma = self.std2(
                N.array(lbd_syst)[N.isfinite(lbd_syst)], lbd)
            flux_syst_sigma = self.std2(
                N.array(flux_syst)[N.isfinite(flux_syst)], flux)
            velocity_syst_bin = ((binning) * c) / (N.sqrt(12)
                                                   * infodict['lrest'])

            velocity_sigma = N.sqrt(velocity_sigma ** 2
                                    + velocity_syst_sigma**2
                                    + velocity_syst_bin**2)
            lbd_sigma = N.sqrt(lbd_sigma ** 2 + lbd_syst_sigma**2
                               + (binning)**2)
            flux_sigma = N.sqrt(flux_sigma ** 2 + flux_syst_sigma**2)

            velo[infodict['name'] + '.syst'] = float(velocity_syst_sigma)
            velo[infodict['name'] + '.err'] = float(velocity_sigma)
            velo[infodict['name'] + '_lbd.syst'] = float(lbd_syst_sigma)
            velo[infodict['name'] + '_lbd.err'] = float(lbd_sigma)
            velo[infodict['name'] + '_flux.syst'] = float(flux_syst_sigma)
            velo[infodict['name'] + '_flux.err'] = float(flux_sigma)
            velo[infodict['name'] + '.binsyst'] = float(velocity_syst_bin)
            velo[infodict['name'] + '.bin'] = float(binning)

        if velocity_sigma is None:
            return float(velocity)
        else:
            return [float(velocity), float(velocity_sigma)]

    def velocity2(self, infodict, verbose=True, simu=True, syst=True,
                  left=False, right=False):
        """
        Second velocity measurement.

        returns the value and the error of a velocity of an absprtion feature
        infodict should have the following structure :
        {'lmin' : minimum lambda for searching the dip,
        'lmax' : max lambda for searching the dip,
        'lrest' : restframe wavelangth of absorption feature
        'name' : name of the feature}
        the error will be coded as ['name']+'.err'
        and the lambda as ['name']+'_lbd'

        Get the first minimum after the blue edge
        """
        # shortcut
        velo = self.velocityvalues

        # Initialisation
        velo[infodict['name'] + '.err'] = N.nan
        velo[infodict['name'] + '.stat'] = N.nan
        velo[infodict['name'] + '.syst'] = N.nan
        velo[infodict['name'] + '_lbd'] = N.nan
        velo[infodict['name'] + '_lbd.stat'] = N.nan
        velo[infodict['name'] + '_lbd.syst'] = N.nan
        velo[infodict['name'] + '_lbd.err'] = N.nan
        velo[infodict['name'] + '_lbd.mean'] = N.nan
        velo[infodict['name'] + '.binsyst'] = N.nan
        velo[infodict['name'] + '.bin'] = N.nan
        velo[infodict['name'] + '.mean'] = N.nan
        if self.init_only:
            return

        c = 299792.458
        print 'try new velocity method'
        lbdm, fluxm, varm = self._extrema_value_in_interval(infodict['lmin'],
                                                            infodict['lmax'],
                                                            self.maxima['x'],
                                                            self.maxima['v'],
                                                            self.maxima['s'],
                                                            extrema='maxima',
                                                            verbose=verbose,
                                                            right=right,
                                                            left=left)
        if lbdm is None:
            if verbose:
                print 'find special peak============'
            lbdm, fluxm, varm = self._find_special_peak(infodict['lmin'],
                                                        infodict['lmax'],
                                                        maxima=True,
                                                        right=right,
                                                        left=left)

        if simu and lbdm is None:
            lbdm, fluxm, varm = self.max_of_interval(infodict['lmin'],
                                                     infodict['lmax'])

        ok = self.minima['x'] > lbdm
        lbd = float(self.minima['x'][ok][0])
        flux = float(self.minima['v'][ok][0])
        var = float(self.minima['s'][ok][0])
        velocity = (infodict['lrest'] - lbd) / infodict['lrest'] * c

        if N.isfinite(velocity):
            velo[infodict['name']] = float(velocity)
            velo[infodict['name'] + '_lbd'] = float(lbd)

        velocity_sigma = None
        if simu:
            # store only in case we are at top level.
            if not N.isfinite(velocity):
                return [float(N.nan), float(N.nan)]

            velocity_simu, lbd_simu = [], []
            for simul in self.simulations:
                try:
                    velocity_simu.append(simul.velocity(infodict,
                                                        simu=False,
                                                        syst=False,
                                                        verbose=False))
                    lbd_simu.append(simul.velocityvalues[infodict['name'] +
                                                         '_lbd'])
                except TypeError:
                    continue

            velocity_sigma = self.std2(
                N.array(velocity_simu)[N.isfinite(velocity_simu)], velocity)
            lbd_sigma = self.std2(N.array(lbd_simu)[N.isfinite(lbd_simu)], lbd)
            velocity_mean = N.mean(
                N.array(velocity_simu)[N.isfinite(velocity_simu)])
            lbd_mean = N.mean(N.array(lbd_simu)[N.isfinite(lbd_simu)])

            velo[infodict['name'] + '.err'] = float(velocity_sigma)
            velo[infodict['name'] + '.stat'] = float(velocity_sigma)
            velo[infodict['name'] + '_lbd.err'] = float(lbd_sigma)
            velo[infodict['name'] + '_lbd.stat'] = float(lbd_sigma)
            velo[infodict['name'] + '_lbd.mean'] = float(lbd_mean)
            velo[infodict['name'] + '.mean'] = float(velocity_mean)

        if syst:
            velocity_syst, lbd_syst = [], []
            for system in self.syst:
                try:
                    velocity_syst.append(system.velocity(infodict, simu=False,
                                                         syst=False,
                                                         verbose=False))
                    lbd_syst.append(system.velocityvalues[infodict['name'] +
                                                          '_lbd'])
                except TypeError:
                    continue

            binning = self.x[1] - self.x[0]
            velocity_syst_sigma = self.std2(
                N.array(velocity_syst)[N.isfinite(velocity_syst)], velocity)
            lbd_syst_sigma = self.std2(N.array(lbd_syst)[N.isfinite(lbd_syst)],
                                       lbd)
            velocity_syst_bin = ((binning) * c) / (N.sqrt(12) *
                                                   infodict['lrest'])

            velocity_sigma = N.sqrt(velocity_sigma ** 2 +
                                    velocity_syst_sigma**2 +
                                    velocity_syst_bin**2)
            lbd_sigma = N.sqrt(
                lbd_sigma ** 2 + lbd_syst_sigma**2 + (binning)**2)

            velo[infodict['name'] + '.syst'] = float(velocity_syst_sigma)
            velo[infodict['name'] + '.err'] = float(velocity_sigma)
            velo[infodict['name'] + '_lbd.syst'] = float(lbd_syst_sigma)
            velo[infodict['name'] + '_lbd.err'] = float(lbd_sigma)
            velo[infodict['name'] + '.binsyst'] = float(velocity_syst_bin)
            velo[infodict['name'] + '.bin'] = float(binning)

        if velocity_sigma is None:
            return float(velocity)
        else:
            return [float(velocity), float(velocity_sigma)]


#==============================================================================
# Function to compute Stephen's ratio, and other general ratios
#==============================================================================


def integration(spec, lbd, v=2000.):
    """."""
    c = 299792.458
    min = lbd * (1 - v / c)
    max = lbd * (1 + v / c)
    step = spec.x[1] - spec.x[0]
    return float(N.sum(spec.y[(spec.x >= min) & (spec.x <= max)]) * step)


def stephen_ratio(specB, specR=None, lbd_6415=6415, lbd_4427=4427):
    """."""
    if specR is None and (specB.x[0] < 4500 and specB.x[-1] > 6400):
        return integration(specB, lbd_6415) / integration(specB, lbd_4427)
    elif specR is not None:
        return integration(specR, lbd_6415) / integration(specB, lbd_4427)
    else:
        print "Warning: Rsjb not computed."
        return 0.


def general_ratio(specB, specR=None, lbd1=6310, lbd2=4390, v=4000):
    """
    General flux ratio.

    new <good> R lbd1=6310, lbd2=4390 with v = 4000
    new <good> R lbd1=6310, lbd2=5130 with v = 2000 in silicon zone
    """
    if specR is None:
        specR = specB
    return integration(specR, lbd1, v=v) / integration(specB, lbd2, v=v)


def get_cranio(x, y, v, smoother='spline_free_knot', verbose=False):
    """."""
    obj = covariance.SPCS(x, y, v)
    if smoother == 'spline_free_knot':
        smoothing = 'sp'
    else:
        smoothing = 'sg'
    # obj.comp_rho_f()
    obj.smooth(smoothing=smoothing)
    obj.make_simu()
    simus = N.array([s.y for s in obj.simus])
    cr = Craniometer(x, y, v * obj.factor_used)
    cr.smooth(rho=obj.rho, smoother=smoother, s=obj.s, hsize=obj.w,
              verbose=verbose)
    cr.cranio_generator(rho=obj.rho, simus=simus, verbose=verbose)
    cr.find_extrema()
    return cr

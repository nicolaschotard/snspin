#!/usr/bin/env python
######################################################################
# Filename:      SNfPhrenology.py
# Version:       $Revision: 1.64 $
# Description:
# Author:        $Author: nchotard $
# Created at:    Thur 30 17:15:00 2008
# $Id: SNfPhrenology.py,v 1.64 2015/06/15 11:37:26 nchotard Exp $
######################################################################

import sys
import time
import numpy as N
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
import matplotlib.pyplot as P

import smooth_tools
import var_corr_lib as VCL
from ToolBox import Signal


class Craniometer:
    #=========================================================================
    # Initialization function
    #=========================================================================

    def __init__(self, wavelength, flux, variance):
        """ Class to feel bumps on SN spectra and conclude about how they
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
        EWSi4000 = cranio.EW(3830,3963,4034,4150,'SiII4000')
        values will be saved in cranio.EWvalues for EWs computing

        Spectrum initialization.
        self.x = wavelength
        self.y = flux
        self.v = variance
        self.s = []
        self.smoother = None
        self.maxima = None
        self.minima = None
        self.EWvalues = {}
        """
        self.x = wavelength
        self.y = flux
        self.v = variance
        self.s = []
        self.maxima = None
        self.minima = None
        self.EWvalues = {}
        self.velocityValues = {}
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
        Creates the smoother function and makes a smooth array out of spec.y

        Mode = 0 : number and position are fixed (smoother='spline_fix_knot')
        interpolate.LSQUnivariateSpline(self.x,
                                        self.y,
                                        t=(self.x[::12])[1:],
                                        w = 1/(N.sqrt(self.v)))

        Mode = 1 : number and position of knots are not fixed
        (smoother='spline_free_knot')
        interpolate.UnivariateSpline(self.x,self.y,w = 1/(N.sqrt(self.v)),s=s)

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
        Create a spline with interpolate

        Mode = 0 : number and position are fixed
        interpolate.LSQUnivariateSpline(self.x,
                                        self.y,
                                        t=(self.x[::12])[1:],
                                        w = 1/(N.sqrt(self.v)))

        Mode = 1 : number and position of knots are not fixed
        interpolate.UnivariateSpline(self.x,self.y,w = 1/(N.sqrt(self.v)),s=s)
        """
        rc = 1. - 2. * (rho**2)
        if s is None:
            try:
                s = smooth_tools.spline_find_s(self.x,
                                               self.y,
                                               self.v * rc,
                                               corr=(rho**2) / rc)
            except:
                s = 0.492 * len(self.x)
        try:
            s = s[0]
        except:
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
        """Use savitzky_golay() to apply a savitzky golay filter on the
        spectrum (from pySnurp)
        Input:
        - hsize : half size of the window (default:15)
        - order : order of the polynome used to smooth the spectrum (default:2)
        Output:
        - self.s : smoothing spectrum
        """
        rc = 1. - 2. * (rho**2)
        if hsize is None:
            try:
                hsize = int(smooth_tools.sg_find_num_points(self.x,
                                                            self.y,
                                                            self.v * rc,
                                                            corr=(rho**2) / rc))
            except:
                if verbose:
                    print >> sys.stderr, 'ERROR in computing of best hsize'
                hsize = 15
        if (hsize * 2) + 1 < (order + 2):
            hsize = 10  # order/2.+1
        if self.lim and hsize < self.lim:
            hsize = self.lim   # for oxygen zone only
        if verbose:
            print >> sys.stderr, 'best_w=%i' % hsize
        self.s = Signal.savitzky_golay(self.y,
                                       kernel=(int(hsize) * 2) + 1,
                                       order=order,
                                       derivative=0)
        self.s_deriv = Signal.savitzky_golay(self.y,
                                             kernel=(int(hsize) * 2) + 1,
                                             order=order,
                                             derivative=1)
        self.hsize = hsize
        self.order = order

    def smoother(self, lbd, verbose=False):

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
                flux = N.array([float(self.s[(self.x > (l - 1)) & (self.x < (l + 1))])
                                for l in lbd])
                return flux

    def find_extrema(self, verbose=True):
        """
        Function to find all extrema in a smoothed spectrum
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
        except:
            self.syst = None
            print >> sys.stderr, "ERROR in systematic_error (cranio_generator)!"

    def _correl_simulated_spectra(self, nsimu, rho=0.482, verbose=True):

        def comp_alpha(rho):
            return 0.5 * (1 + N.sqrt(1 - 4 * (rho**2)))

        def comp_beta(rho):
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
        Find all signicative extrema (x,y,v and smooth function) of a spectrum
        or spectrum's zone
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
            if (i > w) and (i < (len(x) - w)):
                window = (x >= x[i - w]) & (x <= x[i + w])
                lbdmin = (x[window])[N.argmin(s[window])]
                return x[i] == lbdmin
            else:
                return False

        def maxima(i, w):
            if (i > w) and (i < (len(x) - w)):
                window = (x >= x[i - w]) & (x <= x[i + w])
                lbdmax = (x[window])[N.argmax(s[window])]
                return x[i] == lbdmax
            else:
                return False

        def signaltonoise(i):
            good = (x > (x[i] - 20)) & (x < (x[i] + 20))
            return (y[i] / N.sqrt(v[i])) \
                / N.mean(y[good] / N.sqrt(v[good])) >= StoN

        # Define arrays: lambda,flux,smooth values and variance
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
        """Compute variance of an intergration"""

        if len(v):
            var_int = v[(x > min) & (x < max)].sum()
        else:
            if verbose:
                print >> sys.stderr, "No variance for this spectrum"
            var_int = N.nan
        return float(var_int)

    def _var_rapport(self, a, b, var_a, var_b, verbose=True):
        """Compute variance for a/b"""

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
        Function to find extrema values (lambda,flux and variance) in a given
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

        except:
            return [None, None, None]

    def _find_special_peak(self, min, max, maxima=False,
                           minima=False, right=False, left=False):

        if maxima == False and minima == False:
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
        la = self.x[(self.x > min) & (self.x < max)]
        fa = self.smoother(la)
        arg = N.argmax(fa)
        l, f = la[arg], fa[arg]
        v = 0.0
        return l, f, v

    def std2(self, x, x0):
        """
        conpute the standard deviation of the distribution x compared with x0
        x0 can be the mean or an other value
        """
        return N.sqrt(N.mean(N.absolute(x - x0)**2))

    def _get_min(self, lbd, plot=False):
        """
        Get the minimum the flux around a given bin, using the derivative
        of the smoothed function. A linear interpolation is made using the given
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

    def RCa(self, verbose=True, simu=True, syst=True):
        """
        Return the value and the error of RCa
        [RCa,RCa_sigma]
        """
        # Initialisation
        self.RCavalues = {'RCa': N.nan, 'RCa.err': N.nan, 'RCa.stat': N.nan,
                          'RCa.syst': N.nan, 'RCa.mean': N.nan,
                          'RCa_lbd': [N.nan, N.nan], 'RCa_flux': [N.nan, N.nan]}
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
            RCa_value = flux2 / flux1
        except:
            if verbose:
                print >> sys.stderr, "ERROR in computing RCa"
            RCa_value = N.nan

        if simu:

            if not N.isfinite(RCa_value):
                return [float(N.nan), float(N.nan)]

            RCa_simu = []
            for simu in self.simulations:
                try:
                    RCa_simu.append(simu.RCa(simu=False, syst=False,
                                             verbose=False))
                except:
                    continue
            RCa_sigma = self.std2(N.array(RCa_simu)[N.isfinite(RCa_simu)],
                                  RCa_value)
            RCa_mean = N.mean(N.array(RCa_simu)[N.isfinite(RCa_simu)])

            if N.isfinite(RCa_value):
                self.RCavalues = {'RCa': float(RCa_value),
                                  'RCa.err': float(RCa_sigma),
                                  'RCa.stat': float(RCa_sigma),
                                  'RCa.mean': float(RCa_mean),
                                  'RCa_lbd': [float(lbd1), float(lbd2)],
                                  'RCa_flux': [float(flux1), float(flux2)]}

        if syst:

            RCa_syst = []
            for system in self.syst:
                try:
                    RCa_syst.append(system.RCa(syst=False, simu=False,
                                               verbose=False))
                except:
                    continue
            RCa_sigma_syst = self.std2(N.array(RCa_syst)[N.isfinite(RCa_syst)],
                                       RCa_value)

            if N.isfinite(RCa_sigma_syst):
                RCa_sigma = float(N.sqrt(RCa_sigma**2 + RCa_sigma_syst**2))
            else:
                RCa_sigma *= 2
            self.RCavalues['RCa.syst'] = float(RCa_sigma_syst)
            self.RCavalues['RCa.err'] = float(RCa_sigma)

            return [float(RCa_value), float(RCa_sigma)]

        if simu == False and syst == False:

            if N.isfinite(RCa_value):
                self.RCavalues = {'RCa': float(RCa_value),
                                  'RCa_lbd': [float(lbd1), float(lbd2)],
                                  'RCa_flux': [float(flux1), float(flux2)],
                                  'RCa.err': N.nan, 'RCa.stat': N.nan,
                                  'RCa.mean': N.nan}
            return RCa_value

    def RCaS(self, verbose=True, simu=True, syst=True):
        """
        Return the value and the error of RCaS
        [RCaS,RCaS_sigma]
        """
        # Initialisation
        self.RCaSvalues = {'RCaS': N.nan, 'RCaS.err': N.nan, 'RCaS.stat': N.nan,
                           'RCaS.syst': N.nan, 'RCaS.mean': N.nan,
                           'RCaS_lbd': [N.nan, N.nan, N.nan, N.nan]}
        if self.init_only:
            return

        min_1 = 3620
        max_1 = 3716
        min_2 = 3887
        max_2 = 4012

        try:
            RCaS_value = (self._integration(self.x, self.y, min=min_2, max=max_2,
                                            verbose=verbose)) / \
                (self._integration(self.x, self.y,
                                   min=min_1,
                                   max=max_1,
                                   verbose=verbose))
        except:
            if verbose:
                print >> sys.stderr, 'ERROR in compute of RCaS'
            RCaS_value = float(N.nan)

        if simu:
            if not N.isfinite(RCaS_value):
                return [float(N.nan), float(N.nan)]

            RCaS_simu = []
            for simu in self.simulations:
                try:
                    RCaS_simu.append(simu.RCaS(simu=False, syst=False,
                                               verbose=False))
                except:
                    continue
            RCaS_sigma = self.std2(N.array(RCaS_simu)[N.isfinite(RCaS_simu)],
                                   RCaS_value)
            RCaS_mean = N.mean(N.array(RCaS_simu)[N.isfinite(RCaS_simu)])

            if N.isfinite(RCaS_value):
                self.RCaSvalues = {'RCaS': float(RCaS_value),
                                   'RCaS.err': float(RCaS_sigma),
                                   'RCaS.stat': float(RCaS_sigma),
                                   'RCaS.mean': float(RCaS_mean),
                                   'RCaS_lbd': [float(min_1),
                                                float(max_1),
                                                float(min_2),
                                                float(max_2)]}

        if syst:

            RCaS_syst = []
            for system in self.syst:
                try:
                    RCaS_syst.append(system.RCaS(simu=False,
                                                 syst=False, verbose=False))
                except:
                    continue
            RCaS_sigma_syst = self.std2(
                N.array(RCaS_syst)[N.isfinite(RCaS_syst)], RCaS_value)

            if N.isfinite(RCaS_sigma_syst):
                RCaS_sigma = float(N.sqrt(RCaS_sigma**2 + RCaS_sigma_syst**2))
            else:
                RCaS_sigma *= 2
            self.RCaSvalues['RCaS.syst'] = float(RCaS_sigma_syst)
            self.RCaSvalues['RCaS.err'] = float(RCaS_sigma)

            return [float(RCaS_value), float(RCaS_sigma)]

        if simu == False and syst == False:

            if N.isfinite(RCaS_value):
                self.RCaSvalues = {'RCaS': float(RCaS_value),
                                   'RCaS_lbd': [min_1, max_1, min_2, max_2]}
            return RCaS_value

    def RCaS2(self, verbose=True, simu=True, syst=True):
        """
        New RCaS where peaks are following
        Return the value and the error of RCaS
        [RCaS,RCaS_sigma]
        """
        interval_1 = 48
        interval_2 = 62.5

        # Initialisation
        self.RCaS2values = {'RCaS2': N.nan, 'RCaS2.err': N.nan,
                            'RCaS2.stat': N.nan,
                            'RCaS2.syst': N.nan,
                            'RCaS2.mean': N.nan,
                            'RCaS2_lbd': [N.nan, N.nan, N.nan, N.nan]}
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

            RCaS2_value = (self._integration(self.x, self.y, min=min_2,
                                             max=max_2, verbose=verbose)) / \
                self._integration(self.x, self.y,
                                  min=min_1,
                                  max=max_1,
                                  verbose=verbose)
        except:
            if verbose:
                print >> sys.stderr, 'ERROR in compute of RCaS2'
            RCaS2_value = float(N.nan)

        if simu:
            if not N.isfinite(RCaS2_value):
                return [float(N.nan), float(N.nan)]

            RCaS2_simu = []
            for simu in self.simulations:
                try:
                    RCaS2_simu.append(simu.RCaS2(simu=False,
                                                 syst=False,
                                                 verbose=False))
                except:
                    continue
            RCaS2_sigma = self.std2(N.array(RCaS2_simu)[N.isfinite(RCaS2_simu)],
                                    RCaS2_value)
            RCaS2_mean = N.mean(N.array(RCaS2_simu)[N.isfinite(RCaS2_simu)])

            if N.isfinite(RCaS2_value):
                self.RCaS2values = {'RCaS2': float(RCaS2_value),
                                    'RCaS2.err': float(RCaS2_sigma),
                                    'RCaS2.stat': float(RCaS2_sigma),
                                    'RCaS2.mean': float(RCaS2_mean),
                                    'RCaS2_lbd': [float(min_1), float(max_1),
                                                  float(min_2), float(max_2)]}

        if syst:
            RCaS2_syst = []
            for system in self.syst:
                try:
                    RCaS2_syst.append(system.RCaS2(simu=False,
                                                   syst=False,
                                                   verbose=False))
                except:
                    continue

            RCaS2_sigma_syst = self.std2(
                N.array(RCaS2_syst)[N.isfinite(RCaS2_syst)], RCaS2_value)

            if N.isfinite(RCaS2_sigma_syst):
                RCaS2_sigma = float(N.sqrt(RCaS2_sigma**2 +
                                           RCaS2_sigma_syst**2))
            else:
                RCaS2_sigma *= 2
            self.RCaS2values['RCaS2.syst'] = float(RCaS2_sigma_syst)
            self.RCaS2values['RCaS2.err'] = float(RCaS2_sigma)

            return [float(RCaS2_value), float(RCaS2_sigma)]

        if simu == False and syst == False:

            if N.isfinite(RCaS2_value):
                self.RCaS2values = {'RCaS2': float(RCaS2_value),
                                    'RCaS2_lbd': [float(min_1),
                                                  float(max_1),
                                                  float(min_2),
                                                  float(max_2)],
                                    'RCaS2.err': N.nan,
                                    'RCaS2.stat': N.nan,
                                    'RCaS2.mean': N.nan}

            return RCaS2_value

    def RCaSbis(self, verbose=True, simu=True):
        """
        Return the value and the error of RCaSbis
        [RCaSbis,RCaSbis_sigma]
        """

        min_1 = 3620
        max_1 = 3716
        min_2 = 3887
        max_2 = 4012

        try:
            RCaS_value = self._integration(self.x, self.y, min=min_2,
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
            RCaS_sigma = N.sqrt(self._var_rapport(a, b, var_a, var_b,
                                                  verbose=verbose))
        except:
            if verbose:
                print >> sys.stderr, 'ERROR in compute of RCaS'
            RCaS_value = float(N.nan)
            RCaS_sigma = float(N.nan)

        return [float(RCaS_value), float(RCaS_sigma)]

    def RCaS2bis(self, verbose=True, simu=True):
        """
        Return the value and the error of RCaS2bis
        [RCaS2bis,RCaS2bis_sigma]
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
            RCaS_value = (self._integration(self.x, self.y, min=min_2,
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
            RCaS_sigma = N.sqrt(self._var_rapport(a, b, var_a, var_b,
                                                  verbose=verbose))
        except:
            if verbose:
                print >> sys.stderr, 'ERROR in compute of RCaS'
            RCaS_value = float(N.nan)
            RCaS_sigma = float(N.nan)

        return [float(RCaS_value), float(RCaS_sigma)]

    def EDCa(self, verbose=True, simu=True, syst=True):
        """
        Return the value and the error of EDCa
        [EDCa,EDCa_sigma]
        """
        self.EDCavalues = {'EDCa': N.nan, 'EDCa.err': N.nan, 'EDCa.stat': N.nan,
                           'EDCa.syst': N.nan, 'EDCa.mean': N.nan,
                           'EDCa_lbd': [N.nan, N.nan]}
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

            EDCa_value = (self._equivalentwidth(self.x, self.y, lbd1=lbd1,
                                                lbd2=lbd2, flux1=flux1,
                                                flux2=flux2,
                                                verbose=verbose)) / \
                (lbd2 - lbd1)
        except:
            if verbose:
                print >> sys.stderr, 'ERROR, none extrema found, try '\
                    'self.find_extrema()'
            EDCa_value = N.nan

        if simu:
            if not N.isfinite(EDCa_value):
                return [float(N.nan), float(N.nan)]

            EDCa_simu = []
            for simu in self.simulations:
                try:
                    EDCa_simu.append(simu.EDCa(simu=False, syst=False,
                                               verbose=False))
                except:
                    continue
            EDCa_sigma = self.std2(N.array(EDCa_simu)[N.isfinite(EDCa_simu)],
                                   EDCa_value)
            EDCa_mean = N.mean(N.array(EDCa_simu)[N.isfinite(EDCa_simu)])

            if N.isfinite(EDCa_value):
                self.EDCavalues = {'EDCa': float(EDCa_value),
                                   'EDCa.err': float(EDCa_sigma),
                                   'EDCa.stat': float(EDCa_sigma),
                                   'EDCa.mean': float(EDCa_mean),
                                   'EDCa_lbd': [lbd1, lbd2]}

        if syst:
            EDCa_syst = []
            for system in self.syst:
                try:
                    EDCa_syst.append(system.EDCa(simu=False,
                                                 syst=False,
                                                 verbose=False))
                except:
                    continue
            EDCa_sigma_syst = self.std2(
                N.array(EDCa_syst)[N.isfinite(EDCa_syst)], EDCa_value)

            if N.isfinite(EDCa_sigma_syst):
                EDCa_sigma = float(N.sqrt(EDCa_sigma**2 + EDCa_sigma_syst**2))
            else:
                EDCa_sigma *= 2
            self.EDCavalues['EDCa.syst'] = float(EDCa_sigma_syst)
            self.EDCavalues['EDCa.err'] = float(EDCa_sigma)

            return [float(EDCa_value), float(EDCa_sigma)]

        if not simu and not syst:
            if N.isfinite(EDCa_value):
                self.EDCavalues = {'EDCa': float(EDCa_value),
                                   'EDCa_lbd': [lbd1, lbd2]}
            return EDCa_value

    def RSi(self, verbose=True, simu=True, syst=True):
        """
        Retun the value and the error of RSi
        [RSi,RSi_sigma]
        """
        # initialisation
        self.RSivalues = {'RSi': N.nan, 'RSi.err': N.nan, 'RSi.stat': N.nan,
                          'RSi.syst': N.nan, 'RSi.mean': N.nan, 'RSi_lbd': N.nan}
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
            except:
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
            except:
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
            except:
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
            RSi_value = d_blue / d_red

        except:
            if verbose:
                print >> sys.stderr, 'ERROR in computing of RSi, no '\
                    'wavelenght to compute RSi or maybe none extrema found, '\
                    'try self.find_extrema()'
            RSi_value = N.nan

        if simu:
            if not N.isfinite(RSi_value):
                return [float(N.nan), float(N.nan)]

            RSi_simu = []
            for simu in self.simulations:
                try:
                    RSi_simu.append(simu.RSi(simu=False, syst=False,
                                             verbose=False))
                except:
                    continue
            RSi_sigma = self.std2(N.array(RSi_simu)[N.isfinite(RSi_simu)],
                                  RSi_value)
            RSi_mean = N.mean(N.array(RSi_simu)[N.isfinite(RSi_simu)])

            if N.isfinite(RSi_value):
                self.RSivalues = {'RSi': float(RSi_value),
                                  'RSi.err': float(RSi_sigma),
                                  'RSi.stat': float(RSi_sigma),
                                  'RSi.mean': float(RSi_mean),
                                  'RSi_lbd': lbd}

        if syst:
            RSi_syst = []
            for system in self.syst:
                try:
                    RSi_syst.append(system.RSi(simu=False, syst=False,
                                               verbose=False))
                except:
                    continue
            RSi_sigma_syst = self.std2(N.array(RSi_syst)[N.isfinite(RSi_syst)],
                                       RSi_value)
            if N.isfinite(RSi_sigma_syst):
                RSi_sigma = float(N.sqrt(RSi_sigma**2 + RSi_sigma_syst**2))
            else:
                RSi_sigma *= 2
            self.RSivalues['RSi.syst'] = float(RSi_sigma_syst)
            self.RSivalues['RSi.err'] = float(RSi_sigma)

            return [float(RSi_value), float(RSi_sigma)]

        if simu == False and syst == False:
            if N.isfinite(RSi_value):
                self.RSivalues = {'RSi': float(RSi_value),
                                  'RSi_lbd': lbd,
                                  'RSi.err': N.nan,
                                  'RSi.stat': N.nan,
                                  'RSi.syst': N.nan,
                                  'RSi.mean': N.nan}

            return RSi_value

    def RSiS(self, verbose=True, simu=True, syst=True):
        """
        Return the value and the error of RSiS
        [RSiS,RSiS_sigma]
        """
        # initialisation
        self.RSiSvalues = {'RSiS': N.nan, 'RSiS.err': N.nan, 'RSiS.stat': N.nan,
                           'RSiS.syst': N.nan, 'RSiS.mean': N.nan,
                           'RSiS_lbd': [N.nan, N.nan], 'RSiS_flux': [N.nan, N.nan]}
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
            RSiS_value = flux1 / flux2
        except:
            if verbose:
                print >> sys.stderr, "ERROR in computing RSiS"
            RSiS_value = N.nan

        if simu:
            if not N.isfinite(RSiS_value):
                return [float(N.nan), float(N.nan)]

            RSiS_simu = []
            for simu in self.simulations:
                try:
                    RSiS_simu.append(simu.RSiS(simu=False, syst=False,
                                               verbose=False))
                except:
                    continue

            RSiS_sigma = self.std2(N.array(RSiS_simu)[N.isfinite(RSiS_simu)],
                                   RSiS_value)
            RSiS_mean = N.mean(N.array(RSiS_simu)[N.isfinite(RSiS_simu)])

            if N.isfinite(RSiS_value):
                self.RSiSvalues = {'RSiS': float(RSiS_value),
                                   'RSiS.err': float(RSiS_sigma),
                                   'RSiS.stat': float(RSiS_sigma),
                                   'RSiS.mean': float(RSiS_mean),
                                   'RSiS_lbd': [float(lbd1), float(lbd2)],
                                   'RSiS_flux': [float(flux1), float(flux2)]}

        if syst:

            RSiS_syst = []
            for system in self.syst:
                try:
                    RSiS_syst.append(system.RSiS(simu=False, syst=False,
                                                 verbose=False))
                except:
                    continue

            RSiS_sigma_syst = self.std2(
                N.array(RSiS_syst)[N.isfinite(RSiS_syst)], RSiS_value)
            if N.isfinite(RSiS_sigma_syst):
                RSiS_sigma = float(N.sqrt(RSiS_sigma**2 + RSiS_sigma_syst**2))
            else:
                RSiS_sigma *= 2
            self.RSiSvalues['RSiS.syst'] = float(RSiS_sigma_syst)
            self.RSiSvalues['RSiS.err'] = float(RSiS_sigma)

            return [float(RSiS_value), float(RSiS_sigma)]

        if simu == False and syst == False:

            if N.isfinite(RSiS_value):
                self.RSiSvalues = {'RSiS': float(RSiS_value),
                                   'RSiS_lbd': [float(lbd1), float(lbd2)],
                                   'RSiS_flux': [float(flux1), float(flux2)],
                                   'RSiS.err': N.nan, 'RSiS.stat': N.nan,
                                   'RSiS.syst': N.nan, 'RSiS.mean': N.nan}
            return RSiS_value

    def RSiSS2(self, verbose=True, simu=True):
        """
        Return the value and the error of RSiSS
        [RSiSS,RSiSS_sigma]
        """
        # initialisation
        self.RSiSSvalues = {'RSiSS': N.nan, 'RSiSS.err': N.nan,
                            'RSiSS.stat': N.nan, 'RSiSS.syst': N.nan,
                            'RSiSS.mean': N.nan,
                            'RSiSS_lbd': [N.nan, N.nan, N.nan, N.nan]}
        if self.init_only:
            return

        min_1 = 5500
        max_1 = 5700
        min_2 = 6200
        max_2 = 6450
        try:
            RSiSS_value = (self._integration(self.x, self.y, min=min_1,
                                             max=max_1, verbose=verbose)) / \
                (self._integration(self.x, self.y,
                                   min=min_2,
                                   max=max_2,
                                   verbose=verbose))
        except:
            if verbose:
                print >> sys.stderr, 'ERROR in RSiSS computing'
            RSiSS_value = float(N.nan)

        if simu:
            if not N.isfinite(RSiSS_value):
                return [float(N.nan), float(N.nan)]

            RSiSS_simu = []
            for simu in self.simulations:
                try:
                    RSiSS_simu.append(simu.RSiSS(simu=False, verbose=False))
                except:
                    continue

            RSiSS_sigma = self.std2(RSiSS_simu, RSiSS_value)
            RSiSS_mean = N.mean(RSiSS_simu)

            if N.isfinite(RSiSS_value):
                self.RSiSSvalues = {'RSiSS': float(RSiSS_value),
                                    'RSiSS.err': float(RSiSS_sigma),
                                    'RSiSS.stat': float(RSiSS_sigma),
                                    'RSiSS.mean': float(RSiSS_mean),
                                    'RSiSS_lbd': [float(min_1),
                                                  float(max_1),
                                                  float(min_2),
                                                  float(max_2)]}

            return [float(RSiSS_value), float(RSiSS_sigma)]

        else:
            if N.isfinite(RSiSS_value):
                self.RSiSSvalues = {'RSiSS': float(RSiSS_value),
                                    'RSiSS_lbd': [float(min_1),
                                                  float(max_1),
                                                  float(min_2),
                                                  float(max_2)],
                                    'RSiSS.err': N.nan, 'RSiSS.stat': N.nan,
                                    'RSiSS.mean': N.nan, 'RSiSS.syst': N.nan}

            return RSiSS_value

    def RSiSS(self, verbose=True, simu=True):
        """
        Return the value and the error of RSiSS
        [RSiSS,RSiSS_sigma]
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
            RSiSS_value = a / b
            RSiSS_sigma = N.sqrt(self._var_rapport(a, b, var_a, var_b,
                                                   verbose=verbose))
        except:
            if verbose:
                print >> sys.stderr, 'ERROR in compute of RSiSS'
            RSiSS_value = float(N.nan)
            RSiSS_sigma = float(N.nan)
        self.RSiSSvalues = {'RSiSS': float(RSiSS_value),
                            'RSiSS.err': float(RSiSS_sigma),
                            'RSiSS_lbd': [float(min_1),
                                          float(max_1),
                                          float(min_2),
                                          float(max_2)]}

        return [float(RSiSS_value), float(RSiSS_sigma)]

    def EW(self, lambda_min_blue, lambda_max_blue, lambda_min_red,
           lambda_max_red, sf, verbose=True, simu=True,
           right1=False, left1=False, right2=False, left2=False,
           sup=False, syst=True, check=True):
        """
        Return the value and the error of an Equivalent Width
        [lambda_min_blue,lambda_max_blue] and [lambda_min_red,lambda_max_red]
        are the interval where the two peaks are searching.
        'sf' is the name (a string) of the spectral feature associated to the EW
        [EW,EW_sigma]
        """
        # shoftcut
        EWV = self.EWvalues

        # Initialisation
        EWV['EW%s' % sf] = N.nan
        EWV['lbd_EW%s' % sf] = [N.nan, N.nan]
        EWV['flux_EW%s' % sf] = [N.nan, N.nan]
        EWV['R%s' % sf] = N.nan
        EWV['flux_sum_norm_EW%s' % sf] = N.nan
        EWV['depth_norm_EW%s' % sf] = N.nan
        EWV['depth_EW%s' % sf] = N.nan
        EWV['depth_EW%s.err' % sf] = N.nan
        EWV['depth_EW%s.stat' % sf] = N.nan
        EWV['depth_EW%s.syst' % sf] = N.nan
        EWV['depth_EW%s.mean' % sf] = N.nan
        EWV['surf_norm_EW%s' % sf] = N.nan
        EWV['surf_EW%s.err' % sf] = N.nan
        EWV['surf_EW%s.stat' % sf] = N.nan
        EWV['surf_EW%s.syst' % sf] = N.nan
        EWV['surf_EW%s.mean' % sf] = N.nan
        EWV['width_EW%s' % sf] = N.nan
        EWV['EW%s.err' % sf] = N.nan
        EWV['EW%s.stat' % sf] = N.nan
        EWV['EW%s.syst' % sf] = N.nan
        EWV['flux_sum_norm_EW%s.err' % sf] = N.nan
        EWV['flux_sum_norm_EW%s.stat' % sf] = N.nan
        EWV['flux_sum_norm_EW%s.syst' % sf] = N.nan
        EWV['depth_norm_EW%s.err' % sf] = N.nan
        EWV['depth_norm_EW%s.stat' % sf] = N.nan
        EWV['depth_norm_EW%s.syst' % sf] = N.nan
        EWV['width_EW%s.err' % sf] = N.nan
        EWV['width_EW%s.stat' % sf] = N.nan
        EWV['width_EW%s.syst' % sf] = N.nan
        EWV['EW%s.mean' % sf] = N.nan
        EWV['R%s.mean' % sf] = N.nan
        EWV['R%s.stat' % sf] = N.nan
        EWV['R%s.syst' % sf] = N.nan
        EWV['R%s.err' % sf] = N.nan
        EWV['flux_sum_norm_EW%s.mean' % sf] = N.nan
        EWV['depth_norm_EW%s.mean' % sf] = N.nan
        EWV['width_EW%s.mean' % sf] = N.nan
        EWV['EW%s.med' % sf] = N.nan
        EWV['fmean_EW%s' % sf] = N.nan
        EWV['fmean_EW%s.err' % sf] = N.nan
        EWV['fmean_EW%s.stat' % sf] = N.nan
        EWV['fmean_EW%s.syst' % sf] = N.nan
        EWV['fmean_EW%s.mean' % sf] = N.nan

        if self.init_only:
            return

        # Function to compute the EW value and find its parameters ============
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
                except:
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
                except:
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
            #if sup and len(x[x<0]) > 5: EW_value = N.nan
            # else: EW_value = self._equivalentwidth(self.x,self.y,
            # lbd1=lbd1,lbd2=lbd2,flux1=flux1,flux2=flux2,verbose=verbose)
            EW_value = self._equivalentwidth(self.x,
                                             self.y,
                                             lbd1=lbd1,
                                             lbd2=lbd2,
                                             flux1=flux1,
                                             flux2=flux2,
                                             verbose=verbose)
        except:
            if verbose:
                print >> sys.stderr, 'ERROR, no extrema found, '\
                    'try self.find_extrema()'
            EW_value = N.nan
        #======================================================================

        if N.isfinite(EW_value):  # Additional informations
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

            EWV['EW%s' % sf] = float(EW_value)
            EWV['lbd_EW%s' % sf] = [float(lbd1), float(lbd2)]
            EWV['flux_EW%s' % sf] = [float(flux1), float(flux2)]
            EWV['R%s' % sf] = float(flux2 / flux1)
            EWV['flux_sum_norm_EW%s' % sf] = float(flux_norm)
            EWV['depth_norm_EW%s' % sf] = float(depth_n)
            EWV['width_EW%s' % sf] = float(lbd2 - lbd1)
            EWV['depth_EW%s' % sf] = float(depth)
            EWV['surf_EW%s' % sf] = float(surf)
            EWV['fmean_EW%s' % sf] = float(fmean)

        # Compute statistiaue error
        if simu:
            if not N.isfinite(EW_value):
                return [float(N.nan), float(N.nan)]

            EW_simu, R_simu, d_simu, w_simu, f_simu, fm_simu = [
            ], [], [], [], [], []
            dep_simu, sur_simu = [], []
            for simu in self.simulations:
                try:
                    EW_simu.append(simu.EW(lambda_min_blue,
                                           lambda_max_blue,
                                           lambda_min_red,
                                           lambda_max_red,
                                           sf,
                                           simu=False,
                                           syst=False,
                                           verbose=False))
                    R_simu.append(float(simu.EWvalues['R%s' % sf]))
                    f_simu.append(
                        float(simu.EWvalues['flux_sum_norm_EW%s' % sf]))
                    d_simu.append(float(simu.EWvalues['depth_norm_EW%s' % sf]))
                    w_simu.append(float(simu.EWvalues['width_EW%s' % sf]))
                    dep_simu.append(float(simu.EWvalues['depth_EW%s' % sf]))
                    sur_simu.append(float(simu.EWvalues['surf_EW%s' % sf]))
                    fm_simu.append(float(simu.EWvalues['fmean_EW%s' % sf]))
                except:
                    continue
            EW_sigma = self.std2(
                N.array(EW_simu)[N.isfinite(EW_simu)], EW_value)
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

            EW_mean = N.mean(N.array(EW_simu)[N.isfinite(EW_simu)])
            R_mean = N.mean(N.array(R_simu)[N.isfinite(R_simu)])
            f_mean = N.mean(N.array(f_simu)[N.isfinite(f_simu)])
            d_mean = N.mean(N.array(d_simu)[N.isfinite(d_simu)])
            w_mean = N.mean(N.array(w_simu)[N.isfinite(w_simu)])
            dep_mean = N.mean(N.array(dep_simu)[N.isfinite(dep_simu)])
            sur_mean = N.mean(N.array(sur_simu)[N.isfinite(sur_simu)])
            fmean_mean = N.mean(N.array(fm_simu)[N.isfinite(fm_simu)])

            EW_med = N.median(N.array(EW_simu)[N.isfinite(EW_simu)])

            EWV['EW%s.err' % sf] = float(EW_sigma)
            EWV['EW%s.stat' % sf] = float(EW_sigma)
            EWV['R%s.err' % sf] = float(R_sigma)
            EWV['R%s.stat' % sf] = float(R_sigma)
            EWV['flux_sum_norm_EW%s.err' % sf] = float(f_sigma)
            EWV['flux_sum_norm_EW%s.stat' % sf] = float(f_sigma)
            EWV['depth_norm_EW%s.err' % sf] = float(d_sigma)
            EWV['depth_norm_EW%s.stat' % sf] = float(d_sigma)
            EWV['width_EW%s.err' % sf] = float(w_sigma)
            EWV['width_EW%s.stat' % sf] = float(w_sigma)
            EWV['depth_EW%s.err' % sf] = float(dep_sigma)
            EWV['depth_EW%s.stat' % sf] = float(dep_sigma)
            EWV['surf_EW%s.err' % sf] = float(sur_sigma)
            EWV['surf_EW%s.stat' % sf] = float(sur_sigma)
            EWV['fmean_EW%s.err' % sf] = float(fmean_sigma)
            EWV['fmean_EW%s.stat' % sf] = float(fmean_sigma)

            EWV['EW%s.mean' % sf] = float(EW_mean)
            EWV['R%s.mean' % sf] = float(R_mean)
            EWV['flux_sum_norm_EW%s.mean' % sf] = float(f_mean)
            EWV['depth_norm_EW%s.mean' % sf] = float(d_mean)
            EWV['width_EW%s.mean' % sf] = float(w_mean)
            EWV['depth_EW%s.mean' % sf] = float(dep_mean)
            EWV['surf_EW%s.mean' % sf] = float(sur_mean)
            EWV['fmean_EW%s.mean' % sf] = float(fmean_mean)

            EWV['EW%s.med' % sf] = float(EW_med)

        # Compute systematic error
        if syst:
            if not N.isfinite(EW_value):
                return [float(N.nan), float(N.nan)]

            EW_syst, R_syst, d_syst, w_syst, f_syst, fm_syst = [
            ], [], [], [], [], []
            dep_syst, sur_syst = [], []
            for system in self.syst:
                try:
                    EW_syst.append(system.EW(lambda_min_blue,
                                             lambda_max_blue,
                                             lambda_min_red,
                                             lambda_max_red,
                                             sf,
                                             simu=False,
                                             syst=False,
                                             verbose=False))
                    R_syst.append(float(system.EWvalues['R%s' % sf]))
                    f_syst.append(float(system.EWvalues['flux_sum_norm_EW%s' %
                                                        sf]))
                    d_syst.append(
                        float(system.EWvalues['depth_norm_EW%s' % sf]))
                    w_syst.append(float(system.EWvalues['width_EW%s' % sf]))
                    dep_syst.append(float(system.EWvalues['depth_EW%s' % sf]))
                    sur_syst.append(float(system.EWvalues['surf_EW%s' % sf]))
                    fm_syst.append(float(system.EWvalues['fmean_EW%s' % sf]))
                except:
                    continue

            EW_sigma_syst = self.std2(N.array(EW_syst)[N.isfinite(EW_syst)],
                                      EW_value)
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

            if not N.isfinite(EW_sigma_syst):
                EW_sigma_syst = float(0.0)
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

            EW_sigma = N.sqrt(EW_sigma**2 + EW_sigma_syst**2)
            EWV['EW%s.syst' % sf] = float(EW_sigma_syst)
            EWV['EW%s.err' % sf] = float(N.sqrt(EWV['EW%s.err' % sf]**2 +
                                                EW_sigma_syst**2))
            EWV['R%s.syst' % sf] = float(R_sigma_syst)
            EWV['R%s.err' % sf] = float(N.sqrt(EWV['R%s.err' % sf]**2 +
                                               R_sigma_syst**2))
            EWV['flux_sum_norm_EW%s.syst' % sf] = float(f_sigma_syst)
            EWV['flux_sum_norm_EW%s.err' % sf] = float(
                N.sqrt(EWV['flux_sum_norm_EW%s.err' % sf]**2 + f_sigma_syst**2))
            EWV['depth_norm_EW%s.syst' % sf] = float(d_sigma_syst)
            EWV['depth_norm_EW%s.err' % sf] = float(
                N.sqrt(EWV['depth_norm_EW%s.err' % sf]**2 + d_sigma_syst**2))
            EWV['width_EW%s.syst' % sf] = float(w_sigma_syst)
            EWV['width_EW%s.err' % sf] = float(N.sqrt(EWV['width_EW%s.err' % sf]**2
                                                      + w_sigma_syst**2))
            EWV['depth_EW%s.syst' % sf] = float(dep_sigma_syst)
            EWV['depth_EW%s.err' % sf] = float(N.sqrt(EWV['depth_EW%s.err' % sf]**2
                                                      + dep_sigma_syst**2))
            EWV['surf_EW%s.syst' % sf] = float(sur_sigma_syst)
            EWV['surf_EW%s.err' % sf] = float(N.sqrt(EWV['surf_EW%s.err' % sf]**2 +
                                                     sur_sigma_syst**2))
            EWV['fmean_EW%s.syst' % sf] = float(fm_sigma_syst)
            EWV['fmean_EW%s.err' % sf] = float(N.sqrt(EWV['fmean_EW%s.err' % sf]**2
                                                      + fm_sigma_syst**2))

            return [float(EW_value), float(EW_sigma)]

        if simu == False and syst == False:
            return EW_value

    def velocity(self, infodict, verbose=False, simu=True, syst=True,
                 left=False, right=False):
        """ returns the value and the error of a velocity of an absorption
        feature
        infodict should have the following structure :
        {'lmin' : minimum lambda for searching the dip,
        'lmax' : max lambda for searching the dip,
        'lrest' : restframe wavelangth of absorption feature
        'name' : name of the feature}
        the error will be coded as ['name']+'.err'
        and the lambda as ['name']+'_lbd'"""
        # shortcut
        VV = self.velocityValues

        c = 299792.458
        # Initialisation
        VV[infodict['name']] = N.nan
        VV[infodict['name'] + '.err'] = N.nan
        VV[infodict['name'] + '.stat'] = N.nan
        VV[infodict['name'] + '.syst'] = N.nan
        VV[infodict['name'] + '_lbd'] = N.nan
        VV[infodict['name'] + '_lbd.stat'] = N.nan
        VV[infodict['name'] + '_lbd.syst'] = N.nan
        VV[infodict['name'] + '_lbd.err'] = N.nan
        VV[infodict['name'] + '_lbd.mean'] = N.nan
        VV[infodict['name'] + '_flux'] = N.nan
        VV[infodict['name'] + '_flux.stat'] = N.nan
        VV[infodict['name'] + '_flux.syst'] = N.nan
        VV[infodict['name'] + '_flux.err'] = N.nan
        VV[infodict['name'] + '_flux.mean'] = N.nan
        VV[infodict['name'] + '.binsyst'] = N.nan
        VV[infodict['name'] + '.bin'] = N.nan
        VV[infodict['name'] + '.mean'] = N.nan
        VV[infodict['name'] + '.med'] = N.nan
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
        except:
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
            VV[infodict['name']] = float(velocity)
            VV[infodict['name'] + '_lbd'] = float(lbd)
            VV[infodict['name'] + '_flux'] = float(flux)

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
                    lbd_simu.append(simul.velocityValues[infodict['name'] +
                                                         '_lbd'])
                    flux_simu.append(simul.velocityValues[infodict['name'] +
                                                          '_flux'])
                except:
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

            VV[infodict['name'] + '.err'] = float(velocity_sigma)
            VV[infodict['name'] + '.stat'] = float(velocity_sigma)
            VV[infodict['name'] + '.mean'] = float(velocity_mean)
            VV[infodict['name'] + '.med'] = float(velocity_med)
            VV[infodict['name'] + '_lbd.err'] = float(lbd_sigma)
            VV[infodict['name'] + '_lbd.stat'] = float(lbd_sigma)
            VV[infodict['name'] + '_lbd.mean'] = float(lbd_mean)
            VV[infodict['name'] + '_lbd.med'] = float(lbd_med)
            VV[infodict['name'] + '_flux.err'] = float(flux_sigma)
            VV[infodict['name'] + '_flux.stat'] = float(flux_sigma)
            VV[infodict['name'] + '_flux.mean'] = float(flux_mean)
            VV[infodict['name'] + '_flux.med'] = float(flux_med)

        if syst:
            velocity_syst, lbd_syst, flux_syst = [], [], []
            for system in self.syst:
                try:
                    velocity_syst.append(system.velocity(infodict,
                                                         simu=False,
                                                         syst=False,
                                                         verbose=False))
                    lbd_syst.append(system.velocityValues[infodict['name'] +
                                                          '_lbd'])
                    flux_syst.append(system.velocityValues[infodict['name'] +
                                                           '_flux'])
                except:
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

            VV[infodict['name'] + '.syst'] = float(velocity_syst_sigma)
            VV[infodict['name'] + '.err'] = float(velocity_sigma)
            VV[infodict['name'] + '_lbd.syst'] = float(lbd_syst_sigma)
            VV[infodict['name'] + '_lbd.err'] = float(lbd_sigma)
            VV[infodict['name'] + '_flux.syst'] = float(flux_syst_sigma)
            VV[infodict['name'] + '_flux.err'] = float(flux_sigma)
            VV[infodict['name'] + '.binsyst'] = float(velocity_syst_bin)
            VV[infodict['name'] + '.bin'] = float(binning)

        if velocity_sigma is None:
            return float(velocity)
        else:
            return [float(velocity), float(velocity_sigma)]

    def velocity2(self, infodict, verbose=True, simu=True, syst=True,
                  left=False, right=False):
        """
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
        VV = self.velocityValues

        # Initialisation
        VV[infodict['name'] + '.err'] = N.nan
        VV[infodict['name'] + '.stat'] = N.nan
        VV[infodict['name'] + '.syst'] = N.nan
        VV[infodict['name'] + '_lbd'] = N.nan
        VV[infodict['name'] + '_lbd.stat'] = N.nan
        VV[infodict['name'] + '_lbd.syst'] = N.nan
        VV[infodict['name'] + '_lbd.err'] = N.nan
        VV[infodict['name'] + '_lbd.mean'] = N.nan
        VV[infodict['name'] + '.binsyst'] = N.nan
        VV[infodict['name'] + '.bin'] = N.nan
        VV[infodict['name'] + '.mean'] = N.nan
        if self.init_only:
            return

        c = 299792.458
        print 'try new velocity method'
        lbdm, fluxm, varm = self._extrema_value_in_interval(infodict['lmin'],
                                                            infodict['lmax'], self.maxima[
                                                                'x'], self.maxima['v'],
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
            VV[infodict['name']] = float(velocity)
            VV[infodict['name'] + '_lbd'] = float(lbd)

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
                    lbd_simu.append(simul.velocityValues[infodict['name'] +
                                                         '_lbd'])
                except:
                    continue

            velocity_sigma = self.std2(
                N.array(velocity_simu)[N.isfinite(velocity_simu)], velocity)
            lbd_sigma = self.std2(N.array(lbd_simu)[N.isfinite(lbd_simu)], lbd)
            velocity_mean = N.mean(
                N.array(velocity_simu)[N.isfinite(velocity_simu)])
            lbd_mean = N.mean(N.array(lbd_simu)[N.isfinite(lbd_simu)])

            VV[infodict['name'] + '.err'] = float(velocity_sigma)
            VV[infodict['name'] + '.stat'] = float(velocity_sigma)
            VV[infodict['name'] + '_lbd.err'] = float(lbd_sigma)
            VV[infodict['name'] + '_lbd.stat'] = float(lbd_sigma)
            VV[infodict['name'] + '_lbd.mean'] = float(lbd_mean)
            VV[infodict['name'] + '.mean'] = float(velocity_mean)

        if syst:
            velocity_syst, lbd_syst = [], []
            for system in self.syst:
                try:
                    velocity_syst.append(system.velocity(infodict, simu=False,
                                                         syst=False,
                                                         verbose=False))
                    lbd_syst.append(system.velocityValues[infodict['name'] +
                                                          '_lbd'])
                except:
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

            VV[infodict['name'] + '.syst'] = float(velocity_syst_sigma)
            VV[infodict['name'] + '.err'] = float(velocity_sigma)
            VV[infodict['name'] + '_lbd.syst'] = float(lbd_syst_sigma)
            VV[infodict['name'] + '_lbd.err'] = float(lbd_sigma)
            VV[infodict['name'] + '.binsyst'] = float(velocity_syst_bin)
            VV[infodict['name'] + '.bin'] = float(binning)

        if velocity_sigma is None:
            return float(velocity)
        else:
            return [float(velocity), float(velocity_sigma)]


#=========================================================================
# Class to use the craniometer class, and to plot control plots
#=========================================================================
class DrGall:

    """
    Class to manipulate and use the craniometer
    """

    def __init__(self, spec=None, specB=None, specR=None,
                 spec_merged=None, verbose=True):
        """
        Spectrum initialization. 
        Create self.x,self.y,self.v if spec is the merged spectrum
        Create self.xB,self.yB,self.vB for the blue channel if specB is given
        Create self.xR,self.yR,self.vR for the blue channel if specR is given
        """
        self.values_initialization()

        self.xB = None
        self.xR = None
        self.x_merged = None

        if spec is not None:
            if spec.x[0] < 4000 and spec.x[-1] > 6500:
                self.x = N.array(spec.x)
                self.y = N.array(spec.y)
                self.v = N.array(spec.v)
                self.xB = N.array(spec.x)
                self.yB = N.array(spec.y)
                self.vB = N.array(spec.v)
                self.xR = N.array(spec.x)
                self.yR = N.array(spec.y)
                self.vR = N.array(spec.v)
                self.x_merged = N.array(spec.x)
                self.y_merged = N.array(spec.y)
                self.v_merged = N.array(spec.v)
            elif spec.x[0] < 4000 and spec.x[-1] < 6500:
                self.xB = N.array(spec.x)
                self.yB = N.array(spec.y)
                self.vB = N.array(spec.v)
            elif spec.x[0] > 4000 and spec.x[-1] > 6500:
                self.xR = N.array(spec.x)
                self.yR = N.array(spec.y)
                self.vR = N.array(spec.v)
            if verbose:
                print >> sys.stderr, 'Work on merged spectrum'

        elif specB or specR:
            if (specB and specB.x[0] > 4000) or (specR and specR.x[0] < 4000):
                print  >> sys.stderr, 'ERROR, check if B channel is really B '\
                    'channel and not R channel'
                return
            try:
                self.xB = N.array(specB.x)
                self.yB = N.array(specB.y)
                self.vB = N.array(specB.v)
            except:
                pass
            try:
                self.xR = N.array(specR.x)
                self.yR = N.array(specR.y)
                self.vR = N.array(specR.v)
            except:
                pass

            if self.xB is not None and self.xR is not None:
                try:
                    import merged_spectrum
                    spec_merged = merged_spectrum.MergedSpectrum(specB, specR)
                    del merged_spectrum
                    self.x_merged = N.array(spec_merged.x)
                    self.y_merged = N.array(spec_merged.y)
                    self.v_merged = N.array(spec_merged.v)
                except:
                    print >> sys.stderr, 'Merged spectrum failure'

            if verbose:
                if self.xB is not None and self.xR is not None:
                    print >> sys.stderr, 'Work on B and R channel'
                elif self.xB is not None and self.xR is None:
                    print >> sys.stderr, 'Work only on B channel'
                elif self.xB is None and self.xR is not None:
                    print >> sys.stderr, 'Work only on R channel'
                elif self.xB is None \
                        and self.xR is None \
                        and not hasattr(self, 'x'):
                    print >> sys.stderr, 'Work on merged spectrum'
                else:
                    print >> sys.stderr, 'ERROR, no correct input in DrGall. '\
                        'Give me a spectrum (for instance; spec with '\
                        'spec.x,spec.y and spec.v)'
                    sys.exit()

        else:
            print >> sys.stderr, 'ERROR, no correct input in DrGall. Give me a'\
                'spectrum (for instance; spec with spec.x,spec.y and spec.v)'
            sys.exit()

    def values_initialization(self, verbose=False):

        Values = {}
        # Initialisation craniomter
        fake_lbd = range(3000, 1000, 2)
        cranio = Craniometer(fake_lbd,
                             N.zeros(len(fake_lbd)),
                             N.zeros(len(fake_lbd)))
        cranio.init_only = True

        # Create values
        cranio.RCa(verbose=verbose)
        cranio.RCaS(verbose=verbose)
        cranio.RCaS2(verbose=verbose)
        cranio.RSi(verbose=verbose)
        cranio.RSiS(verbose=verbose)
        cranio.RSiSS(verbose=verbose)
        cranio.EW(3504, 3687, 3887, 3990, 'CaIIHK', verbose=verbose)
        cranio.EW(3830, 3963, 4034, 4150, 'SiII4000', verbose=verbose)
        cranio.EW(4034, 4150, 4452, 4573, 'MgII', verbose=verbose)
        cranio.EW(5085, 5250, 5500, 5681, 'SIIW', verbose=verbose)
        cranio.EW(5085, 5250, 5250, 5450, 'SIIW_L', verbose=verbose)
        cranio.EW(5250, 5450, 5500, 5681, 'SIIW_R', verbose=verbose)
        cranio.EW(5550, 5681, 5850, 6015, 'SiII5972', verbose=verbose)
        cranio.EW(5850, 6015, 6250, 6365, 'SiII6355', verbose=verbose)
        cranio.EW(7100, 7270, 7720, 8000, 'OI7773', verbose=verbose)
        cranio.EW(7720, 8000, 8300, 8800, 'CaIIIR', verbose=verbose)
        cranio.EW(4400, 4650, 5050, 5300, 'Fe4800', verbose=verbose)
        cranio.velocity({'lmin': 3963,
                         'lmax': 4034,
                         'lrest': 4128,
                         'name': 'vSiII_4128'},
                        verbose=verbose)
        cranio.velocity({'lmin': 5200,
                         'lmax': 5350,
                         'lrest': 5454,
                         'name': 'vSiII_5454'},
                        verbose=verbose)
        cranio.velocity({'lmin': 5351,
                         'lmax': 5550,
                         'lrest': 5640,
                         'name': 'vSiII_5640'},
                        verbose=verbose)
        cranio.velocity({'lmin': 5700,
                         'lmax': 5900,
                         'lrest': 5972,
                         'name': 'vSiII_5972'},
                        verbose=verbose)
        cranio.velocity({'lmin': 6000,
                         'lmax': 6210,
                         'lrest': 6355,
                         'name': 'vSiII_6355'},
                        verbose=verbose)

        # Update values
        Values.update(cranio.RCavalues)
        Values.update(cranio.RCaSvalues)
        Values.update(cranio.RCaS2values)
        Values.update(cranio.RSivalues)
        Values.update(cranio.RSiSvalues)
        Values.update(cranio.RSiSSvalues)
        Values.update(cranio.velocityValues)
        Values.update(cranio.EWvalues)

        self.Values = Values

    def calcium_computing(self, factor=1.05, rhoB=0.479, nsimu=1000,
                          smoother="sgfilter", sBCa=None, sBSi=None,
                          sBMg=None, wBCa=None, wBSi=None, wBMg=None,
                          verbose=False):
        """
        Function to compute and return all spectral indicators in the calcium
        zone (Blue part of the spectrum, B channel)
        """
        # Test if computing is possible
        if self.xB is None:
            print >> sys.stderr, 'ERROR, impossible to compute spectral '\
                'indictors defined in calcium zone (maybe no B channel)'
            indicators = {'EDCa': [N.nan, N.nan],
                          'RCa': [N.nan, N.nan],
                          'RCaS': [N.nan, N.nan],
                          'RCaS2': [N.nan, N.nan],
                          'EWCaIIHK': [N.nan, N.nan],
                          'EWSiII4000': [N.nan, N.nan],
                          'EWMgII': [N.nan, N.nan]}
            return indicators

        # Create zone and craniometers
        Cazone = (self.xB > 3450) & (self.xB < 4070)
        Sizone = (self.xB > 3850) & (self.xB < 4150)
        Mgzone = (self.xB > 4000) & (self.xB < 4610)

        self.cranio_BCa = get_cranio(self.xB[Cazone],
                                     self.yB[Cazone],
                                     self.vB[Cazone],
                                     smoother=smoother,
                                     verbose=verbose)
        self.cranio_BSi = get_cranio(self.xB[Sizone],
                                     self.yB[Sizone],
                                     self.vB[Sizone],
                                     smoother=smoother,
                                     verbose=verbose)
        self.cranio_BMg = get_cranio(self.xB[Mgzone],
                                     self.yB[Mgzone],
                                     self.vB[Mgzone],
                                     smoother=smoother,
                                     verbose=verbose)

        try:
            RCa = self.cranio_BCa.RCa(verbose=verbose)
            self.Values.update(self.cranio_BCa.RCavalues)
            if verbose:
                print 'RCa computing done, RCa =', RCa
        except:
            RCa = [N.nan, N.nan]
            if verbose:
                print 'ERROR in RCa computing, RCa =', RCa

        try:
            RCaS = self.cranio_BCa.RCaS(verbose=verbose)
            self.Values.update(self.cranio_BCa.RCaSvalues)
            if verbose:
                print 'RCaS computing done, RCaS =', RCaS
        except:
            RCaS = [N.nan, N.nan]
            if verbose:
                print 'ERROR in RCaS computing, RCaS =', RCaS

        try:
            RCaS2 = self.cranio_BCa.RCaS2(verbose=verbose)
            self.Values.update(self.cranio_BCa.RCaS2values)
            if verbose:
                print 'RCaS2 computing done, RCaS2 =', RCaS2
        except:
            RCaS2 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in RCaS2 computing, RCaS2 =', RCaS2

        try:
            EWCaIIHK = self.cranio_BCa.EW(3504, 3687, 3830, 3990,
                                          'CaIIHK',
                                          sup=True,
                                          right1=True,
                                          verbose=verbose)
            self.Values.update(self.cranio_BCa.EWvalues)
            if verbose:
                print 'EWCaIIHK computing done, EWCaIIHK =', EWCaIIHK
        except:
            EWCaIIHK = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWCaIIHK computing, EWCaIIHK =', EWCaIIHK

        try:
            EWSiII4000 = self.cranio_BSi.EW(3830, 3990, 4030, 4150,
                                            'SiII4000',
                                            sup=True,
                                            verbose=verbose)
            self.Values.update(self.cranio_BSi.EWvalues)
            if verbose:
                print 'EWSiII4000 computing done, EWSiII4000 =', EWSiII4000
        except:
            EWSiII4000 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWSiII4000 computing EWSiII4000 =', EWSiII4000

        try:
            EWMgII = self.cranio_BMg.EW(4030, 4150, 4450, 4650,
                                        'MgII',
                                        sup=True,
                                        left2=True,
                                        verbose=verbose)
            self.Values.update(self.cranio_BMg.EWvalues)
            if verbose:
                print 'EWMgII computing done, EWMgII = ', EWMgII
        except:
            EWMgII = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWMgII computing, EWMgII =', EWMgII

        try:
            vSiII_4000 = self.cranio_BSi.velocity({'lmin': 3963,
                                                   'lmax': 4034,
                                                   'lrest': 4128,
                                                   'name': 'vSiII_4128'},
                                                  verbose=verbose)
            self.Values.update(self.cranio_BSi.velocityValues)
            if verbose:
                print 'vSiII_4128 computing done, vSiII_4000 =', vSiII_4000
        except:
            vSiII_4000 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in vSiII_4128 computing, vSiII_4000', vSiII_4000

        if verbose:
            print >> sys.stderr, 'Computing on calcium zone for this '\
                'spectrum done\n'

        indicators = {'RCa': RCa,
                      'RCaS2': RCaS2,
                      'EWCaIIHK': EWCaIIHK,
                      'EWSiII4000': EWSiII4000,
                      'EWMgII': EWMgII,
                      'vSiII4128': vSiII_4000}

        del self.cranio_BCa.simulations
        del self.cranio_BCa.syst
        del self.cranio_BSi.simulations
        del self.cranio_BSi.syst
        del self.cranio_BMg.simulations
        del self.cranio_BMg.syst

        return indicators

    def silicon_computing(self, factor=1.23, rhoR=0.484, nsimu=1000,
                          smoother="sgfilter", sR1=None, sR2=None, sR3=None,
                          sR4=None, wR1=None, wR2=None, wR3=None, wR4=None,
                          verbose=False):
        """
        Function to compute and retunr all spectral indicators in the silicon
        zone
        """
        # Test if computing is possible
        if self.xR is None:
            print >> sys.stderr, 'ERROR, impossible to compute spectral '\
                'indictors defined in calcium zone (maybe no R channel)'
            indicators = {'EDCa': [N.nan, N.nan],
                          'RCa': [N.nan, N.nan],
                          'RCaS': [N.nan, N.nan],
                          'RCaS2': [N.nan, N.nan],
                          'EWCaIIHK': [N.nan, N.nan],
                          'EWSiII4000': [N.nan, N.nan],
                          'EWMgII': [N.nan, N.nan],
                          'vSiII_5972': [N.nan, N.nan],
                          'vSiII_6355': [N.nan, N.nan]}
            return indicators

        # Create zone and craniometers
        zone1 = (self.xR > 5500) & (self.xR < 6400)
        zone2 = (self.xR > 5060) & (self.xR < 5700)
        zone3 = (self.xR > 5500) & (self.xR < 6050)
        zone4 = (self.xR > 5800) & (self.xR < 6400)
        zone5 = (self.xR > 5480) & (self.xR < 6500)
        self.cranio_R1 = get_cranio(self.xR[zone1],
                                    self.yR[zone1],
                                    self.vR[zone1],
                                    smoother=smoother,
                                    verbose=verbose)  # RSi,RSiS
        self.cranio_R2 = get_cranio(self.xR[zone2],
                                    self.yR[zone2],
                                    self.vR[zone2],
                                    smoother=smoother,
                                    verbose=verbose)  # EWSIIW
        self.cranio_R3 = get_cranio(self.xR[zone3],
                                    self.yR[zone3],
                                    self.vR[zone3],
                                    smoother=smoother,
                                    verbose=verbose)  # EWSiII5972
        self.cranio_R4 = get_cranio(self.xR[zone4],
                                    self.yR[zone4],
                                    self.vR[zone4],
                                    smoother=smoother,
                                    verbose=verbose)  # EWSiII6355
        self.cranio_R5 = get_cranio(self.xR[zone5],
                                    self.yR[zone5],
                                    self.vR[zone5],
                                    smoother=smoother,
                                    verbose=verbose)  # RSiSS

        try:
            RSi = self.cranio_R1.RSi(verbose=verbose)
            self.Values.update(self.cranio_R1.RSivalues)
            if verbose:
                print 'RSi computing done, RSi =', RSi
        except:
            RSi = [N.nan, N.nan]
            if verbose:
                print 'ERROR in RSi computing, RSi =', RSi

        try:
            RSiS = self.cranio_R1.RSiS(verbose=verbose)
            self.Values.update(self.cranio_R1.RSiSvalues)
            if verbose:
                print 'RSiS computing done, RSiS =', RSiS
        except:
            RSiS = [N.nan, N.nan]
            if verbose:
                print 'ERROR in RSiS computing, RSiS =', RSiS

        try:
            RSiSS = self.cranio_R5.RSiSS(verbose=verbose)
            self.Values.update(self.cranio_R5.RSiSSvalues)
            if verbose:
                print 'RSiSS computing done, RSiSS =', RSiSS
        except:
            RSiSS = [N.nan, N.nan]
            if verbose:
                print 'ERROR in RSiSS computing, RSiSS =', RSiSS

        try:
            EWSIIW = self.cranio_R2.EW(5050, 5285, 5500, 5681,
                                       'SIIW',
                                       sup=True,
                                       # right1=True,
                                       verbose=verbose)
            if verbose:
                print 'EWSIIW computing done, EWSIIW =', EWSIIW
        except:
            EWSIIW = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWSIIW computing, EWSIIW =', EWSIIW

        try:
            EWSIIW_L = self.cranio_R2.EW(5085, 5250, 5250, 5450,
                                         'SIIW_L',
                                         sup=True,
                                         right1=True,
                                         verbose=verbose)
            if verbose:
                print 'EWSIIW_L computing done, EWSIIW_L =', EWSIIW_L
        except:
            EWSIIW_L = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWSIIW_L computing, EWSIIW_L =', EWSIIW_L

        try:
            EWSIIW_R = self.cranio_R2.EW(5250, 5450, 5500, 5681,
                                         'SIIW_R',
                                         sup=True,
                                         verbose=verbose)
            if verbose:
                print 'EWSIIW_R computing done, EWSIIW_R =', EWSIIW_R
        except:
            EWSIIW_R = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWSIIW_R computing, EWSIIW_R =', EWSIIW_R

        try:
            self.Values.update(self.cranio_R2.EWvalues)
        except:
            pass

        try:
            EWSiII5972 = self.cranio_R3.EW(5550, 5681, 5850, 6015,
                                           'SiII5972',
                                           sup=True,
                                           right2=True,
                                           verbose=verbose)
            self.Values.update(self.cranio_R3.EWvalues)
            if verbose:
                print 'EWSiII5972 computing done, EWSiII5972 =', EWSiII5972
        except:
            EWSiII5972 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWSiII5972 computing, EWSiII5972 =', EWSiII5972
        try:
            EWSiII6355 = self.cranio_R4.EW(5850, 6015, 6250, 6365,
                                           'SiII6355',
                                           right1=True,
                                           sup=True,
                                           verbose=verbose)
            self.Values.update(self.cranio_R4.EWvalues)
            if verbose:
                print 'EWSiII6355 computing done, EWSiII6355 =', EWSiII6355
        except:
            EWSiII6355 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWSiII6355 computing, EWSiII6355 =', EWSiII6355

        try:
            vSiII_5454 = self.cranio_R2.velocity({'lmin': 5200,
                                                  'lmax': 5350,
                                                  'lrest': 5454,
                                                  'name': 'vSiII_5454'},
                                                 verbose=verbose)
            self.Values.update(self.cranio_R2.velocityValues)
            if verbose:
                print 'vSiII_5454 computing done, vSiII_5454 =', vSiII_5454
        except:
            vSiII_5454 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in vSiII_5454 computing, vSiII_5454 =', vSiII_5454

        try:
            vSiII_5640 = self.cranio_R2.velocity({'lmin': 5351,
                                                  'lmax': 5550,
                                                  'lrest': 5640,
                                                  'name': 'vSiII_5640'},
                                                 verbose=verbose)
            self.Values.update(self.cranio_R2.velocityValues)
            if verbose:
                print 'vSiII_5640 computing done, vSiII_5640 =', vSiII_5640
        except:
            vSiII_5640 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in vSiII_5640 computing, vSiII_5640 =', vSiII_5640

        try:
            vSiII_5972 = self.cranio_R3.velocity({'lmin': 5700,
                                                  'lmax': 5875,
                                                  'lrest': 5972,
                                                  'name': 'vSiII_5972'},
                                                 verbose=verbose)
            self.Values.update(self.cranio_R3.velocityValues)
            if verbose:
                print 'vSiII_5972 computing done, vSiII_5972 =', vSiII_5972
        except:
            vSiII_5972 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in vSiII_5972 computing, vSiII_5972 =', vSiII_5972

        try:
            vSiII_6355 = self.cranio_R4.velocity({'lmin': 6000,
                                                  'lmax': 6210,
                                                  'lrest': 6355,
                                                  'name': 'vSiII_6355'},
                                                 verbose=verbose)
            # vSiII_6355 = self.cranio_R4.velocity2({'lmin':5850,'lmax':6015,
            # 'lrest':6355,'name':'vSiII_6355'},verbose=verbose)
            self.Values.update(self.cranio_R4.velocityValues)
            if verbose:
                print 'vSiII_6355 computing done, vSiII_6355 =', vSiII_6355
        except:
            vSiII_6355 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in vSiII_6355 computing, vSiII_6355 =', vSiII_6355

        if verbose:
            print >> sys.stderr, 'Computing on silicon zone for this spectrum done'
            print ''.center(100, '=')

        indicators = {'RSi': RSi,
                      'RSiS': RSiS,
                      'RSiSS': RSiSS,
                      'EWSIIW': EWSIIW,
                      'EWSiII5972': EWSiII5972,
                      'EWSiII6355': EWSiII6355,
                      'vSiII_5972': vSiII_5972,
                      'vSiII_6355': vSiII_6355}

        del self.cranio_R1.simulations
        del self.cranio_R2.simulations
        del self.cranio_R3.simulations
        del self.cranio_R4.simulations
        del self.cranio_R1.syst
        del self.cranio_R2.syst
        del self.cranio_R3.syst
        del self.cranio_R4.syst

        return indicators

    def oxygen_computing(self, factor=1.23, rhoR=0.484, nsimu=1000,
                         w=None, s=None, smoother="sgfilter", verbose=True):
        """
        Function to compute and return spectral indicators in the end of
        the spectrum
        """
        # Test if the computation will be possible
        if self.xR is None:
            print >> sys.stderr, 'ERROR, impossible to compute spectral '\
                'indictors defined in oxygen zone (maybe no R channel)'
            indicators = {'EWOI7773': [N.nan, N.nan],
                          'EWCaIIIR': [N.nan, N.nan]}
            return indicators

        # Create zone and craniometers
        zone = (self.xR > 6500) & (self.xR < 8800)
        self.cranio_O = get_cranio(self.xR[zone],
                                   self.yR[zone],
                                   self.vR[zone],
                                   smoother=smoother,
                                   verbose=verbose)  # EWOI7773 and CaIIIR

        try:
            EWOI7773 = self.cranio_O.EW(7100, 7270, 7720, 8000,
                                        'OI7773',
                                        sup=True,
                                        verbose=verbose)
            if verbose:
                print 'EWOI7773 computing done, EWOI7773 =', EWOI7773
        except:
            EWOI7773 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWOI7773 computing, EWOI7773 =', EWOI7773

        try:
            EWCaIIIR = self.cranio_O.EW(7720, 8000, 8300, 8800,
                                        'CaIIIR',
                                        sup=True,
                                        verbose=verbose)
            if verbose:
                print 'EWCaIIIR computing done, EWCaIIIR =', EWCaIIIR
        except:
            EWCaIIIR = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWCaIIIR computing, EWCaIIIR =', EWCaIIIR

        try:
            self.Values.update(self.cranio_O.EWvalues)
        except:
            pass

        if verbose:
            print >> sys.stderr, 'Computing on oxygen zone for this '\
                'spectrum done'
            print ''.center(100, '=')

        indicators = {'EWOI7773': EWOI7773,
                      'EWCaIIIR': EWCaIIIR}

        del self.cranio_O.simulations
        del self.cranio_O.syst

        return indicators

    def iron_computing(self, factor=1.23, rhoR=0.484, nsimu=1000,
                       smoother="sgfilter", w=None, s=None, verbose=True):
        """
        Function to compute and return spectral indicators on the middle of
        the spectrum (iron zone)
        """
        # Test if the computation will be possible
        if self.x_merged is None:
            print >> sys.stderr, 'ERROR, impossible to compute spectral '\
                'indictors defined in iron zone (maybe no R or B channel)'
            indicators = {'EWFe4800': [N.nan, N.nan]}
            return indicators

        # Create zone and craniometers
        zone = (self.x_merged > 4350) & (self.x_merged < 5350)
        self.cranio_F = get_cranio(self.x_merged[zone],
                                   self.y_merged[zone],
                                   self.v_merged[zone],
                                   smoother=smoother,
                                   verbose=verbose)  # EWFe4800

        try:
            EWFe4800 = self.cranio_F.EW(4450, 4650, 5050, 5285,
                                        'Fe4800',
                                        sup=True,
                                        left2=True,
                                        verbose=verbose)
            if verbose:
                print 'EWFe4800 computing done, EWFe4800 =', EWFe4800
        except:
            EWFe4800 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWFe4800 computing, EWFe4800 =', EWFe4800

        try:
            self.Values.update(self.cranio_F.EWvalues)
        except:
            pass

        if verbose:
            print >> sys.stderr, 'Computing on iron zone for this spectrum done'
            print ''.center(100, '=')

        indicators = {'EWFe4800': EWFe4800}

        del self.cranio_F.simulations
        del self.cranio_F.syst

        return indicators

    def initialize_parameters(self, verbose=True):
        """
        Function to initialize parameters use to make the control_plot
        """
        try:
            RSi = self.cranio_R1.RSivalues['RSi']
        except:
            RSi = float(N.nan)

        try:
            RSiS = self.cranio_R1.RSiSvalues['RSiS']
        except:
            RSiS = float(N.nan)

        try:
            RSiSS = self.cranio_R5.RSiSSvalues['RSiSS']
        except:
            RSiSS = float(N.nan)

        try:
            RCa = self.cranio_BCa.RCavalues['RCa']
        except:
            RCa = float(N.nan)

        try:
            RCaS = self.cranio_BCa.RCaSvalues['RCaS']
        except:
            RCaS = float(N.nan)

        try:
            RCaS2 = self.cranio_BCa.RCaS2values['RCaS2']
        except:
            RCaS2 = float(N.nan)

        try:
            EDCa = self.cranio_BCa.EDCavalues['EDCa']
        except:
            EDCa = float(N.nan)

        try:
            EWCaIIHK = self.cranio_BCa.EWvalues['EWCaIIHK']
        except:
            EWCaIIHK = float(N.nan)

        try:
            EWSiII4000 = self.cranio_BSi.EWvalues['EWSiII4000']
        except:
            EWSiII4000 = float(N.nan)

        try:
            EWMgII = self.cranio_BMg.EWvalues['EWMgII']
        except:
            EWMgII = float(N.nan)

        try:
            EWSIIW = self.cranio_R2.EWvalues['EWSIIW']
        except:
            EWSIIW = float(N.nan)

        try:
            EWSIIW_L = self.cranio_R2.EWvalues['EWSIIW_L']
        except:
            EWSIIW_L = float(N.nan)

        try:
            EWSIIW_R = self.cranio_R2.EWvalues['EWSIIW_R']
        except:
            EWSIIW_R = float(N.nan)

        try:
            EWSiII5972 = self.cranio_R3.EWvalues['EWSiII5972']
        except:
            EWSiII5972 = float(N.nan)

        try:
            EWSiII6355 = self.cranio_R4.EWvalues['EWSiII6355']
        except:
            EWSiII6355 = float(N.nan)

        try:
            vSiII_5972 = self.cranio_R3.velocityValues['vSiII_5972']
        except:
            vSiII_5972 = float(N.nan)

        try:
            vSiII_6355 = self.cranio_R4.velocityValues['vSiII_6355']
        except:
            vSiII_6355 = float(N.nan)

        return RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII,\
            EWSIIW, EWSiII5972, EWSiII6355, vSiII_5972, vSiII_6355, EWSIIW_L,\
            EWSIIW_R

    #=========================================================================
    # Functions to plot control_plot of spectral indicators computing
    #=========================================================================

    def plot_cranioBCa(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where RCa,RCaS,RCas2,EDCa and EWCaIIHK are computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW,\
            EWSiII5972, EWSiII6355, vSiII_5972, vSiII_6355, EWSIIW_L, EWSIIW_R = metrics
        cr = self.cranio_BCa

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
            save = True
        else:
            save = False

        ax.plot(cr.x, cr.y, color='k', label='Flux')
        try:
            ax.plot(cr.x, cr.s, color='r', label='Interpolated flux')
        except:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        try:  # Plot the RCaS vspan
            ax.axvspan(cr.RCaSvalues['RCaS_lbd'][0],
                       cr.RCaSvalues['RCaS_lbd'][1],
                       ymin=0, ymax=1, facecolor='y', alpha=0.25)
            ax.axvspan(cr.RCaSvalues['RCaS_lbd'][2],
                       cr.RCaSvalues['RCaS_lbd'][3],
                       ymin=0, ymax=1, facecolor='y', alpha=0.25)
        except:
            print >> sys.stderr, "No parameters to plot RCaS zone"

        try:  # Plot the EWCaIIHK points and lines
            lbd_line = cr.x[(cr.x >= cr.EWvalues['lbd_EWCaIIHK'][0])
                            & (cr.x <= cr.EWvalues['lbd_EWCaIIHK'][1])]
            p_line = N.polyfit([cr.EWvalues['lbd_EWCaIIHK'][0],
                                cr.EWvalues['lbd_EWCaIIHK'][1]],
                               [cr.smoother(cr.EWvalues['lbd_EWCaIIHK'])[0],
                                cr.smoother(cr.EWvalues['lbd_EWCaIIHK'])[1]], 1)
            ax.scatter(cr.RCavalues['RCa_lbd'],
                       cr.smoother(cr.RCavalues['RCa_lbd']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except:
            print >> sys.stderr, "No parameters to plot EWCaIIHK zone"

        try:  # Plot the RCa lines
            for x, y in zip(cr.RCavalues['RCa_lbd'],
                            cr.smoother(cr.RCavalues['RCa_lbd'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot RCa zone"

        # Annotate the Ca zone with spectral indicators values
        try:
            ax.annotate('RCa=%.2f, RCaS=%.2f, RCaS2=%.2f' %
                        (RCa, RCaS, RCaS2), xy=(0.01, 0.01),
                        xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
            ax.annotate('EWCaIIHK=%.2f' %
                        (EWCaIIHK), xy=(0.01, 0.95), xycoords='axes fraction',
                        xytext=(0.01, 0.95), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except:
            pass
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=3450, xmax=4070)

        if save:
            fig.savefig('calcium_' + filename)

    def plot_cranioBSi(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where EWSi4000 is computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW,\
            EWSiII5972, EWSiII6355, vSiII_5972, vSiII_6355, EWSIIW_L, EWSIIW_R = metrics
        cr = self.cranio_BSi

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
            save = True
        else:
            save = False

        ax.plot(cr.x, cr.y, color='k', label='Flux')
        try:
            ax.plot(cr.x, cr.s, color='r', label='Interpolated flux')
        except:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.EWvalues['lbd_EWSiII4000'][0])
                            & (cr.x <= cr.EWvalues['lbd_EWSiII4000'][1])]
            p_line = N.polyfit([cr.EWvalues['lbd_EWSiII4000'][0],
                                cr.EWvalues['lbd_EWSiII4000'][1]],
                               [cr.smoother(cr.EWvalues['lbd_EWSiII4000'])[0],
                                cr.smoother(cr.EWvalues['lbd_EWSiII4000'])[1]],
                               1)
            ax.scatter(cr.EWvalues['lbd_EWSiII4000'],
                       cr.smoother(cr.EWvalues['lbd_EWSiII4000']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except:
            print >> sys.stderr, "No parameters to plot EDCa straight line"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWSiII4000'],
                            cr.smoother(cr.EWvalues['lbd_EWSiII4000'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot RCa vlines"

        # Annotate the Ca zone with spectral indicators values
        try:
            ax.annotate('EWSiII4000=%.2f' %
                        (EWSiII4000), xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except:
            pass
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=3850, xmax=4150)

        if save:
            fig.savefig('EWSiII4000_' + filename)

    def plot_cranioBMg(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where EWMgII is computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW,\
            EWSiII5972, EWSiII6355, vSiII_5972, vSiII_6355, EWSIIW_L, EWSIIW_R = metrics
        cr = self.cranio_BMg

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
            save = True
        else:
            save = False

        ax.plot(cr.x, cr.y, color='k', label='Flux')
        try:
            ax.plot(cr.x, cr.s, color='r', label='Interpolated flux')
        except:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.EWvalues['lbd_EWMgII'][0])
                            & (cr.x <= cr.EWvalues['lbd_EWMgII'][1])]
            p_line = N.polyfit([cr.EWvalues['lbd_EWMgII'][0],
                                cr.EWvalues['lbd_EWMgII'][1]],
                               [cr.smoother(cr.EWvalues['lbd_EWMgII'])[0],
                                cr.smoother(cr.EWvalues['lbd_EWMgII'])[1]], 1)
            ax.scatter(cr.EWvalues['lbd_EWMgII'],
                       cr.smoother(cr.EWvalues['lbd_EWMgII']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except:
            print >> sys.stderr, "No parameters to plot EWMgII straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWMgII'],
                            cr.smoother(cr.EWvalues['lbd_EWMgII'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot EWMgII vlines"

        # Annotate the Ca zone with spectral indicators values
        try:
            ax.annotate('EWMgII=%.2f' %
                        (EWMgII), xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except:
            pass
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=4000, xmax=4600)

        if save:
            fig.savefig('EWMgII_' + filename)

    def plot_cranioR1R5(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where RCa,RCaS,RCas2,EDCa and EWCaIIHK are computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW,\
            EWSiII5972, EWSiII6355, vSiII_5972, vSiII_6355, EWSIIW_L, EWSIIW_R = metrics
        cr1 = self.cranio_R1
        cr5 = self.cranio_R5

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
            save = True
        else:
            save = False

        ax.plot(cr5.x, cr5.y, color='k', label='Flux')
        try:
            ax.plot(cr1.x, cr1.s, color='r', label='Interpolated flux')
        except:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        # try: #Plot the RSiSS vspan
        ax.axvspan(cr5.RSiSSvalues['RSiSS_lbd'][0],
                   cr5.RSiSSvalues['RSiSS_lbd'][1],
                   ymin=0, ymax=1, facecolor='y', alpha=0.25)
        ax.axvspan(cr5.RSiSSvalues['RSiSS_lbd'][2],
                   cr5.RSiSSvalues['RSiSS_lbd'][3],
                   ymin=0, ymax=1, facecolor='y', alpha=0.25)
        # except: print >> sys.stderr, "No parameters to plot RSiSS zone"

        try:  # Plot the RSi points and lines
            lbd_line1 = cr1.x[(cr1.xR >= cr1.RSivalues['RSi_lbd'][0])
                              & (cr1.x <= cr1.RSivalues['RSi_lbd'][2])]
            lbd_line2 = cr1.x[(cr1.x >= cr1.RSivalues['RSi_lbd'][2])
                              & (cr1.x <= cr1.RSivalues['RSi_lbd'][4])]
            p_line1 = N.polyfit([cr1.RSivalues['RSi_lbd'][0],
                                 cr1.RSivalues['RSi_lbd'][2]],
                                [cr1.smoother(cr1.RSivalues['RSi_lbd'])[0],
                                 cr1.smoother(cr1.RSivalues['RSi_lbd'])[2]], 1)
            p_line2 = N.polyfit([cr1.RSivalues['RSi_lbd'][2],
                                 cr1.RSivalues['RSi_lbd'][4]],
                                [cr1.smoother(cr1.RSivalues['RSi_lbd'])[2],
                                 cr1.smoother(cr1.RSivalues['RSi_lbd'])[4]], 1)
            ax.scatter(cr1.RSivalues['RSi_lbd'],
                       cr1.smoother(cr1.RSivalues['RSi_lbd']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line1, N.polyval(p_line1, lbd_line1), color='g')
            ax.plot(lbd_line2, N.polyval(p_line2, lbd_line2), color='g')
        except:
            print >> sys.stderr, "No parameters to plot RSi zone"

        try:  # Plot the RSi and RSiS lines
            for x, y in zip(cr1.RSivalues['RSi_lbd'],
                            cr1.smoother(cr1.RSivalues['RSi_lbd'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot RSi and RSiS lines"

        # Annotate the Ca zone with spectral indicators values
        try:
            ax.annotate('RSi=%.2f, RSiS=%.2f, RSiSS=%.2f' %
                        (RSi, RSiS, RSiSS), xy=(0.01, 0.01),
                        xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except:
            pass

        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=5480, xmax=6500)

        if save:
            fig.savefig('silicon_' + filename)

    def plot_cranioR2(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where EWSIIW is computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW,\
            EWSiII5972, EWSiII6355, vSiII_5972, vSiII_6355, EWSIIW_L, EWSIIW_R = metrics
        cr = self.cranio_R2

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
            save = True
        else:
            save = False

        ax.plot(cr.x, cr.y, color='k', label='Flux')
        try:
            ax.plot(cr.x, cr.s, color='r', label='Interpolated flux')
        except:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        # For EWSiW
        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.EWvalues['lbd_EWSIIW'][0])
                            & (cr.x <= cr.EWvalues['lbd_EWSIIW'][1])]
            p_line = N.polyfit([cr.EWvalues['lbd_EWSIIW'][0],
                                cr.EWvalues['lbd_EWSIIW'][1]],
                               [cr.smoother(cr.EWvalues['lbd_EWSIIW'])[0],
                                cr.smoother(cr.EWvalues['lbd_EWSIIW'])[1]], 1)
            ax.scatter(cr.EWvalues['lbd_EWSIIW'],
                       cr.smoother(cr.EWvalues['lbd_EWSIIW']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except:
            print >> sys.stderr, "No parameters to plot EWSIIW straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWSIIW'],
                            cr.smoother(cr.EWvalues['lbd_EWSIIW'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot EWSIIW vlines"

        # For EWSiW_L
        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.EWvalues['lbd_EWSIIW_L'][0])
                            & (cr.x <= cr.EWvalues['lbd_EWSIIW_L'][1])]
            p_line = N.polyfit([cr.EWvalues['lbd_EWSIIW_L'][0],
                                cr.EWvalues['lbd_EWSIIW_L'][1]],
                               [cr.smoother(cr.EWvalues['lbd_EWSIIW_L'])[0],
                                cr.smoother(cr.EWvalues['lbd_EWSIIW_L'])[1]], 1)
            ax.scatter(cr.EWvalues['lbd_EWSIIW_L'],
                       cr.smoother(cr.EWvalues['lbd_EWSIIW_L']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except:
            print >> sys.stderr, "No parameters to plot EWSIIW_L straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWSIIW_L'],
                            cr.smoother(cr.EWvalues['lbd_EWSIIW_L'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot EWSIIW_L vlines"

        # For EWSiW_R
        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.EWvalues['lbd_EWSIIW_R'][0])
                            & (cr.x <= cr.EWvalues['lbd_EWSIIW_R'][1])]
            p_line = N.polyfit([cr.EWvalues['lbd_EWSIIW_R'][0],
                                cr.EWvalues['lbd_EWSIIW_R'][1]],
                               [cr.smoother(cr.EWvalues['lbd_EWSIIW_R'])[0],
                                cr.smoother(cr.EWvalues['lbd_EWSIIW_R'])[1]], 1)
            ax.scatter(cr.EWvalues['lbd_EWSIIW_R'],
                       cr.smoother(cr.EWvalues['lbd_EWSIIW_R']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except:
            print >> sys.stderr, "No parameters to plot EWSIIW_R straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWSIIW_R'],
                            cr.smoother(cr.EWvalues['lbd_EWSIIW_R'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot EWSIIW_R vlines"

        # Annotate the Ca zone with spectral indicators values
        try:
            ax.annotate('EWSIIW=%.2f' %
                        (EWSIIW), xy=(0.01, 0.07), xycoords='axes fraction',
                        xytext=(0.01, 0.07), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
            ax.annotate('EWSIIW_L=%.2f,EWSIIW_R=%.2f' %
                        (EWSIIW_L, EWSIIW_R), xy=(0.01, 0.01),
                        xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except:
            pass

        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=5060, xmax=5700)
        print filename
        if save:
            fig.savefig('EWSIIW_' + filename)

    def plot_cranioR3R4(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where EWSIIW is computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW,\
            EWSiII5972, EWSiII6355, vSiII_5972, vSiII_6355, EWSIIW_L, EWSIIW_R = metrics
        cr3 = self.cranio_R3
        cr4 = self.cranio_R4

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
            save = True
        else:
            save = False

        ax.plot(cr3.x, cr3.y, color='k', label='Flux')
        ax.plot(cr4.x, cr4.y, color='k', label='Flux')
        try:
            ax.plot(cr3.x, cr3.s, color='r', label='Interpolated flux')
        except:
            print >> sys.stderr, "No smothing function computed for '\
            'EWSiII5972, so no smoothing function ploted"
        try:
            ax.plot(cr4.x, cr4.s, color='b', label='Interpolated flux')
        except:
            print >> sys.stderr, "No smothing function computed for '\
            'EWSiII6355, so no smoothing function ploted"

        try:  # Plot points and straight lines
            lbd_line = cr3.x[(cr3.x >= cr3.EWvalues['lbd_EWSiII5972'][0]) &
                             (cr3.x <= cr3.EWvalues['lbd_EWSiII5972'][1])]
            p_line = N.polyfit([cr3.EWvalues['lbd_EWSiII5972'][0],
                                cr3.EWvalues['lbd_EWSiII5972'][1]],
                               [cr3.smoother(cr3.EWvalues['lbd_EWSiII5972'])[0],
                                cr3.smoother(cr3.EWvalues['lbd_EWSiII5972'])[1]], 1)
            ax.scatter(cr3.EWvalues['lbd_EWSiII5972'],
                       cr3.smoother(cr3.EWvalues['lbd_EWSiII5972']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except:
            print >> sys.stderr, "No parameters to plot EWSiII5972 straight '\
            'line zone"

        try:  # Plot points and straight lines
            lbd_line = cr4.x[(cr4.x >= cr4.EWvalues['lbd_EWSiII6355'][0]) &
                             (cr4.x <= cr4.EWvalues['lbd_EWSiII6355'][1])]
            p_line = N.polyfit([cr4.EWvalues['lbd_EWSiII6355'][0],
                                cr4.EWvalues['lbd_EWSiII6355'][1]],
                               [cr4.smoother(cr4.EWvalues['lbd_EWSiII6355'])[0],
                                cr4.smoother(cr4.EWvalues['lbd_EWSiII6355'])[1]], 1)
            ax.scatter(cr4.EWvalues['lbd_EWSiII6355'],
                       cr4.smoother(cr4.EWvalues['lbd_EWSiII6355']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except:
            print >> sys.stderr, "No parameters to plot EWSiII6355 straight '\
            'line zone"

        try:  # Plot vlines for EWSiII5972
            for x, y in zip(cr3.EWvalues['lbd_EWSiII5972'],
                            cr3.smoother(cr3.EWvalues['lbd_EWSiII5972'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot EWSiII5972 vlines"

        try:  # Plot vlines for EWSiII6355
            for x, y in zip(cr4.EWvalues['lbd_EWSiII6355'],
                            cr4.smoother(cr4.EWvalues['lbd_EWSiII6355'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot EWSiII6355 vlines"

        # Annotate the Si zone with spectral indicators values
        try:
            ax.annotate('EWSiII5972=%.2f, EWSiII6355=%.2f' %
                        (EWSiII5972, EWSiII6355),
                        xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left', verticalalignment='bottom',
                        fontsize=10)
        except:
            pass

        try:  # Plot vline for vSiII_6355
            ax.axvline(cr4.velocityValues['vSiII_6355_lbd'],
                       color='k', lw=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot SiII6355 vlines"

        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=5500, xmax=6400)

        if save:
            fig.savefig('EWSiII5972_' + filename)

    def plot_spectrum(self, metrics, ax=None, filename='', title=None, verbose=True):

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW,\
            EWSiII5972, EWSiII6355, vSiII_5972, vSiII_6355, EWSIIW_L, EWSIIW_R = metrics

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
        else:
            ax = ax

        # Plot spectrum===============================================

        try:
            ax.plot(self.x, self.y, color='k', label='Flux')
            label = 'Interpolated flux'
            try:
                ax.plot(self.calcium_zone['x'],
                        self.cranio_B.smoother(self.calcium_zone['x']),
                        color='r', label=label)
                label = '_nolegend_'
            except:
                pass
            try:
                ax.plot(self.silicon_zone['x'],
                        self.cranioR.smoother(self.silicon_zone['x']),
                        color='r', label=label)
            except:
                pass
        except:
            label1 = 'Flux'
            label2 = 'Interpolated flux'
            try:
                ax.plot(self.xB, self.yB, color='k', label=label1)
                ax.plot(self.calcium_zone['x'],
                        self.cranio_B.smoother(self.calcium_zone['x']),
                        color='r', label=label2)
                label1 = '_nolegend_'
                label2 = '_nolegend_'
            except:
                pass
            try:
                ax.plot(self.xR, self.yR, color='k', label=label1)
                ax.plot(self.silicon_zone['x'],
                        self.cranioR.smoother(self.silicon_zone['x']),
                        color='r', label=label2)
            except:
                pass
        ax.set_xlabel('Wavelength [AA]')
        ax.set_ylabel('Flux [erg/s/cm2]')
        if title is not None:
            ax.set_title('%s' % title)

    def control_plot(self, filename='', title=None, format=['png'], verbose=True):
        """
        self.cranio.control_plot(filename=filename, title=title)

        Options:
            filename: (string) filename of the png created. Should end in .png
            title: (string) optional title of the control plot. Passing SN
            name and exp_code through it is a good idea.
        """

        # Initialize parameters=====================================

        metrics = self.initialize_parameters()
        if self.xB is not None and self.xR is not None:
            MetricsFig = P.figure(figsize=(14, 12))
            ax1 = MetricsFig.add_subplot(3, 3, 1)
            ax2 = MetricsFig.add_subplot(3, 3, 2)
            ax3 = MetricsFig.add_subplot(3, 3, 3)
            ax4 = MetricsFig.add_subplot(3, 3, 4)
            ax5 = MetricsFig.add_subplot(3, 3, 5)
            ax6 = MetricsFig.add_subplot(3, 3, 6)
            ax7 = MetricsFig.add_subplot(3, 1, 3)
            self.plot_cranioBCa(metrics, ax=ax1, filename=filename)
            self.plot_cranioBSi(metrics, ax=ax2, filename=filename)
            self.plot_cranioBMg(metrics, ax=ax3, filename=filename)
            self.plot_cranioR1R5(metrics, ax=ax4, filename=filename)
            self.plot_cranioR2(metrics, ax=ax5, filename=filename)
            self.plot_cranioR3R4(metrics, ax=ax6, filename=filename)
            self.plot_spectrum(metrics, ax=ax7, filename=filename, title=title)
            ax7.set_ylim(ymin=0)
            ax7.set_xlim(xmin=3000, xmax=7000)
            if filename is None:
                unique_suffix = time.strftime("%Y-%m-%d-%H_%M_%S_UTC",
                                              time.gmtime())
                filename = "control_plot_SNfPhrenology_" + unique_suffix
            for f in format:
                MetricsFig.savefig(filename + '.' + f)
                print >> sys.stderr, "Control plot saved in %s" % filename \
                    + '.' + f

        elif self.xB is not None:
            print >> sys.stderr, 'Worked on the B channel only'
            MetricsFig = P.figure(figsize=(12, 8))
            ax1 = MetricsFig.add_subplot(2, 3, 1)
            ax2 = MetricsFig.add_subplot(2, 3, 2)
            ax3 = MetricsFig.add_subplot(2, 3, 3)
            ax7 = MetricsFig.add_subplot(2, 1, 2)
            self.plot_cranioBCa(metrics, ax=ax1, filename=filename)
            self.plot_cranioBSi(metrics, ax=ax2, filename=filename)
            self.plot_cranioBMg(metrics, ax=ax3, filename=filename)
            self.plot_spectrum(metrics, ax=ax7, filename=filename, title=title)
            ax7.set_ylim(ymin=0)
            ax7.set_xlim(xmin=self.xB[0], xmax=self.xB[-1])
            if filename is None:
                unique_suffix = time.strftime("%Y-%m-%d-%H_%M_%S_UTC",
                                              time.gmtime())
                filename = "control_plot_SNfPhrenology_" + unique_suffix
            if title is not None:
                ax7.set_title('%s, Calcium zone' % title)
            else:
                ax7.set_title('Calcium zone')
            for f in format:
                MetricsFig.savefig(filename + '.' + f)
                print >> sys.stderr, "Control plot saved in %s" % filename \
                    + '.' + f

        elif self.xR is not None:
            print >> sys.stderr, 'Worked on the R channel only'
            MetricsFig = P.figure(figsize=(12, 8))
            ax4 = MetricsFig.add_subplot(2, 3, 1)
            ax5 = MetricsFig.add_subplot(2, 3, 2)
            ax6 = MetricsFig.add_subplot(2, 3, 3)
            ax7 = MetricsFig.add_subplot(2, 1, 2)
            self.plot_cranioR1R5(metrics, ax=ax4, filename=filename)
            self.plot_cranioR2(metrics, ax=ax5, filename=filename)
            self.plot_cranioR3R4(metrics, ax=ax6, filename=filename)
            self.plot_spectrum(metrics, ax=ax7, filename=filename, title=title)
            ax7.set_ylim(ymin=0)
            ax7.set_xlim(xmin=self.xR[0], xmax=7000)
            if filename is None:
                unique_suffix = time.strftime("%Y-%m-%d-%H_%M_%S_UTC",
                                              time.gmtime())
                filename = "control_plot_SNfPhrenology_" + unique_suffix
            if title is not None:
                ax7.set_title('%s, Silicon zone' % title)
            else:
                ax7.set_title('Silicon zone')
            for f in format:
                MetricsFig.savefig(filename + '.' + f)
                print >> sys.stderr, "Control plot saved in %s" % filename \
                    + '.' + f
        P.close()

    def plot_oxygen(self, filename='', title=None, format=['png'], verbose=True):

        cr = self.cranio_O

        fig = P.figure()
        ax = fig.add_subplot(111)
        ax.plot(cr.x, cr.y, 'k', label='Flux')
        ax.plot(cr.x, cr.s, 'r', label='Interpolated flux')
        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.EWvalues['lbd_EWOI7773'][0])
                            & (cr.x <= cr.EWvalues['lbd_EWOI7773'][1])]
            p_line = N.polyfit([cr.EWvalues['lbd_EWOI7773'][0],
                                cr.EWvalues['lbd_EWOI7773'][1]],
                               [cr.smoother(cr.EWvalues['lbd_EWOI7773'])[0],
                                cr.smoother(cr.EWvalues['lbd_EWOI7773'])[1]], 1)
            ax.scatter(cr.EWvalues['lbd_EWOI7773'],
                       cr.smoother(cr.EWvalues['lbd_EWOI7773']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except:
            print >> sys.stderr, "No parameters to plot EWOI7773 straight line"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWOI7773'],
                            cr.smoother(cr.EWvalues['lbd_EWOI7773'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot EWOI7773 vlines\n"

        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.EWvalues['lbd_EWCaIIIR'][0])
                            & (cr.x <= cr.EWvalues['lbd_EWCaIIIR'][1])]
            p_line = N.polyfit([cr.EWvalues['lbd_EWCaIIIR'][0],
                                cr.EWvalues['lbd_EWCaIIIR'][1]],
                               [cr.smoother(cr.EWvalues['lbd_EWCaIIIR'])[0],
                                cr.smoother(cr.EWvalues['lbd_EWCaIIIR'])[1]], 1)
            ax.scatter(cr.EWvalues['lbd_EWCaIIIR'],
                       cr.smoother(cr.EWvalues['lbd_EWCaIIIR']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except:
            print >> sys.stderr, "No parameters to plot EWCaIIIR straight line"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWCaIIIR'],
                            cr.smoother(cr.EWvalues['lbd_EWCaIIIR'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot EWCaIIIR vlines\n"

        # Try to Annotate with spectral indicators values
        try:
            ax.annotate('EWOI7773=%.2f, EWCaIIIR=%.2f' %
                        (cr.EWvalues['EWOI7773'],
                         cr.EWvalues['EWCaIIIR']),
                        xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left', verticalalignment='bottom',
                        fontsize=10)
        except:
            pass

        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=cr.x[0], xmax=cr.x[-1])
        ax.set_xlabel('Wavelength [AA]')
        ax.set_ylabel('Flux [erg/s/cm2]')
        if title is not None:
            ax.set_title(title)
        for f in format:
            fig.savefig(filename + '.' + f)
            print >> sys.stderr, "Control plot for oxygen zone saved in %s" \
                % filename + '.' + f
        P.close()

    def plot_iron(self, filename='', title=None, format=['png'], verbose=True):

        cr = self.cranio_F

        fig = P.figure()
        ax = fig.add_subplot(111)
        ax.plot(cr.x, cr.y, 'k', label='Flux')
        ax.plot(cr.x, cr.s, 'r', label='Interpolated flux')
        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.EWvalues['lbd_EWFe4800'][0])
                            & (cr.x <= cr.EWvalues['lbd_EWFe4800'][1])]
            p_line = N.polyfit([cr.EWvalues['lbd_EWFe4800'][0],
                                cr.EWvalues['lbd_EWFe4800'][1]],
                               [cr.smoother(cr.EWvalues['lbd_EWFe4800'])[0],
                                cr.smoother(cr.EWvalues['lbd_EWFe4800'])[1]], 1)
            ax.scatter(cr.EWvalues['lbd_EWFe4800'],
                       cr.smoother(cr.EWvalues['lbd_EWFe4800']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except:
            print >> sys.stderr, "No parameters to plot EWFe4800 straight line"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWFe4800'],
                            cr.smoother(cr.EWvalues['lbd_EWFe4800'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except:
            print >> sys.stderr, "No parameters to plot EWFe4800 vlines\n"

        # Try to Annotate with spectral indicators values
        try:
            ax.annotate('EWFe4800=%.2f' % (cr.EWvalues['EWFe4800']),
                        xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left', verticalalignment='bottom',
                        fontsize=10)
        except:
            pass

        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=cr.x[0], xmax=cr.x[-1])
        ax.set_xlabel('Wavelength [AA]')
        ax.set_ylabel('Flux [erg/s/cm2]')
        if title is not None:
            ax.set_title(title)
        for f in format:
            fig.savefig(filename + '.' + f)
            print >> sys.stderr, "Control plot for iron zone saved in %s" %\
                filename + '.' + f
        P.close()

#==============================================================================
# Function to compute Stephen's ratio, and other general ratios
#==============================================================================


def integration(spec, lbd, v=2000.):
    c = 299792.458
    min = lbd * (1 - v / c)
    max = lbd * (1 + v / c)
    step = spec.x[1] - spec.x[0]
    return float(N.sum(spec.y[(spec.x >= min) & (spec.x <= max)]) * step)


def stephen_ratio(specB, specR=None, lbd_6415=6415, lbd_4427=4427):
    if specR is None and (specB.x[0] < 4500 and specB.x[-1] > 6400):
        return integration(specB, lbd_6415) / integration(specB, lbd_4427)
    elif specR is not None:
        return integration(specR, lbd_6415) / integration(specB, lbd_4427)
    else:
        print "Warning: Rsjb not computed."
        return 0.


def general_ratio(specB, specR=None, lbd1=6310, lbd2=4390, v=4000):
    """
    new <good> R lbd1=6310,lbd2=4390 with v = 4000
    new <good> R lbd1=6310,lbd2=5130 with v = 2000 in silicon zone
    """
    if specR is None:
        specR = specB
    return integration(specR, lbd1, v=v) / integration(specB, lbd2, v=v)


def get_cranio(x, y, v, smoother='spline_free_knot', verbose=False):
    obj = VCL.SPCS(x, y, v)
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

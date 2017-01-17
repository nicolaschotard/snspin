#!/usr/bin/env python

"""Spectral covariance matrix (variance and correlation) study."""

import pylab as P
import numpy as N
import scipy as S
from scipy import interpolate

from snspin.tools import io
from snspin.tools import statistics
from snspin.spectrum.smoothing import savitzky_golay as sg
from snspin.spectrum.smoothing import spline_find_s
# from snspin.spectrum.smoothing import sg_find_num_points


class SPCS(object):

    """SpecVarCorrStudy."""

    def __init__(self, x, y, v, specid='', obj='',
                 verbose=False, rhob=0.32, rhor=0.40, factor=0.69):
        """."""
        # Set the data
        self.data = {'x': x, 'y': y, 'v': v * factor, 'factor_used': factor}
        self.data['rho'] = rhob if S.mean(x) < 5150 else rhor
        self.data['rho_used'] = 'rhob' if S.mean(x) < 5150 else 'rhor'

        # Set infomations about the spectrum
        self.specid = specid
        self.object = obj
        self.verbose = verbose
        if verbose:
            print 'Working on ', self.specid, ', from', self.object
            print '%i bins from %.2f to %.2f A' \
                  % (len(self.data['x']), self.data['x'][0], self.data['x'][-1])

        # Initialize attributes
        self.smoothing = self.s = self.w = self.ndist = None
        self.simus = []
        self.smooth_results = {}

    def smooth(self, smoothing='sp', s=None, w=15, findp=True):
        """."""
        # Set the smoothing parameters
        self.smoothing = smoothing
        if self.smoothing == 'sp' and findp:
            try:
                self.s = spline_find_s(self.data['x'], self.data['y'], self.data['v'], corr=self.data['rho'])
            except:
                self.s = 0.5
                print "WARNING: Smoothing failed. s=0.5 by default"
        elif self.smoothing == 'sg' and findp:
            try:
                # this looks like a simple relation between average S/N, and works pretty well
                self.w = int(-10 * N.log(N.median(self.data['y'] / N.sqrt(self.data['v']))) + 52)
                if self.w < 3:
                    self.w = 3
                # self.w = int(sg_find_num_points(self.data['x'], self.data['y'],
                #                                 self.data['v'], corr=self.data['rho']))
            except:
                self.w = 15
                print "WARNING: Smoothing failed. w=15 by default"
        self.s = s if self.s is None else self.s
        self.w = w if self.w is None else self.w

        # Smooth, compute the pull and the derive values
        self.smooth_results['ysmooth'] = smooth_spec(self.data['x'], self.data['y'], self.data['v'],
                                                     sfunc=smoothing, s=self.s, w=self.w)
        self.smooth_results['chi2'] = S.sum((self.data['y'] - self.smooth_results['ysmooth'])**2 /
                                            self.data['v']) / (len(self.data['x']) - 1)
        self.smooth_results['pull'] = comp_pull(self.data['y'], self.smooth_results['ysmooth'], self.data['v'])
        self.smooth_results['pull_mean'] = S.mean(self.smooth_results['pull'])
        self.smooth_results['pull_std'] = S.std(self.smooth_results['pull'])
        # self.data['rho'] = autocorr(self.pull, k=1, full=False)
        self.smooth_results['residuals'] = self.data['y'] - self.smooth_results['ysmooth']

        if self.verbose:
            print 'Smoothing function used:', self.smoothing
            if self.smoothing == 'sg':
                print 'w = ', self.w
            if self.smoothing == 'sp':
                print 's = ', self.s
            print 'Chi2 = ', self.smooth_results['chi2']
            print 'Mean(pull) = ', self.smooth_results['pull_mean']
            print 'Std(pull) = ', self.smooth_results['pull_std']
            print 'Corr = ', self.data['rho']

        if self.verbose:
            print "Factor used: ", self.data['factor_used']
            print "Correlation coefficient:", self.data['rho']

    def make_simu(self, nsimu=1000):
        """
        Build a simulated set of data.

        Make some simulation for which everything will be computing as well nsimu is the number
        of simulations
        if factor is set (1 by default), then self.data['v']*=factor
        if factor is set to None, then self.data['v']*=self.factor
        (run comp_factor before)
        if corr is set to 1, then the pixel will be correlated using the
        correlation parameter found after the smoothing
        corr can also be set to a float value
        """
        # Compute the ramdom noisea
        if self.data['rho'] is None:
            ndist = S.random.randn(nsimu, len(self.data['x']))
        else:
            ndist = corr_noise(self.data['rho'], nbin=len(self.data['x']), nsimu=nsimu)

        # Create the simulated spectra
        simus = ndist * (S.sqrt((self.data['v']))) + self.smooth_results['ysmooth']

        # Save the random distribution
        self.ndist = ndist

        # Smooth and compute stuffs for the simulated spectra (pull,rho...)
        for sim in simus:
            si = SPCS(self.data['x'], sim, self.data['v'], verbose=False)
            si.smooth(smoothing=self.smoothing,
                      s=self.s, w=self.w, findp=False)
            self.simus.append(si)

        if self.verbose:
            print "\nSmoothing used:", self.smoothing
            if self.smoothing == 'sg':
                print 'w = ', self.w
            if self.smoothing == 'sp':
                print 's = ', self.s
            print "Factor used: ", self.data['factor_used']
            print "Correlation coefficient:", self.data['rho']
            print nsimu, "simulated spectra have been created"
            print "             Real spectrum   Simulations"
            print "Mean pull        %.3f         %.3f" \
                % (self.smooth_results['pull_mean'],
                   S.mean([s.smooth_results['pull_mean'] for s in self.simus]))
            print "Std  pull        %.3f         %.3f" \
                % (self.smooth_results['pull_std'],
                   S.mean([s.smooth_results['pull_std'] for s in self.simus]))
            print "Mean chi2        %.3f         %.3f" \
                % (self.smooth_results['chi2'],
                   S.mean([s.smooth_results['chi2'] for s in self.simus]))
            print "Mean rho         %.3f         %.3f" \
                  % (self.data['rho'], S.mean([s.rho for s in self.simus]))

    def do_plots(self, allp=False, lim=2):
        """Make some plots."""
        plot_spec(self.data['x'], self.data['y'], self.data['v'], self.smooth_results['ysmooth'],
                  title=self.object + ', ' + self.specid)
        plot_pull(self.data['x'], self.smooth_results['pull'], title=self.object + ', ' + self.specid)
        if hasattr(self, 'simus'):
            self.plot_simu_distri()
            if not allp:
                return None
            for i, s in enumerate(self.simus):
                if i < lim:
                    s.do_plots()
                else:
                    return

    def plot_simu_distri(self):
        """Plot a few parameter distributions for the simulations."""
        # set the data
        chi2 = S.concatenate([[s.chi2 for s in self.simus], [self.smooth_results['chi2']]])
        corr = S.concatenate([[s.rho for s in self.simus], [self.data['rho']]])
        pmean = S.concatenate([[s.pull_mean for s in self.simus],
                               [self.smooth_results['pull_mean']]])
        pstd = S.concatenate([[s.smooth_results['pull_std'] for s in self.simus],
                              [self.smooth_results['pull_std']]])

        data = [chi2, corr, pmean, pstd]
        names = ['chi2', 'corr', 'pmean', 'pstd']

        # make the figure
        for d, n in zip(data, names):
            P.figure()
            P.hist(d[:-1], bins=S.sqrt(len(d[:-1])) * 2, color='b', alpha=0.5)
            P.axvline(d[-1], color='k')
            P.title(n)


# Definitions ============================================================


def smooth_spec(x, y, v, s=None, w=15, sfunc='sp', order=2, verbose=False):
    """Smooth a given spectrum."""
    if sfunc == 'sp':
        if s == None:
            sp = interpolate.LSQUnivariateSpline(x, y, t=(x[::12])[1:], w=1 / (S.sqrt(v)))
        else:
            if isinstance(s, list):
                s = s[0]
            if s <= 1:
                s *= len(x)
            sp = interpolate.UnivariateSpline(x, y, w=1 / (S.sqrt(v)), s=s)
        ysmooth = sp(x)
    elif sfunc == 'sg':
        kernel = (int(w) * 2) + 1
        if kernel <= order + 2:
            if verbose:
                print "<smooth_spec> WARNING: w  not > order+2 "\
                      "(%d <= %d+2). Replaced it by first odd number "\
                      "above order+2" % (kernel, order)
            kernel = int(order / 2) * 2 + 3
        ysmooth = sg(y, kernel=kernel, order=order)
    return ysmooth


def comp_pull(y, ysmooth, v):
    """Compute the pull."""
    return (y - ysmooth) / S.sqrt(v)


def corr_noise(rho, nbin=10, nsimu=10):
    """Create a correlated gaussian noise array."""

    # Check if rho is between 0 and 0.5
    if rho < 0:
        print 'rho<0, creation of an uncorrelated noise'
        return S.random.randn(nsimu, nbin)
    elif rho > 0.5:
        print 'rho>0.5, set it to 0.5 by default'
        rho = 0.5
    else:
        pass

    # Compute alpha and beta
    r = rho / (1. + 2. * rho)
    alpha = 0.5 * (1. + S.sqrt(1. - 4. * r))
    beta = 0.5 * (1. - S.sqrt(1. - 4. * r))

    # Create the correlated noise
    ndist0 = S.random.randn(nsimu, nbin + 1)
    ndist = S.zeros((nsimu, nbin))
    for i in range(S.shape(ndist)[0]):
        ndist[i] = alpha * ndist0[i][:-1] + beta * ndist0[i][1:]

    return ndist


def plot_spec(x, y, v, ysmooth, title=''):
    """Plot a spectrum."""
    fig = P.figure(dpi=150)
    ax = fig.add_axes([0.1, 0.08, 0.86, 0.87],
                      xlabel=r'Wavelength [$\AA$]', ylabel='Flux')
    ax.errorbar(x, y, yerr=S.sqrt(v), color='k', alpha=0.1)
    ax.plot(x, y, 'g')
    ax.plot(x, ysmooth, 'r', lw=1.5)
    ax.set_title(title)


def plot_pull(x, pull, title=''):
    """Plot the pull distribution."""
    fig = P.figure(dpi=150)
    ax = fig.add_axes([0.1, 0.08, 0.86, 0.87],
                      xlabel=r'Wavelength [$\AA$]', ylabel='Pull')
    ax.plot(x, pull, 'k',
            label='Mean=%.2f, Std=%.2f' % (S.mean(pull),
                                           S.std(pull)))
    ax.set_title(title)
    ax.legend(loc='best').draw_frame(False)


def plot_corr(corr, title=''):
    """."""
    fig = P.figure(dpi=150)
    ax = fig.add_axes([0.1, 0.08, 0.86, 0.87],
                      xlabel=r'Wavelength [$\AA$]', ylabel='Auto correlation')
    ax.plot(corr, 'k')
    ax.set_title(title)


def autocorr(x, k=1, full=False):
    """Autocorrelation function: R(k)= E[ (X(i)-mu)(X(i+k)-mu) ] / (sigma**2)."""
    n, mu, sig = len(x), S.mean(x), S.std(x)
    def ack(k):
        return S.sum([(x[i] - mu) * (x[i + k] - mu) for i in range(n - k)]) / ((n - k) * sig**2)
    if not full:
        return ack(k)
    else:
        return S.array([ack(j) for j in range(n - 1)])


def plot_smoothed_spec(f, num=5):
    """Plot the smoothed spectrum."""
    d = io.loaddata(f)
    for i in d:
        ob = d[i]
        plot_obj(ob, num=num)


def plot_obj(ob, num=5):
    """Plot spectrum and smoothed spectrum from a SPCS object."""
    fig = P.figure(figsize=(8, 18), dpi=150)
    ax = fig.add_axes([0.08, 0.08, 0.86, 0.87],
                      xlabel=r'Wavelength [$\AA$]', ylabel='Flux')
    ax.plot(ob.x, ob.y, 'k')
    ax.plot(ob.x, ob.ysmooth, 'r', lw=1.5)
    for i, s in enumerate(ob.simus):
        if i >= num:
            return
        cst = (i + 1) * S.mean(ob.y)
        ax.plot(s.x, s.y - cst, 'k')
        ax.plot(s.x, s.ysmooth - cst, 'r', lw=1.5)


def control_case(rho=0, factor=1, nsimu=1000, nbin=300, plot=False):
    """
    Simu.

    Make some simulation for which everything will be computing as well
    nsimu is the number af simulations
    if factor is set (1 by default), then self.data['v']*=factor
    if factor is set to None, then self.data['v']*=self.factor
    (run comp_factor before)
    if corr is set to 1, then the pixel will be correlated using
    the correlation parameter found after the smoothing
    corr can also be set to a float value
    """
    # Creation of (correlated) random noise
    if rho == 0:
        sims = S.random.randn(nsimu, nbin)
    else:
        sims = corr_noise(rho, nbin=nbin, nsimu=nsimu)
    sims *= factor

    # Check if rho is the given one
    rho = N.array([autocorr(N.array(sim), k=1, full=False) for sim in sims])
    if plot:
        P.hist(rho, histtype='step', color='b',
               alpha=0.5, bins=statistics.hist_nbin(rho))
        P.title('Mean=%.2f, Std=%.2f' % (N.mean(rho), N.std(rho)))
    else:
        return rho

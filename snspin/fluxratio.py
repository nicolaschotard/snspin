#!/usr/bin/env python

"""
Compute flux ratios.

Work for a given spectrum, a collection of spectra, or the IDR. Some plot function comes with it.
"""

import numpy as N
import pylab as P

from snspin.tools import statistics
from ToolBox import Hubblefit, MPL


LIGHT_VELOCITY = 299792.458  # km/s


# Class ==================================================================


class OneSpecFLuxRatio(object):

    """Compute flux ratios for an input spectrum."""

    def __init__(self, x, y, v, velocity=2000, wmin=3350, wmax=8800, num=None):
        """
        Initialization is made with a regular spectrum:
        x: wavelength
        y: flux
        v: variance
        velocity: for the velocity space
        num: number of bin instead of velocity (in log space)
        """
        self.velocity = float(velocity)
        self.wmin = float(wmin)
        self.wmax = float(wmax)
        if num is not None:
            self.num_bins = int(num)
        else:
            self.num_bins = None

        self.x = N.array(x)
        self.y = N.array(y)
        self.v = N.array(v)
        self.step = self.x[1] - self.x[0]

        self.resampling()
        self.map_ratios()
        self.clean_map()

    def resampling(self):

        # number of bin
        if self.num_bins is None:
            self.num_bins = int(round(N.log10(float(self.wmax) /
                                              float(self.wmin)) /
                                      N.log10(1 + self.velocity /
                                              LIGHT_VELOCITY) + 1))

        # bin edges and centers
        self.binedges = N.logspace(N.log10(self.wmin),
                                   N.log10(self.wmax),
                                   self.num_bins)
        self.bincenters = (self.binedges[0:-1] + self.binedges[1:]) / 2

        # integrated functions
        intf = lambda minw, maxw: N.mean(self.y[(self.x >= minw) &
                                                (self.x < maxw)]) * self.step
        intv = lambda minw, maxw: N.mean(self.v[(self.x >= minw) &
                                                (self.x < maxw)]) * self.step**2

        # Save resampled data
        self.new_x = self.bincenters
        self.new_y = N.array([intf(self.binedges[i], self.binedges[i + 1]) for i
                              in range(len(self.binedges) - 1)])
        self.new_v = N.array([intv(self.binedges[i], self.binedges[i + 1]) for i
                              in range(len(self.binedges) - 1)])
        self.new_e = N.sqrt(self.new_v)

    def map_ratios(self):
        """
        Create a non symetric matrix of resampled spectral flux ratios.
        """
        self.ratios = N.array([y / self.new_y for y in self.new_y])
        self.ratiose = N.array([ratios * N.sqrt((self.new_e[i] /
                                                 self.new_y[i])**2
                                                + (self.new_e / self.new_y)**2)
                                for i, ratios in enumerate(self.ratios)])

    def clean_map(self):
        """
        Clean the maps (ratio and ratiose) for the negative values.
        Make them all equal to nan values
        """
        self.ratios = N.array([N.where((r >= 0) & (re >= 0), r, [N.nan] * len(r))
                               for r, re in zip(self.ratios, self.ratiose)])
        self.ratiose = N.array([N.where((r >= 0) & (re >= 0), re, [N.nan] * len(r))
                                for r, re in zip(self.ratios, self.ratiose)])


class CorrMap(object):

    """Plot correlation maps."""
    
    def __init__(self, maps, param, mapse=None, names=None,
                 x=None, method='pearson', criteria=0.5):
        """
        Input:
        maps: collection of maps, i.e map or flux ratios.
        param: a parameter with which the correlation is computed.
        mapse: a collections of maps corresponding to the errors on maps.
        names: a list of object names, same length as the number of maps.
        x: a third parameter correponding to the maps bining
        method: 'pearson' or 'spearman' correlation coefficient.
        criteria: limit for the classification.

        Output:
        - a plot function of the correlation map
        - a classification of the best correlation,
          with the corresponding values

        Example for the flux ratio case:
        For N SNE, and n bin:
            shape(maps)  = (N,n,n)
            shape(param) = (N,)
            shape(x)     = (n,)   
        """

        self.maps = N.array(maps)
        if mapse is not None:
            self.mapse = N.array(mapse)
        else:
            self.mapse = None
        self.param = N.array(param)
        self.method = method
        self.numb = range(N.shape(self.maps)[1])
        self.names = names
        if x == None:
            self.x = self.numb
        else:
            self.x = N.array(x)
        self.criteria = float(criteria)

        # Compute the correlation map
        self._comp_corr_map()

        # Get the classification
        self._classify()

    def _comp_corr_map(self):
        self.corr_map = N.array([[self._comp_corr(i, j) for j in self.numb]
                                 for i in self.numb])

    def _comp_corr(self, i, j):

        x, y = self._get_column_vals(i, j), self.param

        if not (x[x == x] - 1).any():  # if i == j
            corr = 0
        else:
            filt = N.isfinite(x) & N.isfinite(y)
            corr = statistics.correlation(N.array(x)[filt], N.array(y)[filt],
                                          method=self.method)
        return corr

    def _get_column_vals(self, i, j, err=False):
        """
        return the value of all the maps for the bin j,k.
        """
        if not err:
            return N.array([mapi[i][j] for mapi in self.maps])
        else:
            return N.array([mapi[i][j] for mapi in self.mapse])

    def _classify(self):

        # Only the absoulte value are interesting
        corr_map = N.absolute(self.corr_map)

        # Get the classification
        cvalue = 1
        self.classification = []
        while cvalue >= self.criteria:
            # Get the coordinates of the max

            i, j = N.argwhere(corr_map == corr_map.max())[0]
            if self.x == None:
                self.classification.append([[i, j], corr_map[i, j]])
            else:
                self.classification.append([[i, j], corr_map[i, j],
                                            [self.x[i], self.x[j]]])

            # Set to the current value
            cvalue = corr_map[i, j]

            # Set this value to zero in the temporary matrix
            corr_map[i, j] = 0

    def print_classification(self, i=20):
        """
        Print the first `i` best correlations.
        """
        # Print header
        print "Correlation method used: %s" % self.method
        print " w1  /  w2      rho       bins"
        for ii, cl in enumerate(self.classification):
            if ii > i:
                continue
            else:
                print "%1.f / %1.f     %.2f     [%i,%i]" %\
                    (cl[2][0], cl[2][1], cl[1], cl[0][0], cl[0][1])

    def plot_matrix(self, cmap=P.cm.jet):

        if self.x != None:
            label = r'Wavelength [nm]'
            wmin = self.x[0]
            wmax = self.x[-1]
            wlength = [wmin, wmax, wmin, wmax]
        else:
            label = None
            wlength = None

        wmin = 3500
        wmax = 8500

        # Plot the matrix
        fig = P.figure('Correlation matrix', dpi=150)
        ax = fig.add_axes([0.08, 0.09, 0.88, 0.86], title='Correlation matrix')
        im = ax.imshow(N.absolute(self.corr_map), cmap=cmap,
                       vmin=0, vmax=1,
                       extent=wlength, origin='lower',
                       interpolation='nearest')
        cb = fig.colorbar(im)
        cb.set_label('Absolute %s correlation' % self.method, size='x-large')
        ax.set_xlabel(label, size='x-large')
        ax.set_ylabel(label, size='x-large')

        # relabel axis to get logspace correct
        b = N.log10(float(wmin) / float(wmax)) / (wmin - wmax)
        a = wmin * 10**(-b * wmin)
        y = N.arange(wmin, wmax, 500)[1:]
        x = N.log10(y / a) / b
        strx = [str(int(tmp) / 10) for tmp in y]
        stry = list(strx)
        ax.set_yticks(x)
        ax.set_yticklabels(stry, fontsize=15)
        strx[-2] = ''
        strx[-4] = ''
        ax.set_xticks(x)
        ax.set_xticklabels(strx, fontsize=15)

    def plot_correlation(self, i, j):
        """
        Plot the correlation between the given parameter and the bin (i,j).
        If self.names has been given, you can browse then here.
        """

        # shortcut
        x = self._get_column_vals(i, j)
        y = self.param
        if self.mapse is not None:
            dx = self._get_column_vals(i, j, err=True)
        else:
            dx = None

        # the figure
        fig = P.figure('2D correlation', dpi=100)
        ax = fig.add_axes([0.08, 0.09, 0.88, 0.86])
        line, = ax.plot(x, y, 'ok')

        # errobars if given
        if dx is not None:
            ax.errorbar(x, y, xerr=dx, color='k', ls='None')

        # labels and title
        ax.set_xlim(xmin=0.2, xmax=2)
        ax.set_xlabel('bin (%i,%i)' % (i, j))
        ax.set_ylabel('Given parameter')
        ax.set_title('Correlation: %.2f' % self._comp_corr(i, j))

        # to browse the data
        if self.names is not None:
            if len(self.names) == len(self.param):
                browser = MPL.PointBrowser(x, y, self.names, line)

# Definitions ============================================================


def FluxRatios(idr, **kwargs):
    """
    Compute the flux ratio values for all the spectra of all the SNe
    stored in a LoadIDR object.

    :param bool gp: compute the flux ratio on the HFK gaussian process spectra
    :param wmin array: min and max wavelengths of the spectra to use.
    :param num int: number of bin for the flux ratio bining.
    :return: the same object, with idr.data.[sn]['fratio.'] data.
    """
    # default values
    gp = False
    wmin, wmax = idr.lmin, idr.lmax
    num = None

    # if in kwargs
    if 'gp' in kwargs:
        gp = kwargs['gp']
    if 'wlim' in kwargs:
        wmin, wmax = kwargs['wlim'][0], kwargs['wlim'][1]
    if 'num' in kwargs:
        num = kwargs['num']

    print "Computing flux ratio for..."
    for i, sn in enumerate(sorted(idr.data)):
        print "%i/%i %s" % (i, len(idr.data.keys()), sn)
        if gp:
            key = 'gp'
        else:
            key = 'data'

        X = idr.data[sn][key + '.X']
        Y = idr.data[sn][key + '.Y']
        V = idr.data[sn][key + '.V']

        Fratio = N.array([OneSpecFLuxRatio(x, y, v, wmin=wmin,
                                           wmax=wmax, num=num)
                          for x, y, v in zip(X, Y, V)])

        idr.data[sn]['fratio.X'] = N.array([fratio.new_x
                                            for fratio in Fratio])
        idr.data[sn]['fratio.Y'] = N.array([fratio.ratios
                                            for fratio in Fratio])
        idr.data[sn]['fratio.V'] = N.array([fratio.ratiose**2
                                            for fratio in Fratio])

        # all the velocity are the same, take only the last one
        idr.data[sn]['fratio.velocity'] = fratio.velocity
        idr.data[sn]['fratio.objects'] = Fratio


def mag_ratio_corr_map(idr, phase=0, window=1, w_mag=4000):
    """
    options:
        phase: spectra choosen as close as possible to this phase...
        window: ... in this windows range.
        w_mag: magnitude is choosen as close as possible to 'w_mag'

    WARNING: the mean value of the magnitude distribution is
             subtracted to blind the analysis.

    """
    d = idr.data  # shortcut

    Xr, Fr, Vr, phases, sne = idr.get_data_at_phase(phase=phase,
                                                    window=window,
                                                    data='fratio')

    # Select magnitude of the closest spectra to the choosen phase
    i = N.argmin(N.abs(d[d.keys()[0]]['mag.X'][0] - w_mag))
    mags = N.array([d[sn]['mag.Y'][N.argmin(N.abs(d[sn]['data.phases']
                                                  - phase))][i] for sn in sne])
    corr = CorrMap(Fr, mags, x=Xr[0])

    return corr


def get_column_vals(maps, i, j):
    return N.array([mapi[i][j] for mapi in maps])


def plot_matrix(map, wlength, title='', blabel='', cmap=P.cm.jet):
    """
    map: the map
    wlength: the wavelength
    title: the title
    blabel: the colorbar label
    cmap: the P.cm.* color map
    """

    label = r'Wavelength [$\AA$]'
    wlength = [wlength[0], wlength[-1], wlength[-1], wlength[0]]

    # Plot the matrix
    fig = P.figure(dpi=100)
    ax = fig.add_axes([0.08, 0.09, 0.88, 0.86], title=title)
    im = ax.imshow(map, cmap=cmap, extent=wlength, interpolation='nearest')
    cb = fig.colorbar(im)
    cb.set_label(blabel, size='x-large')
    ax.set_xlabel(label, size='large')
    ax.set_ylabel(label, size='large')


# End of FluxRatio.py

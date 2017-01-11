#!/usr/bin/env python

"""Class to use spin."""

import sys
import time
import numpy as N
import matplotlib.pyplot as P

from snspin.spin import Craniometer, get_cranio
from snspin.spectrum import merge


class DrGall(object):

    """Class to manipulate and use the craniometer."""

    def __init__(self, spec=None, specb=None, specr=None,
                 spec_merged=None, verbose=True):
        """
        Spectrum initialization.

        Create self.x, self.y, self.v if spec is the merged spectrum
        Create self.xB, self.yB, self.vB for the blue channel if specB is given
        Create self.xr, self.yr, self.vr for the blue channel if specr is given
        """
        self.values_initialization()

        self.xb = None
        self.xr = None
        self.x_merged = None

        if spec is not None:
            if spec.x[0] < 4000 and spec.x[-1] > 6500:
                self.x = N.array(spec.x)
                self.y = N.array(spec.y)
                self.v = N.array(spec.v)
                self.xb = N.array(spec.x)
                self.yb = N.array(spec.y)
                self.vb = N.array(spec.v)
                self.xr = N.array(spec.x)
                self.yr = N.array(spec.y)
                self.vr = N.array(spec.v)
                self.x_merged = N.array(spec.x)
                self.y_merged = N.array(spec.y)
                self.v_merged = N.array(spec.v)
            elif spec.x[0] < 4000 and spec.x[-1] < 6500:
                self.xb = N.array(spec.x)
                self.yb = N.array(spec.y)
                self.vb = N.array(spec.v)
            elif spec.x[0] > 4000 and spec.x[-1] > 6500:
                self.xr = N.array(spec.x)
                self.yr = N.array(spec.y)
                self.vr = N.array(spec.v)
            if verbose:
                print >> sys.stderr, 'Working on merged spectrum'

        elif specb or specr:
            if (specb and specb.x[0] > 4000) or (specr and specr.x[0] < 4000):
                print >> sys.stderr, 'Error, check if B channel is really B '\
                    'channel and not r channel'
                return
            try:
                self.xb = N.array(specb.x)
                self.yb = N.array(specb.y)
                self.vb = N.array(specb.v)
            except ValueError:
                pass
            try:
                self.xr = N.array(specr.x)
                self.yr = N.array(specr.y)
                self.vr = N.array(specr.v)
            except ValueError:
                pass

            if self.xb is not None and self.xr is not None:
                try:
                    spec_merged = merge.MergedSpectrum(specb, specr)
                    self.x_merged = N.array(spec_merged.x)
                    self.y_merged = N.array(spec_merged.y)
                    self.v_merged = N.array(spec_merged.v)
                except ValueError:
                    print >> sys.stderr, 'Merged spectrum failure'

            if verbose:
                if self.xb is not None and self.xr is not None:
                    print >> sys.stderr, 'Work on B and r channel'
                elif self.xb is not None and self.xr is None:
                    print >> sys.stderr, 'Work only on B channel'
                elif self.xb is None and self.xr is not None:
                    print >> sys.stderr, 'Work only on r channel'
                elif self.xb is None \
                        and self.xr is None \
                        and not hasattr(self, 'x'):
                    print >> sys.stderr, 'Work on merged spectrum'
                else:
                    print >> sys.stderr, 'ErrOr, no correct input in DrGall. '\
                        'Give me a spectrum (for instance; spec with '\
                        'spec.x, spec.y and spec.v)'
                    sys.exit()

        else:
            print >> sys.stderr, 'ErrOr, no correct input in DrGall. Give me a'\
                'spectrum (for instance; spec with spec.x, spec.y and spec.v)'
            sys.exit()

    def values_initialization(self, verbose=False):
        """Initialize all values."""
        values = {}
        # Initialisation craniomter
        fake_lbd = range(3000, 10000, 2)
        cranio = Craniometer(fake_lbd,
                             N.ones(len(fake_lbd)),
                             N.ones(len(fake_lbd)))
        cranio.init_only = True

        # Create values
        cranio.rca(verbose=verbose)
        cranio.rcas(verbose=verbose)
        cranio.rcas2(verbose=verbose)
        cranio.rsi(verbose=verbose)
        cranio.rsis(verbose=verbose)
        cranio.rsiss(verbose=verbose)
        cranio.ew(3504, 3687, 3887, 3990, 'caiiHK', verbose=verbose)
        cranio.ew(3830, 3963, 4034, 4150, 'siii4000', verbose=verbose)
        cranio.ew(4034, 4150, 4452, 4573, 'mgii', verbose=verbose)
        cranio.ew(5085, 5250, 5500, 5681, 'SiiW', verbose=verbose)
        cranio.ew(5085, 5250, 5250, 5450, 'SiiW_L', verbose=verbose)
        cranio.ew(5250, 5450, 5500, 5681, 'SiiW_r', verbose=verbose)
        cranio.ew(5550, 5681, 5850, 6015, 'siii5972', verbose=verbose)
        cranio.ew(5850, 6015, 6250, 6365, 'siii6355', verbose=verbose)
        cranio.ew(7100, 7270, 7720, 8000, 'oi7773', verbose=verbose)
        cranio.ew(7720, 8000, 8300, 8800, 'caiiir', verbose=verbose)
        cranio.ew(4400, 4650, 5050, 5300, 'fe4800', verbose=verbose)
        cranio.velocity({'lmin': 3963,
                         'lmax': 4034,
                         'lrest': 4128,
                         'name': 'vsiii_4128'},
                        verbose=verbose)
        cranio.velocity({'lmin': 5200,
                         'lmax': 5350,
                         'lrest': 5454,
                         'name': 'vsiii_5454'},
                        verbose=verbose)
        cranio.velocity({'lmin': 5351,
                         'lmax': 5550,
                         'lrest': 5640,
                         'name': 'vsiii_5640'},
                        verbose=verbose)
        cranio.velocity({'lmin': 5700,
                         'lmax': 5900,
                         'lrest': 5972,
                         'name': 'vsiii_5972'},
                        verbose=verbose)
        cranio.velocity({'lmin': 6000,
                         'lmax': 6210,
                         'lrest': 6355,
                         'name': 'vsiii_6355'},
                        verbose=verbose)

        # Update values
        values.update(cranio.rcavalues)
        values.update(cranio.rcasvalues)
        values.update(cranio.rcas2values)
        values.update(cranio.rsivalues)
        values.update(cranio.rsisvalues)
        values.update(cranio.rsissvalues)
        values.update(cranio.velocityvalues)
        values.update(cranio.ewvalues)

        self.values = values

    def calcium_computing(self, smoother="sgfilter", verbose=False, nsimu=1000):
        """
        Function to compute and return all spectral indicators in the calcium zone.

        (Blue part of the spectrum, B channel)
        """
        # Test if computing is possible
        if self.xb is None:
            print >> sys.stderr, 'ErrOr, impossible to compute spectral '\
                'indictors defined in calcium zone (maybe no B channel)'
            indicators = {'edca': [N.nan, N.nan],
                          'rca': [N.nan, N.nan],
                          'rcas': [N.nan, N.nan],
                          'rcas2': [N.nan, N.nan],
                          'ewcaiiHK': [N.nan, N.nan],
                          'ewsiii4000': [N.nan, N.nan],
                          'ewmgii': [N.nan, N.nan]}
            return indicators

        # Create zone and craniometers
        cazone = (self.xb > 3450) & (self.xb < 4070)
        sizone = (self.xb > 3850) & (self.xb < 4150)
        mgzone = (self.xb > 4000) & (self.xb < 4610)

        self.cranio_bca = get_cranio(self.xb[cazone],
                                     self.yb[cazone],
                                     self.vb[cazone],
                                     smoother=smoother,
                                     verbose=verbose,
                                     nsimu=nsimu)
        self.cranio_bsi = get_cranio(self.xb[sizone],
                                     self.yb[sizone],
                                     self.vb[sizone],
                                     smoother=smoother,
                                     verbose=verbose,
                                     nsimu=nsimu)
        self.cranio_bmg = get_cranio(self.xb[mgzone],
                                     self.yb[mgzone],
                                     self.vb[mgzone],
                                     smoother=smoother,
                                     verbose=verbose,
                                     nsimu=nsimu)

        rca = self.cranio_bca.rca(verbose=verbose)
        try:
            rca = self.cranio_bca.rca(verbose=verbose)
            self.values.update(self.cranio_bca.rcavalues)
            if verbose:
                print 'rca computing done, rca =', rca
        except ValueError:
            rca = [N.nan, N.nan]
            if verbose:
                print 'Error in rca computing, rca =', rca

        try:
            rcas = self.cranio_bca.rcas(verbose=verbose)
            self.values.update(self.cranio_bca.rcasvalues)
            if verbose:
                print 'rcas computing done, rcas =', rcas
        except ValueError:
            rcas = [N.nan, N.nan]
            if verbose:
                print 'Error in rcas computing, rcas =', rcas

        try:
            rcas2 = self.cranio_bca.rcas2(verbose=verbose)
            self.values.update(self.cranio_bca.rcas2values)
            if verbose:
                print 'rcas2 computing done, rcas2 =', rcas2
        except ValueError:
            rcas2 = [N.nan, N.nan]
            if verbose:
                print 'Error in rcas2 computing, rcas2 =', rcas2

        try:
            ewcaiiHK = self.cranio_bca.ew(3504, 3687, 3830, 3990, 'caiiHK', sup=True, right1=True,
                                          verbose=verbose)
            self.values.update(self.cranio_bca.ewvalues)
            if verbose:
                print 'ewcaiiHK computing done, ewcaiiHK =', ewcaiiHK
        except ValueError:
            ewcaiiHK = [N.nan, N.nan]
            if verbose:
                print 'Error in ewcaiiHK computing, ewcaiiHK =', ewcaiiHK

        try:
            ewsiii4000 = self.cranio_bsi.ew(3830, 3990, 4030, 4150,
                                            'siii4000',
                                            sup=True,
                                            verbose=verbose)
            self.values.update(self.cranio_bsi.ewvalues)
            if verbose:
                print 'ewsiii4000 computing done, ewsiii4000 =', ewsiii4000
        except ValueError:
            ewsiii4000 = [N.nan, N.nan]
            if verbose:
                print 'Error in ewsiii4000 computing ewsiii4000 =', ewsiii4000

        try:
            ewmgii = self.cranio_bmg.ew(4030, 4150, 4450, 4650,
                                        'mgii',
                                        sup=True,
                                        left2=True,
                                        verbose=verbose)
            self.values.update(self.cranio_bmg.ewvalues)
            if verbose:
                print 'ewmgii computing done, ewmgii = ', ewmgii
        except ValueError:
            ewmgii = [N.nan, N.nan]
            if verbose:
                print 'Error in ewmgii computing, ewmgii =', ewmgii

        try:
            vsiii_4000 = self.cranio_bsi.velocity({'lmin': 3963,
                                                   'lmax': 4034,
                                                   'lrest': 4128,
                                                   'name': 'vsiii_4128'},
                                                  verbose=verbose)
            self.values.update(self.cranio_bsi.velocityvalues)
            if verbose:
                print 'vsiii_4128 computing done, vsiii_4000 =', vsiii_4000
        except ValueError:
            vsiii_4000 = [N.nan, N.nan]
            if verbose:
                print 'Error in vsiii_4128 computing, vsiii_4000', vsiii_4000

        if verbose:
            print >> sys.stderr, 'Computing on calcium zone for this '\
                'spectrum done\n'

        indicators = {'rca': rca,
                      'rcas2': rcas2,
                      'ewcaiiHK': ewcaiiHK,
                      'ewsiii4000': ewsiii4000,
                      'ewmgii': ewmgii,
                      'vsiii4128': vsiii_4000}

        del self.cranio_bca.simulations
        del self.cranio_bca.syst
        del self.cranio_bsi.simulations
        del self.cranio_bsi.syst
        del self.cranio_bmg.simulations
        del self.cranio_bmg.syst

        return indicators

    def silicon_computing(self, smoother="sgfilter", verbose=False, nsimu=1000):
        """
        Function to compute and retunr all spectral indicators in the silicon
        zone
        """
        # Test if computing is possible
        if self.xr is None:
            print >> sys.stderr, 'Error, impossible to compute spectral '\
                'indictors defined in calcium zone (maybe no r channel)'
            indicators = {'edca': [N.nan, N.nan],
                          'rca': [N.nan, N.nan],
                          'rcas': [N.nan, N.nan],
                          'rcas2': [N.nan, N.nan],
                          'ewcaiiHK': [N.nan, N.nan],
                          'ewsiii4000': [N.nan, N.nan],
                          'ewmgii': [N.nan, N.nan],
                          'vsiii_5972': [N.nan, N.nan],
                          'vsiii_6355': [N.nan, N.nan]}
            return indicators

        # Create zone and craniometers
        zone1 = (self.xr > 5500) & (self.xr < 6400)
        zone2 = (self.xr > 5060) & (self.xr < 5700)
        zone3 = (self.xr > 5500) & (self.xr < 6050)
        zone4 = (self.xr > 5800) & (self.xr < 6400)
        zone5 = (self.xr > 5480) & (self.xr < 6500)
        self.cranio_r1 = get_cranio(self.xr[zone1],
                                    self.yr[zone1],
                                    self.vr[zone1],
                                    smoother=smoother,
                                    nsimu=nsimu,
                                    verbose=verbose)  # rsi, rsis
        self.cranio_r2 = get_cranio(self.xr[zone2],
                                    self.yr[zone2],
                                    self.vr[zone2],
                                    smoother=smoother,
                                    nsimu=nsimu,
                                    verbose=verbose)  # ewSiiW
        self.cranio_r3 = get_cranio(self.xr[zone3],
                                    self.yr[zone3],
                                    self.vr[zone3],
                                    smoother=smoother,
                                    nsimu=nsimu,
                                    verbose=verbose)  # ewsiii5972
        self.cranio_r4 = get_cranio(self.xr[zone4],
                                    self.yr[zone4],
                                    self.vr[zone4],
                                    smoother=smoother,
                                    verbose=verbose)  # ewsiii6355
        self.cranio_r5 = get_cranio(self.xr[zone5],
                                    self.yr[zone5],
                                    self.vr[zone5],
                                    smoother=smoother,
                                    nsimu=nsimu,
                                    verbose=verbose)  # rsiss

        try:
            rsi = self.cranio_r1.rsi(verbose=verbose)
            self.values.update(self.cranio_r1.rsivalues)
            if verbose:
                print 'rsi computing done, rsi =', rsi
        except ValueError:
            rsi = [N.nan, N.nan]
            if verbose:
                print 'Error in rsi computing, rsi =', rsi

        try:
            rsis = self.cranio_r1.rsis(verbose=verbose)
            self.values.update(self.cranio_r1.rsisvalues)
            if verbose:
                print 'rsis computing done, rsis =', rsis
        except ValueError:
            rsis = [N.nan, N.nan]
            if verbose:
                print 'Error in rsis computing, rsis =', rsis

        try:
            rsiss = self.cranio_r5.rsiss(verbose=verbose)
            self.values.update(self.cranio_r5.rsissvalues)
            if verbose:
                print 'rsiss computing done, rsiss =', rsiss
        except ValueError:
            rsiss = [N.nan, N.nan]
            if verbose:
                print 'Error in rsiss computing, rsiss =', rsiss

        try:
            ewSiiW = self.cranio_r2.ew(5050, 5285, 5500, 5681,
                                       'SiiW',
                                       sup=True,
                                       # right1=True,
                                       verbose=verbose)
            if verbose:
                print 'ewSiiW computing done, ewSiiW =', ewSiiW
        except ValueError:
            ewSiiW = [N.nan, N.nan]
            if verbose:
                print 'Error in ewSiiW computing, ewSiiW =', ewSiiW

        try:
            ewSiiW_L = self.cranio_r2.ew(5085, 5250, 5250, 5450,
                                         'SiiW_L',
                                         sup=True,
                                         right1=True,
                                         verbose=verbose)
            if verbose:
                print 'ewSiiW_L computing done, ewSiiW_L =', ewSiiW_L
        except ValueError:
            ewSiiW_L = [N.nan, N.nan]
            if verbose:
                print 'Error in ewSiiW_L computing, ewSiiW_L =', ewSiiW_L

        try:
            ewSiiW_r = self.cranio_r2.ew(5250, 5450, 5500, 5681,
                                         'SiiW_r',
                                         sup=True,
                                         verbose=verbose)
            if verbose:
                print 'ewSiiW_r computing done, ewSiiW_r =', ewSiiW_r
        except ValueError:
            ewSiiW_r = [N.nan, N.nan]
            if verbose:
                print 'Error in ewSiiW_r computing, ewSiiW_r =', ewSiiW_r

        try:
            self.values.update(self.cranio_r2.ewvalues)
        except ValueError:
            pass

        try:
            ewsiii5972 = self.cranio_r3.ew(5550, 5681, 5850, 6015,
                                           'siii5972',
                                           sup=True,
                                           right2=True,
                                           verbose=verbose)
            self.values.update(self.cranio_r3.ewvalues)
            if verbose:
                print 'ewsiii5972 computing done, ewsiii5972 =', ewsiii5972
        except ValueError:
            ewsiii5972 = [N.nan, N.nan]
            if verbose:
                print 'Error in ewsiii5972 computing, ewsiii5972 =', ewsiii5972
        try:
            ewsiii6355 = self.cranio_r4.ew(5850, 6015, 6250, 6365,
                                           'siii6355',
                                           right1=True,
                                           sup=True,
                                           verbose=verbose)
            self.values.update(self.cranio_r4.ewvalues)
            if verbose:
                print 'ewsiii6355 computing done, ewsiii6355 =', ewsiii6355
        except ValueError:
            ewsiii6355 = [N.nan, N.nan]
            if verbose:
                print 'Error in ewsiii6355 computing, ewsiii6355 =', ewsiii6355

        try:
            vsiii_5454 = self.cranio_r2.velocity({'lmin': 5200,
                                                  'lmax': 5350,
                                                  'lrest': 5454,
                                                  'name': 'vsiii_5454'},
                                                 verbose=verbose)
            self.values.update(self.cranio_r2.velocityvalues)
            if verbose:
                print 'vsiii_5454 computing done, vsiii_5454 =', vsiii_5454
        except ValueError:
            vsiii_5454 = [N.nan, N.nan]
            if verbose:
                print 'Error in vsiii_5454 computing, vsiii_5454 =', vsiii_5454

        try:
            vsiii_5640 = self.cranio_r2.velocity({'lmin': 5351,
                                                  'lmax': 5550,
                                                  'lrest': 5640,
                                                  'name': 'vsiii_5640'},
                                                 verbose=verbose)
            self.values.update(self.cranio_r2.velocityvalues)
            if verbose:
                print 'vsiii_5640 computing done, vsiii_5640 =', vsiii_5640
        except ValueError:
            vsiii_5640 = [N.nan, N.nan]
            if verbose:
                print 'Error in vsiii_5640 computing, vsiii_5640 =', vsiii_5640

        try:
            vsiii_5972 = self.cranio_r3.velocity({'lmin': 5700,
                                                  'lmax': 5875,
                                                  'lrest': 5972,
                                                  'name': 'vsiii_5972'},
                                                 verbose=verbose)
            self.values.update(self.cranio_r3.velocityvalues)
            if verbose:
                print 'vsiii_5972 computing done, vsiii_5972 =', vsiii_5972
        except ValueError:
            vsiii_5972 = [N.nan, N.nan]
            if verbose:
                print 'Error in vsiii_5972 computing, vsiii_5972 =', vsiii_5972

        try:
            vsiii_6355 = self.cranio_r4.velocity({'lmin': 6000,
                                                  'lmax': 6210,
                                                  'lrest': 6355,
                                                  'name': 'vsiii_6355'},
                                                 verbose=verbose)
            # vsiii_6355 = self.cranio_r4.velocity2({'lmin':5850, 'lmax':6015,
            # 'lrest':6355, 'name':'vsiii_6355'}, verbose=verbose)
            self.values.update(self.cranio_r4.velocityvalues)
            if verbose:
                print 'vsiii_6355 computing done, vsiii_6355 =', vsiii_6355
        except ValueError:
            vsiii_6355 = [N.nan, N.nan]
            if verbose:
                print 'Error in vsiii_6355 computing, vsiii_6355 =', vsiii_6355

        if verbose:
            print >> sys.stderr, 'Computing on silicon zone for this spectrum done'
            print ''.center(100, '=')

        indicators = {'rsi': rsi,
                      'rsis': rsis,
                      'rsiss': rsiss,
                      'ewSiiW': ewSiiW,
                      'ewsiii5972': ewsiii5972,
                      'ewsiii6355': ewsiii6355,
                      'vsiii_5972': vsiii_5972,
                      'vsiii_6355': vsiii_6355}

        del self.cranio_r1.simulations
        del self.cranio_r2.simulations
        del self.cranio_r3.simulations
        del self.cranio_r4.simulations
        del self.cranio_r1.syst
        del self.cranio_r2.syst
        del self.cranio_r3.syst
        del self.cranio_r4.syst

        return indicators

    def oxygen_computing(self, smoother="sgfilter", verbose=True, nsimu=1000):
        """
        Function to compute and return spectral indicators in the end of
        the spectrum
        """
        # Test if the computation will be possible
        if self.xr is None:
            print >> sys.stderr, 'Error, impossible to compute spectral '\
                'indictors defined in oxygen zone (maybe no r channel)'
            indicators = {'ewoi7773': [N.nan, N.nan],
                          'ewcaiiir': [N.nan, N.nan]}
            return indicators

        # Create zone and craniometers
        zone = (self.xr > 6500) & (self.xr < 8800)
        self.cranio_O = get_cranio(self.xr[zone],
                                   self.yr[zone],
                                   self.vr[zone],
                                   smoother=smoother,
                                   nsimu=nsimu,
                                   verbose=verbose)  # ewoi7773 and caiiir

        try:
            ewoi7773 = self.cranio_O.ew(7100, 7270, 7720, 8000,
                                        'oi7773',
                                        sup=True,
                                        verbose=verbose)
            if verbose:
                print 'ewoi7773 computing done, ewoi7773 =', ewoi7773
        except ValueError:
            ewoi7773 = [N.nan, N.nan]
            if verbose:
                print 'Error in ewoi7773 computing, ewoi7773 =', ewoi7773

        try:
            ewcaiiir = self.cranio_O.ew(7720, 8000, 8300, 8800,
                                        'caiiir',
                                        sup=True,
                                        verbose=verbose)
            if verbose:
                print 'ewcaiiir computing done, ewcaiiir =', ewcaiiir
        except ValueError:
            ewcaiiir = [N.nan, N.nan]
            if verbose:
                print 'Error in ewcaiiir computing, ewcaiiir =', ewcaiiir

        try:
            self.values.update(self.cranio_O.ewvalues)
        except ValueError:
            pass

        if verbose:
            print >> sys.stderr, 'Computing on oxygen zone for this '\
                'spectrum done'
            print ''.center(100, '=')

        indicators = {'ewoi7773': ewoi7773,
                      'ewcaiiir': ewcaiiir}

        del self.cranio_O.simulations
        del self.cranio_O.syst

        return indicators

    def iron_computing(self, smoother="sgfilter", verbose=True, nsimu=1000):
        """
        Function to compute and return spectral indicators on the middle of the spectrum (iron zone).
        """
        # Test if the computation will be possible
        if self.x_merged is None:
            print >> sys.stderr, 'Error, impossible to compute spectral '\
                'indictors defined in iron zone (maybe no r or b channel)'
            indicators = {'ewfe4800': [N.nan, N.nan]}
            return indicators

        # Create zone and craniometers
        zone = (self.x_merged > 4350) & (self.x_merged < 5350)
        self.cranio_fe = get_cranio(self.x_merged[zone],
                                    self.y_merged[zone],
                                    self.v_merged[zone],
                                    smoother=smoother,
                                    nsimu=nsimu,
                                    verbose=verbose)  # ewfe4800

        try:
            ewfe4800 = self.cranio_fe.ew(4450, 4650, 5050, 5285,
                                         'fe4800',
                                         sup=True,
                                         left2=True,
                                         verbose=verbose)
            if verbose:
                print 'ewfe4800 computing done, ewfe4800 =', ewfe4800
        except ValueError:
            ewfe4800 = [N.nan, N.nan]
            if verbose:
                print 'Error in ewfe4800 computing, ewfe4800 =', ewfe4800

        try:
            self.values.update(self.cranio_fe.ewvalues)
        except ValueError:
            pass

        if verbose:
            print >> sys.stderr, 'Computing on iron zone for this spectrum done'
            print ''.center(100, '=')

        indicators = {'ewfe4800': ewfe4800}

        del self.cranio_fe.simulations
        del self.cranio_fe.syst

        return indicators

    def initialize_parameters(self, verbose=True):
        """
        Function to initialize parameters use to make the control_plot
        """
        try:
            rsi = self.cranio_r1.rsivalues['rsi']
        except ValueError:
            rsi = float(N.nan)

        try:
            rsis = self.cranio_r1.rsisvalues['rsis']
        except ValueError:
            rsis = float(N.nan)

        try:
            rsiss = self.cranio_r5.rsissvalues['rsiss']
        except ValueError:
            rsiss = float(N.nan)

        try:
            rca = self.cranio_bca.rcavalues['rca']
        except ValueError:
            rca = float(N.nan)

        try:
            rcas = self.cranio_bca.rcasvalues['rcas']
        except ValueError:
            rcas = float(N.nan)

        try:
            rcas2 = self.cranio_bca.rcas2values['rcas2']
        except ValueError:
            rcas2 = float(N.nan)

        # try:
        #    edca = self.cranio_bca.edcavalues['edca']
        # except ValueError:
        edca = float(N.nan)

        try:
            ewcaiiHK = self.cranio_bca.ewvalues['ewcaiiHK']
        except ValueError:
            ewcaiiHK = float(N.nan)

        try:
            ewsiii4000 = self.cranio_bsi.ewvalues['ewsiii4000']
        except ValueError:
            ewsiii4000 = float(N.nan)

        try:
            ewmgii = self.cranio_bmg.ewvalues['ewmgii']
        except ValueError:
            ewmgii = float(N.nan)

        try:
            ewSiiW = self.cranio_r2.ewvalues['ewSiiW']
        except ValueError:
            ewSiiW = float(N.nan)

        try:
            ewSiiW_L = self.cranio_r2.ewvalues['ewSiiW_L']
        except ValueError:
            ewSiiW_L = float(N.nan)

        try:
            ewSiiW_r = self.cranio_r2.ewvalues['ewSiiW_r']
        except ValueError:
            ewSiiW_r = float(N.nan)

        try:
            ewsiii5972 = self.cranio_r3.ewvalues['ewsiii5972']
        except ValueError:
            ewsiii5972 = float(N.nan)

        try:
            ewsiii6355 = self.cranio_r4.ewvalues['ewsiii6355']
        except ValueError:
            ewsiii6355 = float(N.nan)

        try:
            vsiii_5972 = self.cranio_r3.velocityvalues['vsiii_5972']
        except ValueError:
            vsiii_5972 = float(N.nan)

        try:
            vsiii_6355 = self.cranio_r4.velocityvalues['vsiii_6355']
        except ValueError:
            vsiii_6355 = float(N.nan)

        return rsi, rsis, rsiss, rca, rcas, rcas2, edca, ewcaiiHK, ewsiii4000, ewmgii, \
            ewSiiW, ewsiii5972, ewsiii6355, vsiii_5972, vsiii_6355, ewSiiW_L, \
            ewSiiW_r

    #=========================================================================
    # Functions to plot control_plot of spectral indicators computing
    #=========================================================================

    def plot_craniobca(self, metrics, ax=None, filename=''):
        """Plot zone where rca, rcas, rcas2, edca and ewcaiiHK are computed"""

        rsi, rsis, rsiss, rca, rcas, rcas2, edca, ewcaiiHK, ewsiii4000, ewmgii, ewSiiW, \
            ewsiii5972, ewsiii6355, vsiii_5972, vsiii_6355, ewSiiW_L, ewSiiW_r = metrics
        cr = self.cranio_bca

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
            save = True
        else:
            save = False

        ax.plot(cr.x, cr.y, color='k', label='Flux')
        try:
            ax.plot(cr.x, cr.s, color='r', label='Interpolated flux')
        except ValueError:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        try:  # Plot the rcas vspan
            ax.axvspan(cr.rcasvalues['rcas_lbd'][0],
                       cr.rcasvalues['rcas_lbd'][1],
                       ymin=0, ymax=1, facecolor='y', alpha=0.25)
            ax.axvspan(cr.rcasvalues['rcas_lbd'][2],
                       cr.rcasvalues['rcas_lbd'][3],
                       ymin=0, ymax=1, facecolor='y', alpha=0.25)
        except ValueError:
            print >> sys.stderr, "No parameters to plot rcas zone"

        try:  # Plot the ewcaiiHK points and lines
            lbd_line = cr.x[(cr.x >= cr.ewvalues['lbd_ewcaiiHK'][0])
                            & (cr.x <= cr.ewvalues['lbd_ewcaiiHK'][1])]
            p_line = N.polyfit([cr.ewvalues['lbd_ewcaiiHK'][0],
                                cr.ewvalues['lbd_ewcaiiHK'][1]],
                               [cr.smoother(cr.ewvalues['lbd_ewcaiiHK'])[0],
                                cr.smoother(cr.ewvalues['lbd_ewcaiiHK'])[1]], 1)
            ax.scatter(cr.rcavalues['rca_lbd'],
                       cr.smoother(cr.rcavalues['rca_lbd']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewcaiiHK zone"

        try:  # Plot the rca lines
            for x, y in zip(cr.rcavalues['rca_lbd'],
                            cr.smoother(cr.rcavalues['rca_lbd'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot rca zone"

        # Annotate the ca zone with spectral indicators values
        try:
            ax.annotate('rca=%.2f, rcas=%.2f, rcas2=%.2f' %
                        (rca, rcas, rcas2), xy=(0.01, 0.01),
                        xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
            ax.annotate('ewcaiiHK=%.2f' %
                        (ewcaiiHK), xy=(0.01, 0.95), xycoords='axes fraction',
                        xytext=(0.01, 0.95), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except ValueError:
            pass
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=3450, xmax=4070)

        if save:
            fig.savefig('calcium_' + filename)

    def plot_craniobsi(self, metrics, ax=None, filename=''):
        """Plot zone where ewsi4000 is computed"""

        rsi, rsis, rsiss, rca, rcas, rcas2, edca, ewcaiiHK, ewsiii4000, ewmgii, ewSiiW, \
            ewsiii5972, ewsiii6355, vsiii_5972, vsiii_6355, ewSiiW_L, ewSiiW_r = metrics
        cr = self.cranio_bsi

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
            save = True
        else:
            save = False

        ax.plot(cr.x, cr.y, color='k', label='Flux')
        try:
            ax.plot(cr.x, cr.s, color='r', label='Interpolated flux')
        except ValueError:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.ewvalues['lbd_ewsiii4000'][0])
                            & (cr.x <= cr.ewvalues['lbd_ewsiii4000'][1])]
            p_line = N.polyfit([cr.ewvalues['lbd_ewsiii4000'][0],
                                cr.ewvalues['lbd_ewsiii4000'][1]],
                               [cr.smoother(cr.ewvalues['lbd_ewsiii4000'])[0],
                                cr.smoother(cr.ewvalues['lbd_ewsiii4000'])[1]],
                               1)
            ax.scatter(cr.ewvalues['lbd_ewsiii4000'],
                       cr.smoother(cr.ewvalues['lbd_ewsiii4000']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except ValueError:
            print >> sys.stderr, "No parameters to plot edca straight line"

        try:  # Plot vlines
            for x, y in zip(cr.ewvalues['lbd_ewsiii4000'],
                            cr.smoother(cr.ewvalues['lbd_ewsiii4000'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot rca vlines"

        # Annotate the ca zone with spectral indicators values
        try:
            ax.annotate('ewsiii4000=%.2f' %
                        (ewsiii4000), xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except ValueError:
            pass
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=3850, xmax=4150)

        if save:
            fig.savefig('ewsiii4000_' + filename)

    def plot_craniobmg(self, metrics, ax=None, filename=''):
        """Plot zone where ewmgii is computed"""

        rsi, rsis, rsiss, rca, rcas, rcas2, edca, ewcaiiHK, ewsiii4000, ewmgii, ewSiiW, \
            ewsiii5972, ewsiii6355, vsiii_5972, vsiii_6355, ewSiiW_L, ewSiiW_r = metrics
        cr = self.cranio_bmg

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
            save = True
        else:
            save = False

        ax.plot(cr.x, cr.y, color='k', label='Flux')
        try:
            ax.plot(cr.x, cr.s, color='r', label='Interpolated flux')
        except ValueError:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.ewvalues['lbd_ewmgii'][0])
                            & (cr.x <= cr.ewvalues['lbd_ewmgii'][1])]
            p_line = N.polyfit([cr.ewvalues['lbd_ewmgii'][0],
                                cr.ewvalues['lbd_ewmgii'][1]],
                               [cr.smoother(cr.ewvalues['lbd_ewmgii'])[0],
                                cr.smoother(cr.ewvalues['lbd_ewmgii'])[1]], 1)
            ax.scatter(cr.ewvalues['lbd_ewmgii'],
                       cr.smoother(cr.ewvalues['lbd_ewmgii']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewmgii straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.ewvalues['lbd_ewmgii'],
                            cr.smoother(cr.ewvalues['lbd_ewmgii'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewmgii vlines"

        # Annotate the ca zone with spectral indicators values
        try:
            ax.annotate('ewmgii=%.2f' %
                        (ewmgii), xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except ValueError:
            pass
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=4000, xmax=4600)

        if save:
            fig.savefig('ewmgii_' + filename)

    def plot_cranior1r5(self, metrics, ax=None, filename=''):
        """Plot zone where rca, rcas, rcas2, edca and ewcaiiHK are computed"""

        rsi, rsis, rsiss, rca, rcas, rcas2, edca, ewcaiiHK, ewsiii4000, ewmgii, ewSiiW, \
            ewsiii5972, ewsiii6355, vsiii_5972, vsiii_6355, ewSiiW_L, ewSiiW_r = metrics
        cr1 = self.cranio_r1
        cr5 = self.cranio_r5

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
            save = True
        else:
            save = False

        ax.plot(cr5.x, cr5.y, color='k', label='Flux')
        try:
            ax.plot(cr1.x, cr1.s, color='r', label='Interpolated flux')
        except ValueError:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        # try: #Plot the rsiss vspan
        ax.axvspan(cr5.rsissvalues['rsiss_lbd'][0],
                   cr5.rsissvalues['rsiss_lbd'][1],
                   ymin=0, ymax=1, facecolor='y', alpha=0.25)
        ax.axvspan(cr5.rsissvalues['rsiss_lbd'][2],
                   cr5.rsissvalues['rsiss_lbd'][3],
                   ymin=0, ymax=1, facecolor='y', alpha=0.25)
        # except ValueError: print >> sys.stderr, "No parameters to plot rsiss
        # zone"
        if N.isfinite(cr1.rsivalues['rsi']):
            # Plot the rsi points and lines
            lbd_line1 = cr1.x[(cr1.x >= cr1.rsivalues['rsi_lbd'][0])
                              & (cr1.x <= cr1.rsivalues['rsi_lbd'][2])]
            lbd_line2 = cr1.x[(cr1.x >= cr1.rsivalues['rsi_lbd'][2])
                              & (cr1.x <= cr1.rsivalues['rsi_lbd'][4])]
            p_line1 = N.polyfit([cr1.rsivalues['rsi_lbd'][0],
                                 cr1.rsivalues['rsi_lbd'][2]],
                                [cr1.smoother(cr1.rsivalues['rsi_lbd'])[0],
                                 cr1.smoother(cr1.rsivalues['rsi_lbd'])[2]], 1)
            p_line2 = N.polyfit([cr1.rsivalues['rsi_lbd'][2],
                                 cr1.rsivalues['rsi_lbd'][4]],
                                [cr1.smoother(cr1.rsivalues['rsi_lbd'])[2],
                                 cr1.smoother(cr1.rsivalues['rsi_lbd'])[4]], 1)
            ax.scatter(cr1.rsivalues['rsi_lbd'],
                       cr1.smoother(cr1.rsivalues['rsi_lbd']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line1, N.polyval(p_line1, lbd_line1), color='g')
            ax.plot(lbd_line2, N.polyval(p_line2, lbd_line2), color='g')

            for x, y in zip(cr1.rsivalues['rsi_lbd'], # Plot the rsi and rsis lines
                            cr1.smoother(cr1.rsivalues['rsi_lbd'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        else:
            print >> sys.stderr, "No parameters to plot rsi zone"

        # Annotate the ca zone with spectral indicators values
        try:
            ax.annotate('rsi=%.2f, rsis=%.2f, rsiss=%.2f' %
                        (rsi, rsis, rsiss), xy=(0.01, 0.01),
                        xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except ValueError:
            pass

        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=5480, xmax=6500)

        if save:
            fig.savefig('silicon_' + filename)

    def plot_cranior2(self, metrics, ax=None, filename=''):
        """Plot zone where ewSiiW is computed"""

        rsi, rsis, rsiss, rca, rcas, rcas2, edca, ewcaiiHK, ewsiii4000, ewmgii, ewSiiW, \
            ewsiii5972, ewsiii6355, vsiii_5972, vsiii_6355, ewSiiW_L, ewSiiW_r = metrics
        cr = self.cranio_r2

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
            save = True
        else:
            save = False

        ax.plot(cr.x, cr.y, color='k', label='Flux')
        try:
            ax.plot(cr.x, cr.s, color='r', label='Interpolated flux')
        except ValueError:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        # For ewsiW
        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.ewvalues['lbd_ewSiiW'][0])
                            & (cr.x <= cr.ewvalues['lbd_ewSiiW'][1])]
            p_line = N.polyfit([cr.ewvalues['lbd_ewSiiW'][0],
                                cr.ewvalues['lbd_ewSiiW'][1]],
                               [cr.smoother(cr.ewvalues['lbd_ewSiiW'])[0],
                                cr.smoother(cr.ewvalues['lbd_ewSiiW'])[1]], 1)
            ax.scatter(cr.ewvalues['lbd_ewSiiW'],
                       cr.smoother(cr.ewvalues['lbd_ewSiiW']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewSiiW straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.ewvalues['lbd_ewSiiW'],
                            cr.smoother(cr.ewvalues['lbd_ewSiiW'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewSiiW vlines"

        # For ewsiW_L
        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.ewvalues['lbd_ewSiiW_L'][0])
                            & (cr.x <= cr.ewvalues['lbd_ewSiiW_L'][1])]
            p_line = N.polyfit([cr.ewvalues['lbd_ewSiiW_L'][0],
                                cr.ewvalues['lbd_ewSiiW_L'][1]],
                               [cr.smoother(cr.ewvalues['lbd_ewSiiW_L'])[0],
                                cr.smoother(cr.ewvalues['lbd_ewSiiW_L'])[1]], 1)
            ax.scatter(cr.ewvalues['lbd_ewSiiW_L'],
                       cr.smoother(cr.ewvalues['lbd_ewSiiW_L']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewSiiW_L straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.ewvalues['lbd_ewSiiW_L'],
                            cr.smoother(cr.ewvalues['lbd_ewSiiW_L'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewSiiW_L vlines"

        # For ewsiW_r
        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.ewvalues['lbd_ewSiiW_r'][0])
                            & (cr.x <= cr.ewvalues['lbd_ewSiiW_r'][1])]
            p_line = N.polyfit([cr.ewvalues['lbd_ewSiiW_r'][0],
                                cr.ewvalues['lbd_ewSiiW_r'][1]],
                               [cr.smoother(cr.ewvalues['lbd_ewSiiW_r'])[0],
                                cr.smoother(cr.ewvalues['lbd_ewSiiW_r'])[1]], 1)
            ax.scatter(cr.ewvalues['lbd_ewSiiW_r'],
                       cr.smoother(cr.ewvalues['lbd_ewSiiW_r']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewSiiW_r straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.ewvalues['lbd_ewSiiW_r'],
                            cr.smoother(cr.ewvalues['lbd_ewSiiW_r'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewSiiW_r vlines"

        # Annotate the ca zone with spectral indicators values
        try:
            ax.annotate('ewSiiW=%.2f' %
                        (ewSiiW), xy=(0.01, 0.07), xycoords='axes fraction',
                        xytext=(0.01, 0.07), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
            ax.annotate('ewSiiW_L=%.2f, ewSiiW_r=%.2f' %
                        (ewSiiW_L, ewSiiW_r), xy=(0.01, 0.01),
                        xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except ValueError:
            pass

        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=5060, xmax=5700)
        print filename
        if save:
            fig.savefig('ewSiiW_' + filename)

    def plot_cranior3r4(self, metrics, ax=None, filename=''):
        """Plot zone where ewSiiW is computed"""

        rsi, rsis, rsiss, rca, rcas, rcas2, edca, ewcaiiHK, ewsiii4000, ewmgii, ewSiiW, \
            ewsiii5972, ewsiii6355, vsiii_5972, vsiii_6355, ewSiiW_L, ewSiiW_r = metrics
        cr3 = self.cranio_r3
        cr4 = self.cranio_r4

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
        except ValueError:
            print >> sys.stderr, "No smothing function computed for '\
            'ewsiii5972, so no smoothing function ploted"
        try:
            ax.plot(cr4.x, cr4.s, color='b', label='Interpolated flux')
        except ValueError:
            print >> sys.stderr, "No smothing function computed for '\
            'ewsiii6355, so no smoothing function ploted"

        try:  # Plot points and straight lines
            lbd_line = cr3.x[(cr3.x >= cr3.ewvalues['lbd_ewsiii5972'][0]) &
                             (cr3.x <= cr3.ewvalues['lbd_ewsiii5972'][1])]
            p_line = N.polyfit([cr3.ewvalues['lbd_ewsiii5972'][0],
                                cr3.ewvalues['lbd_ewsiii5972'][1]],
                               [cr3.smoother(cr3.ewvalues['lbd_ewsiii5972'])[0],
                                cr3.smoother(cr3.ewvalues['lbd_ewsiii5972'])[1]], 1)
            ax.scatter(cr3.ewvalues['lbd_ewsiii5972'],
                       cr3.smoother(cr3.ewvalues['lbd_ewsiii5972']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewsiii5972 straight '\
            'line zone"

        try:  # Plot points and straight lines
            lbd_line = cr4.x[(cr4.x >= cr4.ewvalues['lbd_ewsiii6355'][0]) &
                             (cr4.x <= cr4.ewvalues['lbd_ewsiii6355'][1])]
            p_line = N.polyfit([cr4.ewvalues['lbd_ewsiii6355'][0],
                                cr4.ewvalues['lbd_ewsiii6355'][1]],
                               [cr4.smoother(cr4.ewvalues['lbd_ewsiii6355'])[0],
                                cr4.smoother(cr4.ewvalues['lbd_ewsiii6355'])[1]], 1)
            ax.scatter(cr4.ewvalues['lbd_ewsiii6355'],
                       cr4.smoother(cr4.ewvalues['lbd_ewsiii6355']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewsiii6355 straight '\
            'line zone"

        try:  # Plot vlines for ewsiii5972
            for x, y in zip(cr3.ewvalues['lbd_ewsiii5972'],
                            cr3.smoother(cr3.ewvalues['lbd_ewsiii5972'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewsiii5972 vlines"

        try:  # Plot vlines for ewsiii6355
            for x, y in zip(cr4.ewvalues['lbd_ewsiii6355'],
                            cr4.smoother(cr4.ewvalues['lbd_ewsiii6355'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewsiii6355 vlines"

        # Annotate the si zone with spectral indicators values
        try:
            ax.annotate('ewsiii5972=%.2f, ewsiii6355=%.2f' %
                        (ewsiii5972, ewsiii6355),
                        xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left', verticalalignment='bottom',
                        fontsize=10)
        except ValueError:
            pass

        try:  # Plot vline for vsiii_6355
            ax.axvline(cr4.velocityvalues['vsiii_6355_lbd'],
                       color='k', lw=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot siii6355 vlines"

        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=5500, xmax=6400)

        if save:
            fig.savefig('ewsiii5972_' + filename)

    def plot_spectrum(self, metrics, ax=None, title=None):

        rsi, rsis, rsiss, rca, rcas, rcas2, edca, ewcaiiHK, ewsiii4000, ewmgii, ewSiiW, \
            ewsiii5972, ewsiii6355, vsiii_5972, vsiii_6355, ewSiiW_L, ewSiiW_r = metrics

        if ax is None:
            fig = P.figure()
            ax = fig.add_subplot(111)
        else:
            ax = ax

        # Plot spectrum===============================================
        ax.plot(self.x, self.y, color='k')
        ax.set_xlabel('Wavelength [AA]')
        ax.set_ylabel('Flux [erg/s/cm2]')
        if title is not None:
            ax.set_title('%s' % title)

    def control_plot(self, filename='', title=None, format=['png']):
        """
        self.cranio.control_plot(filename=filename, title=title)

        Options:
            filename: (string) filename of the png created. Should end in .png
            title: (string) optional title of the control plot. Passing SN
            name and exp_code through it is a good idea.
        """

        # Initialize parameters=====================================

        metrics = self.initialize_parameters()
        if self.xb is not None and self.xr is not None:
            MetricsFig = P.figure(figsize=(14, 12))
            ax1 = MetricsFig.add_subplot(3, 3, 1)
            ax2 = MetricsFig.add_subplot(3, 3, 2)
            ax3 = MetricsFig.add_subplot(3, 3, 3)
            ax4 = MetricsFig.add_subplot(3, 3, 4)
            ax5 = MetricsFig.add_subplot(3, 3, 5)
            ax6 = MetricsFig.add_subplot(3, 3, 6)
            ax7 = MetricsFig.add_subplot(3, 1, 3)
            self.plot_craniobca(metrics, ax=ax1, filename=filename)
            self.plot_craniobsi(metrics, ax=ax2, filename=filename)
            self.plot_craniobmg(metrics, ax=ax3, filename=filename)
            self.plot_cranior1r5(metrics, ax=ax4, filename=filename)
            self.plot_cranior2(metrics, ax=ax5, filename=filename)
            self.plot_cranior3r4(metrics, ax=ax6, filename=filename)
            self.plot_spectrum(metrics, ax=ax7, title=title)
            ax7.set_ylim(ymin=0)
            ax7.set_xlim(xmin=3000, xmax=7000)
            if filename is None:
                # unique_suffix = time.strftime("%Y-%m-%d-%H_%M_%S_UTC",
                #                               time.gmtime())
                filename = "control_plot"  # _" + unique_suffix
            for f in format:
                MetricsFig.savefig(filename + '.' + f)
                print >> sys.stderr, "Control plot saved in %s" % filename \
                    + '.' + f

        elif self.xb is not None:
            print >> sys.stderr, 'Worked on the b channel only'
            MetricsFig = P.figure(figsize=(12, 8))
            ax1 = MetricsFig.add_subplot(2, 3, 1)
            ax2 = MetricsFig.add_subplot(2, 3, 2)
            ax3 = MetricsFig.add_subplot(2, 3, 3)
            ax7 = MetricsFig.add_subplot(2, 1, 2)
            self.plot_craniobca(metrics, ax=ax1, filename=filename)
            self.plot_craniobsi(metrics, ax=ax2, filename=filename)
            self.plot_craniobmg(metrics, ax=ax3, filename=filename)
            self.plot_spectrum(metrics, ax=ax7, title=title)
            ax7.set_ylim(ymin=0)
            ax7.set_xlim(xmin=self.xb[0], xmax=self.xb[-1])
            if filename is None:
                # unique_suffix = time.strftime("%Y-%m-%d-%H_%M_%S_UTC",
                #                               time.gmtime())
                filename = "control_plot"  # _" + unique_suffix
            if title is not None:
                ax7.set_title('%s, calcium zone' % title)
            else:
                ax7.set_title('calcium zone')
            for f in format:
                MetricsFig.savefig(filename + '.' + f)
                print >> sys.stderr, "Control plot saved in %s" % filename \
                    + '.' + f

        elif self.xr is not None:
            print >> sys.stderr, 'Worked on the r channel only'
            MetricsFig = P.figure(figsize=(12, 8))
            ax4 = MetricsFig.add_subplot(2, 3, 1)
            ax5 = MetricsFig.add_subplot(2, 3, 2)
            ax6 = MetricsFig.add_subplot(2, 3, 3)
            ax7 = MetricsFig.add_subplot(2, 1, 2)
            self.plot_cranior1r5(metrics, ax=ax4, filename=filename)
            self.plot_cranior2(metrics, ax=ax5, filename=filename)
            self.plot_cranior3r4(metrics, ax=ax6, filename=filename)
            self.plot_spectrum(metrics, ax=ax7, title=title)
            ax7.set_ylim(ymin=0)
            ax7.set_xlim(xmin=self.xr[0], xmax=7000)
            if filename is None:
                # unique_suffix = time.strftime("%Y-%m-%d-%H_%M_%S_UTC",
                #                               time.gmtime())
                filename = "control_plot"  # _" + unique_suffix
            if title is not None:
                ax7.set_title('%s, silicon zone' % title)
            else:
                ax7.set_title('silicon zone')
            for f in format:
                MetricsFig.savefig(filename + '.' + f)
                print >> sys.stderr, "Control plot saved in %s" % filename \
                    + '.' + f
        P.close()

    def plot_oxygen(self, filename='', title=None, format=['png']):

        cr = self.cranio_O

        fig = P.figure()
        ax = fig.add_subplot(111)
        ax.plot(cr.x, cr.y, 'k', label='Flux')
        ax.plot(cr.x, cr.s, 'r', label='Interpolated flux')
        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.ewvalues['lbd_ewoi7773'][0])
                            & (cr.x <= cr.ewvalues['lbd_ewoi7773'][1])]
            p_line = N.polyfit([cr.ewvalues['lbd_ewoi7773'][0],
                                cr.ewvalues['lbd_ewoi7773'][1]],
                               [cr.smoother(cr.ewvalues['lbd_ewoi7773'])[0],
                                cr.smoother(cr.ewvalues['lbd_ewoi7773'])[1]], 1)
            ax.scatter(cr.ewvalues['lbd_ewoi7773'],
                       cr.smoother(cr.ewvalues['lbd_ewoi7773']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewoi7773 straight line"

        try:  # Plot vlines
            for x, y in zip(cr.ewvalues['lbd_ewoi7773'],
                            cr.smoother(cr.ewvalues['lbd_ewoi7773'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewoi7773 vlines\n"

        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.ewvalues['lbd_ewcaiiir'][0])
                            & (cr.x <= cr.ewvalues['lbd_ewcaiiir'][1])]
            p_line = N.polyfit([cr.ewvalues['lbd_ewcaiiir'][0],
                                cr.ewvalues['lbd_ewcaiiir'][1]],
                               [cr.smoother(cr.ewvalues['lbd_ewcaiiir'])[0],
                                cr.smoother(cr.ewvalues['lbd_ewcaiiir'])[1]], 1)
            ax.scatter(cr.ewvalues['lbd_ewcaiiir'],
                       cr.smoother(cr.ewvalues['lbd_ewcaiiir']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewcaiiir straight line"

        try:  # Plot vlines
            for x, y in zip(cr.ewvalues['lbd_ewcaiiir'],
                            cr.smoother(cr.ewvalues['lbd_ewcaiiir'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewcaiiir vlines\n"

        # Try to Annotate with spectral indicators values
        try:
            ax.annotate('ewoi7773=%.2f, ewcaiiir=%.2f' %
                        (cr.ewvalues['ewoi7773'],
                         cr.ewvalues['ewcaiiir']),
                        xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left', verticalalignment='bottom',
                        fontsize=10)
        except ValueError:
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

    def plot_iron(self, filename='', title=None, format=['png']):

        cr = self.cranio_fe

        fig = P.figure()
        ax = fig.add_subplot(111)
        ax.plot(cr.x, cr.y, 'k', label='Flux')
        ax.plot(cr.x, cr.s, 'r', label='Interpolated flux')
        try:  # Plot points and straight lines
            lbd_line = cr.x[(cr.x >= cr.ewvalues['lbd_ewfe4800'][0])
                            & (cr.x <= cr.ewvalues['lbd_ewfe4800'][1])]
            p_line = N.polyfit([cr.ewvalues['lbd_ewfe4800'][0],
                                cr.ewvalues['lbd_ewfe4800'][1]],
                               [cr.smoother(cr.ewvalues['lbd_ewfe4800'])[0],
                                cr.smoother(cr.ewvalues['lbd_ewfe4800'])[1]], 1)
            ax.scatter(cr.ewvalues['lbd_ewfe4800'],
                       cr.smoother(cr.ewvalues['lbd_ewfe4800']),
                       s=40, c='g', marker='o', edgecolors='none',
                       label='_nolegend_')
            ax.plot(lbd_line, N.polyval(p_line, lbd_line), color='g')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewfe4800 straight line"

        try:  # Plot vlines
            for x, y in zip(cr.ewvalues['lbd_ewfe4800'],
                            cr.smoother(cr.ewvalues['lbd_ewfe4800'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot ewfe4800 vlines\n"

        # Try to Annotate with spectral indicators values
        try:
            ax.annotate('ewfe4800=%.2f' % (cr.ewvalues['ewfe4800']),
                        xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left', verticalalignment='bottom',
                        fontsize=10)
        except ValueError:
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


def test_code(idr):
    """
    Test the code using an SNf IDR.

    Input is the path to an IDR.
    """
    import cPickle
    from snspin.extern import pySnurp
    if not idr.endswith('/'):
        idr += '/'
    d = cPickle.load(open(idr + 'META.pkl'))
    sn = 'SNF20070818-001'
    spec = '07_235_065_003'
    z = d[sn]['host.zhelio']
    phase = d[sn]['spectra'][spec]['salt2.phase']
    specB = pySnurp.Spectrum(idr + d[sn]['spectra'][spec]['idr.spec_B'])
    specR = pySnurp.Spectrum(idr + d[sn]['spectra'][spec]['idr.spec_R'])
    specM = pySnurp.Spectrum(idr + d[sn]['spectra'][spec]['idr.spec_merged'])
    specB.x /= (1. + z)
    specR.x /= (1. + z)
    specM.x /= (1. + z)
    dg = DrGall(spec=specM, specb=specB, specr=specR)
    calcium = dg.calcium_computing()
    silicon = dg.silicon_computing()
    oxygen = dg.oxygen_computing()
    iron = dg.iron_computing()
    title = sn + ', Rest-Frame Phase=%.1f' % phase
    dg.control_plot(filename="control_plot_name", title=title)
    dg.plot_oxygen(filename="control_plot_name_ox", title=title)
    dg.plot_iron(filename="control_plot_name_fe", title=title)

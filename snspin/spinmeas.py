#!/usr/bin/env python

import sys
import time
import numpy as N
import matplotlib.pyplot as P

from spectrum import merge


class DrGall:

    """
    Class to manipulate and use the craniometer
    """

    def __init__(self, spec=None, specB=None, specR=None,
                 spec_merged=None, verbose=True):
        """
        Spectrum initialization.
        Create self.x, self.y, self.v if spec is the merged spectrum
        Create self.xB, self.yB, self.vB for the blue channel if specB is given
        Create self.xR, self.yR, self.vR for the blue channel if specR is given
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
            except ValueError:
                pass
            try:
                self.xR = N.array(specR.x)
                self.yR = N.array(specR.y)
                self.vR = N.array(specR.v)
            except ValueError:
                pass

            if self.xB is not None and self.xR is not None:
                try:
                    spec_merged = merge.MergedSpectrum(specB, specR)
                    self.x_merged = N.array(spec_merged.x)
                    self.y_merged = N.array(spec_merged.y)
                    self.v_merged = N.array(spec_merged.v)
                except ValueError:
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
                        'spec.x, spec.y and spec.v)'
                    sys.exit()

        else:
            print >> sys.stderr, 'ERROR, no correct input in DrGall. Give me a'\
                'spectrum (for instance; spec with spec.x, spec.y and spec.v)'
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
        except ValueError:
            RCa = [N.nan, N.nan]
            if verbose:
                print 'ERROR in RCa computing, RCa =', RCa

        try:
            RCaS = self.cranio_BCa.RCaS(verbose=verbose)
            self.Values.update(self.cranio_BCa.RCaSvalues)
            if verbose:
                print 'RCaS computing done, RCaS =', RCaS
        except ValueError:
            RCaS = [N.nan, N.nan]
            if verbose:
                print 'ERROR in RCaS computing, RCaS =', RCaS

        try:
            RCaS2 = self.cranio_BCa.RCaS2(verbose=verbose)
            self.Values.update(self.cranio_BCa.RCaS2values)
            if verbose:
                print 'RCaS2 computing done, RCaS2 =', RCaS2
        except ValueError:
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
        except ValueError:
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
        except ValueError:
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
        except ValueError:
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
        except ValueError:
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
                                    verbose=verbose)  # RSi, RSiS
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
        except ValueError:
            RSi = [N.nan, N.nan]
            if verbose:
                print 'ERROR in RSi computing, RSi =', RSi

        try:
            RSiS = self.cranio_R1.RSiS(verbose=verbose)
            self.Values.update(self.cranio_R1.RSiSvalues)
            if verbose:
                print 'RSiS computing done, RSiS =', RSiS
        except ValueError:
            RSiS = [N.nan, N.nan]
            if verbose:
                print 'ERROR in RSiS computing, RSiS =', RSiS

        try:
            RSiSS = self.cranio_R5.RSiSS(verbose=verbose)
            self.Values.update(self.cranio_R5.RSiSSvalues)
            if verbose:
                print 'RSiSS computing done, RSiSS =', RSiSS
        except ValueError:
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
        except ValueError:
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
        except ValueError:
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
        except ValueError:
            EWSIIW_R = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWSIIW_R computing, EWSIIW_R =', EWSIIW_R

        try:
            self.Values.update(self.cranio_R2.EWvalues)
        except ValueError:
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
        except ValueError:
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
        except ValueError:
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
        except ValueError:
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
        except ValueError:
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
        except ValueError:
            vSiII_5972 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in vSiII_5972 computing, vSiII_5972 =', vSiII_5972

        try:
            vSiII_6355 = self.cranio_R4.velocity({'lmin': 6000,
                                                  'lmax': 6210,
                                                  'lrest': 6355,
                                                  'name': 'vSiII_6355'},
                                                 verbose=verbose)
            # vSiII_6355 = self.cranio_R4.velocity2({'lmin':5850, 'lmax':6015,
            # 'lrest':6355, 'name':'vSiII_6355'}, verbose=verbose)
            self.Values.update(self.cranio_R4.velocityValues)
            if verbose:
                print 'vSiII_6355 computing done, vSiII_6355 =', vSiII_6355
        except ValueError:
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
        except ValueError:
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
        except ValueError:
            EWCaIIIR = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWCaIIIR computing, EWCaIIIR =', EWCaIIIR

        try:
            self.Values.update(self.cranio_O.EWvalues)
        except ValueError:
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
        except ValueError:
            EWFe4800 = [N.nan, N.nan]
            if verbose:
                print 'ERROR in EWFe4800 computing, EWFe4800 =', EWFe4800

        try:
            self.Values.update(self.cranio_F.EWvalues)
        except ValueError:
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
        except ValueError:
            RSi = float(N.nan)

        try:
            RSiS = self.cranio_R1.RSiSvalues['RSiS']
        except ValueError:
            RSiS = float(N.nan)

        try:
            RSiSS = self.cranio_R5.RSiSSvalues['RSiSS']
        except ValueError:
            RSiSS = float(N.nan)

        try:
            RCa = self.cranio_BCa.RCavalues['RCa']
        except ValueError:
            RCa = float(N.nan)

        try:
            RCaS = self.cranio_BCa.RCaSvalues['RCaS']
        except ValueError:
            RCaS = float(N.nan)

        try:
            RCaS2 = self.cranio_BCa.RCaS2values['RCaS2']
        except ValueError:
            RCaS2 = float(N.nan)

        try:
            EDCa = self.cranio_BCa.EDCavalues['EDCa']
        except ValueError:
            EDCa = float(N.nan)

        try:
            EWCaIIHK = self.cranio_BCa.EWvalues['EWCaIIHK']
        except ValueError:
            EWCaIIHK = float(N.nan)

        try:
            EWSiII4000 = self.cranio_BSi.EWvalues['EWSiII4000']
        except ValueError:
            EWSiII4000 = float(N.nan)

        try:
            EWMgII = self.cranio_BMg.EWvalues['EWMgII']
        except ValueError:
            EWMgII = float(N.nan)

        try:
            EWSIIW = self.cranio_R2.EWvalues['EWSIIW']
        except ValueError:
            EWSIIW = float(N.nan)

        try:
            EWSIIW_L = self.cranio_R2.EWvalues['EWSIIW_L']
        except ValueError:
            EWSIIW_L = float(N.nan)

        try:
            EWSIIW_R = self.cranio_R2.EWvalues['EWSIIW_R']
        except ValueError:
            EWSIIW_R = float(N.nan)

        try:
            EWSiII5972 = self.cranio_R3.EWvalues['EWSiII5972']
        except ValueError:
            EWSiII5972 = float(N.nan)

        try:
            EWSiII6355 = self.cranio_R4.EWvalues['EWSiII6355']
        except ValueError:
            EWSiII6355 = float(N.nan)

        try:
            vSiII_5972 = self.cranio_R3.velocityValues['vSiII_5972']
        except ValueError:
            vSiII_5972 = float(N.nan)

        try:
            vSiII_6355 = self.cranio_R4.velocityValues['vSiII_6355']
        except ValueError:
            vSiII_6355 = float(N.nan)

        return RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, \
            EWSIIW, EWSiII5972, EWSiII6355, vSiII_5972, vSiII_6355, EWSIIW_L, \
            EWSIIW_R

    #=========================================================================
    # Functions to plot control_plot of spectral indicators computing
    #=========================================================================

    def plot_cranioBCa(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where RCa, RCaS, RCas2, EDCa and EWCaIIHK are computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW, \
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
        except ValueError:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        try:  # Plot the RCaS vspan
            ax.axvspan(cr.RCaSvalues['RCaS_lbd'][0],
                       cr.RCaSvalues['RCaS_lbd'][1],
                       ymin=0, ymax=1, facecolor='y', alpha=0.25)
            ax.axvspan(cr.RCaSvalues['RCaS_lbd'][2],
                       cr.RCaSvalues['RCaS_lbd'][3],
                       ymin=0, ymax=1, facecolor='y', alpha=0.25)
        except ValueError:
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
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWCaIIHK zone"

        try:  # Plot the RCa lines
            for x, y in zip(cr.RCavalues['RCa_lbd'],
                            cr.smoother(cr.RCavalues['RCa_lbd'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
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
        except ValueError:
            pass
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=3450, xmax=4070)

        if save:
            fig.savefig('calcium_' + filename)

    def plot_cranioBSi(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where EWSi4000 is computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW, \
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
        except ValueError:
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
        except ValueError:
            print >> sys.stderr, "No parameters to plot EDCa straight line"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWSiII4000'],
                            cr.smoother(cr.EWvalues['lbd_EWSiII4000'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot RCa vlines"

        # Annotate the Ca zone with spectral indicators values
        try:
            ax.annotate('EWSiII4000=%.2f' %
                        (EWSiII4000), xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except ValueError:
            pass
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=3850, xmax=4150)

        if save:
            fig.savefig('EWSiII4000_' + filename)

    def plot_cranioBMg(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where EWMgII is computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW, \
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
        except ValueError:
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
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWMgII straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWMgII'],
                            cr.smoother(cr.EWvalues['lbd_EWMgII'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWMgII vlines"

        # Annotate the Ca zone with spectral indicators values
        try:
            ax.annotate('EWMgII=%.2f' %
                        (EWMgII), xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
        except ValueError:
            pass
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=4000, xmax=4600)

        if save:
            fig.savefig('EWMgII_' + filename)

    def plot_cranioR1R5(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where RCa, RCaS, RCas2, EDCa and EWCaIIHK are computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW, \
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
        except ValueError:
            print >> sys.stderr, "No smothing function computed, so no '\
            'smoothing function ploted"

        # try: #Plot the RSiSS vspan
        ax.axvspan(cr5.RSiSSvalues['RSiSS_lbd'][0],
                   cr5.RSiSSvalues['RSiSS_lbd'][1],
                   ymin=0, ymax=1, facecolor='y', alpha=0.25)
        ax.axvspan(cr5.RSiSSvalues['RSiSS_lbd'][2],
                   cr5.RSiSSvalues['RSiSS_lbd'][3],
                   ymin=0, ymax=1, facecolor='y', alpha=0.25)
        # except ValueError: print >> sys.stderr, "No parameters to plot RSiSS zone"

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
        except ValueError:
            print >> sys.stderr, "No parameters to plot RSi zone"

        try:  # Plot the RSi and RSiS lines
            for x, y in zip(cr1.RSivalues['RSi_lbd'],
                            cr1.smoother(cr1.RSivalues['RSi_lbd'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot RSi and RSiS lines"

        # Annotate the Ca zone with spectral indicators values
        try:
            ax.annotate('RSi=%.2f, RSiS=%.2f, RSiSS=%.2f' %
                        (RSi, RSiS, RSiSS), xy=(0.01, 0.01),
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

    def plot_cranioR2(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where EWSIIW is computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW, \
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
        except ValueError:
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
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWSIIW straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWSIIW'],
                            cr.smoother(cr.EWvalues['lbd_EWSIIW'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
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
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWSIIW_L straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWSIIW_L'],
                            cr.smoother(cr.EWvalues['lbd_EWSIIW_L'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
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
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWSIIW_R straight '\
            'line zone"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWSIIW_R'],
                            cr.smoother(cr.EWvalues['lbd_EWSIIW_R'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWSIIW_R vlines"

        # Annotate the Ca zone with spectral indicators values
        try:
            ax.annotate('EWSIIW=%.2f' %
                        (EWSIIW), xy=(0.01, 0.07), xycoords='axes fraction',
                        xytext=(0.01, 0.07), textcoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='bottom', fontsize=10)
            ax.annotate('EWSIIW_L=%.2f, EWSIIW_R=%.2f' %
                        (EWSIIW_L, EWSIIW_R), xy=(0.01, 0.01),
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
            fig.savefig('EWSIIW_' + filename)

    def plot_cranioR3R4(self, metrics, ax=None, filename='', verbose=True):
        """Plot zone where EWSIIW is computed"""

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW, \
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
        except ValueError:
            print >> sys.stderr, "No smothing function computed for '\
            'EWSiII5972, so no smoothing function ploted"
        try:
            ax.plot(cr4.x, cr4.s, color='b', label='Interpolated flux')
        except ValueError:
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
        except ValueError:
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
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWSiII6355 straight '\
            'line zone"

        try:  # Plot vlines for EWSiII5972
            for x, y in zip(cr3.EWvalues['lbd_EWSiII5972'],
                            cr3.smoother(cr3.EWvalues['lbd_EWSiII5972'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWSiII5972 vlines"

        try:  # Plot vlines for EWSiII6355
            for x, y in zip(cr4.EWvalues['lbd_EWSiII6355'],
                            cr4.smoother(cr4.EWvalues['lbd_EWSiII6355'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWSiII6355 vlines"

        # Annotate the Si zone with spectral indicators values
        try:
            ax.annotate('EWSiII5972=%.2f, EWSiII6355=%.2f' %
                        (EWSiII5972, EWSiII6355),
                        xy=(0.01, 0.01), xycoords='axes fraction',
                        xytext=(0.01, 0.01), textcoords='axes fraction',
                        horizontalalignment='left', verticalalignment='bottom',
                        fontsize=10)
        except ValueError:
            pass

        try:  # Plot vline for vSiII_6355
            ax.axvline(cr4.velocityValues['vSiII_6355_lbd'],
                       color='k', lw=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot SiII6355 vlines"

        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=5500, xmax=6400)

        if save:
            fig.savefig('EWSiII5972_' + filename)

    def plot_spectrum(self, metrics, ax=None, filename='', title=None, verbose=True):

        RSi, RSiS, RSiSS, RCa, RCaS, RCaS2, EDCa, EWCaIIHK, EWSiII4000, EWMgII, EWSIIW, \
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
            except ValueError:
                pass
            try:
                ax.plot(self.silicon_zone['x'],
                        self.cranioR.smoother(self.silicon_zone['x']),
                        color='r', label=label)
            except ValueError:
                pass
        except ValueError:
            label1 = 'Flux'
            label2 = 'Interpolated flux'
            try:
                ax.plot(self.xB, self.yB, color='k', label=label1)
                ax.plot(self.calcium_zone['x'],
                        self.cranio_B.smoother(self.calcium_zone['x']),
                        color='r', label=label2)
                label1 = '_nolegend_'
                label2 = '_nolegend_'
            except ValueError:
                pass
            try:
                ax.plot(self.xR, self.yR, color='k', label=label1)
                ax.plot(self.silicon_zone['x'],
                        self.cranioR.smoother(self.silicon_zone['x']),
                        color='r', label=label2)
            except ValueError:
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
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWOI7773 straight line"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWOI7773'],
                            cr.smoother(cr.EWvalues['lbd_EWOI7773'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
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
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWCaIIIR straight line"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWCaIIIR'],
                            cr.smoother(cr.EWvalues['lbd_EWCaIIIR'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
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
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWFe4800 straight line"

        try:  # Plot vlines
            for x, y in zip(cr.EWvalues['lbd_EWFe4800'],
                            cr.smoother(cr.EWvalues['lbd_EWFe4800'])):
                ax.vlines(x, 0, y, color='g', linewidth=1, label='_nolegend_')
        except ValueError:
            print >> sys.stderr, "No parameters to plot EWFe4800 vlines\n"

        # Try to Annotate with spectral indicators values
        try:
            ax.annotate('EWFe4800=%.2f' % (cr.EWvalues['EWFe4800']),
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
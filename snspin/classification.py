#!/usr/bin/env python

"""
Make the SNID, Branch and Wang classification sheme table for all the SNe
avalaible in the given file.

References:

* Benetti et al. 2005:
 - http://cdsads.u-strasbg.fr/abs/2005ApJ...623.1011B
* Blondin et al. 2007 et 2012:
 - http://adsabs.harvard.edu/abs/2007ApJ...666.1024B
 - http://adsabs.harvard.edu/abs/2012arXiv1203.4832B
* Branch et al 2006 et 2009:
 - http://cdsads.u-strasbg.fr/abs/2006PASP..118..560B
 - http://cdsads.u-strasbg.fr/abs/2009PASP..121..238B
* Wang et al. 2009:
 - http://adsabs.harvard.edu/abs/2009ApJ...699L.139W

See also http://snovae.in2p3.fr/nchotard/SpectralAnalysis/index.html for details
of the analysis and some results.
"""

__author__ = "Nicolas Chotard <nchotard@ipnl.in2p3.fr>"
__version__ = '$Id: classLib.py,v 1.1 2014/12/30 02:39:19 nchotard Exp $'

import copy
import numpy as N
import pylab as P
import scipy.stats as stats

from ToolBox import MPL, Optimizer
from snspin.tools import io
from snspin.tools import statistics


# Definitions ============================================================


# classifications


def wang_classification(phreno, vcut=12200.0, prange=2.5,
                        plot=False, keepall=False):
    """
    Wang et al. 2009 classification scheme.

    Classify the SNe sanple into two sub-classes based on their SiII 6355
    velocity values:
     - if v < vcut: classified as normal SN Ia (normal)
     - if v >= vcut: classified as hight velocity SN Ia (HV)

    :param dictionnary phreno: the phrenology dictionnary containing the
                               spectral indicators
    :param vcut float: the volocity cut defined the two sub-classes. You can
                       also set this value to None. In this case, it will be
                       computed as the (mean+std) of the velocity distribution.
    :param 1D-array prange: the phase range where the closest spectra to the
                            maximum light will be taken.
    :param bool keepall: if True, keep all the spetra in the given phase range
    :return: a dictionnary containing the results
    """

    print "\nSub-sampling the SNe list with the 'Wang09' velocity criterium."

    # get the velocity values
    if not keepall:
        sne, vSi, vSie, phases = get_si_at_phase(phreno, 'vSiII_6355',
                                                 0, prange)
        sne, EWSi, EWSie, phases = get_si_at_phase(phreno, 'EWSiII6355',
                                                   0, prange, sne=sne)
    else:
        sne, vSi, vSie, phases = get_si_in_range(phreno, 'vSiII_6355',
                                                 [-prange, prange])
        sne = [sn for sn in set(sne)]
        sne, EWSi, EWSie, phases = get_si_in_range(phreno, 'EWSiII6355',
                                                   [-prange, prange], sne=sne)

    # check for None or nan values
    filt = (vSi == vSi) & (EWSi == EWSi) & \
           (vSie == vSie) & (EWSie == EWSie) & \
           (vSi > 0) & (EWSi > 0)

    if len(sne[~filt]) != 0:
        print "\nThe following SNe removed because velocity nan values:"
        print sne[~filt]

    sne = sne[filt]
    EWSi = EWSi[filt]
    vSi = vSi[filt]
    EWSie = EWSie[filt]
    vSie = vSie[filt]
    phases = phases[filt]

    # velocity cut is already given?
    if vcut is None:
        print "\nNo given velocity cut, computing one..."
        vcut = N.mean(vSi) + 3 * N.std(vSi)
        print "The new velocity cut is %.2f" % vcut

    # sub-sample filters
    normf = (vSi < vcut)
    HVf = (vSi >= vcut)

    # save general info
    d = {}
    d['info'] = {'vcut': vcut,
                 'prange': prange,
                 'bysn': {}}

    for sn in sne:
        if sn in sne[normf]:
            d['info']['bysn'][sn] = 'norm'
        else:
            d['info']['bysn'][sn] = 'Hv'

    # get the sub-sample lists
    d['normal'] = {'sne': sne[normf],
                   'vSi': vSi[normf],
                   'vSi.err': vSie[normf],
                   'phases': phases[normf],
                   'EWSi': EWSi[normf],
                   'EWSi.err': EWSie[normf]}
    d['HV'] = {'sne': sne[HVf],
               'vSi': vSi[HVf],
               'vSi.err': vSie[HVf],
               'phases': phases[HVf],
               'EWSi': EWSi[HVf],
               'EWSi.err': EWSie[HVf]}

    # print some info
    print "\nResults:"
    print '\n' + " Normal SNe Ia ".center(40, '=')
    print "  - %i SNe." % len(sne[normf])
    print "  - mean vSi: %.2f +- %.2f (km/s)" %\
          (N.mean(vSi[normf]), N.std(vSi[normf]))
    print "  - mean phase: %.2f +- %.2f (days)" %\
          (N.mean(phases[normf]), N.mean(phases[normf]))
    print '\n' + " HV SNe Ia ".center(40, '=')
    print "  - %i SNe." % len(sne[HVf])
    if len(sne[HVf]) != 0:
        print "  - mean vSi: %.2f +- %.2f (km/s)" %\
              (N.mean(vSi[HVf]), N.std(vSi[HVf]))
        print "  - mean phase: %.2f +- %.2f (days)" %\
              (N.mean(phases[HVf]), N.mean(phases[HVf]))
        print "  - HV SNe list:"
        print "     " + "\n     ".join(sorted(d['HV']['sne']))
    else:
        print "Probably not tail in the velocity distribution..."
    print "\nVelocity cutoff: %.2f km/s" % vcut

    if plot:
        wang_classification_plot(d, phreno, StN=2, square=False)

    return d


def branch_classification(phreno, prange=2.5, plot=False,
                          keepall=False, sne=None):
    """
    Branch et al 2006 et 2009 classification scheme.

    Classify the SNe sanple into four sub-classes based on their SiII 5972
    and SiII 6355 equivalent widths values:
     - Core-normal (CN):
     - Broad line (BL): EWSiII 6355 ~> 105 A
     - Shallow silicon (SS): 
     - Cool (CL): 

    :param dictionnary phreno: the phrenology dictionnary containing the
                               spectral indicators
    :param vcut float: the volocity cut defined the two sub-classes
    :param 1D-array prange: the phase range where the closest spectra to the
                            maximum light will be taken.
    :param 1D-array sne: list of sne.
    :return: a dictionnary containing the results
    """
    print "\nSub-sampling the SNe list with the 'Branch09' EWSi criteria."

    # get the velocity values
    if not keepall:
        sne, EW5, EW5e, phases = get_si_at_phase(phreno, 'EWSiII5972',
                                                 0, prange, sne=sne)
        sne, EW6, EW6e, phases = get_si_at_phase(phreno, 'EWSiII6355',
                                                 0, prange, sne=sne)
    else:
        sne, EW5, EW5e, phases = get_si_in_range(phreno, 'EWSiII5972',
                                                 [-prange, prange])
        sne = [sn for sn in set(sne)]
        sne, EW6, EW6e, phases = get_si_in_range(phreno, 'EWSiII6355',
                                                 [-prange, prange], sne=sne)

    # check for None or nan values
    filt = (EW5 == EW5) & (EW6 == EW6) & \
           (EW5e == EW5e) & (EW6e == EW6e) & \
           (EW5 > 0) & (EW6 > 0)

    if len(sne[~filt]) != 0:
        print "\nThe following SNe removed because velocity nan values:"
        print sne[~filt]

    afilt = lambda x: x[filt]
    sne, EW5, EW5e, EW6, EW6e, phases = map(
        afilt, [sne, EW5, EW5e, EW6, EW6e, phases])
    print "\n %i SNe in this sample." % (len(sne))

    # sub-sample filters
    BLf = (EW6 >= 105) & (EW5 < 30)
    SSf = (EW6 <= 80) & (EW5 < 30)
    CLf = (EW5 >= 30)
    CNf = (~BLf) & (~SSf) & (~CLf)

    # save general info
    d = {}
    d['info'] = {'prange': prange, 'bysn': {}}

    # get the sub-sample lists
    stype = ['CN', 'BL', 'SS', 'CL']
    filts = [CNf, BLf, SSf, CLf]
    for i, st in enumerate(stype):
        d[st] = {}
        d[st]['sne'] = sne[filts[i]]
        d[st]['EWSiII5972'] = EW5[filts[i]]
        d[st]['EWSiII5972.err'] = EW5e[filts[i]]
        d[st]['EWSiII6355'] = EW6[filts[i]]
        d[st]['EWSiII6355.err'] = EW6e[filts[i]]
        d[st]['phases'] = phases[filts[i]]
        for sn in sne[filts[i]]:
            d['info']['bysn'][sn] = st

    # print some info
    print "\nResults:"
    for i, st in enumerate(d):
        if st == 'info':
            continue
        print '\n' + (" %s " % st).center(40, '=')
        print "  - %i SNe." % len(d[st]['sne'])
        print "  - mean Si 5972: %.2f +- %.2f (A)" %\
              (N.mean(d[st]['EWSiII5972']), N.std(d[st]['EWSiII5972']))
        print "  - mean Si 5972: %.2f +- %.2f (A)" %\
              (N.mean(d[st]['EWSiII6355']), N.std(d[st]['EWSiII6355']))
        print "  - mean phase: %.2f +- %.2f (days)" %\
              (N.mean(d[st]['phases']), N.mean(d[st]['phases']))

    if plot:
        branch_classification_plot(copy.deepcopy(d), phreno, StN=2)

    return d


def benetti_classification(phreno, indic='vSiII_6355', pmin=-5, pmax=25,
                           degree=1, StN=4, jack=True, plot=False):
    """
    Benetti et al. 2005 classification sheme.

    Classify the SNe sanple into three sub-classes based on their SiII 6355
    velocity values:
     - faint
     - Low velocity gradient (LVG)
     - High velocity gradient (HVG)

    :param dictionnary phreno: the phrenology dictionnary containing the
                               spectral indicators
    :param 1D-array prange: the phase range where the closest spectra to the
                            maximum light will be taken.
    :return: a dictionnary containing the results
    """
    print "\nSub-sampling the SNe list with the 'Benetti05' classification."
    from ToolBox.Wrappers import SALT2model

    prange = [pmin, pmax]
    
    # some functions
    getallp = lambda d: N.array([d[id]['salt2.phase'] for id in d])
    getallv = lambda d: N.array([d[id]['phrenology.%s' % indic] for id in d])
    getallve = lambda d: N.array(
        [d[id]['phrenology.%s.err' % indic] for id in d])

    # get all the indicator values for all the SNe
    sne = sorted([sn for sn in phreno])
    z = [phreno[sn]['salt2.Redshift'] for sn in sne]
    x1 = [phreno[sn]['salt2.X1'] for sn in sne]
    color = [phreno[sn]['salt2.Color'] for sn in sne]
    x1e = [phreno[sn]['salt2.X1.err'] for sn in sne]
    colore = [phreno[sn]['salt2.Color.err'] for sn in sne]
    ph = [getallp(phreno[sn]['spectra']) / (1. + phreno[sn]['salt2.Redshift'])
          for sn in sne]
    Ind = [getallv(phreno[sn]['spectra']) for sn in sne]
    Inde = [getallve(phreno[sn]['spectra']) for sn in sne]

    results = {}
    for i, sn in enumerate(sne):

        p, ind, inde = ph[i], Ind[i], Inde[i]

        # filt nan values
        filt = (p == p) & (ind > 0) & \
               (ind == ind) & (inde == inde)
        p, ind, inde = p[filt], ind[filt], inde[filt]

        # filt for phase range
        pfilt = (p >= prange[0]) & (p <= prange[1])
        p, ind, inde = p[pfilt], ind[pfilt], inde[pfilt]

        # filt for signal to noise ratio
        sfilt = (ind / inde >= StN)
        p, ind, inde = p[sfilt], ind[sfilt], inde[sfilt]

        # fit it
        if jack:
            minp = degree + 3
        else:
            minp = degree + 2
        if not len(p) >= minp:
            print "%s non fitted, not enought values (%i)." % (sn, len(p))
            continue
        else:
            if not jack:
                res = polyfit(p, ind, dy=inde, degree=degree)
                print "%i/%i %s: %i value in [%.1f,%.1f]. slope: %.2f" %\
                      (i + 1, len(sne), sn, len(p),
                       prange[0], prange[1], res['params'][0])
                res['Jparams'] = res['params']
                res['Jdparams'] = res['dparams']
            else:
                # jackknifing to get a proper error
                ress = []
                for j in range(len(p)):
                    pj = N.concatenate([p[:j], p[j + 1:]])
                    indj = N.concatenate([ind[:j], ind[j + 1:]])
                    indej = N.concatenate([inde[:j], inde[j + 1:]])
                    res = polyfit(pj, indj, dy=indej, degree=degree)
                    ress.append(res)
                res = polyfit(p, ind, dy=inde, degree=degree)
                par0 = N.mean([r['params'][0] for r in ress])
                par1 = N.mean([r['params'][1] for r in ress])
                err0 = N.std([r['params'][0] for r in ress])
                err1 = N.std([r['params'][1] for r in ress])
                res['Jparams'] = [par0, par1]
                res['Jdparams'] = [err0, err1]

                print "%i/%i %s: %i value in [%.1f,%.1f]. slope: %.2f +- %.2f" %\
                      (i + 1, len(sne), sn, len(p), prange[0], prange[1],
                       res['Jparams'][0], res['Jdparams'][0])

        if plot:
            fig = P.figure()
            ax = fig.add_subplot(111,
                                 xlabel='phase',
                                 ylabel=indic,
                                 title=sn + ', s = %.2f += %.2f' %
                                 (res['Jparams'][0], res['Jdparams'][0]))
            ax.plot(p, ind, 'ok', mew=1.5, ms=8)
            ax.errorbar(p, ind, yerr=inde, capsize=None,
                        color='k', lw=1, ls='None')
            ax.plot(p, N.polyval(
                [res['Jparams'][0], res['Jparams'][1]], p), 'r')
            fig.savefig('figures/%s_%s.png' % (sn, indic))
            P.close()

        # save results
        dm15, dm15e = SALT2model.x1_to_dm15(x1[i], x1e[i])
        results[sn] = res
        results[sn]['salt2'] = {'x1': x1[i],
                                'x1.err': x1e[i],
                                'color': color[i],
                                'color.err': colore[i],
                                'dm15': dm15,
                                'dm15.err': dm15e}

    # general info
    results['info'] = {'indic': indic,
                       'prange': prange,
                       'degree': degree,
                       'StN': StN,
                       'jack': jack,
                       'bysn': {}}

    for sn in results:
        if sn == 'info':
            continue
        vsi = results[sn]['Jparams'][0]
        dm15 = results[sn]['salt2']['dm15']

        if dm15 > 1.7:
            t = 'FAINT'
        elif vsi < 65 and dm15 < 1.5:
            t = 'LVG'
        elif vsi > 75 and dm15 < 1.5:
            t = 'HVG'
        elif vsi >= 65 and vsi <= 75 and dm15 < 1.5:
            t = 'LVG-HVG'
        elif vsi > 75 and dm15 >= 1.5 and dm15 <= 1.7:
            t = 'HVG-FAINT'
        elif vsi < 65 and dm15 >= 1.5 and dm15 <= 1.7:
            t = 'LVG-FAINT'
        elif vsi >= 65 and vsi <= 75 and dm15 >= 1.5 and dm15 <= 1.7:
            t = 'LVG-HVG-FAINT'

        results[sn]['type'] = t

    if plot:
        benetti_classification_plot(results, idr=phreno)

    return results


# utilities

def get_si_at_phase(phreno, si, pvalue, prange, sne=None, saltp='salt2'):
    """
    Get the spectral indicators (si) values for the given list of SNe in a phase
    range around a phase value.
    :param dictionnary phreno: the phrenology dictionnary containing the
                               spectral indicators
    :param string si: the spectral indocator name, eg, EWSiII4000 or vSiII_6355.
    :param float pvalue: the central phase, eg, 0 for maximum light.
    :param 1D-array prange: the phase range where the closest spectra to the
                            maximum light will be taken.

    """
    # get the data
    if sne is None:
        sne = sorted([sn for sn in phreno if sn != 'DATASET'])
    else:
        sne = sorted(sne)
    specs = [get_id_at_phase(phreno[sn], pvalue) for sn in sne]
    phases = N.array([phreno[sn]['spectra'][sp][saltp + '.phase']
                      / (1. + phreno[sn]['salt2.Redshift'])
                      for sn, sp in zip(sne, specs)])
    values = [phreno[sn]['spectra'][sp]['phrenology.' + si]
              for sn, sp in zip(sne, specs)]
    valuese = [phreno[sn]['spectra'][sp]['phrenology.' + si + '.err']
               for sn, sp in zip(sne, specs)]

    # apply the phase filter
    pfilter = (N.array(phases) >= (pvalue - prange)) & \
              (N.array(phases) <= (pvalue + prange))
    fsne = N.array(sne)[pfilter]
    fvalues = N.array(values)[pfilter]
    fvaluese = N.array(valuese)[pfilter]
    fphases = N.array(phases)[pfilter]

    return fsne, fvalues, fvaluese, fphases


def get_id_at_phase(d, p, saltp='salt2'):
    """
    Get the id name of the closest spectra to the phase p.
    """
    ids = N.array([i for i in d['spectra']])
    phases = N.array([d['spectra'][i][saltp + '.phase'] for i in ids]) \
        / (1. + d['salt2.Redshift'])
    return ids[N.argmin(N.abs(phases - p))]


def get_si_in_range(phreno, si, prange, sne=None):
    """
    Get all the spectral indicators (si) values for the given list of SNe in
    a phase range around a phase value.

    :param dictionnary phreno: the phrenology dictionnary containing the
                               spectral indicators
    :param string si: the spectral indocator name, eg, EWSiII4000 or vSiII_6355.
    :param 1D-array prange: the phase range where the closest spectra to the
                            maximum light will be taken.

    """
    # get the data
    if sne is None:
        sne = sorted([sn for sn in phreno if sn != 'DATASET'])
    else:
        sne = sorted(sne)

    phases = N.concatenate([[phreno[sn]['spectra'][sp]['salt2.phase']
                             for sp in phreno[sn]['spectra']] for sn in sne])
    values = N.concatenate([[phreno[sn]['spectra'][sp]['phrenology.' + si]
                             for sp in phreno[sn]['spectra']] for sn in sne])
    valuese = N.concatenate([[phreno[sn]['spectra'][sp]['phrenology.' + si + '.err']
                              for sp in phreno[sn]['spectra']] for sn in sne])
    sne = N.concatenate([[sn for sp in phreno[sn]['spectra']] for sn in sne])

    # apply the phase filter
    pfilter = (phases > prange[0]) & (phases < prange[1])
    fsne = N.array(sne)[pfilter]
    fvalues = N.array(values)[pfilter]
    fvaluese = N.array(valuese)[pfilter]
    fphases = N.array(phases)[pfilter]

    return fsne, fvalues, fvaluese, fphases


def merge_phreno_idr(idr, phreno):
    """
    merge the idr dictionnary and the phrenology output.
    """
    ndic = copy.deepcopy(idr)
    for sn in phreno:
        if sn in idr:
            for spid in phreno[sn]['spectra']:
                ndic[sn]['spectra'][spid].update(phreno[sn]['spectra'][spid])
        else:
            print "%s in 'phreno' but not in 'idr'."
    if 'DATASET' in ndic:
        del ndic['DATASET']
    return ndic


def polyfit(x, y, dy=None, degree=1):
    """
    Polynomial fit of degree 1 or 2 (default is 1) for the given input.

    :param 1D-array x: the x array axis
    :param 1D-array y: the y array axis (arror is dy, None by default)
    :param int degree: the polynomial degree. Could be 1 or 2.
    """
    if not isinstance(degree, int ) \
       or not degree in [1, 2]:
        mess = "Error, degree must be 1 or 2."
        raise ValueError(mess)

    # Initialization of the parameters
    if degree == 1:
        params = [1, 0]

        def model(p):
            a, b = p
            return a * x + b
    else:
        params = [1, 1, 0]

        def model(p):
            a, b, c = p
            return a * x**2 + b * x + c

    # Set the data and model
    D = Optimizer.DataSet(y)
    M = Optimizer.Model(model)

    # Run the fit using Minuit
    mnt = Optimizer.Minuit(M, D)
    mnt.init(params)

    # get some results
    params, dparams = mnt.fit()
    dof, chi2, pvalue = mnt.goodness()

    # Save fitter and results
    results = {'params': params,
               'dparams': dparams,
               'CovMat': mnt.covariance(),
               'CorMat': mnt.correlation(),
               'chi2': chi2,
               'chi2pdof': chi2 / dof,
               'dof': dof,
               'pvalue': pvalue,
               'fitter': mnt}

    return results


# plots

def wang_classification_plot(d, idr=None, name=None, StN=3, square=True):
    """
    This plot is usually done in the EWSiII 5972 vs EWSiII 6533 space.

    :param dictionnay d: dictionnary containing the following keys:
       - 'normal': {'sne', 'vSi', 'vSi.err', 'phases', 'EWSi', 'EWSi.err'}
       - 'HV': {'sne', 'vSi', 'vSi.err', 'phases', 'EWSi', 'EWSi.err'}
       with: 
         * sne: the SNe list (normal and HV) 
         * vSi: the SiII 6355 velocity list (n and HV) (and error, .err)
         * EWSi: the SiII 6355 EW list (n and HV) (and error, .err)
         * phases: phases list (n and HV)
    :param bool hist: plot the velocity histogram with the cut if True.
    :param dictionnary idr: the idr dictionnary. If given, the 'bad' sample
                            will be show in red on the plot.
    :param string name: name of the figure is you want to save it (name.ext)

    Note:
     - all the sne, vSi and EWSi array (n or HV) must have the same length,
       and must not contain nan values.

    """
    # get the data
    dn = d['normal']
    dHV = d['HV']
    filtHV = ((dHV['vSi'] / dHV['vSi.err']) > StN) & \
             ((dHV['EWSi'] / dHV['EWSi.err']) > StN)
    filtn = ((dn['vSi'] / dn['vSi.err']) > StN) & \
            ((dn['EWSi'] / dn['EWSi.err']) > StN)
    ntot = len(dn['vSi'][filtn]) + len(dHV['vSi'][filtHV])

    # figure
    fig = P.figure(dpi=120)

    # axe
    ax = fig.add_axes([0.09, 0.09, 0.88, 0.69])
    ax2 = fig.add_axes([0.09, 0.79, 0.88, 0.19], sharex=ax)
    ax.set_xlabel('v SiII 6355 [km/s]', fontsize='x-large')
    ax.set_ylabel(r'EW SiII 6355 [$\AA$]', fontsize='x-large')

    # plot
    ax.plot(dn['vSi'][filtn], dn['EWSi'][filtn], 'o',
            mec='k', mfc='None', mew=1.5, ms=8, label='Normal (%i)' %
            len(dn['vSi'][filtn]))
    ax.errorbar(dn['vSi'][filtn], dn['EWSi'][filtn],
                xerr=dn['vSi.err'][filtn], yerr=dn['EWSi.err'][filtn],
                capsize=None, color='k', lw=1, ls='None')
    ax.plot(dHV['vSi'][filtHV], dHV['EWSi'][filtHV], 's',
            mec='b', mfc='None', mew=1.5, ms=8, label='HV (%i)' %
            len(dHV['vSi'][filtHV]))

    ax.errorbar(dHV['vSi'][filtHV], dHV['EWSi'][filtHV],
                xerr=dHV['vSi.err'][filtHV], yerr=dHV['EWSi.err'][filtHV],
                capsize=None, color='b', lw=1, ls='None')

    ax.axvline(N.mean(dn['vSi'][filtn]), color='k', ls='--')
    ax.axvline(d['info']['vcut'], color='r', ls='--')
    ax.axhline(N.mean(dn['EWSi'][filtn]), color='k', ls='--')

    vsi = N.concatenate([dn['vSi'][filtn], dHV['vSi'][filtHV]])
    bins = statistics.hist_bins(vsi)
    ax2.hist(vsi, lw=1, color='k', bins=bins, normed=True, alpha=0.7)
    loc, scale = stats.norm.fit(vsi)
    xx = N.linspace(8000, 15000, 1000)
    gauss = stats.norm.pdf(xx, loc=loc, scale=scale)
    ax2.fill_between(xx, gauss, color='k', alpha=0.15)
    ax2.set_yticks([])
    ax2.set_ylabel('Normed distr.')

    ax2.xaxis.set_tick_params(label1On=False)

    # legend
    ax.legend(loc='upper left', numpoints=1,
              prop={'size': 'x-large'})  # .draw_frame(False)

    if square:
        ax.axvspan(13600, 16200, ymin=0.44, ymax=0.97,
                   fc='None', ec='b', lw=2, ls='dashdot')
        ax.text(14900, 148, "?", rotation=0, size='x-large',
                va='center', ha='center', color='b')
        ax.set_xlim(xmin=7000, xmax=16400)
        ax.set_ylim(ymin=20, ymax=200)

    # annotate the HV SNe
    for i, sn in enumerate(dHV['sne'][filtHV]):
        if sn == 'PTF09dnp':
            vc = 0.98
        else:
            vc = 1.01
        ax.text(1.01 * dHV['vSi'][filtHV][i], vc * dHV['EWSi'][filtHV][i],
                sn, rotation=20, ha='left', va='bottom',
                size='small', color='b')

    if name is not None:
        fig.savefig(name)
    else:
        sne = N.concatenate([dn['sne'][filtn], dHV['sne'][filtHV]])
        vSi = N.concatenate([dn['vSi'][filtn], dHV['vSi'][filtHV]])
        EWSi = N.concatenate([dn['EWSi'][filtn], dHV['EWSi'][filtHV]])
        line, = ax.plot(vSi, EWSi, 'k.', ms=1)
        browser = MPL.PointBrowser(vSi, EWSi, sne, line)
        print "You should be able to browse the points to get the object name."
        P.show()


def branch_classification_plot(d, idr=None, name=None, StN=3):
    """
    This plot is usually done in the EWSiII 6355 vs vSiII 6533 space.

    :param dictionnay d: dictionnary containing the following keys:
     - CN (Core-normal)
     - BL (Broad line)
     - SS (Shallow silicon): 
     - CL (Cool):
     which much be dictionnary with the following keys:
     {'sne', 'EWSiII5972', 'EWSiII5972.err',
     EWSiII6355', 'EWSiII6355.err', 'phases'}
         * sne: the SNe list (normal and HV) 
         * EWSi*: the SiII * EW list (n and HV) (and error, .err)
         * phases: phases list (n and HV)
    :param dictionnary idr: the idr dictionnary. If given, the 'bad' sample
                            will be show in red on the plot.
    :param string name: name of the figure is you want to save it (name.ext)

    Note:
     - all the sne, vSi and EWSi array (of a given key) must have the same
     length, and must not contain nan values.

    """
    # get the data
    CN = d['CN']
    BL = d['BL']
    SS = d['SS']
    CL = d['CL']

    sne = N.concatenate([CN['sne'], BL['sne'], SS['sne'], CL['sne']])
    EWSi5 = N.concatenate([CN['EWSiII5972'], BL['EWSiII5972'],
                           SS['EWSiII5972'], CL['EWSiII5972']])
    EWSi5e = N.concatenate([CN['EWSiII5972.err'], BL['EWSiII5972.err'],
                            SS['EWSiII5972.err'], CL['EWSiII5972.err']])
    EWSi6 = N.concatenate([CN['EWSiII6355'], BL['EWSiII6355'],
                           SS['EWSiII6355'], CL['EWSiII6355']])
    EWSi6e = N.concatenate([CN['EWSiII6355.err'], BL['EWSiII6355.err'],
                            SS['EWSiII6355.err'], CL['EWSiII6355.err']])

    # figure
    fig = P.figure(dpi=120)

    # axe
    ax = fig.add_axes([0.09, 0.09, 0.69, 0.69])
    ax.set_xlabel(r'EW SiII 6355 [$\AA$]', fontsize='x-large')
    ax.set_ylabel(r'EW SiII 5972 [$\AA$]', fontsize='x-large')

    # plot
    # CN
    ax.plot(CN['EWSiII6355'], CN['EWSiII5972'], 'o',
            mec='k', mfc='k', mew=1.5, ms=8,
            label='CN (%i)' % len(CN['EWSiII6355']))
    ax.errorbar(CN['EWSiII6355'], CN['EWSiII5972'],
                xerr=CN['EWSiII6355.err'], yerr=CN['EWSiII5972.err'],
                color='k', lw=1, ls='None')
    # BL
    ax.plot(BL['EWSiII6355'], BL['EWSiII5972'], 's',
            mec='b', mfc='b', mew=1.5, ms=8,
            label='BL (%i)' % len(BL['EWSiII6355']))
    ax.errorbar(BL['EWSiII6355'], BL['EWSiII5972'],
                xerr=BL['EWSiII6355.err'], yerr=BL['EWSiII5972.err'],
                color='b', lw=1, ls='None')
    # SS
    ax.plot(SS['EWSiII6355'], SS['EWSiII5972'], '^',
            mec='g', mfc='g', mew=1.5, ms=8,
            label='SS (%i)' % len(SS['EWSiII6355']))
    ax.errorbar(SS['EWSiII6355'], SS['EWSiII5972'],
                xerr=SS['EWSiII6355.err'], yerr=SS['EWSiII5972.err'],
                color='g', lw=1, ls='None')
    # CL
    ax.plot(CL['EWSiII6355'], CL['EWSiII5972'], 'd',
            mec='r', mfc='r', mew=1.5, ms=8,
            label='CL (%i)' % len(CL['EWSiII6355']))
    ax.errorbar(CL['EWSiII6355'], CL['EWSiII5972'],
                xerr=CL['EWSiII6355.err'], yerr=CL['EWSiII5972.err'],
                color='r', lw=1, ls='None')

    ax.axhline(30, ls='--', color='k', lw=2)
    ax.vlines(80, ymin=0, ymax=30, linestyle='--', color='k', linewidth=2)
    ax.vlines(105, ymin=0, ymax=30, linestyle='--', color='k', linewidth=2)

    # legend
    ax.legend(loc='upper left', numpoints=1)

    # histo (top)
    ax2 = fig.add_axes([0.09, 0.79, 0.69, 0.19], sharex=ax)
    EW6 = N.concatenate([CN['EWSiII6355'], BL['EWSiII6355'],
                         SS['EWSiII6355'], CL['EWSiII6355']])
    bins = statistics.hist_bins(EW6)
    ax2.hist([CN['EWSiII6355'], BL['EWSiII6355'],
              SS['EWSiII6355'], CL['EWSiII6355']],
             lw=0, color=['k', 'b', 'g', 'r'], bins=bins, normed=True,
             alpha=0.7, histtype='barstacked')
    ax2.set_ylabel('Normed distr.')
    ax2.xaxis.set_tick_params(label1On=False)
    ax2.set_yticks([])

    # histo (right)
    ax3 = fig.add_axes([0.79, 0.09, 0.19, 0.69], sharey=ax)
    EW5 = N.concatenate([CN['EWSiII5972'], BL['EWSiII5972'],
                         SS['EWSiII5972'], CL['EWSiII5972']])
    bins = statistics.hist_bins(EW5)
    ax3.hist([CN['EWSiII5972'], BL['EWSiII5972'],
              SS['EWSiII5972'], CL['EWSiII5972']],
             lw=0, color=['k', 'b', 'g', 'r'], bins=bins, normed=True,
             alpha=0.7, histtype='barstacked',
             orientation='horizontal')
    ax3.set_xlabel('Normed distr.')
    ax3.yaxis.set_tick_params(label1On=False)
    ax3.set_xticks([])

    # annotate the HV SNe
    HVSNe = ['PTF09dnl', 'PTF09dnp', 'SN2004ef',
             'SN2005ir', 'SN2007qe', 'SNF20080920-000']
    SNE = N.concatenate([CN['sne'], BL['sne'], SS['sne'], CL['sne']])
    for i, sn in enumerate(SNE):
        if sn not in HVSNe:
            continue
        if sn.startswith('SNF'):
            sn = sn[6:]
        elif sn.startswith('SN'):
            sn = sn[4:]
        elif sn.startswith('PTF'):
            sn = sn[3:]
        ax.text(1.01 * EW6[i], 1.01 * EW5[i], sn, rotation=20,
                ha='left', va='bottom', size='small', color='b')

    for i, sn in enumerate(CL['sne']):
        if sn.startswith('SNF'):
            sn = sn[6:]
        elif sn.startswith('SN'):
            sn = sn[4:]
        elif sn.startswith('PTF'):
            sn = sn[3:]
        if CL['EWSiII5972'][i] < 60:
            continue
        ax.text(1.01 * CL['EWSiII6355'][i], 1.01 * CL['EWSiII5972'][i],
                sn, rotation=0, size='medium')

    if name is not None:
        fig.savefig(name)
    else:
        line, = ax.plot(EWSi6, EWSi5, 'k.', ms=1)
        browser = MPL.PointBrowser(EWSi6, EWSi5, sne, line)
        print "You should be able to browse the points to get the object name."
        P.show()


def benetti_classification_plot(res, idr=None, name=None, clean=False,
                                vcut=10, x1cut=2.5, StN=3, xaxis='dm15'):
    """
    This plot is usually done in the EWSiII 6355 vs vSiII 6533 space.

    :param dictionnay d: dictionnary containing the following keys:
     - CN (Core-normal)
     - BL (Broad line)
     - SS (Shallow silicon): 
     - CL (Cool):
     which much be dictionnary with the following keys:
     {'sne', 'EWSiII5972', 'EWSiII5972.err',
     EWSiII6355', 'EWSiII6355.err', 'phases'}
         * sne: the SNe list (normal and HV) 
         * EWSi*: the SiII * EW list (n and HV) (and error, .err)
         * phases: phases list (n and HV)
    :param dictionnary idr: the idr dictionnary. If given, the 'bad' sample
                            will be show in red on the plot.
    :param string name: name of the figure is you want to save it (name.ext)
    :param string xaxis: 'dm15' or 'x1'.

    Note:
     - all the sne, vSi and EWSi array (of a given key) must have the same
     length, and must not contain nan values.

    """
    # get the data
    sne = N.array([sn for sn in res if sn != 'info'])
    vsi = N.array([res[sn]['Jparams'][0] for sn in sne])
    vsie = N.array([res[sn]['Jdparams'][0] for sn in sne])
    dm15 = N.array([res[sn]['salt2']['dm15'] for sn in sne])
    dm15e = N.array([res[sn]['salt2']['dm15.err'] for sn in sne])
    x1 = N.array([res[sn]['salt2']['x1'] for sn in sne])
    x1e = N.array([res[sn]['salt2']['x1.err'] for sn in sne])

    if xaxis == 'dm15':
        xval = dm15
        xvale = dm15e
        from ToolBox.Wrappers import SALT2model
        xvalf = (xval >= SALT2model.x1_to_dm15(x1cut))
    elif xaxis == 'x1':
        xval = x1
        xvale = x1e
        xvalf = (xval <= x1cut)
    else:
        raise KeyError("Error: option xaxis must be 'dm15' or 'x1'")

    # figure
    fig = P.figure(dpi=120)

    # axe
    ax = fig.add_axes([0.1, 0.08, 0.88, 0.87])

    # plot
    if clean:
        filt = (-vsi >= vcut) & xvalf & \
               (N.abs(vsi) / vsie >= StN) & (xval / xvale >= StN)
        sne, xval, xvale, x1, x1e, vsi, vsie = sne[filt], \
            xval[filt], xvale[filt], \
            x1[filt], x1e[filt], \
            vsi[filt], vsie[filt]

    ax.plot(xval, -vsi, 'o', mec='k', mfc='None',
            mew=1.5, ms=8, label='All SNe')
    ax.errorbar(xval, -vsi, xerr=xvale, yerr=vsie, capsize=None,
                color='k', lw=1, ls='None')

    # add red points in the middle for the bad sample
    if idr is not None:
        if 'DATASET' in idr:
            del idr['DATASET']
        filt = N.array([((sn in idr) & (idr[sn]['idr.subset'] == 'bad'))
                        for sn in sne], dtype='bool')
        ax.plot(xval[filt], -vsi[filt], 'ro',
                ms=3, label='"Bad" LC (%i)' % (len(xval[filt])))

    # annotate some points
    for i, sn in enumerate(sne):
        if xval[i] > 0.8 and vsi[i] < 0 and not clean:
            continue
        if clean \
                and -vsi[i] > 15 \
                and xval[i] < 1.6 \
                and -vsi[i] < 110:
            continue
        if sn.startswith('SNF'):
            sn = sn[6:]
        elif sn.startswith('SN'):
            sn = sn[4:]
        elif sn.startswith('PTF'):
            sn = sn[3:]
        ax.text(1.01 * xval[i], -1.01 * vsi[i],
                sn, rotation=0, size='x-small')

    # hline
    ax.axhline(-0.01, ls='--', color='k')
    if clean:
        ax.axhspan(65, 75, color='b', alpha=0.1, label='LVG-HVG transition')
        ax.axvspan(1.5, 1.7, color='r', alpha=0.1, label='Faint transition')
        ax.text(0.82, 4, 'LVG', size='x-large', color='g')
        ax.text(1.75, 150, 'FAINT', size='x-large', color='r')
        ax.text(0.9, 150, 'HVG', size='x-large', color='b')

    # legend labels and title
    ax.legend(loc='best', numpoints=1).draw_frame(False)
    ax.set_title('%i SNe, %.1f < p < %.1f ' %
                 (len(sne), res['info']['prange'][0], res['info']['prange'][1]))
    ax.set_xlabel('xval (from SALT2)')
    ax.set_ylabel(r'-$\dot{v}$ [km/s/day]')

    if clean:
        ax.set_ylim(ymin=0)

    if name is not None:
        fig.savefig(name)
    else:
        line, = ax.plot(xval, -vsi, 'k.', ms=1)
        browser = MPL.PointBrowser(xval, -vsi, sne, line)
        print "You should be able to browse the points to get the object name."
        P.show()


def vg_vs_v_plot(diw, dibe, idr=None):

    # get the velocity gradient data
    sne = N.array([sn for sn in dibe if sn != 'info'])

    # get the velocity data
    dn = diw['normal']
    dHV = diw['HV']
    snev = N.concatenate([diw['normal']['sne'], diw['HV']['sne']])
    vsi = N.concatenate([diw['normal']['vSi'], diw['HV']['vSi']])
    vsie = N.concatenate([diw['normal']['vSi.err'], diw['HV']['vSi.err']])

    ft = [False] * len(snev)
    filt = N.array([True and sn in sne or False for sn in snev], dtype='bool')
    snev = snev[filt]
    vsi = vsi[filt]
    vsie = vsie[filt]

    vgsi = N.array([dibe[sn]['Jparams'][0] for sn in snev])
    vgsie = N.array([dibe[sn]['Jdparams'][0] for sn in snev])

    print len(vsi), len(vgsi)
    print vsi
    print vgsi
    # figure
    fig = P.figure(dpi=120)

    # axe
    ax = fig.add_axes([0.1, 0.08, 0.88, 0.87])

    # plot

    ax.plot(vsi, -vgsi, 'o', mec='k', mfc='None', mew=2, label='All SNe')
    ax.errorbar(vsi, -vgsi, xerr=vsie, yerr=vgsie, capsize=None,
                color='k', lw=1, ls='None')

    # legend labels and title
    ax.legend(loc='best', numpoints=1).draw_frame(False)
    # ax.set_title('%i SNe, %.1f < p < %.1f '%\
    #             (len(sne),res['info']['prange'][0],res['info']['prange'][1]))
    ax.set_ylabel(r'-$\dot{v}$ [km/s/day]')
    ax.set_xlabel('vSi 6355')


def vg_v_vs_zplot(diw, dibe, idr=None):

    # get the velocity gradient data
    sne = N.array([sn for sn in dibe if sn != 'info'])

    # get the velocity data
    dn = diw['normal']
    dHV = diw['HV']
    snev = N.concatenate([diw['normal']['sne'], diw['HV']['sne']])
    vsi = N.concatenate([diw['normal']['vSi'], diw['HV']['vSi']])
    vsie = N.concatenate([diw['normal']['vSi.err'], diw['HV']['vSi.err']])

    ft = [False] * len(snev)
    filt = N.array([True and sn in sne or False for sn in snev], dtype='bool')
    snev = snev[filt]
    vsi = vsi[filt]
    vsie = vsie[filt]

    vgsi = N.array([dibe[sn]['Jparams'][0] for sn in snev])
    vgsie = N.array([dibe[sn]['Jdparams'][0] for sn in snev])

    z = N.array([idr[sn]['salt2.Redshift'] for sn in snev])
    # figure
    fig = P.figure(dpi=120)

    # axe
    ax = fig.add_axes([0.1, 0.08, 0.88, 0.87])

    # plot

    ax.plot(vsi, z, 'o', mec='k', mfc='None', mew=2, label='All SNe')
    ax.errorbar(vsi, z, xerr=vsie, capsize=None,
                color='k', lw=1, ls='None')

    # legend labels and title
    ax.legend(loc='best', numpoints=1).draw_frame(False)
    # ax.set_title('%i SNe, %.1f < p < %.1f '%\
    #             (len(sne),res['info']['prange'][0],res['info']['prange'][1]))
    ax.set_ylabel(r'-$\dot{v}$ [km/s/day]')
    ax.set_xlabel('vSi 6355')


def test(wang=True, branch=True, benetti=True, vgrad=True,
         wname=None, brname=None, bename=None, prange=5,
         keepall=False, square=True):

    # import and data loading
    idrpath = '/Users/nicolaschotard/work/data/IDR/ACEv3'
    idr = io.load_anyfile(idrpath + '/META.pkl')
    phreno = io.load_anyfile(idrpath + '/phrenology_ACEv3.pkl')
    nd = merge_phreno_idr(idr, phreno)

    results = []

    # wang
    if wang:
        diw = wang_classification(nd, prange=prange, keepall=keepall)
        wang_classification_plot(diw, idr=idr, name=wname, square=square)
        results.append(diw)

    # branch
    if branch:
        dib = branch_classification(nd, prange=prange, keepall=keepall)
        branch_classification_plot(dib, idr=idr, name=brname, StN=0)
        results.append(dib)

    # benetti
    if benetti:
        dibe = benetti_classification(nd)
        benetti_classification_plot(dibe, idr=idr, clean=True)
        results.append(dibe)

    # v gradient vs. v
    if vgrad:
        diw = wang_classification(nd, prange=prange, keepall=keepall)
        dibe = benetti_classification(nd)
        vg_vs_v_plot(diw, dibe, idr=idr)

        # v vs redshift
        vg_v_vs_zplot(diw, dibe, idr=idr)

    P.show()
    return results


def table(output='html'):
    """
    Get all the classifications and put them into a table (tex or html)
    """
    snid_dir = '/Users/nicolaschotard/work/data/snid/runs/'
    results = test()
    dw, dbr, dbe = results[0], results[1], results[2]

    res_b = io.load_anyfile(snid_dir + 'bsnip/snid_results.pkl')
    res_s = io.load_anyfile(snid_dir + 'snid-2.0/snid_results.pkl')
    res_bp = io.load_anyfile(snid_dir + 'bsnip/snid_results_fixp.pkl')
    res_sp = io.load_anyfile(snid_dir + 'snid-2.0/snid_results_fixp.pkl')

    if output == 'html':
        print "Object   SNID-2.0  SNID-2.0 (fp)  Silv  Silv (fp)  Wang   Branch  Benetti"
    elif output == 'tex':
        print r'\\begin{table}'
        print r'\\caption{Spectral classification.}'
        print r'\\centering'
        print r'\\begin{tabular}{cccccccc}'
        print r'\\\ \hline \hline'
        print r"Object & SNID-2.0 & SNID-2.0 (fp) & Silv & Silv (fp) & Wang "\
              r"& Branch & Benetti \\\ \hline \hline"
    for sn in sorted(res_b):
        if sn not in dw['info']['bysn']:
            dw['info']['bysn'][sn] = ''
        if sn not in dbr['info']['bysn']:
            dbr['info']['bysn'][sn] = ''
        if sn not in dbe:
            dbe[sn] = {'type': ''}

        if output == 'html':
            print '<tr align="center"> <td> %s </td> <td> %s </td> <td> %s '\
                  '</td> <td> %s </td> <td> %s </td> <td> %s </td> <td> %s '\
                  '</td> <td> %s </td> </tr>' %\
                  (sn,
                   res_s[sn]['general']['subtype'],
                   res_sp[sn]['general']['subtype'],
                   res_b[sn]['general']['subtype'],
                   res_bp[sn]['general']['subtype'],
                   dw['info']['bysn'][sn],
                   dbr['info']['bysn'][sn],
                   dbe[sn]['type'])

        elif output == 'tex':
            print r'%s & %s & %s & %s & %s & %s & %s & %s \\\ ' %\
                  (sn,
                   res_s[sn]['general']['subtype'],
                   res_sp[sn]['general']['subtype'],
                   res_b[sn]['general']['subtype'],
                   res_bp[sn]['general']['subtype'],
                   dw['info']['bysn'][sn],
                   dbr['info']['bysn'][sn],
                   dbe[sn]['type'])

    if output == 'tex':
        print r'\hline'
        print r'\end{tabular}'
        print r'\label{table:classification}'
        print r'\end{table}'

"""
<table  cellpadding="6" border="1" align="center">
<caption> SNfactory SNe classifications. See above for detail on the
columns.

<tr>
<th> Templates </th>
<th> SNID-2.0 </th>
<th> SNID-2.0 (fp) </th>
<th> Silverman </th>
<th> Silverman (fp) </th>
<th> Wang 09 </th>
<th> Branch 09 </th>
</tr>
<tr>
<td> Object </td>
<td align="center" colspan="6" > type / sub-type </td>
</tr>

<tr> <td> a</td> <td> a</td> <td> a</td> <td> a</td> <td> a</td> <td>
    a</td> <td> a</td> </tr>
</table> 
"""
# End of classLib.py

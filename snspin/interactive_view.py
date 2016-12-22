#!/usr/bin/env python
######################################################################
## Filename:      interactive_view.py
## Version:       $Revision: 1.28 $
## Description:   Plots indicators vs other parameters for a given production
## Author:        Nicolas Chotard <n.chotard@ipnl.in2p3.fr>
## Author:        $Author: nchotard $
## Created at:    Monday 6 september 10h:10:00 2008
## $Id: interactive_view.py,v 1.28 2012/11/22 03:57:23 nchotard Exp $
######################################################################

from copy import copy

import pylab as P
import numpy as N

from ToolBox import MPL, Hubblefit as H
from ToolBox.Statistics import correlation as Corr
import SnfMetaData

class InterView:
    """missing : a doc string describing the classe members
    self.paramlist ?
    self.newkeys?
    self.pdata : the phrenology dictionnary
    self.hdata : the hubblizer dicitonnary (if any)
   """

    def __init__(self, phreno, idr=None, hubble=None, sne=None, newkeys=[],
                 phase=0, cmap=P.cm.copper_r):
        """phreno : moksha dictionnary or a phrenology dictionnary
        hubble : hubblizer output dictionnary
        sne : a list of sne or a file containing a list of sne.
        newkeys : undocumented feature
        phase : the phase of the selected spectrum inside a supernova in
        case of ambiguities

        Hubblizer parameters
        ['hubblizer.SaltFitMag.err',
         'hubblizer.SaltFitMag',
         'hubblizer.mBfit',
         'hubblizer.dmfit_orig.err',
         'hubblizer.mBfit.err',
         'hubblizer.dmfit_orig',
         'hubblizer.dmfit_corr.err',
         'hubblizer.dmfit_corr',
         'hubblizer.SaltFitMag.fullerr']
         """
        
        if hubble is not None:
            self.params_list = ['salt2.Redshift', 'salt2.Color', 'salt2.X1',
                                'salt2.RestFrameMag_0_B', 'dmfit_orig',
                                'dmfit_corr','salt2.dm15'] + newkeys
        else:
            self.params_list = ['salt2.Redshift', 'salt2.Color', 'salt2.X1',
                                'salt2.RestFrameMag_0_B','salt2.dm15'] + newkeys
        self.newkeys = newkeys
        
        #Open files used to plot or otherwise read input parameters
        if isinstance(phreno, dict):
            self.pdata = phreno
        else:
            self.pdata = SnfMetaData.load(phreno)
        if idr is not None:
            if idr.endswith('pkl'):
                self.pdata = SnfMetaData.load(self.pdata, idr)
            else:
                self.pdata = SnfMetaData.load(self.pdata, idr+'/META.pkl')
          
        if hubble is not None:
            if isinstance(hubble, dict):
                self.hdata=hubble
            else:
                self.pdata = SnfMetaData.load(hubble)
        else: self.hdata = None

        #Create a list of SNe
        if sne is not None:
            if isinstance(sne, str):
                self.listSNe = N.loadtxt(sne, dtype='string')
            else:
                self.listSNe = N.array(sne, dtype='string')
        elif hubble is not None:
            self.listSNe = N.array(self.hdata.keys())
        else:
            self.listSNe = N.array(self.pdata.keys())
        self.listSNe = N.array([ sn for sn in self.listSNe if sn != 'DATASET'])
        self.phase = phase

        #Get names of metrics and errors
        self._get_metrics_name()

        #Organise parameters
        self._organize_parameters()

        #Define propeties
        self.save = False
        self.output = ['png']
        self.colorlegend_name = 'x1'
        self.linfit = False
        self.cutkey = None
        self.inf = -N.inf
        self.sup = N.inf
        self.colorbar = False
        self.text_sne = False
        self.peculiar = []
        self.keepcut = False
        self.values_only = False

        self.cmap=cmap

        #Set the labels
        self._label_names()
        
    def _get_metrics_name(self):

        #Get the list of all parameters in 'spectra'
        for i in range(len(self.listSNe)):
            try:
                list = self.pdata[ self.listSNe[i] ]['spectra'][ (self.pdata[self.listSNe[i]]['spectra'].items())[0][0] ].keys(); break
            except:
                continue
        
        #Keep only spectral indicators
        list.sort()
        self.metrics_name = copy(list)
        for m in list:
            if m[:3]  != 'phr' : self.metrics_name.remove(m); continue
            if m[-3:] == 'err' : self.metrics_name.remove(m); continue
            if m[-4:] == 'stat': self.metrics_name.remove(m); continue
            if m[-4:] == 'syst': self.metrics_name.remove(m); continue
            if m[-4:] == 'mean': self.metrics_name.remove(m); continue
            if m[-4:] == 'flux': self.metrics_name.remove(m); continue
            if m[-3:] == 'lbd' : self.metrics_name.remove(m); continue
            if m[-3:] == 'bin' : self.metrics_name.remove(m); continue
            if m[:15]  == 'phrenology.flux':
                self.metrics_name.remove(m)
                continue
            if m[:14]  == 'phrenology.lbd' :
                self.metrics_name.remove(m)
                continue
            try:
                if len(self.pdata[self.listSNe[i]]['spectra'][(self.pdata[listSNe[i]]['spectra'].keys())[0]][m]) >= 0:
                    self.metrics_name.remove(m)
            except:
                continue
        self.metrics_name = [m[11:] for m in self.metrics_name]

    def _organize_parameters(self):

        print 'Organize parameters=================================='
        listSNe, spectra_at_max, allphases=[], [], []
        for SN in self.listSNe:
            psn = self.pdata[SN]
            hsn = self.hdata[SN]
            if len(psn['spectra']) == 0:
                continue
            
            if self.hdata is not None:
                psn['zerr'] = hsn['host.zhelio.err']
                psn['zcmb'] = hsn['host.zcmb']
                psn['dmfit_orig'] = hsn['hubblizer.dmfit_orig']
                psn['dmfit_corr'] = hsn['hubblizer.dmfit_corr']
                psn['dmfit_orig.err'] = hsn['hubblizer.dmfit_orig.err']
                psn['dmfit_corr.err'] = hsn['hubblizer.dmfit_corr.err']
                psn['salt2.Redshift.err'] = N.sqrt(hsn['host.zhelio.err']**2 \
                                                   + 0.001**2 )
                psn['salt2.Redshift'] = hsn['salt2.Redshift']
                psn['salt2.Redshift.err'] = hsn['salt2.Redshift.err']
                psn['salt2.Color'] = hsn['salt2.Color']
                psn['salt2.Color.err'] = hsn['salt2.Color.err']
                psn['salt2.X1'] = hsn['salt2.X1']
                psn['salt2.X1.err'] = hsn['salt2.X1.err']
                psn['salt2.RestFrameMag_0_B'] = hsn['salt2.RestFrameMag_0_B']
                psn['salt2.RestFrameMag_0_B.err'] = hsn['salt2.RestFrameMag_0_B.err']

            spectra = N.array(psn['spectra'].keys()) 
            if not psn['spectra'][spectra[0]].has_key('obs.phase'):
                for sid in psn['spectra']:
                    psns = psn['spectra'][sid]
                    psns['obs.phase'] = float(psns['obs.mjd'] - \
                                              psn['salt2.DayMax'])   

            if not len(spectra) == 1:
                print 'Get the spectrum near %i...done'%self.phase
                phases = N.array([psn['spectra'][spec]['obs.phase'] for spec in spectra])
                phase = phases[N.argmin(N.abs(phases-self.phase))]
                spec_at_max = spectra[N.argmin(N.abs(phases-self.phase))]
            else:
                spec_at_max = spectra[0]
                phase = psn['spectra'][spec_at_max]['obs.phase']
            

            print SN, spec_at_max
            listSNe.append(SN)
            spectra_at_max.append(spec_at_max)
            allphases.append(phase)
            for newkey in self.newkeys:
                try: psn[newkey]  = float(hsn[newkey])
                except: continue

        self.listSNe = N.array(listSNe)
        self.spectra_at_max = N.array(spectra_at_max)
        self.phases = N.array(allphases)

        params  = {}
        phase, phase_err = [], []
        for key in self.params_list:
            params[key] = []
            params[key+".err"] = []
    
        metrics = {}
        for metric in self.metrics_name:
            metrics[metric] = []
            metrics[metric+'.err'] = []

        for SN, spectrum in zip(self.listSNe, self.spectra_at_max):
            psn = self.pdata[SN]
            if len(psn['spectra']) == 0:
                continue
            else:
                tdic = psn['spectra']
            for key in params:
                try:
                    params[key].append(float(psn[key]))
                except:
                    params[key].append(0.0)
                    
            phase.append(float('%.3f'%tdic[spectrum]['obs.phase']))
            phase_err.append(float(0.0))
            
            for metric in self.metrics_name:
                try:
                    if tdic[spectrum]['phrenology.'+metric] is None:
                        tdic[spectrum]['phrenology.'+metric] = N.nan
                    try:
                        if tdic[spectrum]['phrenology.'+metric+'.err'] is None:
                            pass
                    except:
                        tdic[spectrum]['phrenology.'+metric+'.err'] = float(0)
                except:
                    tdic[spectrum]['phrenology.'+metric] = N.nan
                    tdic[spectrum]['phrenology.'+metric+'.err'] = N.nan
            
                metrics[metric].append(float(tdic[spectrum]['phrenology.' + \
                                                            metric]))
                metrics[metric+'.err'].append( float(tdic[spectrum]['phrenology.'+metric+'.err']))
    
        
        
        for metric in self.metrics_name:
            metrics[metric] = N.array(metrics[metric])
            metrics[metric+'.err']  = N.array(metrics[metric+'.err'] )
    
        # Transform params into Nico's dictionnary format
        for key in self.params_list:
            params[key] = [N.array(params.pop(key)),
                           N.array(params.pop(key+".err"))]
    
        params['phase'] = [N.array(phase), N.array(phase_err)]
        # This is to have some "nice and cute" names 
        params['z'] = params.pop('salt2.Redshift')
        params['color'] = params.pop('salt2.Color')
        params['x1'] = params.pop('salt2.X1')
        params['mb'] = params.pop('salt2.RestFrameMag_0_B')
        params['dm15'] = params.pop('salt2.dm15')

                        
        self.params  = params
        self.metrics = metrics

    def add_ratio(self, key1, key2):
        """
        If ratio option  is given, create a new value with the two first ones
        """
        newkey  = key1 + '_' + key2
        if key1 not in self.metrics_name or key2 not in self.metrics_name:
            print self.metrics_name
            raise ValueError, \
                  'Error, one of the parameters is not in the phrenonoly list'
        else:
            ratio = self.metrics[key1] / self.metrics[key2]
            self.metrics[newkey] = ratio
            self.metrics[newkey+'.err'] = ( ratio ) * \
                                          ( self.metrics[key1+'.err'] \
                                            / self.metrics[key1] \
                                            + self.metrics[key2+'.err'] \
                                            / self.metrics[key2] )
            self.metrics_name = N.concatenate((self.metrics_name, [newkey]))
            print 'New key is added as %s in self.metrics'%newkey

    def _define_filter_colorbar(self):

        if self.cutkey is not None:
            self.filter = self._cutvalues(self.cutkey, inf=self.inf, sup=self.sup)
        else:
            self.filter = N.array(['True'] \
                                  * len(self.params[self.colorlegend_name][0]),
                                  dtype='bool')
        self.colorlegend = self.params[self.colorlegend_name][0][self.filter]

    def plot_parameters(self, key1=None, key2=None, linfit=True,
                        colorlegend='x1', colorbar=True, save=False,
                        output=['png'], cutkey=None, inf=-N.inf,
                        sup=N.inf, textsne=False, peculiar=[],
                        keepcut=False, values_only=False, cmap=False, StN=0):

        """
        You can give one or two keys. If you give to the function only one key,
        it will plot this key versus all the other ones (65 figures...dangerous)
        There is several options:
        linfit: if True, make a linear fit and show the slope and the chi2
        (default is True).
        colorlegend: Name of the parameter used for the color legend in the
        colorbar (default is x1).
        colorbar: If False, no colorbar and no colorlegend (default is True).
        save: True if you want to save your figure (default is False).
        output: Format of your saved figure if the 'save' option is True
        (default is ['png'], you can add some output extensions in this array).
        cutkey: If None, there will be no cut. Else, this key must be in the
        list of key. Must be used with the 'inf' and 'sup' options (default
        is None).
        inf and sup: lower and higher bound of the cut (default is -inf and
        +inf respectively).
        textsne: If True, the name of each SNe will be added just near the
        corresponding point (default is False).
        peculiar: List of peculiar SNe. They will be plotted with an other
        symbol (default is an empty array).
        keepcut: If there is a cut, and the 'keepcut' option is True, the
        cutted values will be plotted in black. Else, they will not be plotted
        (default is False).
        values_only: If it's True, compute the correlation, make the linear
        fit (if linfit is True), and that's all. There will be no plot. Only
        correlation and fit values will be saved.
        """
        self.save = save
        self.output = output
        self.colorlegend_name = colorlegend
        self.linfit = linfit
        self.cutkey = cutkey
        self.inf = inf
        self.sup = sup
        self.colorbar = colorbar
        self.text_sne = textsne
        self.peculiar = peculiar
        self.keepcut = keepcut
        self.values_only = values_only
        self.StN = StN

        #Define filter and colorbar legend
        self._define_filter_colorbar()
            
        if key1 is not None and key2 is not None:
            #Plot only key1 versus key2
            if self.params.has_key(key1) == True \
                   and self.params.has_key(key2) == True:
                param1 = [self.params[key1][0], self.params[key1][1]]
                param2 = [self.params[key2][0], self.params[key2][1]]
                self._plot(param1, param2, key1, key2, cmap=cmap)
            elif self.params.has_key(key1) == True \
                     and self.metrics.has_key(key2) == True:
                metric = [self.metrics[key2], self.metrics[key2+'.err']]
                param = [self.params[key1][0], self.params[key1][1]]
                self._plot(param, metric, key1, key2, cmap=cmap)
            elif self.params.has_key(key2) == True \
                     and self.metrics.has_key(key1) == True:
                metric = [self.metrics[key1], self.metrics[key1+'.err']]
                param = [self.params[key2][0], self.params[key2][1]]
                self._plot(metric, param, key1, key2, cmap=cmap)
            elif self.metrics.has_key(key2) == True \
                     and self.metrics.has_key(key1) == True:
                metric1 = [self.metrics[key1], self.metrics[key1+'.err']]
                metric2 = [self.metrics[key2], self.metrics[key2+'.err']]
                self._plot(metric1, metric2, key1, key2, cmap=cmap)
            else:
                print '%s and/or %s not in the following list of parameters:\n %s \n %s' % \
                    (key1, key2, self.metrics_name, self.params_list)
            return
        elif key1 is not None and key2 == None:
            #Plot only key1 versus all other parameters
            if self.params.has_key(key1) == True: 
                param1 = [self.params[key1][0], self.params[key1][1]]
                for g in self.params:
                    print g
                    param2 = [self.params[g][0], self.params[g][1]] 
                    self._plot(param1, param2, key1, g, cmap=cmap)
                for m in self.metrics_name:
                    print m[11:], m[11:]+'.err'
                    metric = [self.metrics[m], self.metrics[m+'.err']]
                    self._plot(param1, metric, key1, m[11:], cmap=cmap)
            elif self.metrics.has_key(key1) == True:
                print key1, '==========================='
                metric1 = [self.metrics[key1], self.metrics[key1+'.err']]
                for m2 in self.metrics_name:
                    print m2[11:], m2[11:]+'.err'
                    metric2 = [self.metrics[m2], self.metrics[m2+'.err']]
                    self._plot(metric1, metric2, key1, m2[11:], cmap=cmap)
                for g in self.params:
                    print g
                    param2 = [self.params[g][0], self.params[g][1]] 
                    self._plot(metric1, param2, key1, g, cmap=cmap)
        else:
            print "Give me one (key1) or two keys"; return
   
        
    def _cutvalues(self, key, inf=-N.inf, sup=N.inf):
        
        listSNe_cut = []
        filter = []
        if key in self.params: 
            for i, sn in enumerate(self.listSNe):
                if self.params[key][0][i] < inf or self.params[key][0][i] > sup:
                    print sn, self.params[key][0][i]
                    filter.append(0)
                else:
                    listSNe_cut.append(sn)
                    filter.append(1)
        elif key in self.metrics:
            for i, sn in enumerate(self.listSNe):
                if self.metrics[key][i] < inf or self.metrics[key][i] > sup:
                    print sn, self.metrics[key][i]
                    filter.append(0)
                else:
                    filter.append(1)
                    listSNe_cut.append(sn)
        else:
            raise ValueError, \
                  'Error. Name of the cut must be in\n %s\n or in\n %s' % \
                  (self.params.keys(), self.metrics.keys())
        
        print 'list of SNe used after %s cut saved in self.listSNe_cut' % key
        self.listSNe_cut = N.array(listSNe_cut)
        
        return N.array(filter, dtype='bool')

    def _plot(self, x, y, xname, yname, cmap=False):

        if not cmap: cmap = self.cmap

        self.x, self.y = x, y
        
        #Keep only 'good' values
        x = [x[0][self.filter], x[1][self.filter]]
        y = [y[0][self.filter], y[1][self.filter]]
        filt = N.isfinite(x[0])*N.isfinite(y[0])
        if xname in self.metrics and yname in self.metrics:
            filt *= ((x[0]/x[1])>self.StN)*((y[0]/y[1])>self.StN)
        elif xname in self.metrics:
            filt *= (x[0]/x[1])>self.StN
        elif yname in self.metrics:
            filt *= (y[0]/y[1])>self.StN
        else:
            pass
            
        x, y = [x[0][filt], x[1][filt]], [y[0][filt], y[1][filt]]

        #Compute correlation coefficient
        correlation, corred, correp = Corr(x[0], y[0], error=True)
    
        # Simple linear fit
        if self.linfit:
            print 'Linear fit done without error taken into acount'
            self.pol = N.polyfit(x[0], y[0], 1)                

        #If it's True, there will be no plot.
        #Only correlation and fit parameters will be saved.
        if self.values_only:
            return 

        fig = P.figure(dpi=150)
        ax = fig.add_axes([0.10, 0.08, 0.87, 0.84])

        #Plot the linear fit
        if self.linfit:
            ax.plot(x[0], N.polyval(self.pol, x[0]), 'r')

        #Make the colorbar if the option is True
        if self.colorbar:

            color = self.colorlegend[filt]

            #Normalization of the colorbar
            norm = ( color - color.min() ) / ( color - color.min() ).max()
            col = cmap(norm)

            #Plot colored points and error bar
            for i, c in enumerate(col):
                if (self.listSNe)[self.filter][i] not in self.peculiar:
                    #For a non peculiar SN
                    ax.errorbar(x[0][i], y[0][i], xerr=x[1][i], yerr=y[1][i],
                                linestyle='None', capsize=0, ecolor=c,
                                marker='o', mfc=c, mec=c)
                else:
                    #For a peculiar SN
                    ax.errorbar(x[0][i], y[0][i], xerr=x[1][i], yerr=y[1][i],
                                linestyle='None', capsize=0, ecolor=c,
                                marker='D', mfc=c, mec=c)

            #Inverse the color of the colorbar for 'x1' only
            scat = ax.scatter(x[0], y[0], c=color, edgecolor='none',
                              cmap=(cmap), visible=True)

            #Plot the colorbar
            cb = fig.colorbar(scat,format='%.3f')

            #Legend of the colorbar
            cb.set_label(self._get_label(self.colorlegend_name))
            
        else: #If no colorbar
            for i in range(len(x[0])):
                #Plot black points and error bar
                if (self.listSNe)[self.filter][i] not in self.peculiar:
                    #For a non peculiar SN
                    ax.errorbar(x[0][i], y[0][i], xerr=x[1][i], yerr=y[1][i],
                                linestyle='None', capsize=0, ecolor='k',
                                marker='o', mfc='k', mec='k')
                else:
                    #For a peculiar SN
                    ax.errorbar(x[0][i], y[0][i], xerr=x[1][i], yerr=y[1][i],
                                linestyle='None', capsize=0, ecolor='k',
                                marker='D', mfc='k', mec='k')

        #Plot not only non cuted points
        if self.keepcut and (~self.filter).any():
            filt2 = N.isfinite(self.x[0][~self.filter]) * \
                    N.isfinite(self.y[0][~self.filter])
            ax.errorbar(self.x[0][~self.filter][filt2],
                        self.y[0][~self.filter][filt2],
                        xerr=self.x[1][~self.filter][filt2],
                        yerr=self.y[1][~self.filter][filt2],
                        linestyle='None', capsize=0, ecolor='k',
                        marker='s', mfc='k', mec='k')
            #Plot the name of each cuted SN
            if self.text_sne:
                for xx, yy, sn in zip(self.x[0][~self.filter][filt2],
                                    self.y[0][~self.filter][filt2],
                                    (self.listSNe)[~self.filter][filt2]):
                    print sn, xx, yy
                    if sn[:3] == 'SNF':
                        ax.text(xx, yy, ' %s'%sn[5:], rotation=45, size='x-small')
                    else:
                        ax.text(xx, yy, ' %s'%sn[4:], rotation=45, size='x-small')

            filt3 = N.isfinite(self.x[0])*N.isfinite(self.y[0])
            scat = ax.scatter(self.x[0][filt3], self.y[0][filt3], c='k',
                              edgecolor='none', visible=False)
            browser = MPL.PointBrowser(self.x[0][filt2],
                                       self.y[0][filt2],
                                       self.text_sne[filt3], scat)
        else:
            scat = ax.scatter(x[0],y[0],c='k',edgecolor='none')
            browser = MPL.PointBrowser(x[0], y[0],
                                       self.listSNe[self.filter][filt], scat)

        #Change the name of some parameters
        xname = self._get_label(xname)
        yname = self._get_label(yname)

        #Labels and annotations
        ax.set_xlabel('%s' %xname  )
        ax.set_ylabel('%s' %yname )
        ax.set_title(r'%s vs %s (%i objects)' % \
                     (xname, yname, len(N.array(x[0]))))
        ax.annotate(r'$\rho=%.2f^{+%.2f}_{-%.2f}$' % \
                    (correlation, correp, corred),
                    xy=(0.01,  0.99), xycoords='axes fraction',
                    xytext=(0.01, 0.99), textcoords='axes fraction',
                    horizontalalignment='left',
                    verticalalignment='top', fontsize=12)
        if self.linfit:
            ax.annotate('Slope=%.3f'%self.pol[0],
                        xy=(0.98, 0.99), xycoords='axes fraction',
                        xytext=(0.98, 0.99), textcoords='axes fraction',
                        horizontalalignment='right',
                        verticalalignment='top', fontsize=10)

        #Plot the name of each SN
        if self.text_sne:
            for xx, yy, sn in zip(x[0], y[0], (self.listSNe)[self.filter][filt]):
                print sn, xx, yy
                if sn[:3] == 'SNF':
                    ax.text(xx, yy, ' %s'%sn[5:],  rotation=45, size='x-small')
                else:
                    ax.text(xx, yy, ' %s'%sn[4:], rotation=45, size='x-small')

        #Plot some lines
        if xname == '$\Delta \mu_B$':
            xname='DeltaMB' #Change the name for the saved file
            ax.axvline(0, color='k', linewidth=1, label = '_nolegend_' )
        if xname == '$\Delta \mu_B^c$':
            xname='DeltaMBcorr'
            ax.axvline(0, color='k', linewidth=1, label = '_nolegend_' )
        if yname == '$\Delta \mu_B$':
            yname='DeltaMB'
            ax.axhline(0, color='k', linewidth=1, label = '_nolegend_' )
        if yname == '$\Delta \mu_B^c$':
            yname='DeltaMBcorr'
            ax.axhline(0, color='k', linewidth=1, label = '_nolegend_' )

        #Save or show the figure
        if self.save:
            for out in self.output:
                fig.savefig(xname+'_vs_'+yname+'.'+out)
            fig.close()

    def sort_by(self, key):

        """
        Print the list of SN sorted by the given value.
        Input: Name of the value (x1 or color for instance)
        """
        print len(self.listSNe), len(self.params['color'][0])
        for i, sn in enumerate(self.listSNe):
            print sn, self.params['color'][0][i]
        listSNe, values = [], []
        if key in self.params.keys():
            order = N.argsort(self.params[key][0])
            for SN, value in zip(self.listSNe[order], self.params[key][0][order]):
                listSNe.append(SN)
                values.append(value)
        elif key in self.metrics.keys():
            order = N.argsort(self.metrics[key])
            for SN, value in zip(self.listSNe[order], self.metrics[key][order]):
                listSNe.append(SN)
                values.append(value)
        else:
            print 'This key is not in the following list of parameters:\n'
            print self.params.keys()
            print self.metrics.keys()

        return map(N.array, [listSNe, values])

    def multiplots(self,  list_keys=[], list_keys2=False, space=0.04, left=0.08,
                   right=0.97, bottom=0.05, top=0.95, errorbar=True,
                   colorlegend=False, histtype= 'step',
                   coords=[0.35,0.1,0.6,0.8], title='',
                   cut=[-N.inf,N.inf], corrcol=False, dpi=150,
                   cmap=False, corr=False, StN=0):

        """
        space: space between each axe
        left, right, bottom, top: space between axes and the edges of the
        figure (% of axis)
        cut: every values not in this range will be cut
        """
        if not cmap:
            cmap = self.cmap
        if not list_keys2:
            list_keys2 = list_keys

        nvar, nvar2 = len(list_keys), len(list_keys2)
        size = (N.sqrt(len(list_keys2))*4, N.sqrt(len(list_keys))*4)
        
        fig = P.figure(figsize=size, dpi=dpi)
        #fig = P.figure(dpi=dpi)
        fig.subplots_adjust(hspace=space, wspace=space, right=0.85, left=0.09)
        
        if colorlegend:
            if colorlegend in self.metrics:
                color = self.metrics[colorlegend]
            elif colorlegend in self.params:
                color = self.params[colorlegend][0]
            elif colorlegend == 'corr':
                corrcol = True
                color = []
            else:
                raise 'Colorlegend not in the list of parameters'
            
        for i, key1 in enumerate(list_keys):
                
            if key1 in self.metrics:
                x = [self.metrics[key1], self.metrics[key1+'.err']]
            elif key1 in self.params:
                x = [self.params[key1][0], self.params[key1][1]]
            filter1 = N.isfinite(x[0])
            if key1 in self.metrics:
                filter1 *= (x[0]/x[1])>=StN

            for j, key2 in enumerate(list_keys2):
                k = i*nvar2 + j + 1

                if key2 in self.metrics:
                    y = [self.metrics[key2], self.metrics[key2+'.err']]
                elif key2 in self.params:
                    y = [self.params[key2][0], self.params[key2][1]]
            
                filter2 = N.isfinite(y[0])
                if key2 in self.metrics:
                    filter2 *= (y[0]/y[1])>=StN
                filter = filter1*filter2

                if list_keys2 != list_keys:
                    ax = fig.add_subplot(nvar, nvar2, k)
                elif i==j:
                    ax = fig.add_subplot(nvar, nvar, k)
                    ax.hist(y[0][filter],
                            bins=N.sqrt(len(x[0][filter])),
                            fc='k', ec='k', histtype=histtype)
                    mask=N.array([True]*len(y[0][filter]), dtype='bool')
                    xmean=y[0][filter][mask].mean()/10

                    ax.set_xlim(xmin=y[0][filter][mask].min() - xmean,
                                xmax=y[0][filter][mask].max() + xmean)
                    
                    P.setp(ax.get_xticklabels()+ax.get_yticklabels(),
                               fontsize=6)
                    if len(list_keys)>5:
                        self._set_labels(fig, ax, str(j), str(i))
                    else:
                        self._set_labels(fig, ax, list_keys2[j], list_keys[i])
                    continue
                elif j<i:
                    ax = fig.add_subplot(nvar, nvar2, k)
                    pass 
                else:
                    continue

                if colorlegend and not corrcol:
                    mask = (y[0][filter]>=cut[0])&(y[0][filter]<=cut[1]) & \
                           (x[0][filter]>=cut[0])&(x[0][filter]<=cut[1])
                    
                    tmp_col = color[filter][mask]
                    #Normalization of the colorbar
                    norm = ( tmp_col - tmp_col.min() ) \
                           / ( tmp_col - tmp_col.min() ).max()
                    
                    #Inverse the color of points for 'x1' only
                    col=cmap(norm)
                    
                    #Plot colored points and error bar
                    for l, c in enumerate(col):
                        ax.errorbar(y[0][filter][mask][l],
                                    x[0][filter][mask][l],
                                    xerr=y[1][filter][mask][l],
                                    yerr=x[1][filter][mask][l],
                                    linestyle='None', capsize=0, ecolor=c,
                                    marker='.', mfc=c, mec=c)
                        
                    #Inverse the color of the colorbar for 'x1' only
                    scat = ax.scatter(y[0][filter][mask],
                                      x[0][filter][mask],
                                      c=tmp_col,
                                      edgecolor='none',
                                      cmap=(cmap), visible=False)
                else:
                    mask= (y[0][filter]>=cut[0]) & \
                          (y[0][filter]<=cut[1]) & \
                          (x[0][filter]>=cut[0]) & \
                          (x[0][filter]<=cut[1])
                    if corrcol:
                        correlation, corred, correp = Corr(y[0][filter][mask],
                                                           x[0][filter][mask],
                                                           error=True)
                        col = cmap(N.abs(correlation))
                        color.append(N.abs(correlation))
                        print list_keys2[j], list_keys[i], '%.2f + %.2f - %.2f'%\
                              (correlation, correp, corred)
                    else:
                        col='k'
                    ax.plot(y[0][filter][mask], x[0][filter][mask], '.', color=col)
                    if errorbar:
                        ax.errorbar(y[0][filter][mask],
                                    x[0][filter][mask],
                                    xerr=y[1][filter][mask],
                                    yerr=x[1][filter][mask], marker='.',
                                    linestyle='None', capsize=0, ecolor=col,
                                    mfc=col, mec=col)
                if corr or corrcol:
                    correlation, corred, correp = Corr(y[0][filter][mask],
                                                       x[0][filter][mask],
                                                       error=True)
                    ax.annotate(r'$%.2f\pm %.2f$' % \
                                (correlation, N.mean(corred, correp)),
                                xy=(0.01, 0.99),
                                xycoords='axes fraction', xytext=(0.01, 0.99),
                                textcoords='axes fraction',
                                horizontalalignment='left',
                                verticalalignment='top', fontsize=12)

                xmean=y[0][filter][mask].mean()/10
                ymean=x[0][filter][mask].mean()/10
                xmin = y[0][filter][mask].min()-xmean
                xmax = y[0][filter][mask].max()+xmean
                ymin = x[0][filter][mask].min()-ymean
                ymax = x[0][filter][mask].max()+ymean
                if list_keys != list_keys:
                    ax.set_xlim(xmin=xmin, xmax=xmax)
                ax.set_ylim(ymin=ymin, ymax=ymax)
                            
                P.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                           fontsize=6)

                if len(list_keys)>5:
                    self._set_labels(fig, ax, str(j), str(i))
                else:
                    self._set_labels(fig, ax, list_keys2[j], list_keys[i])

                        
        if colorlegend:                
            cax = fig.add_axes(coords)
            if corrcol:
                scat = cax.scatter([0]*k, [0]*k, c=N.linspace(0,1,k),
                                   edgecolor='none', cmap=(cmap), visible=False)
            else:
                scat = cax.scatter(y[0], x[0], c=tmp_col,
                                   edgecolor='none', cmap=(cmap), visible=False)
            #Plot the colorbar
            cb = fig.colorbar(scat,format='%.1f') #,orientation='horizontal')
            #Legend of the colorbar
            if colorlegend == 'corr':
                colorlegend = 'Absolute Pearson correlation'
            cb.set_label(self._get_label(colorlegend))
            cax.set_axis_off()

        if title:
            fig.text(0.5, 0.99, title,horizontalalignment='left',
                     verticalalignment='top', fontsize=18)

        if list_keys2 == list_keys and len(list_keys)>5:
            j=0
            for i,key1 in enumerate(list_keys):
                fig.text(0.75, 0.9-j,str(i)+' : '+self._get_label(key1),
                         horizontalalignment='left', verticalalignment='top',
                         fontsize=10)
                j+=0.05
        
        fig.canvas.draw()

    def _set_labels(self,fig,ax,xname,yname):
        #Change the name of some parameters
        xname = self._get_label(xname)
        yname = self._get_label(yname)
        
        if ax.is_last_row():
            fig.canvas.draw()
            if not xname.startswith('v'):
                ax.set_xticks(ax.get_xticks()[1:-1])
            else:
                xticks=ax.get_xticks()
                ax.set_xticks([xticks[1],xticks[len(xticks)/2],xticks[-2]])
            ax.set_xlabel(xname,size='small')
        else:
            ax.set_xticklabels([])
            
        if ax.is_first_col():
            fig.canvas.draw()
            if not yname.startswith('v'):
                ax.set_yticks(ax.get_yticks()[1:-1])
            else:
                yticks=ax.get_yticks()
                ax.set_yticks([yticks[1],yticks[len(yticks)/2],yticks[-2]])
            ax.set_ylabel(yname,size='small')
        else:
            ax.set_yticklabels([])
        
    def _label_names(self):
        self.labels={}
        R=r'\cal{R}'
        
        self.labels['dmfit_orig'] = r'$\Delta \mu_B$'
        self.labels['dmfit_corr'] = r'$\Delta \mu_B^c$'
        self.labels['x1']         = r'$x_1$'
        self.labels['color']      = r'$c$'
        self.labels['dm15']       = r'$\Delta m_{15}$'
        
        self.labels['RSi']        = r'$\Re_{Si}$'
        self.labels['RSiS']       = r'${%s}_{SiS}$'%R
        self.labels['RSiSS']      = r'${%s}_{SiSS}$'%R
        self.labels['RCa']        = r'${%s}_{Ca}$'%R
        self.labels['RCaS']       = r'${%s}_{CaS}$'%R

        self.labels['EWCaIIHK']   = r'EWCa II H&K'
        self.labels['EWSiII4000'] = r'EWSi II $\lambda4131$'
        self.labels['EWMgII']     = r'EWMg II'
        self.labels['EWFe4800']   = r'EWFe $\lambda4800$'
        self.labels['EWSIIW']     = r'EWS II W'
        self.labels['EWSiII5972'] = r'EWSi II $\lambda5972$'
        self.labels['EWSiII6355'] = r'EWSi II $\lambda6355$'
        self.labels['EWOI7773']   = r'EWO I $\lambda7773$'
        self.labels['EWCaIIIR']   = r'EWCa II IR'
        
        self.labels['RCaIIHK']    = r'${%s}$Ca II H&K'%R
        self.labels['RSiII4000']  = r'${%s}$Si II $\lambda4131$'%R
        self.labels['RMgII']      = r'${%s}$Mg II'%R
        self.labels['RFe4800']    = r'${%s}$Fe $\lambda4800$'%R
        self.labels['RSIIW']      = r'${%s}$S II W'%R
        self.labels['RSiII5972']  = r'${%s}$Si II $\lambda5972$'%R
        self.labels['RSiII6355']  = r'${%s}$Si II $\lambda6355$'%R
        self.labels['ROI7773']    = r'${%s}$O I $\lambda7773$'%R
        self.labels['RCaIIIR']    = r'${%s}$Ca II IR'%R

        self.labels['vSiII_4128'] = r'v Si II $\lambda4128$'
        self.labels['vSiII_5454'] = r'v S II $\lambda5454$'
        self.labels['vSiII_5640'] = r'v S II $\lambda5640$'
        self.labels['vSiII_5972'] = r'v Si II $\lambda5972$'
        self.labels['vSiII_6355'] = r'v Si II $\lambda6355$'

        self.labels['Rsjb']       = r'${%s}_{642/443}$'%R

        self.labels['phase']      = 'Phases'
      
    def _get_label(self,name):
        if name in self.labels:
            return self.labels[name]
        else:
            return name

    def signal_to_noise(self,name):
        if not name in self.metrics and name not in self.params:
            print "Error, wrong name given, try again"
        else:
            if name in self.metrics_name:
                x,xerr=self.metrics[name],self.metrics[name+'.err']
            else:
                self.params[name][0]/self.params[1]
            fig = P.figure(dpi=150)
            ax = fig.add_axes([0.10,0.08,0.87,0.84])
            ax.hist(self.metrics[name] / self.metrics[name+'.err'],
                    histtype='step',color='k')
            sn=self.metrics[name]/self.metrics[name+'.err']
            print 'Mean signal to noise:',N.mean(sn),N.std(sn)    

 

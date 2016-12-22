#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
## Filename:          XLF.py 
## Version:           $Revision: 1.2 $
## Description:       
## Author:            Nicolas Chotard <nchotard@ipnl.in2p3.fr>
## Author:            $Author: nchotard $
## Created at:        $Date: 2012/04/17 06:35:08 $
## Modified at:       17-04-2012 11:30:45
## $Id: XLF.py,v 1.2 2012/04/17 06:35:08 nchotard Exp $
################################################################################

"""
Goal:
- a  class which would compute an 'XLF spectrum' from a given spectrum
 (with diffrent set of parameters)
- sone functions to:
    - 

Some though about the uncertainties (for a given velocity space):
There should be dependant on:
    - the smoothing parameter, for the systematic part
    - the statistic part on the initial uncertainties
    - and ??
"""

__author__  = "Nicolas Chotard <nchotard@ipnl.in2p3.fr>"
__version__ = '$Id: XLF.py,v 1.2 2012/04/17 06:35:08 nchotard Exp $'


import numpy as N
import pylab as P
import sys
from scipy.integrate import trapz

import LoadIDR as L
import pca as PCA
from ToolBox import Statistics,MPL

class OneSpecXLF:

    def __init__(self,x,y,v=None,err=False,wmin=3350,wmax=8800):
        """
        Initialization is made with a regular spectrum:
        x: wavelength
        y: flux
        v: variance
        """
        self.x = N.array(x)
        self.y = N.array(y)
        if v is not None:
            self.v = N.array(v)
        else:
            self.v = None

        self.smooth(wmin=wmin,wmax=wmax)
        self.calcXLF()
        if err:
            self.calcXLF_err()
        
    def smooth(self, v=3000, wmin=3350, wmax=8800,
               sampling=500, d=3, c=299972.0):
        """
        Smooth the flux data with a Savitsky-Golay filter in velocity space.
        
        v : fit polynomial to +- wavelength*v/c wavelength range
                   default 3000 (with of the Savitzky-Golay window)
        w : wavelength sampling (default 500 km/s bins from 3350 to 8800)
        d : polynomial order (degree) to fit (default 3)
        c : light velocity in km/s
        """
        # compute the number of bins
        num = int(N.log(wmax/wmin)/N.log((2*c/sampling+1.)/(2*c/sampling-1.)))
        # 500 km/s bins
        x_smooth = N.logspace(N.log10(wmin), N.log10(wmax), num)
        # check nan values
        notNaN = (self.y == self.y) 

        y_smooth = []
        for xi in x_smooth:
            dw = xi*v/c
            ii = (xi-dw < self.x) & (self.x < xi + dw) & notNaN
            ngood = len(self.x[ii])
            if ngood > 2*d: #checking for overfitting
                #smoothing
                coeff = N.polyfit(self.x[ii], self.y[ii], d)
                y_smooth.append(N.polyval(coeff, xi))
            else:
                print >> sys.stderr, "Failed to calc XLF for w=%.1f"%xi
                y_smooth.append(N.NaN)
                
        self.velocity_smooth = v
        self.x_smooth = x_smooth
        self.y_smooth = N.array(y_smooth)

    def calcXLF(self, velocity=6000.0, c=299972.0):
        """
        Calculate XLF along the same wavelength steps as was used for smoothing
        """
        # XLF widths array before cut
        dw = self.x_smooth * velocity / c  
        # remove edges wavelengths
        filt = (self.x_smooth-dw > self.x_smooth[0]) & \
               (self.x_smooth+dw < self.x_smooth[-1])
        # apply the filter. Create the XLFs central wavelengths
        wXLF = self.x_smooth[filt]  
        
        dw = wXLF * velocity / c  # XLF widths array after cut
        wlo = wXLF - dw           # XFLs wavelength 'blue' wavelengths
        whi = wXLF + dw           # XFLs wavelength 'red' wavelengths
        
        flo,fhi,xlf = [],[],[]
        for xlo, xhi in zip(wlo, whi):
            # where the XLF is computed
            filt = (self.x_smooth > xlo) & (self.x_smooth < xhi) 
            ww = N.concatenate(((xlo,), self.x_smooth[filt], (xhi,)))
            # corresponding interpolated flux
            yy = N.interp(ww, self.x_smooth, self.y_smooth) 
            
            # compute the XLF (Equivalent width-like)
            Flo, Fhi = yy[0], yy[-1]
            # integration with the trapeze method
            Fline = 0.5 * (Flo + Fhi) * (xhi - xlo) 
            Ftot = trapz(x=ww, y=yy)
            xlf.append((Ftot-Fline) / Ftot)

            # for plot purpose
            flo.append(yy[0])
            fhi.append(yy[-1])

        self.xlf_x = wXLF
        self.xlf = N.array(xlf)
        self.xlf_velocity = velocity

        # for plot purpose
        self.wlo = N.array(wlo)
        self.whi = N.array(whi)
        self.flo = N.array(flo)
        self.fhi = N.array(fhi)
        
    def calcXLF_err(self):
        pass

    def XLF(self, w, err=False, spectra=False):
        """
        Returns XLF at wavelength w from interpolation of precalculated XLF
        """
        xlf = N.interp(w,self.xlf_x,self.xlf)
        return xlf

def XLFs(idr):
    """
    Compute the XLF values for all the spectra of all the SNe
    stored in a LoadIDR object.
    Return the same object, with idr.data.[sn]['xlf.'] data. 
    """
    for sn in idr.data:
        print "Computing XLFs for",sn
        X,Y = idr.data[sn]['data.X'],idr.data[sn]['data.Y']
        XLF = N.array([OneSpecXLF(x, y, wmin=idr.lmin, wmax=idr.lmax)
                        for x,y in zip(X,Y)])
        idr.data[sn]['xlf.X'] = N.array([xlf.xlf_x for xlf in XLF])
        idr.data[sn]['xlf.Y'] = N.array([xlf.xlf   for xlf in XLF])
        # WRONG variance!!!
        idr.data[sn]['xlf.V'] = N.array([xlf.xlf/10   for xlf in XLF])
        #all the velocity are the same, take only the last one
        idr.data[sn]['xlf.velocity'] = xlf.xlf_velocity 
        idr.data[sn]['xlf.objects'] = XLF    
                    
def plot_XLFs(X,Y,phase,dpi=80,axes=[0.09,0.08,0.87,0.9]):
    """
    X: wavelength array of all the spectra [x1,x2,x3,...]
    Y: flux array of all the spectra [y1,y2,y3,...]
    with x1 and y1 corresponds to the first spectrum and so on
    """
    for x,y,p in zip(X,Y,phase):
        fig = P.figure(dpi=dpi)
        ax = fig.add_axes(axes, xlabel='Wavelength [A]',ylabel='XLF')
        ax.plot(x,y,label='%.1f'%p)
        ax.legend(loc='best').draw_frame(False)

def plot_XLFs_time(X,Y,phase,dpi=80,axes=[0.09,0.08,0.87,0.9]):
    """
    X: wavelength array of all the spectra [x1,x2,x3,...]
    Y: flux array of all the spectra [y1,y2,y3,...]
    with x1 and y1 corresponds to the first spectrum and so on
    """
    Y = Y.T
    col = P.cm.jet(range(len(X[0])))
    fig = P.figure(dpi=dpi)
    ax = fig.add_axes(axes, xlabel='Rest-phase',ylabel='XLF')
    for y,c in zip(Y,col):
        ax.plot(phase,y,color=c)
        #ax.legend(loc='best').draw_frame(False)

def plot_XLFs_time_all_sne(idr,wl=4000,dpi=80,axes=[0.09,0.08,0.87,0.87]):
    """
    
    """
    fig = P.figure(dpi=dpi)
    ax = fig.add_axes(axes, xlabel='Rest-phase',ylabel='XLF')
    col = P.cm.jet(N.linspace(0,1,len(idr.data)))
    
    for i,sn in enumerate(idr.data):

        #Check if the given wavelength exists for the current SN
        wls = idr.data[sn]['xlf.X'][0]
        if wl > wls.max() or wl < wls.min():
            print "skipping %s"%sn; continue
            
        b=N.argmin(N.abs(wls-wl))
        phases = idr.data[sn]['data.phases']
        xlf = N.array([idr.data[sn]['xlf.Y'][j][b]
                         for j,p in enumerate(phases)])
        ax.plot(phases,xlf,color=col[i])
    ax.set_title('Wavelength is %i [A]'%int(wl))

def fit(X,Y,V,P,):
    """
    Simple model, for a SN i:
    M(lbd,t,i) = alpha(lbd,t) * XLF(lbd,t,i) + Ebmv(i)*Rv(i)*A(lbd)/A(V)

    Input:
    X: the wavelength
    Y: the flux
    V: the variance
    P: the phases of each spectrum
    """
    pass


def mag_vs_XLF(idr,phase=0,window=1,w_mag=4000,w_XLF=4000):
    """
    Do a simple plot of magnitude versus XLF.

    options:
        phase: spectra choosen as close as possible to this phase...
        window: ... in this windows range.
        w_mag: magnitude is choosen as close as possible to 'w_mag'
        x_XLF: XLF choosen as close as possible to 'x_XLF'
    
    WARNING: the mean value of the magnitude distribution is subtracted
             to blind the analysis.
    """
    #shortcut
    d = idr.data
    p = 'data.phase'
    
    #Select the wavelengths
    i = N.argmin(N.abs(d[d.keys()[0]]['mag.X'][0] - w_mag))
    j = N.argmin(N.abs(d[d.keys()[0]]['xlf.X'][0] - w_XLF))
        
    #Select the sne having a spectrum in the selcted phase windows
    sne = [sn for sn in d 
           if len(d[sn][p][N.abs(d[sn][p] - phase) < window]) > 0]

    #Select the closest spectra to the choosen phase
    mags = N.array([d[sn]['mag.Y'][N.argmin(N.abs(d[sn][p] - phase))][i]
                    for sn in sne])
    XLFs = N.array([d[sn]['xlf.Y'][N.argmin(N.abs(d[sn][p] - phase))][j]
                    for sn in sne])
    corr,correrrm,correrrp = Statistics.correlation(mags,XLFs,error=True)
    
    #Make the figure
    fig = P.figure()
    ax = fig.add_subplot(111)
    ax.plot(XLFs,mags-N.mean(mags),'ok')
    ax.annotate(r'$\rho$=%.2f$\pm$%.2f'%(corr,(correrrm+correrrp)/2.),
                (0.02,0.95), xycoords='axes fraction')
    ax.set_xlabel(r'XLF [$\AA$], wl=%.i'%w_XLF)
    ax.set_ylabel(r'$\delta$M [mag], wl=%.i'%w_XLF)
    P.show()

def param_vs_XLF(idr,param,phase=0,window=2.5,w_XLF=4000):
    """
    Do a simple plot of magnitude versus XLF.

    options:
        phase: spectra choosen as close as possible to this phase...
        window: ... in this windows range.
        w_mag: magnitude is choosen as close as possible to 'w_mag'
        x_XLF: XLF choosen as close as possible to 'x_XLF'
    
    WARNING: the mean value of the magnitude distribution is
             subtracted to blind the analysis.
    """
    # shortcuts
    d   = idr.data
    idr = idr.idr
    p = 'data.phases'
    
    # select the wavelengths
    j = N.argmin(N.abs(d[d.keys()[0]]['xlf.X'][0] - w_XLF))
        
    # select the sne having a spectrum in the selcted phase windows
    sne = [sn for sn in d if len(d[sn][p][N.abs(d[sn][p] - phase) \
                                          < window]) > 0]

    # select the closest spectra to the choosen phase
    para = N.array([idr[sn][param] for sn in sne])
    XLFs = N.array([d[sn]['xlf.Y'][N.argmin(N.abs(d[sn][p] - phase))][j]
                    for sn in sne])
    corr,correrrm,correrrp = Statistics.correlation(para,XLFs,error=True)
    
    # mMake the figure
    fig = P.figure()
    ax = fig.add_subplot(111,xlabel=r'XLF [$\AA$], wl=%.i'%w_XLF,ylabel=param)
    ax.plot(XLFs,para,'ok')
    ax.annotate(r'$\rho$=%.2f$\pm$%.2f'%(corr,(correrrm+correrrp)/2.),
                (0.02,0.95), xycoords='axes fraction')
    P.show()

def mag_vs_XLFcorr(idr,phase=0,window=1,w_mag=4000):
    """
    Do a simple plot of corr(magnitude,XLF) vs wavelgenth

    options:
        phase: spectra choosen as close as possible to this phase...
        window: ... in this windows range.
        w_mag: magnitude is choosen as close as possible to 'w_mag'
        x_XLF: XLF choosen as close as possible to 'x_XLF'
    
    WARNING: the mean value of the magnitude distribution is subtracted
             to blind the analysis.
    """
    # shortcut
    d = idr.data 
    p = 'data.phases'
    
    # select the wavelengths
    i = N.argmin(N.abs(d[d.keys()[0]]['mag.X'][0] - w_mag))
        
    # select the sne having a spectrum in the selcted phase windows
    sne = [sn for sn in d if len(d[sn][p][N.abs(d[sn][p] -\
                                                phase) < window]) > 0]

    # select the closest spectra to the choosen phase
    mags  = N.array([d[sn]['mag.Y'][N.argmin(N.abs(d[sn][p] - phase)) ][i]
                     for sn in sne])
    corrs,corrserr = [],[]
    for j in range(len(d[d.keys()[0]]['xlf.X'][0])):
        xlf = N.array([d[sn]['xlf.Y'][N.argmin(N.abs(d[sn][p] - phase)) ][j]
                       for sn in sne])
        corr,correrrm,correrrp = Statistics.correlation(mags,xlf,error=True)
        corrs.append(corr)
        corrserr.append(N.mean([correrrm,correrrp]))

    corrs,corrserr = map(N.array,[corrs,corrserr])
    #Make the figure
    fig = P.figure()
    ax = fig.add_subplot(111)
    ax.plot(d[d.keys()[0]]['xlf.X'][0],corrs,'k')
    MPL.errorband(ax, d[d.keys()[0]]['xlf.X'][0],corrs,corrserr, color='k')
    ax.set_xlabel(r'Wavelegnth [$\AA$]')
    ax.set_ylabel(r'$\rho$(XLF,mag)')
    P.show()

def map_corr_XLF(idr,phase=0,window=2.5,w_mag=4000,plotpoints=True):
    """
    """
    # shortcut
    d = idr.data 
    p = 'data.phases'
    
    # select the wavelengths
    i = N.argmin(N.abs(d[d.keys()[0]]['mag.X'][0] - w_mag))
        
    # select the sne having a spectrum in the selcted phase windows
    sne = [sn for sn in d if len(d[sn][p][N.abs(d[sn][p] -\
                                                phase) < window]) > 0]
    print "Number of sne kept:",len(sne)

    #Take the XLF for a given phase
    XLFs  = N.array([d[sn]['xlf.Y'][N.argmin(N.abs(d[sn][p] - phase)) ]
                     for sn in sne])
    mags  = N.array([d[sn]['mag.Y'][N.argmin(N.abs(d[sn][p] - phase)) ][i]
                     for sn in sne])
    
    wlength = d[sn]['xlf.X'][0]
    matrix = N.abs(N.corrcoef(XLFs.T))
    ylabel = r'$\rho$'
    title  = r"XLFs correlation map of %i sne at p=%i$\pm$%i days"%\
             (len(sne),phase,window)
    cmap = P.cm.jet

    values = [N.diag(matrix,k=i) for i in range(len(matrix))]
    
    means=map(N.mean,values)
    stds=map(N.std,values)
    med,nmad=N.array([Statistics.median_stats(x) for x in values]).T
    
    fig = P.figure(dpi=150)
    ax = fig.add_axes([0.1,0.08,0.88,0.90])
    
    #Plot the values for each wavelength difference
    if plotpoints:
        for i in range(len(wlength)):
            ax.plot([wlength[i]-wlength[0]]*len(values[i]),values[i],
                    'ok',alpha=0.1)
    
    #Plot the mean and the median
    ax.errorbar(wlength-wlength[0],means,yerr=stds,color='r',label='mean')
    ax.errorbar(wlength-wlength[0],med,yerr=nmad,color='c',label='median')
    
    #Set the title and labels
    ax.set_xlabel(r'$\Delta \lambda$ $[\AA]$',size='x-large')
    ax.set_ylabel(ylabel,size='x-large')
    #ax.set_title(title)
    
    #Set legend and limits
    ax.legend(loc='best')
    ax.set_xlim(xmin=-20,xmax=wlength[-1]-wlength[0]+20)
    
    #Plot the matrix
    wlength=[wlength[0],wlength[-1],wlength[-1],wlength[0]]
    fig = P.figure(dpi=150)
    ax = fig.add_axes([0.08,0.09,0.88,0.86],title=title)
    im = ax.imshow(matrix,cmap=cmap,extent=wlength,interpolation='nearest')
    cb = fig.colorbar(im)
    cb.set_label(ylabel,size='x-large')
    ax.set_xlabel(r'Wavelength [$\AA$]',size='large')
    ax.set_ylabel(r'Wavelength [$\AA$]',size='large')
    
    P.show()
    
def plot_XLFs_measurement(xlf_obj,title='toto'):
    """
    Comtrol plot for XLF measurements, for a given spectrum.
    """
    
    x = N.linspace(3000,10000,1000)
    y = N.random.randn(1000)
    fig = P.figure(dpi=150)
    ax1 = fig.add_axes([0.08,0.09,0.88,0.39],
                       xlabel=r'Wavelength [$\AA$]',ylabel='toto')
    ax2 = fig.add_axes([0.08,0.45,0.88,0.50],ylabel='toto',title=title)
    ax1.plot(xlf_obj.xlf_x,xlf_obj.xlf)
    ax2.plot(xlf_obj.x_smooth,xlf_obj.y_smooth,'k')
    for i in range(len(xlf_obj.wlo)):
        ax2.plot([xlf_obj.wlo[i],xlf_obj.whi[i]],
                 [xlf_obj.flo[i],xlf_obj.fhi[i]],'r')
    ax2.set_xticks([])
    ax1.set_yticks(ax1.get_yticks()[1:-1])
    ax2.set_yticks(ax2.get_yticks()[1:-1])
    ax1.axhline(0,color='k')
    P.show()

def compare_result():
    """
    Comparer les resultats des XLFs avec differente valeur de la largeur,
    ainsi que meme valeur de la largeur mais avec different smoothing
    (utiliser une interpollation des XLFs aux bonnes lingueur d'onde.).
    """
    pass

def apply_pca(idr,phase=0,window=2.5):
    """
    apply a pca over the XLF
    XLF are taken for spectra having a phase between phase +/ window
    at most one for each SN
    """
    XLFs = L.get_data_at_phase(idr,phase=phase,window=window,data='XLF')

    pca = PCA.PCA(XLFs)

    return pca
    

# End of XLF.py

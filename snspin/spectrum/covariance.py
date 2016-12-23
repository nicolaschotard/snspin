#/bin/env python

"""
Spectral covariance matrix (variance and correlation) study
"""

from glob import glob
import pylab as P
import numpy as N
import scipy as S
from scipy import interpolate as I
from scipy import optimize

import pySnurp
from snspin.tools import statistics
from snspin.tools.smoothing import savitzky_golay as sg
from snspin.tools.smoothing import spline_find_s
from snspin.tools.smoothing import sg_find_num_points
from snspin.tools import io

class SPCS(object):

    """
    SpecVarCorrStudy.
    """

    def __init__(self, x, y, v, specid='', obj='',
                 verbose=False, rhoB=0.32, rhoR=0.40, factor=0.69):
        """."""
        #Set the data
        self.x = x
        self.y = y
        self.v = v * factor
        self.factor_used = factor
        if S.mean(x) < 5150:
            self.rho = rhoB
            self.rho_used = 'rhoB'
        else:
            self.rho = rhoR
            self.rho_used = 'rhoR'       

        #Set infomations about the spectrum
        self.specid = specid
        self.object = obj
        self.verbose = verbose
        if verbose:
            print 'Working on ', self.specid, ', from', self.object
            print '%i bins from %.2f to %.2f A' \
                  % (len(self.x), self.x[0], self.x[-1])

    def smooth(self, smoothing='sp', s=None, w=15, findp=True):
        """."""
        #Set the smoothing parameters
        self.smoothing = smoothing
        if self.smoothing == 'sp' and findp:
            try:
                self.s = spline_find_s(self.x, self.y, self.v, corr=self.rho)
            except:
                self.s = 0.5
                print "WARNING: Smoothing failed. s=0.5 by default"
        elif self.smoothing == 'sg' and findp:
            try:
                # this looks like a simple relation between average S/N
                # works pretty well
                print N.median(self.y/N.sqrt(self.v))
                self.w = int(-10*N.log(N.median(self.y/N.sqrt(self.v))) + 52)
                if self.w < 3:
                    self.w = 3
                #self.w = int(sg_find_num_points(self.x, self.y,
                #                                self.v, corr=self.rho))
            except:
                self.w = 15
                print "WARNING: Smoothing failed. w=15 by default"
        if not hasattr(self, 's'):
            self.s = s
        if not hasattr(self, 'w'):
            self.w = w

        #Smooth, compute the pull and the derive values
        self.ysmooth = smooth_spec(self.x, self.y, self.v,
                                   sfunc=smoothing, s=self.s, w=self.w)
        self.chi2 = S.sum((self.y-self.ysmooth)**2/self.v) / (len(self.x)-1)
        self.pull = comp_pull(self.y, self.ysmooth, self.v)
        self.pull_mean = S.mean(self.pull)
        self.pull_std = S.std(self.pull)
        #self.rho = autocorr(self.pull, k=1, full=False)
        self.residuals = self.y - self.ysmooth

        if self.verbose:
            print 'Smoothing function used:', self.smoothing
            if self.smoothing == 'sg': print 'w = ', self.w
            if self.smoothing == 'sp': print 's = ', self.s
            print 'Chi2 = ', self.chi2
            print 'Mean(pull) = ', self.pull_mean
            print 'Std(pull) = ', self.pull_std
            print 'Corr = ', self.rho

        if self.verbose:
            print "Factor used: ", self.factor
            print "Correlation coefficient:", self.rho

    def make_simu(self, nsimu=1000):
        """
        Build a simulated set of data.
        
        Make some simulation for which everything will be computing as well nsimu is the number 
        of simulations
        if factor is set (1 by default), then self.v*=factor
        if factor is set to None, then self.v*=self.factor
        (run comp_factor before)
        if corr is set to 1, then the pixel will be correlated using the
        correlation parameter found after the smoothing
        corr can also be set to a float value
        """
        #Compute the ramdom noisea
        if self.rho is None:
            ndist = S.random.randn(nsimu, len(self.x))
        else:
            ndist = corr_noise(self.rho, nbin=len(self.x), nsimu=nsimu)

        #Create the simulated spectra
        simus = ndist*(S.sqrt((self.v))) + self.ysmooth

        #Save the random distribution
        self.ndist = ndist

        #Smooth and compute stuffs for the simulated spectra (pull,rho...)
        self.simus = []
        for sim in simus:
            si=SPCS(self.x, sim, self.v, verbose=False)
            si.smooth(smoothing=self.smoothing, s=self.s, w=self.w, findp=False)
            self.simus.append(si)

        if self.verbose:
            print "\nSmoothing used:", self.smoothing
            if self.smoothing == 'sg': print 'w = ', self.w
            if self.smoothing == 'sp': print 's = ', self.s
            print "Factor used: ", factor
            print "Correlation coefficient:", self.rho
            print nsimu, "simulated spectra have been created"
            print "             Real spectrum   Simulations"
            print "Mean pull        %.3f         %.3f" \
                  % (self.pull_mean, S.mean([s.pull_mean for s in self.simus]))
            print "Std  pull        %.3f         %.3f" \
                  % (self.pull_std, S.mean([s.pull_std for s in self.simus]))
            print "Mean chi2        %.3f         %.3f" \
                  % (self.chi2, S.mean([s.chi2 for s in self.simus]))
            print "Mean rho         %.3f         %.3f" \
                  % (self.rho, S.mean([s.rho for s in self.simus]))

    def do_plots(self, all=False, lim=2):
        """."""
        plot_spec(self.x, self.y, self.v, self.ysmooth,
                  title=self.object+', '+self.specid)
        plot_pull(self.x, self.pull, title=self.object+', '+self.specid)
        if hasattr(self, 'simus'):
            self.plot_simu_distri()
            if all:
                for i, s in enumerate(self.simus):
                    if i < lim:
                        s.do_plots()
                    else:
                        return

    def plot_simu_distri(self):
        """."""
        #set the data
        ch, co, pm, ps=[], [], [], []
        chi2 = S.concatenate([[s.chi2 for s in self.simus], [self.chi2]])
        corr = S.concatenate([[s.rho for s in self.simus], [self.rho]])
        pmean = S.concatenate([[s.pull_mean for s in self.simus],
                               [self.pull_mean]])
        pstd = S.concatenate([[s.pull_std for s in self.simus],
                              [self.pull_std]])

        data = [chi2, corr, pmean, pstd]
        names= ['chi2', 'corr', 'pmean', 'pstd']

        #make the figure
        for d, n in zip(data, names):
            P.figure()
            P.hist(d[:-1], bins=S.sqrt(len(d[:-1]))*2, color='b', alpha=0.5)
            P.axvline(d[-1], color='k')
            P.title(n)

class SPCS_test(object):
                
    """
    SpecVarCorrStudy
    """
    
    def __init__(self, x, y, v, specid='', obj='', verbose=False):
        """."""
        #Set the data
        self.x = x
        self.y = y
        self.v = v
        self.factor=1

        #Set infomations about the spectrum
        self.specid = specid
        self.object = obj
        self.verbose = verbose
        if verbose:
            print 'Working on ', self.specid, ', from', self.object
            print '%i bins from %.2f to %.2f A' \
                  % (len(self.x), self.x[0], self.x[-1])
        
    def smooth(self, smoothing='sp', s=None, w=15,
               findp=False, rho=0, factor=1):
        """."""
        #Set the smoothing parameters
        self.smoothing = smoothing
        if factor == 1: v=self.v
        elif factor == 0: v=self.v*self.factor
        else: v=self.v*factor

        if self.smoothing == 'sp' and findp:
            self.s = spline_find_s(self.x, self.y, v, corr=rho)
        elif self.smoothing == 'sg' and findp:
            self.w = int(sg_find_num_points(self.x, self.y, v, corr=rho))
        if not hasattr(self,'s'):
            self.s = s
        if not hasattr(self,'w'):
            self.w = w

        #Smooth, compute the pull and the derived values
        self.ysmooth = smooth_spec(self.x, self.y, v,
                                   sfunc=smoothing, s=self.s, w=self.w)
        self.chi2 = S.sum((self.y-self.ysmooth)**2/v) / (len(self.x)-1)
        self.pull = comp_pull(self.y, self.ysmooth, v)
        self.pull_mean = S.mean(self.pull)
        self.pull_std = S.std(self.pull)
        self.residuals = self.y - self.ysmooth
        #self.rho = autocorr(self.pull,k=1,full=False)
        self.rho = autocorr(self.residuals, k=1, full=False)

        if self.verbose:
            print 'Smoothing function used:',self.smoothing
            if self.smoothing == 'sg':
                print 'w = ',self.w
            if self.smoothing == 'sp':
                print 's = ',self.s
            print 'Chi2 = ',self.chi2
            print 'Mean(pull) = ',self.pull_mean
            print 'Std(pull) = ',self.pull_std
            print 'Corr = ',self.rho

    def make_simu(self, nsimu=0, factor=1., rho=0,  ndist=None):
        """
        Make some simulation for which everything will be computing as well
        nsimu is the number af simulations
        if factor is set (1 by default), then self.v*=factor
        if factor is set to None, then self.v*=self.factor
        (run comp_factor before)
        if corr is set to 1, then the pixel will be correlated using
        the correlation parameter found after the smoothing
        corr can also be set to a float value
        """
        if factor == None:
            factor = self.factor

        #Compute the ramdom noise
        if ndist != None:
            pass
        elif rho == 0:
            ndist = S.random.randn(nsimu, len(self.x))
        #elif rho == 1:
        #    ndist = corr_noise(self.rho,nbin=len(self.x),nsimu=nsimu)
        else:
            ndist = corr_noise(rho, nbin=len(self.x), nsimu=nsimu)

        #Create the simulated spectra
        #simus = ndist*(S.sqrt((S.mean(self.v)*factor))) + self.ysmooth
        simus = ndist*(S.sqrt((self.v*factor))) + self.ysmooth

        #Save the random distribution
        self.ndist = ndist

        #Smooth and compute stuffs for the simulated spectra (pull,rho...)
        self.simus = []
        for sim in simus:
            si=SPCS_test(self.x, sim, self.v, verbose=False)
            si.smooth(smoothing=self.smoothing, s=self.s, w=self.w)
            #,rho=rho) #,factor=factor)
            self.simus.append(si)

        if self.verbose:
            print "\nSmoothing used:",self.smoothing
            if self.smoothing == 'sg':
                print 'w = ',self.w
            if self.smoothing == 'sp':
                print 's = ',self.s
            print "Factor used: ",factor
            print "Correlation coefficient:",rho
            print nsimu,"simulated spectra have been created"
            print "             Real spectrum   Simulations"
            print "Mean pull        %.3f         %.3f" \
                  % (self.pull_mean, S.mean([s.pull_mean for s in self.simus]))
            print "Std  pull        %.3f         %.3f" \
                  % (self.pull_std, S.mean([s.pull_std for s in self.simus]))
            print "Mean chi2        %.3f         %.3f" \
                  % (self.chi2, S.mean([s.chi2 for s in self.simus]))
            print "Mean rho         %.3f         %.3f" \
                  % (self.rho, S.mean([s.rho for s in self.simus]))

    def comp_rho_f(self, smoothing='sp', verbose=False):
        """."""
        self.verbose=verbose
        self.smooth(smoothing=smoothing, findp=True)
        
        def func_rho(rho):
            self.smooth(smoothing=self.smoothing, rho=self.rho)
            p = S.absolute(rho-self.rho)
            return p

        def func_fac(f):
            if hasattr(self, 'ndist'):
                ndist = self.ndist
            else:
                ndist = None
            self.make_simu(rho=self.rho, nsimu=200, factor=f[0], ndist=ndist)
            p = S.absolute(S.mean([s.rho for s in self.simus]) - self.rho)
            del self.simus
            return p

        self.rho = optimize.fmin(func_rho, self.rho, disp=False)[0]
        self.factor = optimize.fmin(func_fac, self.pull_std, disp=False)[0]

        self.make_simu(rho=self.rho, factor=self.factor, nsimu=1000)
        self.verbose=True
        if self.verbose:
            print "Factor used: ",self.factor
            print "Correlation coefficient:",self.rho

    def comp_rho_f_bof(self, smoothing='sp', verbose=False):
        """."""
        self.verbose=verbose
        self.smooth(smoothing=smoothing, findp=True)
        
        def func_rho(rho):
            self.smooth(smoothing=self.smoothing,
                        rho=self.rho, factor=1./(1.+2*rho))
            #ratio = S.absolute(self.pull_std-1.)
            ratio = S.absolute(rho-self.rho)
            return ratio

        self.rho = optimize.fmin(func_rho, self.rho, disp=False)[0]
        self.factor=1./(1.+2*self.rho)

        #self.smooth(smoothing=self.smoothing,rho=self.rho,
        #factor=self.factor,findp=True)

        #self.rho = optimize.fmin(func_rho,self.rho,disp=False)[0]
        #self.factor=1.+2*self.rho**2

        #self.smooth(smoothing=self.smoothing,rho=self.rho,
        #factor=self.factor,findp=True)
        
        self.make_simu(rho=self.rho, factor=self.factor, nsimu=1000)

        if self.verbose:
            print "Factor used: ",self.factor
            print "Correlation coefficient:",self.rho

    def do_plots(self, all=False, lim=5):
        """."""
        plot_spec(self.x, self.y, self.v, self.ysmooth,
                  title=self.object+','+self.specid)
        plot_pull(self.x, self.pull, title=self.object+','+self.specid)
        if hasattr(self, 'simus'):
            self.plot_simu_distri()
            if all:
                for i,s in enumerate(self.simus):
                    print i,lim
                    if i < lim:
                        s.do_plots()
                    else:
                        return

    def plot_simu_distri(self):
        """."""
        #set the data
        ch, co, pm, ps=[], [], [], []
        chi2 = S.concatenate([[s.chi2 for s in self.simus], [self.chi2]])
        corr = S.concatenate([[s.rho for s in self.simus], [self.rho]])
        pmean = S.concatenate([[s.pull_mean for s in self.simus],
                               [self.pull_mean]])
        pstd = S.concatenate([[s.pull_std for s in self.simus],
                              [self.pull_std]])

        data = [chi2,corr,pmean,pstd]
        names= ['chi2','corr','pmean','pstd']

        #make the figure
        for d,n in zip(data,names):
            P.figure()
            P.hist(d[:-1], bins=S.sqrt(len(d[:-1]))*2, color='b', alpha=0.5)
            P.axvline(d[-1], color='k')
            P.title(n)

# Definitions ==================================================================
def open_spec(spec_file, xmin=0, xmax=10000, z=None):
    """
    Load a spetrum using pySnurp
    Return: x,y,v,obejct,specid
    """
    spec = pySnurp.Spectrum(spec_file, keepFits=False)
    if z != None:
        spec.deredshift(z)
    mask = (spec.x >= xmin) & (spec.x <= xmax)
    return spec.x[mask], spec.y[mask], spec.v[mask], \
           spec.readKey('OBJECT'), spec.name.split('/')[-1]

def load_SPCS(file, xmin=0, xmax=10000, z=None, verbose=False):
    """."""
    x, y, v, obejct, specid = open_spec(file, xmin=xmin, xmax=xmax, z=z)
    return SPCS(x, y, v, obejct, specid, verbose=verbose)

def load_SPCS_test(file, xmin=0, xmax=10000, z=None, verbose=False):
    """."""
    x, y, v, obejct, specid = open_spec(file, xmin=xmin, xmax=xmax, z=z)
    return SPCS_test(x, y, v, obejct, specid, verbose=verbose)
    
def smooth_spec(x, y, v, s=None, w=15, sfunc='sp', order=2, verbose=False):
    """."""
    if sfunc == 'sp':
            if s == None: 
                sp = I.LSQUnivariateSpline(x, y, t=(x[::12])[1:],
                                           w = 1/(S.sqrt(v)))
            else:
                try:
                    s=s[0]
                except:
                    pass
                if s <=1:
                    s*= len(x)
                sp = I.UnivariateSpline(x, y, w = 1/(S.sqrt(v)), s=s)
            ysmooth = sp(x)
    elif sfunc == 'sg':
            kernel = (int(w)*2)+1
            if kernel <= order+2:
                if verbose:
                    print "<smooth_spec> WARNING: w  not > order+2 "\
                          "(%d <= %d+2). Replaced it by first odd number "\
                          "above order+2"%(kernel, order)
                kernel = int(order/2) * 2 + 3
            ysmooth = sg(y, kernel=kernel, order=order)
        return ysmooth
    
def comp_pull(y, ysmooth, v):
    """Compute the pull"""
    return (y-ysmooth)/S.sqrt(v)

def corr_noise(rho, nbin=10, nsimu=10):
    """Create a correlated gaussian noise array"""

    #Check if rho is between 0 and 0.5
    if rho < 0:
        print 'rho<0, creation of an uncorrelated noise'
        return S.random.randn(nsimu, nbin)
    elif rho > 0.5:
        print 'rho>0.5, set it to 0.5 by default'
        rho = 0.5
    else: pass

    #Compute alpha and beta
    r = rho / (1.+2.*rho) 
    alpha = 0.5*(1.+S.sqrt(1.-4.*r))
    beta  = 0.5*(1.-S.sqrt(1.-4.*r))

    #Create the correlated noise
    ndist0 = S.random.randn(nsimu, nbin+1)
    ndist = S.zeros((nsimu, nbin))
    for i in range(S.shape(ndist)[0]):
        ndist[i] = alpha*ndist0[i][:-1] + beta*ndist0[i][1:]
        
    return ndist

def plot_spec(x, y, v, ysmooth, title=''):
    fig = P.figure(dpi=150)
    ax = fig.add_axes([0.1,0.08,0.86,0.87],
                      xlabel=r'Wavelength [$\AA$]', ylabel='Flux')
    ax.errorbar(x, y, yerr=S.sqrt(v), color='k', alpha=0.1)
    ax.plot(x, y, 'g')
    ax.plot(x, ysmooth, 'r', lw=1.5)
    ax.set_title(title)

def plot_pull(x, pull, title=''):
    """."""    
    fig = P.figure(dpi=150)
    ax = fig.add_axes([0.1,0.08,0.86,0.87],
                      xlabel=r'Wavelength [$\AA$]', ylabel='Pull')
    ax.plot(x, pull, 'k',
            label='Mean=%.2f, Std=%.2f'%(S.mean(pull),
                                         S.std(pull)))
    ax.set_title(title)
    ax.legend(loc='best').draw_frame(False)

def plot_corr(corr, title=''):
    """."""    
    fig = P.figure(dpi=150)
    ax = fig.add_axes([0.1,0.08,0.86,0.87],
                      xlabel=r'Wavelength [$\AA$]', ylabel='Auto correlation')
    ax.plot(corr, 'k')
    ax.set_title(title)

def autocorr(x, k=1, full=False):
    """
    R(k)= E[ (X(i)-mu)(X(i+k)-mu) ] / (sigma**2)
    """
    n, mu, sg = len(x), S.mean(x), S.std(x)
    ack = lambda k: S.sum([ (x[i]-mu)*(x[i+k]-mu)
                            for i in range(n-k) ]) / ((n-k)*sg**2)
    if not full:
        return ack(k)
    else:
        return S.array([ ack(j) for j in range(n-1)])

def plot_smoothed_spec(f, num=5):
    """."""        
    d = io.loaddata(f)
    for i in d:
        ob=d[i]
        plot_obj(ob, num=num)

def plot_obj(ob, num=5):
    """."""        
    fig = P.figure(figsize=(8,18), dpi=150)
    ax = fig.add_axes([0.08, 0.08, 0.86, 0.87],
                      xlabel=r'Wavelength [$\AA$]', ylabel='Flux')
    ax.plot(ob.x, ob.y, 'k')
    ax.plot(ob.x, ob.ysmooth, 'r', lw=1.5)
    for i,s in enumerate(ob.simus):
        if i >= num:
            return
        cst = (i+1)*S.mean(ob.y)
        ax.plot(s.x, s.y - cst, 'k')
        ax.plot(s.x, s.ysmooth - cst, 'r', lw=1.5)

def run_over_dir(dir):
    """."""
    files = glob(dir+'*.pkl')
    for f in files:
        plot_smoothed_spec(f)

def control_case(rho=0, factor=1, nsimu=1000, nbin=300, plot=False):
        """
        Simu.

        Make some simulation for which everything will be computing as well
        nsimu is the number af simulations
        if factor is set (1 by default), then self.v*=factor
        if factor is set to None, then self.v*=self.factor
        (run comp_factor before)
        if corr is set to 1, then the pixel will be correlated using
        the correlation parameter found after the smoothing
        corr can also be set to a float value
        """
        #Creation of a variance array (uncorrelated random noise)
        x = N.arange(nbin)
        v = S.random.randn(nbin)**2
        zeros = N.zeros(nbin)

        #Creation of (correlated) random noise
        if rho == 0:
            sims = S.random.randn(nsimu, nbin)
        else:
            sims = corr_noise(rho, nbin=nbin, nsimu=nsimu)
        sims *= factor
        
        #Check if rho is the given one
        Rho = N.array([autocorr(N.array(sim), k=1, full=False) for sim in sims])
        if plot:
            P.hist(Rho, histtype='step', color='b',
                   alpha=0.5, bins=statistics.hist_nbin(Rho))
            P.title('Mean=%.2f, Std=%.2f'%(N.mean(Rho), N.std(Rho)))
        else:
            return Rho

def control_case_rho_var(rhos=[0, 0.1, 0.2, 0.3, 0.4, 0.5], factor=1):
    """."""
    col = P.cm.jet(S.linspace(0, 1, len(rhos)))
    fig = P.figure(dpi=150)
    ax = fig.add_axes([0.1, 0.08, 0.86, 0.87], xlabel=r'$\rho$', ylabel='N')
    ax.set_title(r'Variation of $\rho$ for a controled case')
    for i, rho in enumerate(rhos):
        Rho = control_case(rho=rho, factor=factor)
        p = ax.hist(Rho, histtype='step', color=col[i],
                    bins=statistics.hist_nbin(Rho), lw=1.5,
                    label=r'$\rho=%.2f$'%rho)
        if i == 0: m = p[0].max() + 15
        ax.axvline(rho, color=col[i], ls=':', lw=2)
        ax.annotate('%.2f'%N.mean(Rho), (rho, m),
                    xycoords='data', horizontalalignment='right')
        ax.annotate('%.2f'%N.std(Rho), (rho, m-5),
                    xycoords='data', horizontalalignment='right')
    ax.annotate('Mean', (-0.15, m), xycoords='data',
                horizontalalignment='left')
    ax.annotate('Std', (-0.15, m-5), xycoords='data',
                horizontalalignment='left')
    ax.set_ylim(ymax=m+5)
    #ax.legend(loc='upper right').draw_frame(False)
    ax.legend(loc='best').draw_frame(False)

def spec_autocorr(dir='./'):
    """."""
    dirs = glob(dir+'data_*/')
    rho, factor, lbd, sn, params = [], [], [], [], []
    for d in dirs:
        print d
        files = glob(d+'*.txt')
        for f in files:
            try:
                r, f, l, s, p = N.loadtxt(f, unpack=True)
            except:
                continue
            rho.extend(r)
            factor.extend(f)
            lbd.extend(l)
            sn.extend(s)
            params.extend(p)

    P.figure()
    P.plot(rho, factor, 'og')
    P.plot(rho, N.array(rho)+0.3, 'k')
    rho, factor, lbd, sn, params = map(N.array, [rho, factor, lbd, sn, params])
    sort = N.argsort(lbd)
    lbd0 = lbd[sort][0]
    r, f, s = [], [], []
    fig = P.figure(dpi=150)
    ax = fig.add_axes([0.1,0.08,0.86,0.87], xlabel=r'$\rho$', ylabel='N')
    #ax.set_title(r'Variation of $\rho$ for a controled case')
    fig2 = P.figure(dpi=150)
    ax2 = fig2.add_axes([0.1,0.08,0.86,0.87], xlabel=r'factor', ylabel='N')
    #ax2.set_title(r'Variation of $\rho$ for a controled case')
    fig3 = P.figure(dpi=150)
    ax3 = fig3.add_axes([0.1,0.08,0.86,0.87], xlabel=r'$s$', ylabel='N')
    #ax3.set_title(r'Variation of $\rho$ for a controled case')
    col = P.cm.jet(S.linspace(0,1,9))
    j = 0
    for i,lb in enumerate(lbd[sort]):
        if lb==lbd0 and i != len(lbd)-1:
            r.append(float(rho[sort][i]))
            try: f.append(float(factor[sort][i]))
            except: f.append(0)
            s.append(float(params[sort][i]))
        else:
            ax.hist(r, histtype='step', color=col[j],
                    alpha=0.8, bins=statistics.hist_nbin(r),
                    lw=1.5, label='%i,  %.2f, %.2f'%(int(lbd0),
                                                     N.median(r),
                                                     statistics.nMAD(r)))
            ax2.hist(f, histtype='step', color=col[j], alpha=0.8,
                     bins=statistics.hist_nbin(f), lw=1.5,
                     label='%i, %.2f, %.2f'%(int(lbd0), N.median(f),
                                             statistics.nMAD(f)))
            ax3.hist(s, histtype='step', color=col[j], alpha=0.8,
                     bins=statistics.hist_nbin(s), lw=1.5,
                     label='%i, %.2f, %.2f'%(int(lbd0), N.median(s),
                                             statistics.nMAD(s)))
            fig4 = P.figure(dpi=150)
            ax4 = fig4.add_axes([0.1,0.08,0.86,0.87],
                                xlabel=r'$\rho$', ylabel='N')
            ax4.plot(r, f, 'o', color=col[j], alpha=0.8,
                     label='%i, %.2f, %.2f'%(int(lbd0), N.median(s),
                                             statistics.nMAD(s)))
            ax4.set_xlim(xmin=0, xmax=0.5)
            ax4.set_ylim(ymin=0.5, ymax=1)
            lbd0 = lb
            r = []
            r.append(rho[sort][i])
            f = []
            try:
                f.append(float(factor[sort][i]))
            except:
                f.append(0)
            s = []
            s.append(float(params[sort][i]))
            j += 1
            
    ax.legend(loc='upper left').draw_frame(False)
    ax2.legend(loc='upper left').draw_frame(False)
    ax3.legend(loc='upper left').draw_frame(False)
    ax4.legend(loc='upper left').draw_frame(False)
    ax.set_title(r'Mean $\lambda$, Median, nMAD')
    ax2.set_title(r'Mean $\lambda$, Median, nMAD')
    ax3.set_title(r'Mean $\lambda$, Median, nMAD')
    ax4.set_title(r'Mean $\lambda$, Median, nMAD')

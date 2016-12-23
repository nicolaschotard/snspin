#!/usr/bin/env python
################################################################################
## Filename:          calc_metrics_new.py
## Version:           $Revision: 1.69 $
## Description:       calculate metrics and appends them to moksha output
## Author:            Nicolas Chotard <nchotard@ipnl.in2p3.fr>
## Author:            $Author: nchotard $
## Created at:        $Date: 2016/01/12 11:01:22 $
## Modified at:       12-01-2016 12:01:03
## $Id: calc_metrics.py,v 1.69 2016/01/12 11:01:22 nchotard Exp $
################################################################################

"""
Code to apply Phrenology measurements on data that can come from:
   + An IDR : default behavior
   + Our DB : if the --DB option is set.
   + A local fits file : If spectra are fed in via the argument line.
     If so, they overwrite all options (they are stored in option.specs)

   Creates a directory with the name of the supernova in which the results
   and plot directory will be placed.

"""

__author__  = "Nicolas Chotard <nchotard@ipnl.in2p3.fr>"
__version__ = '$Id: calc_metrics.py,v 1.69 2016/01/12 11:01:22 nchotard Exp $'

import sys
import os
import re
import time
import optparse

# use Agg by default (do not try to show plot windows - extremely slow over ssh)
import matplotlib
matplotlib.use('Agg')
import numpy as np

import SNfPhrenology
import SALTmodel
import SnfUtil
import SnfMetaData

from ToolBox import Spectrum, IO
#+ Dependencies that we would like to remove
#+ pySnurp is in the IFU CVS. It would be nice to remove it since there
#+ is Toolbox.Spectrum to replace it
import pySnurp 

code_name = os.path.basename(__file__) + ' '

def read_option():
    usage = "usage: %prog [options] specnames"
    parser = optparse.OptionParser(usage=usage)
    
    #- Input options for idr mode
    idr = optparse.OptionGroup(parser, "idr")
    idr.add_option("--idr", dest="idr",
                   default="/afs/in2p3.fr/group/snovae/snprod1/IDR/current/",
                   help = "Sets the idr run to use")
    idr.add_option("--subset", dest="subset", action="append",
                   default=None,
                   help = "Limits the IDR to the subsets selected")
    parser.add_option_group(idr)

    #- Input options for DB mode
    db = optparse.OptionGroup(parser, "DB")
    db.add_option("--DB", action="store_true", default=False,
                  help="Needs to be set to run in DB mode")
    db.add_option('--specID', help="Will work only on a this particular \
    spectrum in the list. To use with --target (ex: 07_098_116_003).",
                  default=None)
    db.add_option('-v', '--inver', type='int',
                  help="Flux calibration prod version to get spectra from",
                  default=None)
    db.add_option('-j', '--injob',
                  help="Flux calibration job name to get spectra from",
                  default=None)
    parser.add_option_group(db)

    #- General input options
    inp = optparse.OptionGroup(parser, "Input")
    inp.add_option('-t', '--target',
                   dest="target",
                   action="append",
                   help="Target name, or filename containing a target list. \
                   Can be used multiple times", 
                   default=None)
    inp.add_option("-x", "--exclude",
                   dest="exclude",
                   action="append", 
                   help='Excluded target, or filename containing a list of \
                   targets. Can be used multiple times', 
                   default=None)
    parser.add_option_group(inp)

    #- Output specific options
    out = optparse.OptionGroup(parser, "Output")
    out.add_option("-o","--out_file",
                   help="phrenology output file name (extension can be .yaml,\
                   .yml or .pkl) [%default]",default = None)
    out.add_option("--pkl", dest = "out_pickle",
                   help = "Name of the pickle file if you want to save the \
                   results in yaml *and* pickle at the same time",
                   action="store_true",default = False )
    out.add_option("--more_output", dest = "save_pickles",
                   action = "store_true",
                   help = "Save pickle file with DrGall for each SN.",
                   default = False  )
    out.add_option("-d", "--output_dir",
                   help="Path to the output directory where the phrenology \
                   data, and the control_plot directory will be saved",
                   default="./")


    out.add_option("--no_unique_suffix", action="store_true", default=False)
    parser.add_option_group(out)

    #- Redshift and reddening tweaking options
    red = optparse.OptionGroup(parser, "Dereddening")
#    red.add_option("--AB", dest = "AB", type = "string", help = "Name of the
#    key corresponded to the AB value used to derredened spectra. If --AB
#    yourkey, 'yourkey.AB' is use as key", default = None)
    red.add_option("--ebmv", type = "float",
                   help="Value of ebmv used to deredden *all* the spectra \
                   [%default]", default=None)
    red.add_option("--Rv", type = "float",
                   help="Value of Rv used to deredden spectra [%default]",
                   default=3.1)
#    red.add_option("-c", dest = "color", action="store_true", help = "If set,
#    all spectra will be derredened with the SALT2 color.", default=False)
    red.add_option('-z', '--redshift', type='float',
                   help="Force a redshift",
                   default = None)
    parser.add_option_group(red)

    #- Technical Phrenology options
    adv = optparse.OptionGroup(parser, "Advanced")
    adv.add_option("--smoother","-s",dest="smoother",
                   help='smoother used to smooth the spectrum \
                   (default=%default, other can be "spline")',
                   default='sgfilter')
    adv.add_option("--nsimu",dest="nsimu",type='int',
                   help='Number of simulation used to compute statistical \
                   error [%default]',default=1000)
    parser.add_option_group(adv)

    #- Where and what to plot. Or not plot.
    plt = optparse.OptionGroup(parser, "Plot")
    plt.add_option("--dir",
                   help = "Name of the directory where control plots are saved\
                   [%default]", default = "control_plots"  )
    plt.add_option("-f", '--format',
                   help = "Format of the control plot (png,eps,ps,pdf...) \
                   [%default]",
                   default = "png"  )
    plt.add_option("--noplot", dest="plot", action="store_false",
                   help="Don't produce control plots", default=True)
    parser.add_option_group(plt)


###    deprecated = optparse.OptionGroup(parser, "Deprecated")
###    deprecated.add_option("-k", "--keep_all", default=False,
###    action = "store_true",
###                   help = "If set it will merge the yaml file read in and
###    the phrenologie output")
###    deprecated.add_option("--local", dest = "local", help = "Name of the
###    directory where all the spectra are saved", default = False  )

    option, args = parser.parse_args()

    # spectra fits as args
    option.specs = args

    if option.smoother == 'spline':
        option.smoother += '_free_knot'
#    if option.color and option.ebmv:
#        parser.error('Error, give me only ebmv option or color option.')
#    if option.AB is not None:
#        if option.ebmv:
#            parser.error('Error, give me only ebmv option or AB option.')
#        elif option.color:
#            parser.error('Error, give me only color option or AB option.')

    if option.DB \
           and (option.target is None \
                or (option.inver is None \
                    and option.injob is None)):
        error = 'You need to set a target name (-t) and a prod version (-v)'
        error += 'or job name (-j) to use the DB'
        parser.error(error)
    if option.specID is not None:
        option.specID = option.specID.split(',')
    assert not option.DB or SnfUtil.hasDB(), 'Cannot connect to the DB!'

    option.command_line = " ".join(sys.argv)
    option.prefix = sys.argv[0]
    option.unique_suffix = time.strftime("%Y-%m-%d-%H_%M_%S_UTC",
                                         time.gmtime())

    if option.target is not None:
        target_set = set()
        for target_name in option.target:
            if os.path.exists(target_name) and not os.path.isdir(target_name):
                print "Reading targets from %s" % target_name
                target_set = target_set.union( set(read_ascii(target_name)) )
            else:
                target_set.add(target_name)
        option.target = target_set

    if option.exclude is not None:
        target_set = set()
        for target_name in option.exclude:
            if os.path.exists(target_name):
                print "Reading targets from %s" % target_name
                target_set = target_set.union( set(read_ascii(target_name)) )
            else:
                target_set.add(target_name)
        option.exclude = target_set

    return option



def read_ascii(filename):
    """
    Read ascii files and returns a list of lines.
    Empty lines and lines beginning with # are not returned.
    All comments starting with # in a line are dropped.
    All lines are stripped.
    """
    fh = open(filename, "r")
    line_list = []
    for line in fh.readlines():
        line = (re.sub("#.+$","",line)).strip()        
        if not len(line):
            continue
        line_list.append(line)
    return line_list

def read_from_idr(option):
    """
    Creates an SnfMetaData dictionnary out of the idr pointed at by option.idr

    For compatibility with the other options, option.data_dir is set to
    option.idr
    """
    # define the subsets to load
    all_subsets = ["training", "validation", "auxiliary", "bad","good"]
    if option.target is not None:
        option.subset = None
    elif option.subset is "all":
        option.subset = all_subsets
    elif option.subset is not None:
        if not option.subset in all_subsets:
            raise ValueError("option.subset must be in ",all_subsets)
    else:
        # default
        option.subset = ["training", "validation"]
    print option.idr, option.subset, option.target
    # load the data
    d = SnfMetaData.load_idr(option.idr,
                             subset=option.subset,
                             targets=option.target)

    option.data_dir = option.idr
    return d

def read_from_DB(option):
    """
    Creates an SnfMetaData dictionnary out of the DB.

    The path to the files is supposed to be absolute, therefore
    option.data_dir is set to "".
    """

    from processing.process.models import Process
    tgt = list(option.target)[0]

    # get SALT2 yaml
    salt2 = Process.objects.filter(Fclass=700, XFclass__in=[100, 720, 800],
                                   Pose_FK__Exp_FK__Run_FK__TargetId_FK__Name=\
                                   tgt).order_by('-IdProcess')
    if option.inver is not None:
        salt2 = salt2.filter(Version=option.inver)
    if option.injob is not None:
        salt2 = salt2.filter(Job_FK__Name__contains=option.injob)

    assert salt2.count(), 'no SALT2 YAML found in the DB!'
    salt2 = salt2[0]
    salt2data = IO.load_anyfile(salt2.File_FK.FullPath)
    # make a compatible dict with the rest of the code
    data = {'salt2.yaml': str(salt2.File_FK.FullPath)}
    for k, v in salt2data['saltparams'].iteritems():
        data['salt2.'+k] = v[0]
        if v[1] != -1:
            data['salt2.'+k+'.err'] = v[1]
    data['host.zhelio'] = option.redshift \
                          and option.redshift \
                          or salt2data['target']['redshift']
    data['target.mwebv'] = salt2data['target']['mwebv']

    # loop on spectra used by SALT2
    specs = salt2.Parent_FK.get(Fclass=680,
                                XFclass=salt2.XFclass).Parent_FK.all().order_by('-IdProcess')
    spectra = dict([(str(i), {})
                    for i in sorted(set([s.File_FK.Name[:14] for s in specs
                                         ]))])
    # check if only a few spectra are asked for
    if option.specID is not None:
        for spec in spectra.keys():
            if spec not in option.specID:
                spectra.pop(spec)

    for spec in list(spectra):
        s = specs.filter(IdProcess__startswith=spec.replace('_', ''))
        if s[0].Pose_FK.Exp_FK.Run_FK.Kind in ['HostGalaxy', 'FinalRef'] \
               or spec not in salt2data['data']['exp']:
            continue

        if s.filter(Channel=4).count(): # blue
            specB = str(s.filter(Channel=4)[0].File_FK.FullPath)
        else:
            specB = None
        if s.filter(Channel=2).count(): # red
            specR = str(s.filter(Channel=2)[0].File_FK.FullPath)
        else:
            specR = None

        # mjd from SALT
        w = salt2data['data']['exp'].index(spec)
        mjd = salt2data['data']['mjd'][w]
        spectra[spec] = {'idr.spec_B': specB,
                         'idr.spec_R': specR,
                         'obs.mjd' : mjd}

    # put it all together
    data['spectra'] = spectra
    d = {tgt : data}

    option.data_dir = ""
    return SnfMetaData.SnfMetaData(d)

def read_from_fits(option):
    """
    Creates an SnfMetaData compatible dictionnary out of the spectrum
    fits header.
    
    Sets option.data_dir to ./ since it is expected that option.spec gives
    the path to the file from the local directory.

    FIXME: it would be better not to use pySnurp in order to remove the
    dependency completely
    """
    from copy import copy
    d = {}
    default_chan = {'idr.spec_B': None,
                    'idr.spec_R': None,
                    'idr.spec_merged': None}
    for inspec in option.specs:
        print "read", inspec
        spec = pySnurp.Spectrum(inspec,keepFits=False)
        obj = spec.readKey('OBJECT')
        assert option.redshift is not None \
               or SnfUtil.hasDB(), 'no DB found, you need to set --redshift'
        d.setdefault(obj, {}).setdefault('host.zhelio',
                                         option.redshift is not None \
                                         and option.redshift \
                                         or SnfUtil.redshift(obj))
        assert option.ebmv is not None \
               or SnfUtil.hasDB(), 'no DB found, you need to set --ebmv'
        d.setdefault(obj, {}).setdefault('target.mwebv',
                                         option.ebmv is not None \
                                         and option.ebmv \
                                         or SnfUtil.sfd_ebmv(obj))

        d[obj].setdefault('spectra', {}).setdefault(spec.readKey('OBSID'),
                                                    copy(default_chan))

        channel = spec.readKey('CHANNEL')
        if channel == 'Blue channel' or channel == 'B':
            channel = 'B'
        elif channel == 'R':
            pass
        elif channel == 'B+R':
            channel = 'BR'
        else:
            raise ValueError('Error in the channel of the given spectrum:%s'\
                             % channel)

        d[obj]['spectra'][spec.readKey('OBSID')]['idr.spec_'+channel] = inspec

    # only check last spectrum for absolute path
    if os.path.isabs(inspec):
        option.data_dir = ''
    else:
        option.data_dir = "./"
    return SnfMetaData.SnfMetaData(d)

class SimpleSpectrum:
    """
    Simple class to store the data in an object instead of in 3 arrays.
    This is simply a hook needed until DrGall learns how to ingest arrays.

    FIXME: this should not stay here for ever
    prediction: This will stay here for ever (SZF, 14/11/2011)
    """
    def __init__(self, x,y, v=None):
        self.x = x
        self.y = y
        self.v = v
        self.step = self.x[1] - self.x[0]
        self.name = "dummy"

def read_spectrum(filename, z_helio=None, mwebv=None, Rv=3.1):
    """
    Reads spectra and returns a simple object with x,y,v as attribute,
    as needed by DrGall

    FIXME: it would be nice if DrGall could ingest 3 arrays instead,
    since it copies them in its own internal structure anyway.
    """

    try:
        x,y,v = Spectrum.read_fits(filename, var=True)
    except:
        print "Problem reading %s" % filename
        return None

    if mwebv is not None:
        y,v = Spectrum.deredden(x,y, variance=v, ebmv=mwebv, Rv=Rv)
    if z_helio is not None:
        x,y,v = Spectrum.deredshift(x,y, z_helio, variance=v)

    return SimpleSpectrum(x,y,v=v)

# In read_spectrum before =======
#+ FIXME: --AB option is broken
#    if option.AB:
#        spec.deredden(option.host_ebmv)

# option removed
#    if salt2color is not None:
#        y,v = SALTmodel.decolor_flux(x,y,v,salt2color)
# ==========

# Not used anymore
#def ebmv(AB,Rv=3.1,ab=0.9982 ,bb=1.0495):
#    return (1./Rv)*(AB/(ab+(bb/Rv)))

#=======================================
if __name__ == '__main__':

    option = read_option()

    #- Read the meta-data depending on the method used
    #- If you add a method here, be sure to return a proper SnfMetaData
    if option.DB:
        d = read_from_DB(option)
    elif option.specs:
        d = read_from_fits(option)
    # search IDR by default
    elif os.path.isdir(option.idr):
        if option.ebmv is not None:
            print 'Warning: you are reading from an IDR but set ebmv = %f'%\
                  option.ebmv
            print 'for *all* targets'
        d = read_from_idr(option)

    print "%i targets loaded."%len(d.keys())
    
    #- Target selection
    if option.target is not None:
        target_set = option.target
    else:
        target_set = set(d.keys())

    if option.exclude is not None:
        target_set = target_set.difference(option.exclude)

    # filter targets
    d.set_filter(target__name__in = list(target_set))

    # Create SNfPhrenology data structure
    d_phrenology = SnfMetaData.SnfMetaData()

    if not os.path.exists(option.output_dir):
        print >> sys.stderr, code_name + "creating %s" % option.output_dir
        os.mkdir(option.output_dir)

    #-  Loop over targets
    for target_name, z_helio, mwebv, color,\
        B_filename, R_filename, merge_filename,\
        expId, phase in zip(*d.spectra("target.name",
                                       "host.zhelio",
                                       "target.mwebv",
                                       "salt2.Color",
                                       "idr.spec_B",
                                       "idr.spec_R",
                                       "idr.spec_merged",
                                       "obs.exp",
                                       "salt2.phase")):
        #+ z_helio -> option.z
        #+ mwebv -> option.mwebv
        #+ color -> option.salt2color removed!
        tmpdir = os.path.join(option.output_dir, target_name)
        if not os.path.exists(tmpdir):
            print >> sys.stderr, code_name + "creating %s" % tmpdir
            os.mkdir(tmpdir)

        #- If needed, create the directory to save control plots
        plotdir = os.path.join(option.output_dir, target_name, option.dir)
        if not os.path.exists(plotdir) and option.plot:
            print >> sys.stderr, code_name + "creating %s" % plotdir
            os.mkdir(plotdir)

        # did we set an ebmv for de-reddening for *all* targets?
        if option.ebmv is not None and option.ebmv != mwebv:
            mwebv = option.ebmv

        print "reading %s" % expId
        # merged
        try:
            spec_merge = read_spectrum(os.path.join(option.data_dir,
                                                    merge_filename),
                                       z_helio=z_helio,
                                       mwebv=mwebv,
                                       Rv=option.Rv)
        except AttributeError:
            spec_merge = None
        # B channel
        try:
            spec_B = read_spectrum(os.path.join(option.data_dir, B_filename),
                                   z_helio=z_helio, mwebv=mwebv, Rv=option.Rv)
        except AttributeError:
            spec_B = None
        # R channel
        try:
            spec_R = read_spectrum(os.path.join(option.data_dir, R_filename),
                                   z_helio=z_helio, mwebv=mwebv, Rv=option.Rv)
        except AttributeError:
            spec_R = None

        if spec_merge is None and spec_B is None and spec_R is None:
            print "WARNING: No spectrum for this %s, pass."%expId
            continue
        
        DrGall = SNfPhrenology.DrGall(spec=spec_merge,
                                      specB=spec_B,
                                      specR=spec_R)

        if spec_merge is not None or spec_B is not None:
            calcium = DrGall.calcium_computing(nsimu=option.nsimu,
                                               smoother=option.smoother,
                                               verbose=True)
        else:
            print "There is no B channel"
    
        if spec_merge is not None or spec_R is not None:
            silicon = DrGall.silicon_computing(nsimu=option.nsimu,
                                               smoother=option.smoother,
                                               verbose=True)
            oxygen = DrGall.oxygen_computing(nsimu=option.nsimu,
                                             smoother=option.smoother,
                                             verbose=True)
        else:
            print "There is no R channel"

        if spec_merge is not None \
               or (spec_B is not None \
                   and spec_R is not None):
            if spec_merge is not None:
                Rsjb = [SNfPhrenology.stephen_ratio(spec_merge), 0.0]
            else:
                Rsjb = [SNfPhrenology.stephen_ratio(spec_B, spec_R), 0.0]
            iron = DrGall.iron_computing(nsimu=option.nsimu,
                                         smoother=option.smoother,
                                         verbose=True)
        else:
            Rsjb = [np.nan] * 2

        #Add 'phrenology' on each key of the dictionnary        
        DrGall.Values.update({'Rsjb': float(Rsjb[0]),
                              'Rsjb.err': float(Rsjb[1])})
        for key in DrGall.Values.keys():
            DrGall.Values['phrenology.%s' % key] = DrGall.Values.pop(key)
        
        #Update final dictionnary
        d_phrenology.add_to_spec(target_name, expId,
                                 items=DrGall.Values, autoerr=False)

        #Control plots
        suffix = not option.no_unique_suffix \
                 and ('.' + option.unique_suffix) \
                 or ''
        tgtexp =  target_name + '.' + expId
        cpname = "control_plot.SNfPhrenology." + tgtexp + suffix
        cpname_ox = "control_plot.SNfPhrenology.oxygen_zone." + tgtexp + suffix
        cpname_fe = "control_plot.SNfPhrenology.iron_zone." + tgtexp + suffix
        control_plot_name = os.path.join(plotdir, cpname)
        control_plot_name_ox = os.path.join(plotdir, cpname_ox)
        control_plot_name_fe = os.path.join(plotdir, cpname_fe)

        if option.plot:
            format = option.format.split(',')
            title = target_name+', Rest-Frame Phase=%.1f' % phase
            try:
                DrGall.control_plot(filename=control_plot_name,
                                    title=title,
                                    format=format)
            except Exception, err:
                print "<%s> WARNING: control_plot had a problem:"%\
                      code_name, err
            try:
                DrGall.plot_oxygen(filename=control_plot_name_ox,
                                   title=title,
                                   format=format)
            except Exception, err:
                print "<%s> WARNING: control_plot for oxygen had a problem:"%\
                      code_name, err
            try:
                DrGall.plot_iron(filename=control_plot_name_fe,
                                 title=title,
                                 format=format)
            except Exception, err:
                print "<%s> WARNING: control_plot for iron had a problem:"%\
                      code_name, err
        print '\n'

        if option.save_pickles:
            fname = "data.SNfPhrenology."+ tgtexp + suffix + ".pkl"
            filename = os.path.join(plotdir, fname)
            IO.dump_anyfile(DrGall, filename)

    if option.out_file is None:
        filename = "phrenology_output.yaml"
    else:
        filename = option.out_file
    pkl_filename = filename.replace("yaml", "pkl")

    f = os.path.join(option.output_dir, filename)
    if filename.endswith('pkl'):
        pass
    elif  filename.endswith('yml'):
        pkl_filename = filename.replace("yml", "pkl")
    elif filename.endswith('yaml'):
        pkl_filename = filename.replace("yaml", "pkl")
    else:
        print "option out_file didn't end in .pkl, .yaml or .yml (%s)"%\
              option.out_file
        print "adding the pkl extension!"
        f += ".pkl"

    IO.dump_anyfile(d_phrenology, pkl_filename)
    
    #if option.out_pickle is not None:
    #    print d_phrenology
    #    IO.dump_anyfile(d_phrenology,
    #                    os.path.join(option.output_dir, pkl_filename))

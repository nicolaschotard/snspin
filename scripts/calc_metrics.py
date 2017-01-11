#!/usr/bin/env python

"""
Code to apply Phrenology measurements on data that can come from:
   + An IDR : default behavior
   + A local fits file : If spectra are fed in via the argument line.
     If so, they overwrite all options (they are stored in option.specs)

   Creates a directory with the name of the supernova in which the results
   and plot directory will be placed.

"""

import cPickle
import optparse
import sys
import os
import re

# use Agg by default (do not try to show plot windows - extremely slow over ssh)
import matplotlib
matplotlib.use('Agg')
import numpy as np

from snspin import spin
from snspin import spinmeas
from snspin.extern import pySnurp
from snspin.extern import SnfMetaData

code_name = os.path.basename(__file__) + ' '

def read_option():
    usage = "usage: %prog [options] specnames"
    parser = optparse.OptionParser(usage=usage)

    #- Input options for idr mode
    idr = optparse.OptionGroup(parser, "idr")
    idr.add_option("--idr", dest="idr",
                   default="/afs/in2p3.fr/group/snovae/snprod1/IDR/current/",
                   help="Sets the idr run to use")
    idr.add_option("--subset", dest="subset", action="append",
                   default=None,
                   help="Limits the IDR to the subsets selected")
    parser.add_option_group(idr)

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
    inp.add_option('--expid',
                   dest="expid",
                   action="append",
                   help="Exposure ID. Can be used multiple times",
                   default=None)
    parser.add_option_group(inp)

    #- Output specific options
    out = optparse.OptionGroup(parser, "Output")
    out.add_option("-o", "--out_file",
                   help="snspin output file name (extension can be .yaml,\
                   .yml or .pkl) [%default]", default=None)
    out.add_option("--pkl", dest="out_pickle",
                   help="Name of the pickle file if you want to save the \
                   results in yaml *and* pickle at the same time",
                   action="store_true", default=False)
    out.add_option("--more_output", dest="save_pickles",
                   action="store_true",
                   help="Save pickle file with DrGall for each SN.",
                   default=False)
    out.add_option("-d", "--output_dir",
                   help="Path to the output directory where the snspin \
                   data, and the control_plot directory will be saved",
                   default="./")
    parser.add_option_group(out)

    #- Redshift and reddening tweaking options
    red = optparse.OptionGroup(parser, "Dereddening")
    red.add_option("--ebmv", type="float",
                   help="Value of ebmv used to deredden *all* the spectra \
                   [%default]", default=None)
    red.add_option("--Rv", type="float",
                   help="Value of Rv used to deredden spectra [%default]",
                   default=3.1)
    red.add_option('-z', '--redshift', type='float',
                   help="Force a redshift",
                   default=None)
    parser.add_option_group(red)

    #- Technical Snspin options
    adv = optparse.OptionGroup(parser, "Advanced")
    adv.add_option("--smoother", "-s", dest="smoother",
                   help='smoother used to smooth the spectrum \
                   (default=%default, other can be "spline")',
                   default='sgfilter')
    adv.add_option("--nsimu", dest="nsimu", type='int',
                   help='Number of simulation used to compute statistical \
                   error [%default]', default=1000)
    parser.add_option_group(adv)

    #- Where and what to plot. Or not plot.
    plt = optparse.OptionGroup(parser, "Plot")
    plt.add_option("--dir",
                   help="Name of the directory where control plots are saved\
                   [%default]", default="control_plots" )
    plt.add_option("-f", '--oformat', help="Format of the control plot (png, eps, ps, pdf) \
                   [%default]", default="png")
    plt.add_option("--noplot", dest="plot", action="store_false",
                   help="Don't produce control plots", default=True)
    parser.add_option_group(plt)

    option, args = parser.parse_args()

    # spectra fits as args
    option.specs = args

    if option.smoother == 'spline':
        option.smoother += '_free_knot'

    option.command_line = " ".join(sys.argv)
    option.prefix = sys.argv[0]

    if option.target is not None:
        targets = set()
        for target_name in option.target:
            if os.path.exists(target_name) and not os.path.isdir(target_name):
                print "Reading targets from %s" % target_name
                targets = targets.union(set(read_ascii(target_name)))
            else:
                targets.add(target_name)
        option.target = targets

    if option.exclude is not None:
        targets = set()
        for target_name in option.exclude:
            if os.path.exists(target_name):
                print "Reading targets from %s" % target_name
                targets = targets.union(set(read_ascii(target_name)))
            else:
                targets.add(target_name)
        option.exclude = targets

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
        line = (re.sub("#.+$", "", line)).strip()        
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
            raise ValueError("option.subset must be in ", all_subsets)
    else:
        # default
        option.subset = ["training", "validation"]

    # load the data
    option.data_dir = option.idr
    return SnfMetaData.load_idr(option.idr, subset=option.subset, targets=option.target)


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
        spec = pySnurp.Spectrum(inspec, keepFits=False)
        obj = spec.readKey('OBJECT')
        assert option.redshift is not None, 'you need to set --redshift'
        d.setdefault(obj, {}).setdefault('host.zhelio', option.redshift)
        assert option.ebmv is not None, 'you need to set --ebmv'
        d.setdefault(obj, {}).setdefault('target.mwebv', option.ebmv)

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

def read_spectrum(filename, z_helio=None, mwebv=None, Rv=3.1):
    """
    Reads spectra and returns a simple object with x,y,v as attribute,
    as needed by DrGall
    """
    spec = pySnurp.Spectrum(filename)
    if mwebv is not None:
        spec.deredden(mwebv, Rv=Rv)
    if z_helio is not None:
        spec.deredshift(z_helio)
    return spec


if __name__ == '__main__':

    option = read_option()

    #- Read the meta-data depending on the method used
    #- If you add a method here, be sure to return a proper SnfMetaData
    if option.specs:
        d = read_from_fits(option)
    # search IDR by default
    elif os.path.isdir(option.idr):
        if option.ebmv is not None:
            print 'Warning: you are reading from an IDR but set ebmv = %f'%\
                  option.ebmv
            print 'for *all* targets'
        d = read_from_idr(option)

    print "INFO: %i target(s) loaded.\n"%len(d.keys())

    #- Target selection
    if option.target is not None:
        targets = option.target
    else:
        targets = set(d.keys())

    if option.exclude is not None:
        targets = targets.difference(option.exclude)

    # filter targets
    d.set_filter(target__name__in=list(targets))

    # Create spin data structure
    dspin = SnfMetaData.SnfMetaData()

    if not os.path.exists(option.output_dir):
        print >> sys.stderr, code_name + "creating %s" % option.output_dir
        os.mkdir(option.output_dir)

    #-  Loop over targets
    for target_name, z_helio, mwebv, color,\
        b_filename, r_filename, merge_filename,\
        expId, phase in zip(*d.spectra("target.name",
                                       "host.zhelio",
                                       "target.mwebv",
                                       "salt2.Color",
                                       "idr.spec_B",
                                       "idr.spec_R",
                                       "idr.spec_merged",
                                       "obs.exp",
                                       "salt2.phase")):
        if option.expid is not None and expId not in option.expid:
            continue
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

        print "=" * 80
        print "INFO: Reading %s, from %s" % (expId, target_name)
        print "INFO: Correcting from MW extinction (E(B-V)=%.2f)" %mwebv
        print "INFO: Going to rest-frame using z=%.3f" % z_helio
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
            spec_b = read_spectrum(os.path.join(option.data_dir, b_filename),
                                   z_helio=z_helio, mwebv=mwebv, Rv=option.Rv)
        except AttributeError:
            spec_b = None
        # R channel
        try:
            spec_r = read_spectrum(os.path.join(option.data_dir, r_filename),
                                   z_helio=z_helio, mwebv=mwebv, Rv=option.Rv)
        except AttributeError:
            spec_r = None

        if spec_merge is None and spec_b is None and spec_r is None:
            print "WARNING: No spectrum for this %s, pass."%expId
            continue

        DrGall = spinmeas.DrGall(spec=spec_merge, specb=spec_b, specr=spec_r)

        if spec_merge is not None or spec_b is not None:
            calcium = DrGall.calcium_computing(nsimu=option.nsimu,
                                               smoother=option.smoother,
                                               verbose=True)
        else:
            print "There is no B channel"

        if spec_merge is not None or spec_r is not None:
            silicon = DrGall.silicon_computing(nsimu=option.nsimu,
                                               smoother=option.smoother,
                                               verbose=True)
            oxygen = DrGall.oxygen_computing(nsimu=option.nsimu,
                                             smoother=option.smoother,
                                             verbose=True)
        else:
            print "There is no R channel"

        if spec_merge is not None \
               or (spec_b is not None \
                   and spec_r is not None):
            if spec_merge is not None:
                rsjb = [spin.stephen_ratio(spec_merge), 0.0]
            else:
                rsjb = [spin.stephen_ratio(spec_b, spec_r), 0.0]
            iron = DrGall.iron_computing(nsimu=option.nsimu,
                                         smoother=option.smoother,
                                         verbose=False)
        else:
            rsjb = [np.nan] * 2

        #Add 'snspin' on each key of the dictionnary
        DrGall.values.update({'Rsjb': float(rsjb[0]),
                              'Rsjb.err': float(rsjb[1])})
        for key in DrGall.values.keys():
            DrGall.values['snspin.%s' % key] = DrGall.values.pop(key)

        #Update final dictionnary
        dspin.add_to_spec(target_name, expId,
                          items=DrGall.values, autoerr=False)

        #Control plots
        tgtexp = target_name + '.' + expId
        cpname = tgtexp + ".control_plot"
        cpname_ox = tgtexp + ".control_plot.oxygen_zone"
        cpname_fe = tgtexp + ".control_plot.iron_zone"
        control_plot_name = os.path.join(plotdir, cpname)
        control_plot_name_ox = os.path.join(plotdir, cpname_ox)
        control_plot_name_fe = os.path.join(plotdir, cpname_fe)

        if option.plot:
            print "\nINFO: Making control plots"
            title = target_name+', Rest-Frame Phase=%.1f' % phase
            try:
                DrGall.control_plot(filename=control_plot_name, title=title, oformat=option.oformat)
            except Exception, err:
                print "<%s> WARNING: control_plot had a problem:"%\
                      code_name, err
            try:
                DrGall.plot_oxygen(filename=control_plot_name_ox, title=title, oformat=option.oformat)
            except Exception, err:
                print "<%s> WARNING: control_plot for oxygen had a problem:"%\
                      code_name, err
            try:
                DrGall.plot_iron(filename=control_plot_name_fe,
                                 title=title,
                                 oformat=option.oformat)
            except Exception, err:
                print "<%s> WARNING: control_plot for iron had a problem:"%\
                      code_name, err
        print '\n'

        if option.save_pickles:
            fname = tgtexp + ".spin_data" + ".pkl"
            filename = os.path.join(plotdir, fname)
            cPickle.dump(DrGall, open(filename, 'w'))

    filename = "snspin_output.yaml" if option.out_file is None else option.out_file
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

    cPickle.dump(dspin, open(pkl_filename, 'w'))

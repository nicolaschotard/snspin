#!/usr/bin/env python

"""
Code to apply SNe Ia spectral indicator measurements.

Data can come from:
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
    """Read all options."""
    usage = "usage: %prog [options] specnames"
    parser = optparse.OptionParser(usage=usage)

    # Input options
    inp = optparse.OptionGroup(parser, "Input")
    inp.add_option("--idr", default=None, help="Path to an SNfactory IDR")
    inp.add_option("--subset", default=None,
                   help="Limits the IDR to the subsets selected (coma separated)")
    inp.add_option('--target', action="append",
                   help="Target name, or filename containing a target list. \
                   Can be used multiple times", default=None)
    inp.add_option("--exclude",
                   help='Excluded target, or filename containing a list of \
                   targets (coma separeted)', default=None)
    inp.add_option('--expid', help="Exposure ID (coma separated)",
                   default=None)
    parser.add_option_group(inp)

    #- Output specific options
    out = optparse.OptionGroup(parser, "Output")
    out.add_option("--output", default=None,
                   help="snspin output file name (will be save as .pkl) [%default]",)
    out.add_option("--more_output", dest="save_pickles",
                   action="store_true",
                   help="Save pickle file with DrGall for each SN.",
                   default=False)
    out.add_option("--odir", default="./",
                   help="Path to the output directory where the snspin \
                   data, and the control_plot directory will be saved")
    parser.add_option_group(out)

    #- Redshift and reddening tweaking options
    red = optparse.OptionGroup(parser, "Corrections")
    red.add_option("--ebmv", type="float",
                   help="Ebmv value used to deredden *all* the input spectra \
                   [%default]", default=None)
    red.add_option("--rv", type="float",
                   help="Rv value used to deredden spectra [%default]",
                   default=3.1)
    red.add_option('--redshift', type='float', help="Force a redshift",
                   default=None)
    parser.add_option_group(red)

    #- Technical Snspin options
    adv = optparse.OptionGroup(parser, "Advanced")
    adv.add_option("--smooth",
                   help='Smoother used to smooth the spectrum \
                   (default=%default, other can be "spline")',
                   default='sgfilter')
    adv.add_option("--nsimu", dest="nsimu", type='int',
                   help='Number of simulation used to compute statistical \
                   error [%default]', default=1000)
    parser.add_option_group(adv)

    #- Where and what to plot. Or not plot.
    plt = optparse.OptionGroup(parser, "Plot")
    plt.add_option("--pformat",
                   help="Format of the control plot (png, eps, ps, pdf) \
                   [%default]", default="png")
    plt.add_option("--noplot", dest="plot", action="store_false",
                   help="Do not produce the control plots", default=True)
    parser.add_option_group(plt)

    opts, args = parser.parse_args()

    # spectra fits as args
    opts.specs = args

    if opts.smooth == 'spline':
        opts.smooth += '_free_knot'

    opts.command_line = " ".join(sys.argv)
    opts.prefix = sys.argv[0]

    if opts.target is not None:
        tgs = set()
        for tgn in opts.target:
            if os.path.exists(tgn) and not os.path.isdir(tgn):
                print "Reading targets from %s" % tgn
                tgs = tgs.union(set(read_ascii(tgn)))
            else:
                tgs.add(tgn)
        opts.target = tgs

    if opts.exclude is not None:
        tgs = set()
        for tgn in opts.exclude:
            if os.path.exists(tgn):
                print "Reading targets from %s" % tgn
                tgs = tgs.union(set(read_ascii(tgn)))
            else:
                tgs.add(tgn)
        opts.exclude = tgs

    return opts


def read_ascii(aname):
    """
    Read ascii files and returns a list of lines.

    Empty lines and lines beginning with # are not returned.
    All comments starting with # in a line are dropped.
    All lines are stripped.
    """
    fh = open(aname, "r")
    line_list = []
    for line in fh.readlines():
        line = (re.sub("#.+$", "", line)).strip()
        if not len(line):
            continue
        line_list.append(line)
    return line_list


def read_from_idr(opts):
    """
    Create an SnfMetaData dictionnary out of the idr pointed at by option.idr.

    For compatibility with the other options, option.data_dir is set to
    option.idr
    """
    # define the subsets to load
    all_subsets = ["training", "validation", "auxiliary", "bad","good"]
    if opts.target is not None:
        opts.subset = None
    elif opts.subset is "all":
        opts.subset = all_subsets
    elif opts.subset is not None:
        if not opts.subset in all_subsets:
            raise ValueError("option.subset must be in ", all_subsets)
    else:
        # default
        opts.subset = ["training", "validation"]

    # load the data
    opts.data_dir = opts.idr
    return SnfMetaData.load_idr(opts.idr, subset=opts.subset, targets=opts.target)


def read_from_fits(opts):
    """
    Create an SnfMetaData compatible dictionnary out of the spectrum fits header.

    Sets opts.data_dir to ./ since it is expected that opts.spec gives
    the path to the file from the local directory.
    """
    from copy import copy
    data = {}
    default_chan = {'idr.spec_B': None,
                    'idr.spec_R': None,
                    'idr.spec_merged': None}
    for inspec in opts.specs:
        print "INFO: Reading input spectrum", inspec
        spec = pySnurp.Spectrum(inspec, keepfits=False)
        obj = spec.readKey('OBJECT')
        data.setdefault(obj, {}).setdefault('host.zhelio', opts.redshift \
                                            if opts.redshift is not None \
                                            else 0)
        data.setdefault(obj, {}).setdefault('target.mwebv', opts.ebmv \
                                            if opts.ebmv is not None \
                                            else 0)
        data[obj].setdefault('spectra', {}).setdefault(spec.readKey('OBSID'),
                                                       copy(default_chan))
        channel = spec.readKey('CHANNEL')
        if channel == 'Blue channel' or channel == 'B':
            channel = 'B'
        elif channel == 'R':
            pass
        elif channel == 'B+R':
            if opts.redshift is None:
                channel = 'restframe'
            else:
                channel = 'merged'
        else:
            raise ValueError('Error in the channel of the given spectrum:%s'\
                             % channel)

        data[obj]['spectra'][spec.readKey('OBSID')]['idr.spec_'+channel] = inspec
    # only check last spectrum for absolute path
    if os.path.isabs(inspec):
        opts.data_dir = ''
    else:
        opts.data_dir = "./"
    return SnfMetaData.SnfMetaData(data)

def read_spectrum(aname, z_helio=None, mwebv=None, Rv=3.1):
    """
    Read a spectrum.

    Returns a simple object with x,y,v as attribute, as needed by DrGall
    """
    spec = pySnurp.Spectrum(aname)
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
    else:
        raise IOError("No valid input given")    

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

    if not os.path.exists(option.odir):
        print >> sys.stderr, code_name + "creating %s" % option.odir
        os.mkdir(option.odir)

    #-  Loop over targets
    for target_name, z_helio, mwebv, color,\
        b_filename, r_filename, merge_filename, restframe_filename,\
        expId, phase in zip(*d.spectra("target.name",
                                       "host.zhelio",
                                       "target.mwebv",
                                       "salt2.Color",
                                       "idr.spec_B",
                                       "idr.spec_R",
                                       "idr.spec_merged",
                                       "idr.spec_restframe",
                                       "obs.exp",
                                       "salt2.phase")):
        if option.expid is not None and expId not in option.expid:
            continue
        #+ z_helio -> option.z
        #+ mwebv -> option.mwebv
        #+ color -> option.salt2color removed!
        tmpdir = os.path.join(option.odir, target_name)
        if not os.path.exists(tmpdir):
            print >> sys.stderr, code_name + "creating %s" % tmpdir
            os.mkdir(tmpdir)

        #- If needed, create the directory to save control plots
        plotdir = os.path.join(option.odir, target_name)
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
        spec_merge = merge_filename
        if merge_filename is None:
            spec_merge = None
        else:  # merged spectrum
            spec_merge = read_spectrum(os.path.join(option.data_dir, merge_filename),
                                       z_helio=z_helio, mwebv=mwebv, Rv=option.rv)
        if b_filename is None:
            spec_b = None
        else:  # B channel
            spec_b = read_spectrum(os.path.join(option.data_dir, b_filename),
                                   z_helio=z_helio, mwebv=mwebv, Rv=option.rv)
        if r_filename is None:
            spec_r = None
        else:  # R channel
            spec_r = read_spectrum(os.path.join(option.data_dir, r_filename),
                                   z_helio=z_helio, mwebv=mwebv, Rv=option.rv)
        if restframe_filename is None:
            spec_rf = None
        else:  # R channel
            spec_rf = read_spectrum(os.path.join(option.data_dir, restframe_filename))
        if spec_merge is None and spec_b is None and spec_r is None:
            if spec_rf is None:
                print "WARNING: No spectrum for %s, pass."%expId
                continue
            else:
                spec_merge = spec_rf

        DrGall = spinmeas.DrGall(spec=spec_merge, specb=spec_b, specr=spec_r)

        if spec_merge is not None or spec_b is not None:
            calcium = DrGall.calcium_computing(nsimu=option.nsimu,
                                               smoother=option.smooth,
                                               verbose=True)
        else:
            print "There is no B channel"

        if spec_merge is not None or spec_r is not None:
            silicon = DrGall.silicon_computing(nsimu=option.nsimu,
                                               smoother=option.smooth,
                                               verbose=True)
            oxygen = DrGall.oxygen_computing(nsimu=option.nsimu,
                                             smoother=option.smooth,
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
                                         smoother=option.smooth,
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
                DrGall.control_plot(filename=control_plot_name, title=title, oformat=option.pformat)
            except Exception, err:
                print "<%s> WARNING: control_plot had a problem:"%\
                      code_name, err
            try:
                DrGall.plot_oxygen(filename=control_plot_name_ox, title=title, oformat=option.pformat)
            except Exception, err:
                print "<%s> WARNING: control_plot for oxygen had a problem:"%\
                      code_name, err
            try:
                DrGall.plot_iron(filename=control_plot_name_fe,
                                 title=title,
                                 oformat=option.pformat)
            except Exception, err:
                print "<%s> WARNING: control_plot for iron had a problem:"%\
                      code_name, err
        print '\n'

        if option.save_pickles:
            fname = tgtexp + ".spin_data" + ".pkl"
            filename = os.path.join(plotdir, fname)
            cPickle.dump(DrGall, open(filename, 'w'))

    output = "snspin_output.pkl" if option.output is None else option.output + ".pkl"
    cPickle.dump(dspin, open(output, 'w'))

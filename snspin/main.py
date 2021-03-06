"""Main entry points for scripts."""

import cPickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import os
import numpy

from snspin import spinmeas
from snspin.tools import io


"""
Code to apply SNe Ia spectral indicator measurements.

Data can come from:
   + An IDR : default behavior
   + A local fits file : If spectra are fed in via the argument line.
     If so, they overwrite all options (they are stored in option.specs)

   Creates a directory with the name of the supernova in which the results
   and plot directory will be placed.
"""


def read_option(argv=None):
    """Read all options."""
    description = """Run the spectral indicator calculations."""
    prog = "spincalc.py"
    usage = """%s [options]""" % prog
    parser = ArgumentParser(prog=prog, usage=usage, description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    # Input options
    inp = parser.add_argument_group("Input")
    inp.add_argument("--idr", default=None, help="Path to an SNfactory IDR")
    inp.add_argument('--target', action="append",
                     help="Target name, or filename containing a target list. \
                     Can be used multiple times", default=None)
    inp.add_argument("--exclude",
                     help='Excluded target, or filename containing a list of \
                     targets (coma separeted)', default=None)
    inp.add_argument('--expid', help="Exposure ID (coma separated, or file)",
                     default=None)
    inp.add_argument('--specs',
                     help="Spectrum or list of spectra (coma separated, or file)",
                     default=None)

    # Output specific options
    out = parser.add_argument_group("Output")
    out.add_argument("--output",
                     help="snspin output file name (will be save as .pkl)")
    out.add_argument("--more_output", dest="save_pickles",
                     action="store_true", default=False,
                     help="Save pickle file with DrGall for each SN.")
    out.add_argument("--odir", default="./",
                     help="Path to the output directory where the snspin \
                     data, and the control_plot directory will be saved")

    # Redshift and reddening tweaking options
    red = parser.add_argument_group("Corrections")
    red.add_argument("--ebmv", type=float,
                     help="Ebmv value used to deredden *all* the input spectra")
    red.add_argument("--rv", type=float,
                     help="Rv value used to deredden spectra",
                     default=3.1)
    red.add_argument('--redshift', type=float, help="Force a redshift")

    # Technical snspin options
    adv = parser.add_argument_group("Advanced")
    adv.add_argument("--smooth",
                     help='Smoother used to smooth the spectrum \
                     (other can be "spline")',
                     default='sgfilter')
    adv.add_argument("--nsimu", dest="nsimu", type=int,
                     help='Number of simulation used to compute statistical \
                     error', default=1000)

    # Where and what to plot. Or not plot.
    plt = parser.add_argument_group("Plot")
    plt.add_argument("--pformat", default="png",
                     help="Format of the control plot (png, eps, ps, pdf)")
    plt.add_argument("--noplot", dest="plot", action="store_false",
                     help="Do not produce the control plots", default=True)
    print argv
    opts = parser.parse_args(argv)

    if opts.smooth == 'spline':
        opts.smooth += '_free_knot'

    opts.command_line = " ".join(sys.argv)
    opts.prefix = sys.argv[0]

    if opts.target is not None:
        tgs = set()
        for tgn in opts.target:
            if os.path.exists(tgn) and not os.path.isdir(tgn):
                print "INFO: Reading targets from %s" % tgn
                tgs = tgs.union(set(numpy.loadtxt(tgn, dtype='string', ndmin=1)))
            else:
                tgs.add(tgn)
        opts.target = tgs

    if opts.exclude is not None:
        tgs = set()
        for tgn in opts.exclude:
            if os.path.exists(tgn):
                print "INFO: Reading excluded targets from %s" % tgn
                tgs = tgs.union(set(numpy.loadtxt(tgn, dtype='string', ndmin=1)))
            else:
                tgs.add(tgn)
        opts.exclude = tgs

    if opts.expid is not None:
        if os.path.exists(opts.expid) and not os.path.isdir(opts.expid):
            print "INFO: Reading expids from %s" % opts.expid
            opts.expid = numpy.loadtxt(opts.expid, dtype='string', ndmin=1)
        else:
            opts.expid = opts.expid.split(",")
        print "INFO: %i expids loaded" % len(opts.expid)

    return opts


def read_from_fits(opts):
    """
    Create a compatible dictionnary out of the spectrum fits header.

    Sets opts.data_dir to ./ since it is expected that opts.spec gives
    the path to the file from the local directory.
    """
    from copy import copy
    data = {}
    default_chan = {'idr.spec_B': None,
                    'idr.spec_R': None,
                    'idr.spec_merged': None}
    for inspec in opts.specs:
        # only check last spectrum for absolute path
        if not hasattr(opts, "data_dir"):
            opts.data_dir = '' if os.path.isabs(inspec) else "./"
        print "INFO: Reading input spectrum", inspec
        spec = io.Spectrum(inspec, keepfits=False)
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
        phs = inspec.split('/')[-1].split('_')[1]
        phs = - int(phs[1:]) / 1000. if phs[0] == 'M' else int(phs[1:]) / 1000.
        data[obj]['spectra'][spec.readKey('OBSID')]['salt2.phase'] = phs
    return data


def _add_value_err(data, param, value, autoerr):
    """Parse if value is (value, err) pair and add to dictionary data."""
    if autoerr and hasattr(value, '__iter__') and len(value) == 2:
        data[param] = value[0]
        data[param + '.err'] = value[1]
    else:
        data[param] = value


def add_to_spec(dic, tgt, exp, param=None, value=None, items=None, autoerr=True):
    """
    Add spectrum-level value(s) to a dictionnary, creating target and spectrum entries if needed.

    tgt : target name or list of target names
    exp : exposure code YY_DAY_RUN_EXP or list of exp codes

    either:
        param : parameter name or list of parameter names
        value : parameter values or list of parameter values
    or:
        items : dictionary of parameter:value pairs

    if any values are 2-length lists or tuples, they will be interpreted
    as (value, error) pairs resulting in param and param.err entries
    Set autoerr=False to override this auto-error interpretation.
    """
    def _makeiterable(obj):
        """Transform a single object into an list."""
        return (obj,) if not hasattr(obj, '__iter__') else obj

    tgt = _makeiterable(tgt)
    exp = _makeiterable(exp)

    for name, ex in zip(tgt, exp):
        if name not in dic:
            dic[name] = {'spectra': dict(), 'target.name': name}
        if 'spectra' not in dic[name]:
            dic[name]['spectra'] = dict()
        if ex not in dic[name]['spectra']:
            dic[name]['spectra'][ex] = {
                'target.name': name, 'obs.exp': ex}

        info = dic[name]['spectra'][ex]

        # Loop over parameter, value pairs
        if param is not None and value is not None:
            # Convert single elements into lists first
            param = _makeiterable(param)
            value = _makeiterable(value)
            for p, v in zip(param, value):
                _add_value_err(info, p, v, autoerr)

        # Add any pre-built dictionary items
        if items is not None:
            for p, v in items.iteritems():
                _add_value_err(info, p, v, autoerr)


def read_spectrum(aname, z=None, mwebv=None, Rv=3.1):
    """
    Read a spectrum.

    Returns a simple object with x,y,v as attribute, as needed by DrGall
    """
    spec = io.Spectrum(aname)
    if mwebv is not None:
        spec.deredden(mwebv, Rv=Rv)
    if z is not None:
        spec.deredshift(z)
    return spec


def spincalc(argv=None):
    """Run the spectral indicator calculations."""
    option = read_option(argv)
    if option.specs:  # Read the meta-data depending on the method used
        idata = read_from_fits(option)
    elif os.path.isdir(option.idr):  # search IDR by default
        if option.ebmv is not None:
            print 'Warning: you are reading from an IDR but set ebmv = %f for *all* targets' % \
                option.ebmv
        option.data_dir = option.idr
        idata = io.loaddata(option.idr + "/META.pkl")
    else:
        raise IOError("No valid input given")

    # Target selection
    targets = option.target if option.target is not None else set(idata.keys())
    targets = targets.difference(option.exclude) if option.exclude is not None else targets
    print "INFO: %i target(s) loaded.\n" % len(targets)

    # Reduce initial dictionanry to the selected target list
    idata = {tg: idata[tg] for tg in targets}

    # Create the spin data structure
    dspin = {}

    if not os.path.exists(option.odir):
        print "INFO: creating %s" % option.odir
        os.mkdir(option.odir)

    #  Loop over targets
    for target in sorted(idata):
        z_helio, mwebv = idata[target]["host.zhelio"], idata[target]["target.mwebv"]
        for expId in sorted(idata[target]["spectra"]):
            b_filename = idata[target]["spectra"][expId]["idr.spec_B"]
            r_filename = idata[target]["spectra"][expId]["idr.spec_R"]
            merge_filename = idata[target]["spectra"][expId]["idr.spec_merged"]
            restframe_filename = idata[target]["spectra"][expId]["idr.spec_restframe"]
            phase = idata[target]["spectra"][expId]["salt2.phase"]
            if option.expid is not None and expId not in option.expid:
                continue
            tmpdir = os.path.join(option.odir, target)
            if not os.path.exists(tmpdir):
                print "INFO: creating %s" % tmpdir
                os.mkdir(tmpdir)

            # If needed, create the directory to save control plots
            plotdir = os.path.join(option.odir, target)
            if not os.path.exists(plotdir) and option.plot:
                print "INFO: creating %s" % plotdir
                os.mkdir(plotdir)

            # did we set an ebmv for de-reddening for *all* targets?
            if option.ebmv is not None and option.ebmv != mwebv:
                mwebv = option.ebmv

            print "=" * 80
            print "INFO: Reading %s, from %s" % (expId, target)
            print "INFO: Correcting from MW extinction (E(B-V)=%.2f)" %mwebv
            print "INFO: Going to rest-frame using z=%.3f" % z_helio
            spec_merge = merge_filename
            if merge_filename is None:
                spec_merge = None
            else:  # merged spectrum
                spec_merge = read_spectrum(os.path.join(option.data_dir, merge_filename),
                                           z=z_helio, mwebv=mwebv, Rv=option.rv)
            if b_filename is None:
                spec_b = None
            else:  # B channel
                spec_b = read_spectrum(os.path.join(option.data_dir, b_filename),
                                       z=z_helio, mwebv=mwebv, Rv=option.rv)
            if r_filename is None:
                spec_r = None
            else:  # R channel
                spec_r = read_spectrum(os.path.join(option.data_dir, r_filename),
                                       z=z_helio, mwebv=mwebv, Rv=option.rv)
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
                silicon = DrGall.silicon_computing(nsimu=option.nsimu, smoother=option.smooth,
                                                   verbose=True)
                oxygen = DrGall.oxygen_computing(nsimu=option.nsimu, smoother=option.smooth,
                                                 verbose=True)
            else:
                print "There is no R channel"

            if spec_merge is not None or (spec_b is not None and spec_r is not None):
                iron = DrGall.iron_computing(nsimu=option.nsimu, smoother=option.smooth,
                                             verbose=False)

            #Add 'snspin' on each key of the dictionnary
            for key in DrGall.values.keys():
                DrGall.values['snspin.%s' % key] = DrGall.values.pop(key)

            #Update final dictionnary
            add_to_spec(dspin, target, expId, items=DrGall.values, autoerr=False)

            #Control plots
            tgtexp = target + '.' + expId
            cpname = tgtexp + ".control_plot"
            cpname_ox = tgtexp + ".control_plot.oxygen_zone"
            cpname_fe = tgtexp + ".control_plot.iron_zone"
            control_plot_name = os.path.join(plotdir, cpname)
            control_plot_name_ox = os.path.join(plotdir, cpname_ox)
            control_plot_name_fe = os.path.join(plotdir, cpname_fe)

            if option.plot:
                print "\nINFO: Making control plots"
                title = target+', Rest-Frame Phase=%.1f' % phase
                try:
                    DrGall.control_plot(filename=control_plot_name, title=title,
                                        oformat=option.pformat)
                except Exception, err:
                    print "WARNING: control_plot had a problem:", err
                try:
                    DrGall.plot_oxygen(filename=control_plot_name_ox, title=title,
                                       oformat=option.pformat)
                except Exception, err:
                    print "WARNING: control_plot for oxygen had a problem:", err
                try:
                    DrGall.plot_iron(filename=control_plot_name_fe,
                                     title=title,
                                     oformat=option.pformat)
                except Exception, err:
                    print "WARNING: control_plot for iron had a problem:", err
            print '\n'

            if option.save_pickles:
                fname = tgtexp + ".spin_data" + ".pkl"
                filename = os.path.join(plotdir, fname)
                cPickle.dump(DrGall, open(filename, 'w'))

    output = "snspin_output.pkl" if option.output is None else option.output + ".pkl"
    cPickle.dump(dspin, open(output, 'w'))

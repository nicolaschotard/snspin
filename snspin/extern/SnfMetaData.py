#!/usr/bin/env python

"""
SnfMetaData: Utility classes and functions for SNF target and spectra metadata.

The basic structure tracks per-target and per-spectrum meatadata in the
following hierarchy.  The SnfMetaData class and associated functions
enable merging of metadata from multiple sources to simplify cross-analysis
comparision.

TargetName1 :
    host.zhelio  : xx.yy
    target.mwebv : xx.yy
    target.name  : TargetName1   # autofilled for you
    spectra :
        YY_DAY_RUN_EXP :
            obs.exp  : YY_DAY_RUN_EXP   # autofilled for you
            obs.mjd  : xx.yy
            obs.flux : /path/to/file.fits
            obs.var  : /path/to/varfile.fits
            Seb.RSi  : xx.yy
            Nico.RSi : xx.yy
            SJB.Rx   : xx.yy
        YY_DAY_RUN_EXP :
            # ...
TargetName2 :
    # ...

You are encouraged (though not strictly required) to use namespaces for your
metadata variables, e.g. Nico.RSi and Seb.RSi instead of just RSi.

To merge data from multiple sources:

    meta = SnfMetaData('moksha_output.yaml')
    meta.merge('flux_ratios.yaml')

To apply filters, use set/add_filter():

    meta.add_filter(salt2__phase__within = [-5, 5])

To access spectrum-level metadata (one entry per spectrum):

    name, exp, phase = meta.spectra('target.name', 'obs.exp', 'salt2.phase')

To access target-level metadata (one entry per target):

    name, phase, R = meta.targets('target.name', 'salt2.phase', 'SJB.Rx')

In the case of target-level metadata, a spectrum selector picks which spectrum
to use.  The default selector gets the spectrum nearest max.  e.g. in the
above example, it returns the target name, and the phase and flux ratio
for the spectrum nearest max for that target.  To modify this selector, see
documentation for:

    SnfMetaData.set_phase_selector()      # selection based upon phase
    SnfMetaData.set_spectrum_selector()   # more generic


WARNING:
   d.targets() returns an empty array if salt2.phase doesn't exist
   d.spectra() returns the list of spectra without applying the filters

"""

import sys
import os.path
import pickle
import re
import copy
import warnings
from math import isnan
import numpy
import yaml


NaN = float('nan')


def _makeiterable(obj):
    """Transform a single object into an list."""
    return (obj,) if not hasattr(obj, '__iter__') else obj


def load(*files):
    """
    Load metadata from files and return SnfMetaData object.
    
    This is a typing shortcut for SnfMetaData(files).
    """
    metadata = SnfMetaData(*files)
    return metadata


class SnfMetaData(dict):

    """
    Class to organize, merge, and manipulate target and spec metadata.

    You can traverse the data as a normal dictionary or use targets() and
    spectra() accessors to get metadata on a per-target or a per-spectrum
    level.  You can combine spectrum-level and target-level metadata.
    e.g.

    SnfMetaData.targets('target.name', 'obs.exp', 'salt2.phase')
    --> returns 3 arrays, with one entry each per target.  For each target,
        only one spectrum is selected (currently by phase closest to max)
        and the obs.exp and salt2.phase for that spectrum is selected.

    SnfMetaData.spectra('host.zhelio', 'obs.exp', 'salt2.phase')
    --> returns 3 arrays, with one entry each per spectrum.  The host.zhelio
        (which is target-level) is expanded to have one entry per spectrum.

    Usage example:

    data = SnfMetaData('targets.yaml', 'SiIIvelocities.yaml')
    data.set_filter(host__zhelio__lt = 0.1)
    mwebv, phase, vsi = data.spectra('target.mwebv', 'salt2.phase', 'RT.SiV')
    pylab.plot(phase, vsi)

    print "Spectra nearest max:"
    for name, exp, phase in zip(*data.targets('target.name', 'obs.exp', 'salt2.phase')):
        print name, exp, phase
    """

    # ----------------------------------------------------------------
    # Methods for creating and loading data into SnfMetaData object

    def __init__(self, *files, **kwargs):
        """
        Initialize metadata from a list of files, or a list of dictionaries.

        Optional keyword args:

        - phase_name : keyword to be used for phase selection (see
          'set_phase_selector')
        """
        # Selection filters
        self._filters = dict()      # { key:(filter,label) }
        # Keep track of # of input sources
        self._sources = {'all': 0}
        # Keep 'DATASET' as an hidden attribute, so that keys are
        # really all targets
        self._dataset = {}

        # allow for a different phase selector name
        self.set_phase_selector(
            phase_name=kwargs.get('phase_name', 'salt2.phase'))

        for data in files:
            self.merge(data)

    def __getitem__(self, item):
        """
        Ensure backward compatibility.

        For when 'DATASET' was stored in  main dictionnary along other targets.
        """
        if item == 'DATASET':
            warnings.warn("'DATASET' key is deprecated", DeprecationWarning)
            return self._dataset
        else:
            return dict.__getitem__(self, item)

    def __setitem__(self, item, value):
        """Ensure backward compatibility.

        For  when 'DATASET' was stored in main dictionnary along other targets.
        """
        if item == 'DATASET':
            warnings.warn("'DATASET' key is deprecated", DeprecationWarning)
            self._dataset = value
        else:
            return dict.__setitem__(self, item, value)

    def __delitem__(self, item):
        """Ensure backward compatibility.

        For when 'DATASET' was stored in main dictionnary along other targets.
        """
        if item == 'DATASET':
            warnings.warn("'DATASET' key is deprecated", DeprecationWarning)
            self._dataset = {}
        else:
            dict.__delitem__(self, item)

    def copy(self, key_list=None, sn_list=None):
        """
        Returns a copy of the SnfMetaData.

        To get a full copy:
        d_copy = d.copy()

        To only copy certain keywords:
        d_copy = d.copy("salt2.*")
        or
        d_copy = d.copy(".*.spec")
        or
        d_copy = d_copy(["salt2", "idr"])

        Note that since it does a regexp containment search, this last
        example will copy all the salt2.* and idr.* keywords

        A side effect is that if you want to copy only
        my_prefix.my_indicator and its error,
        d_copy = d.copy("my_prefix.my_indicator") will copy over
        my_prefix.my_indicator and my_prefix.my_indicator.err
        """
        if key_list is None:
            d_copy = copy.deepcopy(self)
        else:
            # To make sure that if the user only enters one key word,
            # the search will not be made over all the characters of
            # the keyword
            key_list = _makeiterable(key_list)

            d_copy = SnfMetaData()
            # This would be faster but we want to allow for regexp searches
            # target_key = set(d.metakeys(only_target=True))
            # target_keep_key = target_key.intersection(key_list)

            # loops over all the available keywords

            # Checks if it matches any of the regexps provided. If so
            # adds it to the list and jumps to the next keyword (it
            # can only be kept once, even if it matches more than one
            # condition)

            l_target_key_keep = []
            for target_key in self.metakeys(only_target=True):
                for key in key_list:
                    if re.search(key, target_key):
                        l_target_key_keep.append(target_key)
                        break

            l_spectrum_key_keep = []
            for spectrum_key in self.metakeys(only_spectrum=True):
                for key in key_list:
                    if re.search(key, spectrum_key):
                        l_spectrum_key_keep.append(spectrum_key)
                        break

            # Now that we know which keywords to keep, we loop over SNe:
            l_sn_keep = set(self.keys())
            if sn_list is not None:
                l_sn_keep = l_sn_keep.intersection(sn_list)

            for sn_name in l_sn_keep:
                self.set_filter(target__name=sn_name)
                for l_value in self.targets(*l_target_key_keep):
                    for value_name, value in zip(l_target_key_keep, l_value):
                        d_copy.add_to_target(sn_name,
                                             param=value_name, value=value)

                for l_value in zip(*self.spectra("obs.exp",
                                                 *l_spectrum_key_keep)):
                    for i, value_name in enumerate(l_spectrum_key_keep):
                        d_copy.add_to_spec(sn_name, l_value[0],
                                           param=value_name, value=l_value[i + 1])

        return d_copy

    def merge(self, data, conflict='last'):
        """
        Merge additional metadata files/dictionaries into current structure.

        input data :
            - dictionary structure to merge, or
            - yaml filename (.yaml or .yml), or
            - pickle filename (.pkl or .pickle)

        conflict = 'first', 'last', or 'panic' for conflicting entry resolution
            'first' : first entry wins
            'last'  : last (i.e. new) entry wins [default]
            'panic' or other : raise exception
        """
        # If data is a string, interpret as a filename
        if isinstance(data, basestring):
            f = open(data)
            if data.endswith('.yml') or data.endswith('.yaml'):
                data = yaml.load(f)
            elif data.endswith('.pkl') or data.endswith('.pickle'):
                data = pickle.load(f)
            else:
                raise IOError("File %s unknown extension" % data)
            f.close()

        self._sources['all'] += 1
        # Merge target level items
        for name, tgtinfo in data.iteritems():
            # the DATASET level will overrun any previous instance of itself
            if name == 'DATASET':
                self._dataset = tgtinfo.copy()
                continue

            if name in self._sources:
                self._sources[name] += 1
            else:
                self._sources[name] = 1

            if name in self:
                for key, value in tgtinfo.iteritems():
                    if key == 'spectra':
                        continue
                    elif key not in self[name] or conflict == 'last':
                        self[name][key] = value
                    else:
                        if conflict == 'first':
                            print >> sys.stderr, \
                                "Skipping key %s already in target %s" % \
                                (key, name)
                        else:
                            raise ValueError(
                                '%s %s already in metadata' % (name, key))
            else:
                # previous IDRs do not have idr.saltprefix (this will
                # eventually be deprecated)
                if 'idr.saltprefix' not in tgtinfo:
                    # search for a meaningful value
                    for key in tgtinfo:
                        if 'DayMax' in key and len(key.split('.')) > 1:
                            tgtinfo['idr.saltprefix'] = key.split('.')[0]
                            break
                    # default to salt2 if everything else failed
                    tgtinfo.setdefault('idr.saltprefix', 'salt2')
                self[name] = tgtinfo.copy()
                continue

            # Merge spectrum level items
            if 'spectra' not in tgtinfo:
                continue
            if 'spectra' not in self[name]:
                self[name]['spectra'] = dict()
            for exp, specinfo in tgtinfo['spectra'].iteritems():
                if exp in self[name]['spectra']:
                    for key, value in specinfo.iteritems():
                        if (key not in self[name]['spectra'][exp] or
                                conflict == 'last'):
                            self[name]['spectra'][exp][key] = value
                        else:
                            if conflict == 'first':
                                print "Skipping key %s already in %s/%s" % \
                                      (key, name, exp)
                            else:
                                raise ValueError(
                                    '%s %s %s already in metadata' %
                                    (name, exp, key))
                else:
                    self[name]['spectra'][exp] = specinfo.copy()

        # Update phases, target.name, and obs.exp for any new entries
        self._expand()

    def add_to_target(self, target,
                      param=None, value=None, items=None, autoerr=True):
        """
        Add target-level value(s) to the MetaData, creating target
        entry if needed

        target : target name or list of target names

        either:
            param : parameter name or list of parameter names
            value : parameter values or list of parameter values
        or:
            items : dictionary of parameter:value pairs

        if any values are 2-length lists or tuples, they will be interpreted
        as (value, error) pairs resulting in param and param.err entries.
        Set autoerr=False to override this auto-error interpretation.
        """
        target = _makeiterable(target)

        for name in target:
            if name not in self:
                self[str(name)] = {'spectra': dict(), 'target.name': str(name)}

            # Convert single elements into lists first
            param = _makeiterable(param)
            value = _makeiterable(value)

            # Loop over parameter, value pairs
            if param is not None and value is not None:
                for p, v in zip(param, value):
                    self._add_value_err(self[name], p, v, autoerr)

            # Add any pre-built dictionary items
            if items is not None:
                for p, v in items.iteritems():
                    self._add_value_err(self[name], p, v, autoerr)

    def add_to_spec(self, target, exp,
                    param=None, value=None, items=None, autoerr=True):
        """
        Add spectrum-level value(s) to the MetaData, creating target and
        spectrum entries if needed.

        target : target name or list of target names
        exp    : exposure code YY_DAY_RUN_EXP or list of exp codes

        either:
            param : parameter name or list of parameter names
            value : parameter values or list of parameter values
        or:
            items : dictionary of parameter:value pairs

        if any values are 2-length lists or tuples, they will be interpreted
        as (value, error) pairs resulting in param and param.err entries
        Set autoerr=False to override this auto-error interpretation.
        """
        target = _makeiterable(target)
        exp = _makeiterable(exp)

        for name, ex in zip(target, exp):
            if name not in self:
                self[name] = {'spectra': dict(), 'target.name': name}
            if 'spectra' not in self[name]:
                self[name]['spectra'] = dict()
            if ex not in self[name]['spectra']:
                self[name]['spectra'][ex] = {
                    'target.name': name, 'obs.exp': ex}

            info = self[name]['spectra'][ex]

            # Loop over parameter, value pairs
            if param is not None and value is not None:
                # Convert single elements into lists first
                param = _makeiterable(param)
                value = _makeiterable(value)
                for p, v in zip(param, value):
                    self._add_value_err(info, p, v, autoerr)

            # Add any pre-built dictionary items
            if items is not None:
                for p, v in items.iteritems():
                    self._add_value_err(info, p, v, autoerr)

    # ----------------------------------------------------------------
    # Methods for accessing the metadata

    def targets(self, *args, **kwargs):
        """
        If no args, return list of target names matching current filters.
        With args, look for keys in targets and spectra, returning one
        value per target.  For spectra level keys, only one spectrum is
        selected per target, using the spec_selector.

        Optional keyword args:

        - sort_by = keyword : returned arrays are sorted by this value
        - squeeze = True: unpack single-item list of arrays
        - other kwargs: propagated to _get_values(), in particular:

          - noNaN    = True : trim any entry where any element is NaN
          - noSpecOK = True : allow entries from targets which have 0 spectra
          - structured=True : return a structured array rather than
                              a list of arrays
        """
        sort_by = kwargs.pop("sort_by", None)
        squeeze = kwargs.pop("squeeze", True)

        if not args:  # Return list of target names by default = meta.targets()
            return_list = self._get_values('target', 'target.name', **kwargs)
        else:
            return_list = self._get_values('target', *args, **kwargs)

        if sort_by in self.metakeys():
            # sort_list is the array we want to sort by.  Use a stable
            # merge, especially useful for spectra.
            sort_list = self._get_values('target', sort_by, **kwargs)
            # return_list is a struct. arr.
            if kwargs.get("structured", False):
                i_sorted = numpy.argsort(sort_list[sort_by], kind="mergesort")
                return_list = return_list[i_sorted]
            else:                       # return_list is a list of arrays
                i_sorted = numpy.argsort(sort_list[0], kind="mergesort")
                return_list = [l[i_sorted] for l in return_list]
        elif sort_by is not None:
            raise ValueError("Cannot sort by '%s', unknown key" % sort_by)

        if squeeze and len(return_list) == 1:
            return_list = return_list[0]

        return return_list

    def spectra(self, *args, **kwargs):
        """
        If no args, return list of spectra (no filter applied).
        With args, look for keys in targets and spectra, returning one
        value per spectrum (duplicating target level items as necessary)

        Optional keyword args:

        - sort_by = keyword : returned arrays are sorted by this value
        - squeeze = True: unpack single-item list of arrays
        - other kwargs: propagated to _get_values(), in particular:

          - noNaN    = True : trim any entry where any element is NaN
          - noSpecOK = True : allow entries from targets which have 0 spectra
          - structured=True : return a structured array rather than
                              a list of arrays
        """
        sort_by = kwargs.pop("sort_by", None)
        squeeze = kwargs.pop("squeeze", True)  # Backward compatibility

        if not args:  # Return *all* spectra-level dicts (no filter applied)
            return_list = [specinfo
                           for tgtinfo in self.itervalues()
                           for specinfo in tgtinfo['spectra'].itervalues()]
        else:
            return_list = self._get_values('spectrum', *args, **kwargs)

        if sort_by in self.metakeys():
            # sort_list is the array we want to sort by.  Use a stable
            # merge, especially useful for spectra.
            sort_list = self._get_values('spectrum', sort_by, **kwargs)
            # return_list is a struct. arr.
            if kwargs.get("structured", False):
                i_sorted = numpy.argsort(sort_list[sort_by], kind="mergesort")
                return_list = return_list[i_sorted]
            else:                       # return_list is a list of arrays
                i_sorted = numpy.argsort(sort_list[0], kind="mergesort")
                return_list = [l[i_sorted] for l in return_list]
        elif sort_by is not None:
            raise ValueError("Cannot sort by '%s', unknown key" % sort_by)

        if squeeze and len(return_list) == 1:
            return_list = return_list[0]

        return return_list

    def dataset(self, *args):
        """
        If no args, return list of dataset-level keywords. With args return
        the corresponding dataset-level value.
        """
        if not args:
            return sorted(self._dataset.keys())
        else:
            return [self._dataset[arg] for arg in args]

    # ----------------------------------------------------------------
    # Output methods

    def dump(self):
        """
        Return yaml dump of current data.

        Filters are *not* applied.
        """

        return yaml.dump(dict(self), default_flow_style=False)

    def tabulate(self, *args, **kwargs):
        """
        Return a string with a table of values with filters applied.

        Inputs:

        - args = metadata columns to include in the table
        - keyword options:

          - colwidth = column width
          - only_target = True : target-level instead of spectrum
            level (useful for cases where you don't have any spectrum)
          - other kwargs are propagated to selector (targets or spectra)

        Output: string with a table of columns.
        """
        colwidth = kwargs.pop('colwidth', 16)
        only_target = kwargs.pop('only_target', False)

        kwargs.setdefault('squeeze', False)
        if only_target:
            data = self.targets(*args, **kwargs)
        else:
            data = self.spectra(*args, **kwargs)

        rows = list()
        header = '# ' + str(args[0]).ljust(colwidth - 3) + ' '
        header += ' '.join([str(x).ljust(colwidth) for x in args[1:]])
        rows.append(header)

        # Add rows built column by column
        rows.extend([' '.join(str(row[icol]).ljust(colwidth) for row in data)
                     for icol in range(len(data[0]))])

        return '\n'.join(rows)

    def tabulate_rst(self, *args, **kwargs):
        """
        reStructuredText table of values with filters applied
        (requires ToolBox.ReST.rst_table).

        Inputs:

        - args = metadata columns to include in the table
        - kwargs (optional):

          - only_target = True : target-level instead of spectrum level
          - other kwargs are propagated to selector (targets or spectra)

        Output: table-formatted string
        """
        from ToolBox.ReST import arr2rst

        only_target = kwargs.pop('only_target', False)

        kwargs.setdefault('squeeze', False)
        kwargs.setdefault('structured', True)  # Return a structured array
        if only_target:
            data = self.targets(*args, **kwargs)
        else:
            data = self.spectra(*args, **kwargs)

        return arr2rst(data)

    def metakeys(self, grep=None, only_target=False, only_spectrum=False):
        """
        Return a list all metadata keys in this dictionary

        if grep is set to a string, will only return the keys
        containing the string.

        Note that grep uses re.search() and thus understand regexps
        For example, in order to get all the salt2 keywords:
        d.metakeys("salt2.*")

        Two options to only keep target level or spectrum level
        keywords are provided:

        - d.metakeys(only_target=True)
        - d.metakeys(only_spectrum=True)
        """

        keys = set()
        for tgtinfo in self.itervalues():
            for key in tgtinfo:
                # Add spectrum-level keys
                if key == 'spectra' and not only_target:
                    keys.update(key
                                for specinfo in tgtinfo['spectra'].itervalues()
                                for key in specinfo)
                # Add target-level keys
                elif not only_spectrum:
                    keys.add(key)

        if not only_spectrum:
            # Target-level special key
            keys.add('nspec')

        keys = filter(lambda key: key is not None, keys)
        # Regexp selection on keys
        if grep:
            keys = set(key for key in keys if re.search(grep, key))

        return sorted(keys)

    # ----------------------------------------------------------------
    # Filter and phase/spectrum selection methods

    def set_spectrum_selector(self, function, **kwargs):
        """
        Set the spectrum selector function to use when picking one
        spectrum per target (e.g. spectrum with phase nearest max).

        The function should take a dictionary of spectra from a single
        target, and return a new dictionary with only the spectrum/spectra
        of interest.

        If you only want to cut on quantities, use filters instead
        (see set/add_filter).  Spectrum selectors are intended for
        more generic selections which require comparisons between
        multiple spectra, e.g., the default spectrum selector returns
        the spectrum closest to salt2.phase = 0.
        """
        self._spec_selector = function
        self._spec_selector_kwargs = kwargs

    def set_phase_selector(self, phase_nearest=0.0, phase_name='salt2.phase'):
        """
        Set the spectrum selector to pick only the spectrum nearest in
        phase to "phase_nearest".

        If you also want to require a phase range, use

            meta.add_filter(salt2__phase__within = [phase_min, phase_max])

        set_spectrum_selector() provides even more generic functionality
        for picking a subset of spectra from a set of spectra.
        """

        self.set_spectrum_selector(
            phase_selector,
            phase_nearest=phase_nearest, phase_name=phase_name)

    def set_filter(self, **kwargs):
        """
        Reset any previous filters, then create new filter set.

        See add_filter(...) for full syntax options
        """
        self._filters = dict()
        self.add_filter(**kwargs)

    def add_filter(self, **kwargs):
        """
        Add/replace filters to current filter set, e.g.

            meta.add_filter(idr__subset = 'training',
                            host__zhelio__lt = 0.1,
                            salt2__phase__within = [-5,+5] )

        Will add filters to require:

        * idr.subset == 'training'
        * host.zhelio < 0.1
        * -5 <= salt2.phase < +5

        Only entries matching those cuts will be included when looping
        over the metadata with meta.targets() and meta.spectra() .

        '__' in kwargs gets expanded into '.' or used to separate
        special endings such as __lt, __within, etc.

        Use set_filter(...) to replace all filters with a new one.

        Examples:

        Use a generic function as a filter:

            meta.add_filter(salt2__X1 = lambda x: x<-1 or x>1)

        Value is in a discreet set:

            meta.add_filter(idr__subset__in = ['training', 'validation'])

        Value is within a range:

            meta.add_filter(salt2__phase__within = [-5, 5])

        Regular expression match:

            meta.add_filter(target__name__re = '^SNF')

        Numerical operations (Fortran/Django-like):
        lt, le, lte, gt, ge, gte, eq, ne

            meta.add_filter(salt2__phase__lt = 0)

        Clear a filter on a specific key:

            meta.add_filter(salt2__phase = None)

        Note that each key can only have one filter.  e.g.

            # Bad: second cut overrides the first one
            meta.add_filter(host__zhelio__gt=0.015)
            meta.add_filter(host__zhelio__lt=0.1)

            # Use this instead
            meta.add_filter(host__helio__within = [0.015, 0.1])

        Available filters:

        * x__eq = val: x == val (default)
        * x__ne = val: x != val
        * x__lt = val: x < val
        * x__le = val: x <= val (syn: lte)
        * x__gt = val: x > val
        * x__ge = val: x >= val (syn: gte)
        * x__inside  = (v1,v2): v1 <= x < v2 (syn: within)
        * x__outside = (v1,v2): not (v1 <= x < v2)
        * x__in = list: x in list
        * x__exclude = list: x not in list
        * x__re = regexp: search(regexp, x)
        """
        # Implementation note:
        # We can only add one filter at a time, because the function
        # caching syntax like
        #   filters[key] = lambda x: x<value
        # only works as intended if value never changes again.
        # So if multiple filters are provided, loop over them and make
        # a recursive call so that only one is saved per call.

        if len(kwargs) > 1:
            for key, value in kwargs.iteritems():
                self.add_filter(**{key: value})
        else:
            for key, value in kwargs.iteritems():
                key = key.replace('__', '.')

                # A None value means clear that filter
                if value is None and key in self._filters:
                    del self._filters[key]
                    continue
                # If value is a function, use it
                elif hasattr(value, '__call__'):
                    keyword = key
                    xfilter = value
                    label = value.__name__ \
                        if hasattr(value, '__name__') \
                        else 'Undefined selection'
                # Check for Fortran/Django-like suffixes and make
                # lambda filters
                elif key.endswith('.eq'):
                    keyword = key[:-3]
                    xfilter = lambda x: x == value
                    label = '%s == %s' % (keyword, value)
                elif key.endswith('.ne'):
                    keyword = key[:-3]
                    xfilter = lambda x: x != value
                    label = '%s != %s' % (keyword, value)
                elif key.endswith('.lt'):
                    keyword = key[:-3]
                    xfilter = lambda x: x < value
                    label = '%s < %s' % (keyword, value)
                elif key.endswith('.le'):
                    keyword = key[:-3]
                    xfilter = lambda x: x <= value
                    label = '%s <= %s' % (keyword, value)
                elif key.endswith('.lte'):
                    keyword = key[:-4]
                    xfilter = lambda x: x <= value
                    label = '%s <= %s' % (keyword, value)
                elif key.endswith('.gt'):
                    keyword = key[:-3]
                    xfilter = lambda x: x > value
                    label = '%s > %s' % (keyword, value)
                elif key.endswith('.ge'):
                    keyword = key[:-3]
                    xfilter = lambda x: x >= value
                    label = '%s >= %s' % (keyword, value)
                elif key.endswith('.gte'):
                    keyword = key[:-4]
                    xfilter = lambda x: x >= value
                    label = '%s >= %s' % (keyword, value)
                # Other filter types
                elif key.endswith('.within'):  # tuple of (min, max) range
                    keyword = key[:-7]
                    xfilter = lambda x: (value[0] <= x < value[1])
                    label = '%s <= %s < %s' % (value[0], keyword, value[1])
                elif key.endswith('.inside'):  # tuple of (min, max) range
                    keyword = key[:-7]
                    xfilter = lambda x: (value[0] <= x < value[1])
                    label = '%s <= %s < %s' % (value[0], keyword, value[1])
                elif key.endswith('.outside'):  # tuple of (min, max) range
                    keyword = key[:-8]
                    xfilter = lambda x: not value[0] <= x < value[1]
                    label = 'not (%s <= %s < %s)' % (
                        value[0], keyword, value[1])
                elif key.endswith('.in'):     # discreet list
                    keyword = key[:-3]
                    xfilter = lambda x: x in value
                    label = '%s in %s' % (keyword, value)
                elif key.endswith('.exclude'):    # discreet list
                    keyword = key[:-8]
                    xfilter = lambda x: x not in value
                    label = '%s not in %s' % (keyword, value)
                elif key.endswith('.re'):     # regular expression match
                    keyword = key[:-3]
                    xfilter = lambda x: bool(re.search(value, x))
                    label = 'search(%s, %s)' % (value, keyword)
                else:                         # default is equality
                    keyword = key
                    xfilter = lambda x: x == value
                    label = '%s == %s' % (keyword, value)

                if keyword in self._filters:
                    print "WARNING: overriding previous filter " \
                          "'%s' with '%s'" % \
                          (self._filters[keyword][1], label)
                # Keep track of filters
                self._filters[keyword] = (xfilter, label)

    def show_filter(self):

        if self._filters:
            print "%d filter(s):" % len(self._filters)
            for key, (xfilter, label) in self._filters.iteritems():
                print "   Filter on %s: %s" % (key, label)
        else:
            print "No filter"

    def apply_cuts(self, cut_zero_spec=True, verbose=False):
        """
        Permanently remove entries which don't pass current filters.

        cut_zero_spec : remove any targets which don't have any spectra
                        which pass the cuts
        """
        # Loop over targets making list of bad ones.
        # Wait until end to remove them to not mess up the iteration
        bad_targets = set()                # Set of targets to be discarded
        for name, tgtinfo in self.iteritems():  # Loop over targets
            for key, (xfilter, label) in self._filters.iteritems():
                if key in tgtinfo and not xfilter(tgtinfo[key]):
                    bad_targets.add(name)
                    break

            # If target is bad, no need to check spectra
            if name in bad_targets:
                continue

            # Find which spectra don't pass the cuts
            bad_spectra = set(
                expid
                for expid, specinfo in tgtinfo['spectra'].iteritems()
                for key, (xfilter, label) in self._filters.iteritems()
                if key in specinfo and not xfilter(specinfo[key])
            )

            for expid in bad_spectra:   # Remove bad spectra from current target
                if verbose:
                    print "Removing spectrum", expid, "from target", name
                del tgtinfo['spectra'][expid]

            # If no spectra left, remove the target
            if cut_zero_spec and not tgtinfo['spectra']:
                bad_targets.add(name)

        for name in bad_targets:        # Remove bad targets
            if verbose:
                print "Removing target", name
            del self[name]

    # ----------------------------------------------------------------
    # Internal SnfMetaData utility functions

    def _add_value_err(self, data, param, value, autoerr):
        """Parse if value is (value, err) pair and add to dictionary data"""
        if autoerr and hasattr(value, '__iter__') and len(value) == 2:
            data[param] = value[0]
            data[param + '.err'] = value[1]
        else:
            data[param] = value

    def _expand(self, datemax='salt2.DayMax',
                dateobs='obs.mjd', phasename='salt2.phase'):
        """Auto-expand salt2.phase, target.name, obs.exp"""
        # phase
        for tgtinfo in self.itervalues():
            if 'host.zhelio' in tgtinfo:
                z = tgtinfo['host.zhelio']
            elif 'host.redshift' in tgtinfo:
                z = tgtinfo['host.redshift']
            elif 'redshift' in tgtinfo:
                z = tgtinfo['redshift']
            else:
                z = 0.0
            # adapt datemax to 'idr.saltprefix'
            if 'idr.saltprefix' in tgtinfo:
                datemax = tgtinfo['idr.saltprefix'] + '.DayMax'
            if datemax in tgtinfo:
                tgtdatemax = tgtinfo[datemax]
                # adapt phasename to 'idr.saltprefix'
                if 'idr.saltprefix' in tgtinfo:
                    phasename = tgtinfo['idr.saltprefix'] + '.phase'
                for specinfo in tgtinfo['spectra'].itervalues():
                    if phasename not in specinfo and dateobs in specinfo:
                        dobs = specinfo[dateobs]
                        specinfo[phasename] = (dobs - tgtdatemax) / (1 + z)

        # target.name, obs.exp
        for name, target in self.iteritems():
            target['target.name'] = name
            if 'spectra' not in target:
                target['spectra'] = dict()
                continue
            for exp, spec in target['spectra'].iteritems():
                spec['target.name'] = name
                spec['obs.exp'] = exp

    def _good_spectra(self, tgtinfo):
        """
        Given the spectral entries from a single target,
        return those which pass the current filters.
        """

        # At this point, potentially all spectra are good
        goodspecs = set(tgtinfo['spectra'].keys())
        nspecs = len(goodspecs)

        for key, (xfilter, label) in self._filters.iteritems():
            # First check target level keywords pass filters. If one
            # of the key is target level but wrong, no spectrum can be
            # good. OTOH, if a target level keyword pass filters, no
            # need to test all spectra.
            if key in tgtinfo:
                if not xfilter(tgtinfo[key]):  # No more hope for this target
                    return dict()
                else:
                    continue                  # Move on to next filter

            # At this point, we have keys that were not at the target
            # level and must thus be tested for at the spectrum level.
            # We assume that everything is good and will remove what
            # is not.
            for expid, specinfo in tgtinfo['spectra'].iteritems():
                if expid not in goodspecs:  # the spectrum was already discarded
                    continue
                if key in specinfo:  # we found the key in this spectrum
                    if not xfilter(specinfo[key]):
                        # but the spectrum doesn't pass the filter: we pop it
                        goodspecs.remove(expid)
                    # else: the spectrum had the key and passed the
                    # test: we keep it
                else:
                    # The key was not a target level key and is not in
                    # this spectrum: the spectrum is bad
                    goodspecs.remove(expid)

        # goodspecs initially had all the spectra, and now contains
        # the spectra that passed the all the filters
        if len(goodspecs) == nspecs:      # No selection applied
            return tgtinfo['spectra']   # Return all initial spectra
        else:                           # Return selected spectra
            return dict((spec, tgtinfo['spectra'][spec]) for spec in goodspecs)

    def _get_values(self, level, *args, **kwargs):
        """
        Return values for requested metadata items (args) at either
        the per-target (level='target') or per-spectrum
        (level='spectrum') level.  Set noNaN=True to exclude NaN
        entries.  Used by SnfMetaData.targets() and SnfMetaData.spectra().

        Special arg: 'nspec' to return number of filtered spectra.

        Optional keyword args:

        - noNaN    = True : trim any entry where any element is NaN
        - noSpecOK = True : allow entries from targets which have 0 spectra
        - structured=True : return a structured array rather than
                            a list of arrays (for level='target' only)
        - other kwargs are considered as *temporary* filters
        """

        noNaN = kwargs.pop('noNaN', False)
        noSpecOK = kwargs.pop('noSpecOK', False)
        structured = kwargs.pop('structured', False)

        if kwargs:                      # Apply addition *temporary* filters
            oldFilters = self._filters.copy()
            self.add_filter(**kwargs)

        results = dict()                # { arg:[values] }
        # Cache the spectra corresponding to current filters
        # (including temporary filters)
        cache = dict()                  # { name:[spectra] }

        for arg in args:
            results[arg] = list()       # Initialize list of values

            for name, tgtinfo in self.iteritems():
                if arg == 'nspec':
                    # Number of *filtered* spectra, i.e. *before*
                    # spectrum selection (e.g. spec closest to max).
                    # Cannot be cached, as it does not correspond to
                    # the *selected* spectra.
                    tgtinfo['nspec'] = len(self._good_spectra(tgtinfo))

                if name not in cache:  # Look for spectra and cache result
                    cache[name] = self._good_spectra(tgtinfo)  # Filter spectra
                    if level == 'target':                     # Select spectra
                        cache[name] = self._spec_selector(
                            cache[name], **self._spec_selector_kwargs)
                spectra = cache[name]  # List of filtered & selected spectra

                if not spectra:         # No selected spectra
                    if noSpecOK:        # ...yet store results
                        print "WARNING: No spectrum found for target", name
                        if arg in tgtinfo:  # Target-level keyword
                            results[arg].append(tgtinfo[arg])
                        else:  # Could be spectrum-level keyword, but no spec!
                            results[arg].append(NaN)
                elif level == 'target':   # One entry per target
                    if arg in tgtinfo:
                        results[arg].append(tgtinfo[arg])
                    else:  # There might be more than one selected spectrum
                        vals = [spectra[expid].get(arg, NaN)
                                for expid in sorted(spectra)]
                        if len(vals) == 1:
                            vals = vals[0]
                        results[arg].append(vals)
                else:                   # One entry per spectrum
                    if arg in tgtinfo:  # One target key per spectrum
                        results[arg].extend([tgtinfo[arg]] * len(spectra))
                    else:
                        results[arg].extend(
                            [spectra[expid].get(arg, NaN)
                             for expid in sorted(spectra)])

        if kwargs:                      # Revert to initial filter
            self._filters = oldFilters

        if noNaN:                       # Remove NaN entries if needed
            # Look for indices of targets containing at least a NaN value
            iNaN = set(i
                       for values in results.itervalues()
                       for i, value in enumerate(values)
                       if isnan(value))  # NaN detection

            for i in sorted(iNaN, reverse=True):  # Remove tagged targets
                for arg in results:
                    del results[arg][i]

        # Convert results into list of numpy arrays
        final_results = [numpy.array(results[arg]) for arg in args]

        if structured:  # Convert a list of arrays into a structured array
            # Construct structured dtype [ (arg, dtype, shape) ]. Note
            # that on target level, some entries may have multiple
            # values (say 2), hence the need for array elements of
            # shape arr.shape[1:]=(2,). This assumes that all arrays
            # of final_results have the same length, which is possible
            # only at target-level.
            dtypes = [(arg, arr.dtype, arr.shape[1:])
                      for arg, arr in zip(args, final_results)]
            final_results = numpy.array(zip(*final_results), dtype=dtypes)

        return final_results

# end of SnfMetaData class

# -------------------------------------------------------------------------

# Spectrum selectors take a dictionary of spectra and a set of keyword args
# and returns a new dictionary with the selected spectra.
# SnfMetaData will cache the function and the keywords to use when calling it.

# Design note:
# Originally we implemented this as a function which returned a function.
# The returned function was to be used without keyword args.  This prevented
# SnfMetaData objects from being pickled because the custom function didn't
# live in any namespace that could be imported upon loading.  This new
# implementation gives the same functionality while supporting pickling.
# Stephen, 2008-01-26

def phase_selector(spectra, phase_nearest=0.0, phase_name='salt2.phase'):
    """
    Selects spectra based upon phase nearest to "phase_nearest"
    """

    if not spectra:
        return dict()

    phases = []                         # [(|delta_phase|, expid)]
    for expid, specinfo in spectra.iteritems():
        # Check if specinfo[phase_name] is missing, None, or NaN
        phase = specinfo.get(phase_name, None)
        if phase is None or isnan(phase):
            continue
        else:
            phases.append((abs(specinfo[phase_name] - phase_nearest), expid))

    if not phases:       # No matching phase
        return dict()
    else:                # Look for closest phase, i.e. smallest delta
        best_expid = min(phases)[1]
        return {best_expid: spectra[best_expid]}

# -------------------------------------------------------------------------
# Internal Data Release function(s)


def load_idr(metafile, salt=False, subset=('training', 'validation'),
             targets=None, saltprefix='salt2'):
    """
    Load metadata from an Internal Data Release (IDR) and auto-expand
    filenames based upon IDR subset, target name, and prefix .
    Returns an SnfMetaData objects.

    If salt=True, also load salt2 fit metadata

    subset option:
        string X : only load targets with idr.subset == X
        list   X : only load targets with idr.subset in X
        None     : load all targets

    targets: optional list of targets to keep
    """

    if os.path.isdir(metafile):
        if os.path.exists(metafile + '/META.pkl'):
            metafile += '/META.pkl'
        else:
            metafile += '/META.yaml'

    meta = SnfMetaData(metafile)
    meta.idrpath = os.path.dirname(metafile)
    meta.idrname = os.path.basename(meta.idrpath)

    # Auto-expand IDR paths
    for name, tgtinfo in meta.iteritems():
        subdir = tgtinfo['idr.subset']
        name = tgtinfo['target.name']
        for expid, specinfo in tgtinfo['spectra'].iteritems():
            fileprefix = specinfo['idr.prefix']
            if fileprefix is not None:
                prefix = '%s/%s/%s' % (subdir, name, fileprefix)
                specinfo['idr.spec_B'] = '%s_B.fits' % prefix
                specinfo['idr.spec_R'] = '%s_R.fits' % prefix
                specinfo['idr.spec_merged'] = '%s_merged.fits' % prefix
                specinfo['idr.spec_restframe'] = '%s_restframe.fits' % prefix
                specinfo['idr.spec_logbin'] = '%s_logbin.fits' % prefix

                # Confirm files actually exist
                for key in ('idr.spec_B', 'idr.spec_R', 'idr.spec_merged',
                            'idr.spec_restframe', 'idr.spec_logbin'):
                    filename = os.path.join(meta.idrpath, specinfo[key])
                    if not os.path.exists(filename):
                        # print 'Spec file missing', filename
                        del specinfo[key]

    # Load salt2 info if requested and needed
    if salt and not metafile.endswith('.pkl'):
        for name, tgtinfo in meta.iteritems():
            name = tgtinfo['target.name']
            subdir = tgtinfo['idr.subset']
            saltfile = '%s/%s/%s_%s.yaml' % (subdir, name, name, saltprefix)
            saltfile = os.path.join(meta.idrpath, saltfile)
            if os.path.exists(saltfile):
                meta.merge(saltfile)
            else:
                # print 'SALT2 file missing', saltfile
                pass

    if targets:
        try:
            if os.path.exists(targets):   # Read list of targets from file
                targets = numpy.loadtxt(targets, dtype='string')
        except (TypeError, IOError,):     # 'targets' is a (list of) target(s)
            targets = _makeiterable(targets)
        remove_targets = set(meta.keys()) - \
            set(targets)  # Discard other targets
    else:
        remove_targets = set()

    if subset:                         # Filter by subset if requested
        subset = _makeiterable(subset)
        for name, tgtinfo in meta.iteritems():
            if tgtinfo['idr.subset'] not in subset:
                # print '%s subset %s not in %s' % \
                # (name, tgtinfo['idr.subset'], str(subset))
                remove_targets.add(name)

    for name in remove_targets:         # Remove targets as requested
        del meta[name]

    if not meta:
        print "WARNING: successive selections result in an empty IDR"

    return meta

if __name__ == '__main__':

    md = SnfMetaData('testdata.yaml')

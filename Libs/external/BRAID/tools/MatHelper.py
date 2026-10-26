"Helps with interfacing with matlab"

import scipy.io as sio
import numpy as np
from pathlib import Path
import builtins
builtin_types = tuple(getattr(builtins, t) for t in dir(builtins) if isinstance(getattr(builtins, t), type) and getattr(builtins, t) is not object and getattr(builtins, t) is not type(None))

def make_sure_parent_dir_exists(file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

def loadmat(file_path, variable_names=None):
    "Loads a mat file as a dictionary"
    try:
        # mat_dict = sio.loadmat(file_path, matlab_compatible=True, variable_names=variable_names)
        mat_dict = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True, chars_as_strings=True, variable_names=variable_names)
    except NotImplementedError: # Runs for v7.3: 'Please use HDF reader for matlab v7.3 files'
        # hd = h5py.File(file_path) # Not nice, let's use mat73
        import mat73
        mat_dict = mat73.loadmat(file_path, use_attrdict=True)

    return _check_keys(mat_dict)

def savemat(file_path, data_dict, NoneReplacement = np.array([]), **args):
    "Saves a dictionary as mat file"
    # Check out dir
    make_sure_parent_dir_exists(file_path)
    data_dict = replaceNone(data_dict, NoneReplacement) # None cannot be saved!
    sio.savemat(file_path, data_dict, **args)
    return

def replaceNone(d, replacement):
    if isinstance(d, dict):
        for key in d:
            d[key] = replaceNone(d[key], replacement)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            d[i] = replaceNone(d[i], replacement)
    elif isinstance(d, np.ndarray):
        if len(d) > 0 and not np.issubdtype(d.dtype, np.number):
            for i in range(d.size):
                d.itemset( i, replaceNone( d.item(i), replacement ) )
    elif d is not None and isinstance(d, object) and not isinstance(d, builtin_types): # For custom classes, make sure they don't have None attributes
        for field in dir(d): 
            if not field.startswith('__') and not 'tensorflow' in str(type(d)):
                # try:
                    d.__setattr__( field, replaceNone( d.__getattribute__(field), replacement ) )
                # except:
                #     pass
    else:
        if d is None:
            d = replacement
    return d



# From https://stackoverflow.com/a/8832212/2275605
def _check_keys(d):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in d:
        if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
        elif isinstance(d[key], np.ndarray) and len(d[key]) > 0 and isinstance(d[key].item(0), sio.matlab.mio5_params.mat_struct):
            for i in range( d[key].size ):
                if isinstance(d[key].item(i), sio.matlab.mio5_params.mat_struct):
                    d[key].itemset( i, _todict( d[key].item(i) ) )
            # with np.nditer(d[key], flags=["refs_ok"], op_flags = ['readwrite']) as it:
            #     for x in it:
            #         # x[...] = _check_keys(x)
            #         x[...] = x
        else:
            pass

    return d

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    d = {}
    for key in matobj._fieldnames:
        elem = matobj.__dict__[key]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            d[key] = _todict(elem)
        elif isinstance(elem, np.ndarray) and elem.size > 0 and isinstance(elem.item(0), sio.matlab.mio5_params.mat_struct):
            for i in range( elem.size ):
                if isinstance(elem.item(i), sio.matlab.mio5_params.mat_struct):
                    elem.itemset(i, _todict( elem.item(i) ) )
            d[key] = elem
        else:
            d[key] = elem
    return d

def load_cd_dataset(file_path):
    "Loads a file with allCVRes var as produced by mainSimScript.m"
    return loadmat(file_path)

import numpy as np
from numpy.typing import ArrayLike, NDArray
import time
from functools import wraps
from .data import FileVar
import logging
module_logger = logging.getLogger(__name__)

def log_path(file_path: FileVar) -> str:
    """Compute the path of the log file

    Parameters
    ----------
    file_path : FileVar
        path of the current file

    Returns
    -------
    log_path : str
        log path
    """
    log_name = ''.join(file_path.FILE.split('.')[:-1]+['.log'])
    return file_path.PATH.__add__(log_name).PATH

def reorganize_index(idxes: tuple | NDArray, axis: int | None, shape: tuple) -> tuple:
    """Convert a 1D positions in a nD positions

    Parameters
    ----------
    idxes : tuple | NDArray
        1D indexes
    axis : int | None
        chosen axis
    shape : tuple
        the shape of the selected array

    Returns
    -------
    new_idxes : tuple[int]
        computed indexes
    """
    if axis is None:
        return np.unravel_index(idxes,shape)
    else:
        axes = [ np.arange(s) for s in shape]
        axes[axis] = idxes
        return tuple(axes) 

def find_argmax(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    """Compute the position of the max in a ndimensional array

    Parameters
    ----------
    obj : ArrayLike
        selected array
    axis : ArrayLike | None, optional
        chosen axis along which compute the maximum, by default None

    Returns
    -------
    argmax : ArrayLike
        maxima positions
    """
    obj = np.asarray(obj)
    maxpos = np.argmax(obj, axis=axis)
    if len(obj.shape) == 1:
        return maxpos
    else:
        return reorganize_index(maxpos,axis=axis,shape=obj.shape)
        

def find_max(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    """Compute the maxima along axis

    Parameters
    ----------
    obj : ArrayLike
        selected array
    axis : ArrayLike | None, optional
        chosen axis, by default None

    Returns
    -------
    max : ArrayLike
        maxima along `axis`
    """
    obj = np.asarray(obj)
    return obj[find_argmax(obj,axis=axis)]

def find_argmin(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    """Compute the position of the nin in a ndimensional array

    Parameters
    ----------
    obj : ArrayLike
        selected array
    axis : ArrayLike | None, optional
        chosen axis along which compute the minimum, by default None

    Returns
    -------
    argmin : ArrayLike
        minima positions
    """
    obj = np.asarray(obj)
    minpos = np.argmin(obj,axis=axis)
    if len(obj.shape) == 1:
        return minpos
    else:
        return reorganize_index(minpos,axis=axis,shape=obj.shape)

def find_min(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    """Compute the minima along axis

    Parameters
    ----------
    obj : ArrayLike
        selected array
    axis : ArrayLike | None, optional
        chosen axis, by default None

    Returns
    -------
    max : ArrayLike
        minima along `axis`
    """
    obj = np.asarray(obj)
    return obj[find_argmin(obj,axis=axis)]

def timeit(func):
    """Source: https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk"""
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function `{func.__name__}`: {total_time:.4f} s')
        return result
    return timeit_wrapper

def distance(p1: tuple[int,int] | np.ndarray, p2: tuple[int,int] | np.ndarray) -> float | np.ndarray:
    """Compute the Euclidean distance between two projectionist

    Parameters
    ----------
    p1 : tuple[int,int] | np.ndarray
        point 1
    p2 : tuple[int,int] | np.ndarray
        point 2

    Returns
    -------
    distance : float | np.ndarray
        Euclidean distance
    """
    module_logger.info('Prova')
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


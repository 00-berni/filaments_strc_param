from typing import Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray
import time
from functools import wraps
from .data import FileVar
from .display import quickplot, plt

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
    return file_path.DIR.__add__(log_name).PATH

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
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def __compute_pxs(distances: np.ndarray, precision: int = 15) -> np.ndarray:
    """Compute pixel coordinates for each distance value

    Parameters
    ----------
    distances : np.ndarray
        set of distances
    precision : int, optional
        the accuracy at which coordinates are computed, by default 15

    Returns
    -------
    pxs : np.ndarray
        pixel coordinates for each distance

    Raises
    ------
    ValueError
        _description_

    Notes
    -----
    The method computes the pixel coordinates for each distance by taking `(d**2 - arange(1,d)**2)**1/2`
    `for d in distances` and then selecting the descrete values only. However, it fails: the Python accuracy
    is around the 16th digit, so it aprroximates the square root
    >>> a = numpy.sqrt(2)
    >>> a**2
    float64(2.0000000000000004)

    So to avoid mistakes, the method round `d**2` at the digit `10**(-precision)` and check whether the
    distance obtained from coordinates so estimated is compatible with the corresponding original one 
    (with a certain uncertainty). If it is not, the method will reduce the precision and check again. 
    """
    # compute the corresponding coordinates for each distance
    pxs = np.array([np.sqrt(np.round(d**2,decimals=precision)-np.arange(1,d)**2) for d in distances], dtype='object')
    tmp_dst = distances.copy()
    # select integer values only
    tmp_pxs = np.array([p[np.mod(p,1) == 0].astype(int) for p in pxs],dtype='object')
    rtol = 10**(-precision)     #: the uncertainty for the check
    # check the values
    check_pos = np.array([ ~np.all(np.isclose(d-np.sqrt(p**2+p[::-1]**2),np.zeros(len(p)),rtol=rtol)) for p,d in zip(tmp_pxs,tmp_dst)],dtype=bool)
    while np.any(check_pos):    #: adjust the accuracy
        precision -= 1
        if precision < 0:
            raise ValueError('Precision cannot be negative')
        # compute the values again with a lower accuracy
        tmp_pxs = np.array([np.sqrt(np.round(d**2,decimals=precision)-np.arange(1,d)**2) for d in tmp_dst], dtype='object')
        tmp_pxs = np.array([p[np.mod(p,1) == 0].astype(int) for p in tmp_pxs],dtype='object')
        # set the uncertainty and update the check
        rtol = 10**(-precision)
        check_pos[check_pos] = np.array([ ~np.all(np.isclose(d-np.sqrt(p**2+p[::-1]**2),np.zeros(len(p)),rtol=rtol)) for p,d in zip(tmp_pxs[check_pos],tmp_dst[check_pos])],dtype=bool)
    # store the values
    pxs = tmp_pxs
    return pxs        


def __discrete_dist(field: np.ndarray, distances: ArrayLike, order: int | None = None, mode: Literal['all','tpcf','sf'] = 'all', precision: int = 14) -> list[np.ndarray | None]:
    """Compute the tpcf for integer distances

    Parameters
    ----------
    field : np.ndarray
        data frame
    distances : ArrayLike
        integer distances
    precision : int, optional
        the accuracy at which coordinates are computed, by default 14
        see the documentation of `stuff.compute_pxs()`

    Returns
    -------
    correlations : np.ndarray
        the corresponding correlation values
    """
    distances = np.asarray(distances)
    distances = distances.astype(int)
    results = [None,None]
    # compute the coordinates of each pixel in the frame
    xdim, ydim = field.shape    #: sizes of the frame
    x = np.arange(xdim)         #: pixel positions along x axis
    y = np.arange(ydim)         #: pixel positions along y axis
    yy, xx = np.meshgrid(y,x)
    pxs = __compute_pxs(distances=distances,precision=precision)
    if mode == 'all' or mode == 'tpcf':
        correlations = np.array([
                                np.sum(field[x[:-d],:] * field[x[:-d]+d,:]) +    
                                np.sum(field[:,y[:-d]] * field[:,y[:-d]+d]) +
                                np.sum([np.sum(
                                                field[xx[ :-i,:-j] , yy[ :-i,:-j]]  *  field[xx[ :-i,:-j]+i , yy[ :-i,:-j]+j] + 
                                                field[xx[i:  ,:-j] , yy[i:  ,:-j]]  *  field[xx[i:  ,:-j]-i , yy[i:  ,:-j]+j]
                                                )
                                        for i,j in zip(pxs[idx],pxs[idx][::-1]) if len(pxs[idx]) != 0 
                                        ]) 
                                for d, idx in zip(distances,range(len(distances)))
                                ])
        results[0] = correlations
    if mode == 'all' or mode == 'sf':
        if order is None: order = 2
        # compute correlations array
        structs = np.array([
                            np.sum(np.abs(field[x[:-d],:] - field[x[:-d]+d,:])**order) +    
                            np.sum(np.abs(field[:,y[:-d]] - field[:,y[:-d]+d])**order) +
                            np.sum([np.sum(
                                            np.abs(field[xx[ :-i,:-j] , yy[ :-i,:-j]]  -  field[xx[ :-i,:-j]+i , yy[ :-i,:-j]+j])**order + 
                                            np.abs(field[xx[i:  ,:-j] , yy[i:  ,:-j]]  -  field[xx[i:  ,:-j]-i , yy[i:  ,:-j]+j])**order
                                            )
                                    for i,j in zip(pxs[idx],pxs[idx][::-1]) if len(pxs[idx]) != 0 
                                    ]) 
                            for d, idx in zip(distances,range(len(distances)))
                            ])
        results[1] = structs
    return results

def __float_dist(field: np.ndarray, distances: ArrayLike, order: int | None = None, mode: Literal['all','tpcf','sf'] = 'all', precision: int = 14) -> list[np.ndarray | None]:
    """Compute the tpcf for not-integer distances

    Parameters
    ----------
    field : np.ndarray
        data frame
    distances : ArrayLike
        not-integer distances
    precision : int, optional
        the accuracy at which coordinates are computed, by default 14
        see the documentation of `stuff.compute_pxs()`

    Returns
    -------
    correlations : np.ndarray
        the corresponding correlation values
    """
    # convert to numpy array type
    distances = np.asarray(distances)
    results = [None, None]
    # compute the coordinates of each pixel in the frame
    xdim, ydim = field.shape    #: sizes of the frame
    yy, xx = np.meshgrid(np.arange(ydim),np.arange(xdim))
    pxs = __compute_pxs(distances=distances,precision=precision)
    pos = np.array([len(p)!=0 for p in pxs])
    correlations = np.zeros(len(distances))
    structs = np.zeros(len(distances))
    if np.any(pos):
        if mode == 'all' or mode == 'tpcf':
            # compute the correlations
            correlations[pos] = np.array([np.sum([
                                                np.sum(
                                                        field[xx[ :-i,:-j] , yy[ :-i,:-j]]  *  field[xx[ :-i,:-j]+i , yy[ :-i,:-j]+j] + 
                                                        field[xx[i:  ,:-j] , yy[i:  ,:-j]]  *  field[xx[i:  ,:-j]-i , yy[i:  ,:-j]+j]
                                                        )
                                                for i,j in zip(p,p[::-1]) 
                                                ]) 
                                        for p in pxs[pos]
                                        ])
            results[0] = correlations 
        if mode == 'all' or mode == 'sf':
            if order is None: order = 2
            # compute the correlations
            structs[pos] = np.array([np.sum([
                                            np.sum(
                                                    np.abs(field[xx[ :-i,:-j] , yy[ :-i,:-j]]  -  field[xx[ :-i,:-j]+i , yy[ :-i,:-j]+j])**order + 
                                                    np.abs(field[xx[i:  ,:-j] , yy[i:  ,:-j]]  -  field[xx[i:  ,:-j]-i , yy[i:  ,:-j]+j])**order
                                                    )
                                            for i,j in zip(p,p[::-1]) 
                                            ]) 
                                        for p in pxs[pos]
                                        ])
            results[1] = structs              
    return results

def compute_tpcf(field: np.ndarray, bins: int | float | np.ndarray, no_zero: bool = False, precision: int = 14, display_plot: bool = True) -> np.ndarray:
    """Compute the Two Point Correlation Function (TPCF) in a frame for a set of distances

    Parameters
    ----------
    field : ndarray
        the data frame
    bins : int | float | ndarray
        the set of distances or a single distance
    no_zero : bool, optional
        if it is `True` the autocorrelation is not considered, by default False
    precision : int, optional
        the accuracy at which coordinates are computed, by default 14
        see the documentation of `stuff.compute_pxs()`
    display_plot : bool, optional
        display the results, by default True

    Returns
    -------
    corr : ndarray
        computed correlations array

    Notes
    -----
    The method takes either a single distance or an array of them as input

    Examples
    --------
    Compute the correlation for a specific distance:
    >>> compute_tpcf(data, 1, display_plot = False)
    array([150.])

    Compute the correlation for a set of distances:
    >>> distances = [0.,1.,1.43,12.]
    >>> compute_tpcf(data, distances, display_plot = False)
    array([200., 150., 0., 42.])

    The result is the same even if the 0 is not in `distances` due to the `no_zero` parameter:
    >>> distances = [1.,1.43,12.]
    >>> compute_tpcf(data, distances, display_plot = False)
    array([200., 150., 0., 42.])
    >>> compute_tpcf(data, distances, no_zero = True, display_plot = False)
    array([150., 0., 42.])
    """
    field = np.copy(field) - field.mean()   #: data after subtracting the mean
    xdim, ydim = field.shape                #: sizes of the frame
    if isinstance(bins,(float,int)):        #: condition for a single distance 
        if bins == 0:
            corr = (field**2).sum()
        elif bins**2 > xdim**2+ydim**2:
            corr = 0
        elif bins.is_integer():
            corr = __discrete_dist(field,[bins],mode='tpcf',precision=precision)[0][0]
        else:
            corr = __float_dist(field,[bins],mode='tpcf',precision=precision)[0][0]
    else:                                   #: condition for array of distances
        # convert to numpy array type
        bins = np.asarray(bins)
        # remove the 0 distance
        bins = bins[bins != 0]
        # initialize the correlation array
        corr = np.zeros(bins.size)
        # correlation is computed for distances less than the diagonal
        pos = bins**2 <= xdim**2 + ydim**2 
        tmp_bins = bins[pos]
        tmp_corr = corr[pos]
        # compute the positions of integer values
        int_pos = np.mod(tmp_bins,1) == 0
        if np.any(int_pos):                 #: compute correlation for discrete distances
            tmp_corr[int_pos]  = __discrete_dist(field,tmp_bins[int_pos],mode='tpcf',precision=precision)[0]
        if np.any(~int_pos):                #: compute correlation for float distances
            tmp_corr[~int_pos] = __float_dist(field=field,distances=tmp_bins[~int_pos],mode='tpcf',precision=precision)[0]
        # store the values in the main arrays
        bins[pos] = tmp_bins
        corr[pos] = tmp_corr 
        if not no_zero:
            # append the 0 value
            corr = np.append([(field**2).sum()],corr)
            bins = np.append([0],bins)
    # plot the results
    if display_plot and not isinstance(bins,(float,int)):
        quickplot((bins,corr),fmt='--.')
        plt.axhline(0,linestyle='dashed',color='black',alpha=0.5)
        plt.show()
    return corr


def tpcf(field: np.ndarray, precision: int = 14, display_plot: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute the TPCF for any possible distance in the frame

    Parameters
    ----------
    field : np.ndarray
        data frame
    precision : int, optional
        the accuracy at which coordinates are computed, by default 14
        see the documentation of `stuff.compute_pxs()`
    display_plot : bool, optional
        display the results, by default True

    Returns
    -------
    distances : np.ndarray
        array of any possible distance in the frame
    correlations : np.ndarray
        the corresponding correlation values
    
    Notes
    -----
    The function applies `stuff.compute_tpcf()` after computing the distances
    """
    xdim, ydim = field.shape
    yy,xx = np.meshgrid(np.arange(ydim),np.arange(xdim))
    # compute all the possible distances in the frame
    distances = np.unique(np.sqrt(xx**2+yy**2)).flatten()
    correlations = compute_tpcf(field=field, bins=distances, no_zero=False, precision=precision, display_plot=display_plot)
    return distances, correlations


def compute_sf(field: np.ndarray, bins: int | float | np.ndarray, order: int, precision: int = 14, display_plot: bool = True) -> np.ndarray:
    """Compute the Two Point Correlation Function (TPCF) in a frame for a set of distances

    Parameters
    ----------
    field : ndarray
        the data frame
    bins : int | float | ndarray
        the set of distances or a single distance
    no_zero : bool, optional
        if it is `True` the autocorrelation is not considered, by default False
    precision : int, optional
        the accuracy at which coordinates are computed, by default 14
        see the documentation of `stuff.compute_pxs()`
    display_plot : bool, optional
        display the results, by default True

    Returns
    -------
    corr : ndarray
        computed correlations array

    Notes
    -----
    The method takes either a single distance or an array of them as input

    Examples
    --------
    Compute the correlation for a specific distance:
    >>> compute_tpcf(data, 1, display_plot = False)
    array([150.])

    Compute the correlation for a set of distances:
    >>> distances = [0.,1.,1.43,12.]
    >>> compute_tpcf(data, distances, display_plot = False)
    array([200., 150., 0., 42.])

    The result is the same even if the 0 is not in `distances` due to the `no_zero` parameter:
    >>> distances = [1.,1.43,12.]
    >>> compute_tpcf(data, distances, display_plot = False)
    array([200., 150., 0., 42.])
    >>> compute_tpcf(data, distances, no_zero = True, display_plot = False)
    array([150., 0., 42.])
    """
    field = np.copy(field) - field.mean()   #: data after subtracting the mean
    xdim, ydim = field.shape                #: sizes of the frame
    if isinstance(bins,(float,int)):        #: condition for a single distance 
        if bins == 0:
            strc = (field**2).sum()
        elif bins**2 > xdim**2+ydim**2:
            strc = 0
        elif bins.is_integer():
            strc = __discrete_dist(field,[bins],order,mode='sf',precision=precision)[1][0]
        else:
            strc = __float_dist(field,[bins],order,mode='sf',precision=precision)[1][0]
    else:                                   #: condition for array of distances
        # convert to numpy array type
        bins = np.asarray(bins)
        # remove the 0 distance
        bins = bins[bins != 0]
        # initialize the correlation array
        strc = np.zeros(bins.size)
        # correlation is computed for distances less than the diagonal
        pos = bins**2 <= xdim**2 + ydim**2 
        tmp_bins = bins[pos]
        tmp_strc = strc[pos]
        # compute the positions of integer values
        int_pos = np.mod(tmp_bins,1) == 0
        if np.any(int_pos):                 #: compute correlation for discrete distances
            tmp_strc[int_pos]  = __discrete_dist(field,tmp_bins[int_pos],order,mode='sf',precision=precision)[1]
        if np.any(~int_pos):                #: compute correlation for float distances
            tmp_strc[~int_pos] = __float_dist(field=field,distances=tmp_bins[~int_pos],order=order,mode='sf',precision=precision)[1]
        # store the values in the main arrays
        bins[pos] = tmp_bins
        strc[pos] = tmp_strc 
    # plot the results
    if display_plot and not isinstance(bins,(float,int)):
        quickplot((bins,strc),fmt='--.')
        plt.axhline(0,linestyle='dashed',color='black',alpha=0.5)
        plt.show()
    return strc

def struc_fun(field: np.ndarray, order: int = 2, precision: int = 14, display_plot: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute the TPCF for any possible distance in the frame

    Parameters
    ----------
    field : np.ndarray
        data frame
    precision : int, optional
        the accuracy at which coordinates are computed, by default 14
        see the documentation of `stuff.compute_pxs()`
    display_plot : bool, optional
        display the results, by default True

    Returns
    -------
    distances : np.ndarray
        array of any possible distance in the frame
    correlations : np.ndarray
        the corresponding correlation values
    
    Notes
    -----
    The function applies `stuff.compute_tpcf()` after computing the distances
    """
    xdim, ydim = field.shape
    yy,xx = np.meshgrid(np.arange(ydim),np.arange(xdim))
    # compute all the possible distances in the frame
    distances = np.unique(np.sqrt(xx**2+yy**2)).flatten()
    structs = compute_sf(field=field, bins=distances, order=order, precision=precision, display_plot=display_plot)
    return distances, structs

def tpcf_n_sf(field: np.ndarray, bins: int | float | np.ndarray, order: int = 2, no_zero: bool = False, precision: int = 14, display_plot: bool = True) -> tuple[np.ndarray, np.ndarray]:
    field = np.copy(field) - field.mean()   #: data after subtracting the mean
    xdim, ydim = field.shape                #: sizes of the frame
    if isinstance(bins,(float,int)):        #: condition for a single distance 
        if bins == 0:
            corr = (field**2).sum()
            strc = 0
        elif bins**2 > xdim**2+ydim**2:
            corr = 0
            strc = 0
        elif bins.is_integer():
            corr, strc = __discrete_dist(field,[bins],order=order,mode='all',precision=precision)
            corr = corr[0]
            strc = strc[1]
        else:
            corr, strc = __float_dist(field,[bins],order=order,mode='all',precision=precision)
            corr = corr[0]
            strc = strc[1]
    else:                                   #: condition for array of distances
        # convert to numpy array type
        bins = np.asarray(bins)
        # remove the 0 distance
        bins = bins[bins != 0]
        # initialize the correlation array
        corr = np.zeros(bins.size)
        strc = np.zeros(bins.size)
        # correlation is computed for distances less than the diagonal
        pos = bins**2 <= xdim**2 + ydim**2 
        tmp_bins = bins[pos]
        tmp_corr = corr[pos]
        tmp_strc = strc[pos]
        # compute the positions of integer values
        int_pos = np.mod(tmp_bins,1) == 0
        if np.any(int_pos):                 #: compute correlation for discrete distances
            tmp_corr[int_pos],  tmp_strc[int_pos]  = __discrete_dist(field,tmp_bins[int_pos],order=order,mode='all',precision=precision)
        if np.any(~int_pos):                #: compute correlation for float distances
            tmp_corr[~int_pos], tmp_strc[~int_pos] = __float_dist(field=field,distances=tmp_bins[~int_pos],order=order,mode='all',precision=precision)
        # store the values in the main arrays
        bins[pos] = tmp_bins
        corr[pos] = tmp_corr 
        strc[pos] = tmp_strc 
        if not no_zero:
            # append the 0 value
            corr = np.append([(field**2).sum()],corr)
            strc = np.append([0],strc)
            bins = np.append([0],bins)
    # plot the results
    if display_plot and not isinstance(bins,(float,int)):
        quickplot((bins,corr),numfig=1,fmt='--.')
        plt.axhline(0,linestyle='dashed',color='black',alpha=0.5)
        quickplot((bins,strc),numfig=2,fmt='--.')
        plt.axhline(0,linestyle='dashed',color='black',alpha=0.5)
        plt.show()
    return corr, strc


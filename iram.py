from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy import ndarray
from numpy.typing import ArrayLike

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import HDUList
import astropy.units as u

from spectral_cube import SpectralCube, Slice

from astroquery.esasky import ESASky
from astroquery.utils import TableList
from astropy.wcs import WCS
from reproject import reproject_interp

import os

class PathVar():
    
    @staticmethod
    def this_dir() -> str:
        return os.path.dirname(__file__)


    def __init__(self, path: str = '') -> None:
        self.PATH = path  if path != '' else PathVar.this_dir()

    def split(self) -> list[str]:
        if self.PATH[0] == os.sep:
            return [os.sep] + self.PATH.split(os.sep)
        else:
            return self.PATH.split(os.sep)
        

    def copy(self) -> 'PathVar':
        return PathVar(self.PATH)

    def __add__(self, path: str) -> 'PathVar':
        new_path = os.path.join(self.PATH,path)
        return PathVar(path=new_path)
     
    def __sub__(self, up: int) -> 'PathVar':
        path_list = self.split()
        new_path = os.path.join(*path_list[:-up])
        return PathVar(path=new_path)
    
    def __str__(self) -> str:
        return self.PATH 
     

class FileVar(PathVar):

    def __init__(self, filename: str | list[str], dirpath: str | PathVar = '') -> None:
        self.DIR  = dirpath if isinstance(dirpath, PathVar) else PathVar(path = dirpath)
        self.FILE = filename

    def path(self) -> str | list[str]:
        filename = self.FILE 
        dirname = self.DIR.copy()
        if isinstance(filename,str): 
            return (dirname + filename).PATH
        else:
            return [(dirname + name).PATH for name in filename]

    def __getitem__(self,item: int) -> str:
        path = self.path()
        if isinstance(path,str): return TypeError('Variable is not subscriptable')
        else: return path[item]
    
    def __setitem__(self,key:int,item:str) -> None:
        if isinstance(self.FILE,str): 
            return TypeError('Variable is not subscriptable')
        else: 
            self.FILE[key] = item


    def __str__(self) -> str:
        return str(self.path())

def reorganize_index(idxes: tuple | ndarray, axis: int | None, shape: tuple) -> tuple:
    if axis is None:
        return np.unravel_index(idxes,shape)
    else:
        axes = [ np.arange(s) for s in shape]
        axes[axis] = idxes
        return tuple(axes) 

def find_argmax(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    obj = np.asarray(obj)
    maxpos = np.argmax(obj, axis=axis)
    if len(obj.shape) == 1:
        return maxpos
    else:
        return reorganize_index(maxpos,axis=axis,shape=obj.shape)
        

def find_max(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    obj = np.asarray(obj)
    return obj[find_argmax(obj,axis=axis)]

def find_argmin(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    obj = np.asarray(obj)
    minpos = np.argmin(obj,axis=axis)
    if len(obj.shape) == 1:
        return minpos
    else:
        return reorganize_index(minpos,axis=axis,shape=obj.shape)

def find_min(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    obj = np.asarray(obj)
    return obj[find_argmin(obj,axis=axis)]

PROJECT_DIR = PathVar()


def hotpx_remove(data: ndarray) -> ndarray:
    """To remove hot pixels from the image

    Parameters
    ----------
    data : ndarray
        spectrum data

    Returns
    -------
    data : ndarrayfrom astropy

    Notes
    -----
    The function replacing `NaN` values from the image, if there are.
    I did not implement this function, I took it from [*astropy documentation*](https://docs.astropy.org/en/stable/convolution/index.html)

    """
    from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
    # check the presence of `NaNs`
    if True in np.isnan(data):
        # build a gaussian kernel for the interpolation
        kernel = Gaussian2DKernel(x_stddev=1)
        # remove the `NaNs`
        data = interpolate_replace_nans(data, kernel)
    return data



def showfits(data: np.ndarray, v: int = -1, title: str = '', n: int = None, norm: str ='linear', dim: list[int] = [10,7], labels: tuple[str,str] = ('',''), ticks: tuple[ndarray[float] | None, ndarray[float] | None] = (None,None), tickslabel: tuple[ ndarray[str | float] | None,  ndarray[str | float] | None] = (None, None),**kwimg) -> None:
    """Function to display the fits image.
    filename
    You can display simply the image or set a figure number and a title.

    :param data: image matrix of fits file
    :type data: np.ndarray
    :param v: cmap parameter: 1 for false colors, 0 for grayscale, -1 for reversed grayscale; defaults to -1
    :type v: int, optional
    :param title: title of the image, defaults to ''
    :type title: str, optional
    :param n: figure number, defaults to None
    :type n: int, optional
    :param dim: figure size, defaults to [10,7]
    :type dim: list[int], optional
    """
    plt.figure(n,figsize=dim)
    plt.title(title)
    if v == 1 : color = 'viridis'
    elif v == 0 : color = 'gray'
    else : color = 'gray_r'
    plt.imshow(data, cmap=color, norm=norm, origin='lower',**kwimg)
    plt.colorbar()
    # plt.xlabel(labels[0])
    # plt.ylabel(labels[1])
    # if_stat = lambda tck : tck[0] is None and tck[1] is None  
    # if if_stat(ticks) and not if_stat(tickslabel):
    #     ticks = (np.arange(*tickslabel[0].shape), np.arange(*tickslabel[1].shape))     
    # plt.xticks(ticks[0],tickslabel[0])
    # plt.yticks(ticks[1],tickslabel[1])


def get_data_fit(path: str, lims: list[int | None] = [None,None,None,None], v: int = -1, title: str = '', n: int = None, dim: list[int] = [10,7], hotpx: bool = True, display_plots: bool = True, **imgargs) -> tuple[HDUList, ndarray]:
    """Function to open fits file and extract data.
    
    It brings the path and extracts the data, giving a row image.
    
    You can set a portion of image and also the correction for hotpx.

    It calls the functions: 
      - `hotpx_remove()`
      - `showfits()`

    :param path: path of the fits file
    :type path: str
    :param lims: edges of the fits, defaults to [None,None,None,None]
    :type lims: list[int | None], optional
    :param hotpx: parameter to remove or not the hot pixels, defaults to True
    :type hotpx: bool, optional
    :param v: cmap parameter: 1 for false colors, 0 for grayscale, -1 for reversed grayscale; defaults to -1
    :type v: int, optional
    :param title: title of the image, defaults to ''
    :type title: str, optional
    :param n: figure number, defaults to None
    :type n: int, optional
    :param dim: figure size, defaults to [10,7]
    :type dim: list[int], optional

    :return: `hdul` list of the chosen fits file and `data` of the spectrum
    :rtype: tuple

    .. note:: `lims` parameter controls the x and y extremes in such the form [lower y, higher y, lower x, higher x]
    """
    # open the file
    hdul = fits.open(path)
    # print fits info
    hdul.info()
    # print header
    hdr = hdul[0].header
    print(' - HEADER -')
    print(hdr.tostring(sep='\n'))
    print()

    # data extraction
    # format -> data[Y,X]
    data = hotpx_remove(hdul[0].data) if hotpx else hdul[0].data
    ly,ry,lx,rx = lims
    data = data[ly:ry,lx:rx]
    # hot px correction
    # Spectrum image
    if display_plots == True: showfits(data, v=v,title=title,n=n,dim=dim, **imgargs) 
    return hdul,data


u_vel = u.km / u.s
u.add_enabled_units(u.def_unit(['K (Tb)'], represents=u.K))


def select_channel(cube: SpectralCube, vel_val: float) -> Slice:
    ch_pos = np.argmin(np.abs(cube.with_spectral_unit(u_vel).spectral_axis - vel_val*u_vel))
    return cube[ch_pos, : , :]

from pandas import read_csv
MBM40_DIR = PROJECT_DIR + 'MBM40'
DATA_FILE = FileVar(filename='data.csv',dirpath=MBM40_DIR)
co_files, hi_files, ir_files = read_csv(DATA_FILE.path()).to_numpy().transpose()
co_paths = FileVar(co_files,MBM40_DIR+'CO')
hi_paths = FileVar(hi_files,MBM40_DIR+'HI')
ir_paths = FileVar(ir_files,MBM40_DIR+'IR')
print(co_paths)
print(hi_paths)
print(ir_paths)

ir60_hdul, ir60_data   = get_data_fit(ir_paths[0],display_plots=False) 
ir100_hdul, ir100_data = get_data_fit(ir_paths[1],display_plots=False) 


ir100_wcs = WCS(ir100_hdul[0].header)

fig = plt.figure(figsize=(13,7))
ax = fig.add_subplot(111,projection=ir100_wcs)
ir_img = ax.imshow(ir100_data,cmap='gray')
plt.colorbar(ir_img)
plt.show()

ir60_wcs = WCS(ir60_hdul[0].header)


fig = plt.figure(figsize=(13,7))
ax = fig.add_subplot(111,projection=ir60_wcs)
ir_img = ax.imshow(ir60_data,cmap='gray',vmax=6)
plt.colorbar(ir_img)
# ax.plot(SkyCoord(),'xr',transform=ax.get_transform("world"))
plt.show()

from scipy.ndimage import sobel

ir60_h = sobel(ir60_data,0)
ir60_v = sobel(ir60_data,1)
filt_ir60 = np.sqrt(ir60_h**2+ir60_v**2)



plt.figure(figsize=(13,5))
plt.subplot(121)
plt.imshow(ir60_v,origin='lower',vmax=0.7,vmin=-0.4)
plt.subplot(122)
plt.imshow(ir60_h,origin='lower',vmax=0.7,vmin=-0.4)
plt.figure(figsize=(13,9))
plt.imshow(filt_ir60,origin='lower',vmax=0.5,vmin=0)
plt.colorbar()
plt.show()

ir60_cut = ir60_h[300:600,400:600]
# plt.figure()
# plt.plot(ir60_h[450])
# plt.figure()
# plt.plot(ir60_v[:,450])
plt.figure(figsize=(17,19))
for i in ir60_cut:
    plt.plot(i[i!=0])
plt.figure(figsize=(17,19))
plt.imshow(ir60_cut,origin='lower',vmax=0.7,vmin=-0.4)
plt.show()
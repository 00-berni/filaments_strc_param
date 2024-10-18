import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from astropy.io import fits
from astropy.io.fits import HDUList


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

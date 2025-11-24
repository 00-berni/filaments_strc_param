import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from .display import show_image
from .typing import *
from .typing import HDUList

def hotpx_remove(data: FloatArray) -> FloatArray:
    """Remove hot pixels from the image

    Parameters
    ----------
    data : FloatArray
        spectrum data

    Returns
    -------
    data : FloatArray
        from astropy

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





def get_data_fit(path: str, lims: list[Optional[int]] = [None,None,None,None], hotpx: bool = True, display_plots: bool = True, **imgargs) -> tuple[HDUList, FloatArray]:
    """Open fits file and extract data. 

    Parameters
    ----------
    path : str
        path of the fits file
    lims : list[int  |  None], optional
        edges of the fits, by default [None,None,None,None]
        `lims` parameter controls the x and y extremes in such the form [lower y, higher y, lower x, higher x]
    hotpx : bool, optional
        parameter to remove or not the hot pixels, by default True
    display_plots : bool, optional
        if `True` a row image is displayed, by default True
    **imgargs
        arguments of `display.show_image()`

    Returns
    -------
    hdul : HDUList
        hdul list of the fits
    data : FloatArray
        fits data
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
    if display_plots == True: show_image(data, **imgargs) 
    return hdul,data

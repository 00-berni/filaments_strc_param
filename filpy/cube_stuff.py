import numpy as np
from astroquery.esasky import ESASky
from astroquery.utils import TableList
from astropy.wcs import WCS
from reproject import reproject_interp
from spectral_cube import SpectralCube, Slice
from .data import U_VEL

def select_channel(cube: SpectralCube, vel_val: float) -> Slice:
    ch_pos = np.argmin(np.abs(cube.with_spectral_unit(U_VEL).spectral_axis - vel_val*U_VEL))
    return cube[ch_pos, : , :]

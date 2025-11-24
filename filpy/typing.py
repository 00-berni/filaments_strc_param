from typing import Literal, Sequence, Optional
from numpy.typing import NDArray, ArrayLike
from numpy import int_, float64, bool_
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from astropy.io.fits import HDUList

IntArray   = NDArray[int_]
FloatArray = NDArray[float64]
BoolArray  = NDArray[bool_]

__all__ = [
           'Literal',
           'Sequence',
           'Optional',
           'NDArray',
           'ArrayLike',
           'IntArray',
           'FloatArray',
           'BoolArray'
          ]

from typing import Literal, Sequence, Optional, Union
from numpy.typing import NDArray, ArrayLike
from numpy import int_, float64, bool_
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from astropy.io.fits import HDUList

IntArray   = NDArray[int_]
FloatArray = NDArray[float64]
BoolArray  = NDArray[bool_]
FloatArrayLike = Union[float, FloatArray]

__all__ = [
           'Literal',
           'Sequence',
           'Optional',
           'Union',
           'NDArray',
           'ArrayLike',
           'FloatArrayLike',
           'IntArray',
           'FloatArray',
           'BoolArray'
          ]

from typing import Literal
from numpy.typing import NDArray
from numpy import int_, float64, bool_

IntArray   = NDArray[int_]
FloatArray = NDArray[float64]
BoolArray  = NDArray[bool_]

__all__ = ['Literal','IntArray','FloatArray','BoolArray']

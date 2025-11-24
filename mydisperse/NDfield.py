#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:16:12 2019

@author: dvibert

This is the native binary format of DisPerSE
"""

from os import path
import struct
import numpy as np
from numpy.typing import NDArray

ND_CHAR   = 1<<0
ND_UCHAR  = 1<<1
ND_SHORT  = 1<<2
ND_USHORT = 1<<3
ND_INT    = 1<<4
ND_UINT   = 1<<5
ND_LONG   = 1<<6
ND_ULONG  = 1<<7
ND_FLOAT  = 1<<8
ND_DOUBLE = 1<<9

NDFIELD_MAX_DIMS = 20

NDFIELD_TAG = b'NDFIELD' 
#NDFIELD_ASCII_TAG = b'ANDFIELD'

datatype_dict = {np.int8:    ND_CHAR,
                 np.uint8:   ND_UCHAR,
                 np.int16:   ND_SHORT,
                 np.uint16:  ND_USHORT,
                 np.int32:   ND_INT,
                 np.uint32:  ND_UINT,
                 np.int64:   ND_LONG,
                 np.uint64:  ND_ULONG,
                 np.float32: ND_FLOAT,
                 np.float64: ND_DOUBLE}


"""
typedef struct NDfield_str
{
 char comment[80];  // a comment on the data
 int dims[NDFIELD_MAX_DIMS];  // dimensions of the grid, must be [ndims,nparticles] when data represents sample particles coordinates (i.e. when fdims_index!=0)
 int ndims;  // number of dimensions of the space
 int n_dims;  // number of meaningfull values in dims array
 int fdims_index; // if 0, the field is a regular grid of dimensions dims, else the file contains the dims[0] coordinates of dims[1] particles.
 int datatype;  // type of the data (one of the ND_... defined above)
 double x0[NDFIELD_MAX_DIMS];  // origin of the bounding box
 double delta[NDFIELD_MAX_DIMS];  // extent of the bounding box
 char dummy[160];  // dummy data, for future extensions or for storing anything you want.

 void *val;  // pointer to data

 long nval;  // total number of particles (fdims_index==1) or pixels (fdims_index==0)
 int datasize;  // size in bytes of datatype type.
} NDfield;
"""


def write_NDfield(data_array: NDArray[np.float64], filename: str, comment: str | None = None, coord: bool = False) -> None:
    """Write a NDfield according to the native binary format of Disperse

    Parameters
    ----------
    data_array : NDArray[np.float64]
        data to save
    filename : str
        name of the future file .NDfield
    comment : str | None, optional
        a comment on the data, by default None
    coord : bool, optional
        if `True` field is sample particles coordinates else it is a regular gird, by default `False`
    """
    # initialize the variables
    if comment is None:
        comment = b''
    else:
        comment = bytes(comment, 'ascii')    
    x0    = np.zeros(NDFIELD_MAX_DIMS)                  #: origin of the bounding box
    delta = np.zeros(NDFIELD_MAX_DIMS)                  #: extent of the bounding box
    dims  = np.zeros(NDFIELD_MAX_DIMS,dtype='int32')    #: dimensions of the grid or [ndims, npoints]
    
    if coord:  
        fdim_index = 1                                  #: set for sample particles coordinates
        ndims = data_array.shape[0]                     #: number of dimensions
        npoints = data_array.shape[1]                   #: number of points
        # update the variables
        dims[:2] = (ndims, npoints)
        x0[:ndims] = data_array.min(axis=1)
        delta[:ndims] = data_array.max(axis=1) - x0[:ndims]
    else:
        fdim_index = 0                                  #: set for regular grid
        ndims = data_array.ndim                         #: number of axes
        # update the variables
        dims[:ndims] = data_array.shape
        x0[:ndims] = 0.
        delta[:ndims] = data_array.shape
    
    # store datatype
    datatype = datatype_dict[data_array.dtype.type]
    
    # check the filename
    name, ext = path.splitext(filename)
    if not ext:
        ext = 'NDfield'
    filename = name + ext
    
    with open(filename, 'wb') as f:
                
        buffer = struct.pack('I16sI', 16,  NDFIELD_TAG, 16)
        f.write(buffer)
        
        nbytes = 4*(NDFIELD_MAX_DIMS+3) + 8*(2*NDFIELD_MAX_DIMS) + (160+80)
        buffer = struct.pack('I80sI', 
                             nbytes,                    #: dummy : 1 int(4B)
                             comment, 
                             ndims                      #: ndims : 1 int(4B)
                             )
        buffer += bytearray(dims.astype('int32'))       #: dims : 20 int(4B)
        buffer += struct.pack('2I',               
                              fdim_index,               
                              datatype                  #: datatype : 1 int(4B)
                             )
        buffer += bytearray(x0.astype('double'))        #: x0 : 20 double(8B)
        buffer += bytearray(delta.astype('double'))     #: delta : 20 double(8B)
        buffer += struct.pack('160xI', nbytes)          #: dummy extra : 160 char(1B)
        f.write(buffer)

        buffer = struct.pack('I', data_array.nbytes)
        buffer += bytearray(data_array.T)
        buffer += struct.pack('I', data_array.nbytes)
        f.write(buffer)


def write_NDfield_coord(pos: NDArray[np.int_], filename: str, mass: NDArray[np.float64] | None = None, comment: str | None = None):
    # store the field
    write_NDfield(pos, filename, comment = comment, coord = True)
    # add the mass field
    if mass is not None:
        name, ext = path.splitext(filename)
        name += '_mass'
        mass_filename = name + ext
        write_NDfield(mass, mass_filename)
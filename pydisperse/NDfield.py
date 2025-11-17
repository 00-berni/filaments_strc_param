#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:16:12 2019

@author: dvibert
"""

from os import path
import struct
import numpy as np

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

def write_NDfield(data_array, filename, comment=None, coord=False):
    
    #tag = (16 - len(NDFIELD_TAG)-1)*b'\x00' + NDFIELD_TAG + b'\x00'
    
    if comment is None:
        comment = b''
    else:
        comment = bytes(comment, 'ascii')
    
    x0 = np.zeros(NDFIELD_MAX_DIMS)
    delta = np.zeros(NDFIELD_MAX_DIMS)
    dims = np.zeros(NDFIELD_MAX_DIMS,dtype='int32')
    
    if coord:
        fdim_index = 1
        ndims = data_array.shape[0]
        npoints = data_array.shape[1]
        dims[:2] = (ndims, npoints)
        x0[:ndims] = data_array.min(axis=1)
        delta[:ndims] = data_array.max(axis=1) - x0[:ndims]
    else:
        fdim_index = 0
        ndims = data_array.ndim
        dims[:ndims] = data_array.shape
        x0[:ndims] = 0.
        delta[:ndims] = data_array.shape
        
    datatype = datatype_dict[data_array.dtype.type]
    
    name, ext = path.splitext(filename)
    if not ext:
        ext = 'NDfield'
    filename = name + ext
    
    with open(filename, 'wb') as f:
                
        buffer = struct.pack('I16sI', 16,  NDFIELD_TAG, 16)
        f.write(buffer)
        
        nbytes = 4*(NDFIELD_MAX_DIMS+3) + 8*(2*NDFIELD_MAX_DIMS) + (160+80)
        buffer = struct.pack('I80sI', 
                             nbytes, # dummy 1 int(4B)
                             comment, 
                             ndims # ndims 1 int(4B)
                             )
        buffer += bytearray(dims.astype('int32')) # dims  20 int(4B)
        buffer += struct.pack('2I',               
                             fdim_index,     # fdim_index (=1 for particles)
                             datatype # datatype 1 int(4B)
                             )
        buffer += bytearray(x0.astype('double')) # x0   20 double(8B)
        buffer += bytearray(delta.astype('double'))  # delta 20 double(8B)
        buffer += struct.pack('160xI', nbytes) # dummy extra  160 char(1B) & dummy 1 int(4B)
        f.write(buffer)

        buffer = struct.pack('I', data_array.nbytes)
        buffer += bytearray(data_array.T)
        buffer += struct.pack('I', data_array.nbytes)
        f.write(buffer)


def write_NDfield_coord(pos, filename, mass=None, comment=None):
    
    write_NDfield(pos, filename, comment=comment, coord=True)
    
    if mass is not None:
        name, ext = path.splitext(filename)
        name += '_mass'
        mass_filename = name + ext
        write_NDfield(mass, mass_filename)
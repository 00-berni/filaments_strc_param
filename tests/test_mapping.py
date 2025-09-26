import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft, fft2, ifft2
from time import time
import argparse
from .test_func import *
from .test_func import distance



def __mapping(coord: tuple[int,int], positions: tuple[np.ndarray,np.ndarray]) -> tuple[tuple[np.ndarray,np.ndarray],tuple[np.ndarray,np.ndarray]]:
    i,j = coord
    # ydim, xdim = data.shape
    # xx, yy = np.meshgrid(np.arange(xdim),np.arange(ydim))
    xx, yy = positions
    xpos, ypos = slice(None,None), slice(None,None)
    if i != 0:
        xidx = (-i,None)
        xsgn = -np.sign(i)
        xpos = slice(*xidx[::xsgn])
    if j != 0:
        yidx = (-j,None)
        ysgn = -np.sign(j)
        ypos = slice(*yidx[::ysgn])
    x = xx[ypos,xpos]
    y = yy[ypos,xpos]
    x_pos = (x,x+i)
    y_pos = (y,y+j)
    return x_pos, y_pos     

def compute(data: np.ndarray):
    data -= np.mean(data)
    ydim, xdim = data.shape
    xx, yy = np.meshgrid(np.arange(xdim),np.arange(ydim))
    corr = np.zeros((*xx.shape,4))
    xsgn = (0,0,-1,-1)
    ysgn = (0,-1,-1,0)
    for m in range(xx.size):
        ii,jj = np.unravel_index(m,xx.shape)
        i = xx[ii,jj]
        j = yy[ii,jj]
        x_pos, y_pos = __mapping((i,j),(xx,yy))
        corr[ii,jj] = [np.sum(data[y_pos[k],x_pos[t]]*data[y_pos[k+1],x_pos[t+1]]) for k,t in zip(ysgn,xsgn)]
    return np.asarray(corr)

if __name__ == '__main__':
    dim = 14
    data = np.zeros((dim,dim))
    ydim, xdim = data.shape
    data[2:4] = 1
    corr = compute(data)
    print(corr.shape)
    tot_corr = np.zeros((2*ydim-1,2*xdim-1))
    tot_corr[ydim-1:,xdim-1:] = corr[:,:,0]
    tot_corr[:ydim,xdim:] = corr[::-1,:,1][:,1:]
    tot_corr[:ydim,:xdim] = corr[::-1,::-1,2]
    tot_corr[ydim:,:xdim] = corr[:,::-1,3][1:,:]
    print(tot_corr[xdim,ydim],np.max(tot_corr))
    plt.figure()
    plt.imshow(data,origin='lower')
    plt.figure()
    plt.imshow(tot_corr,origin='lower')
    plt.show()
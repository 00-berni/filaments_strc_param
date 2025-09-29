import numpy as np
import matplotlib.pyplot as plt 
import argparse
from time import time
from multiprocessing import Pool, cpu_count
from functools import partial
from numpy.typing import ArrayLike

# from .test_func import *
# from .test_func import distance



def __mapping(coord: tuple[int,int], positions: tuple[np.ndarray,np.ndarray]) -> tuple[tuple[np.ndarray,np.ndarray],tuple[np.ndarray,np.ndarray]]:
    i,j = coord
    # ydim, xdim = data.shape
    # xx, yy = np.meshgrid(np.arange(xdim),np.arange(ydim))
    xx, yy = positions
    if i == 0:
        xpos = slice(None,None) 
    elif i > 0:
        xpos = slice(None,-i)
    else:
        xpos = slice(-i,None)

    if j == 0:
        ypos = slice(None,None) 
    elif j > 0:
        ypos = slice(None,-j)
    else:
        ypos = slice(-j,None)

    x = xx[ypos,xpos]
    y = yy[ypos,xpos]
    x_pos = (x,x+i)
    y_pos = (y,y+j)
    return x_pos, y_pos     


def compute(data: np.ndarray):
    tmp_data = data - np.mean(data)
    ydim, xdim = tmp_data.shape
    xx, yy = np.meshgrid(np.arange(xdim),np.arange(ydim))
    corr = np.zeros((*xx.shape,4))
    xsgn = (0,0,-1,-1)
    ysgn = (0,-1,-1,0)
    for m in range(xx.size):
        ii,jj = np.unravel_index(m,xx.shape)
        i = xx[ii,jj]
        j = yy[ii,jj]
        x_pos, y_pos = __mapping((i,j),(xx,yy))
        corr[ii,jj] = [np.sum(tmp_data[y_pos[k],x_pos[t]]*tmp_data[y_pos[k+1],x_pos[t+1]]) for k,t in zip(ysgn,xsgn)]
    return np.asarray(corr)

def step0(inputs: list[ArrayLike]):
    step = inputs[0]
    coord = inputs[1]
    data = inputs[2]

    xsgn = (0,0,-1,-1)
    ysgn = (0,-1,-1,0)
    xx,yy = coord
    ii,jj = np.unravel_index(step,xx.shape)
    i = xx[ii,jj]
    j = yy[ii,jj]
    x_pos, y_pos = __mapping((i,j),(xx,yy))
    corr_i = [np.sum(data[y_pos[k],x_pos[t]]*data[y_pos[k+1],x_pos[t+1]]) for k,t in zip(ysgn,xsgn)]
    return corr_i

def step1(step: int):
    xsgn = (0,0,-1,-1)
    ysgn = (0,-1,-1,0)
    ii,jj = np.unravel_index(step,g_xx.shape)
    i = g_xx[ii,jj]
    j = g_yy[ii,jj]
    x_pos, y_pos = __mapping((i,j),(g_xx,g_yy))
    corr_i = [np.sum(g_tmp_data[y_pos[k],x_pos[t]]*g_tmp_data[y_pos[k+1],x_pos[t+1]]) for k,t in zip(ysgn,xsgn)]
    return corr_i


def parallel_compute(data: np.ndarray, processes: int = cpu_count()-1):
    global g_tmp_data
    g_tmp_data = np.copy(data) - np.mean(data)
    ydim, xdim = data.shape
    global g_xx, g_yy
    g_xx, g_yy = np.meshgrid(np.arange(xdim),np.arange(ydim))
    # print('DATA',data[3])
    # print('PARAL_DATA',tmp_data[3])
    # iterator = [[i, (g_xx,g_yy), tmp_data, corr ] for i in np.arange(xx.size)]
    with Pool(processes=processes) as pool:
        # corr_i = pool.map(step0,iterator)
        corr_i = pool.map(step1, np.arange(g_xx.size))
    return np.asarray(corr_i).reshape(ydim,xdim,4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",help='set log',nargs='*',type=str,action="store",choices=['file','bash','all','DEBUG', 'INFO'],default=None)
    parser.add_argument("--no-diag", help='compute horizontal and vertical only', action='store_false')
    parser.add_argument("-m","--mode", help='mode of the log',type=str, action='store',default='w')
    parser.add_argument("-c","--cores", help='Number of used cores',type=int, action='store',default=11)
    parser.add_argument("-d","--dim", help='field size',type=int, action='store',default=32)
    parser.add_argument("-s","--seed", help='set seed', type=int, action='store',default=10)
    parser.add_argument("-e","--edges", help='set max and min of noise', type=float, required=False, nargs=2, action='store',default=[0,2])
    parser.add_argument("-w","--width", help='set the width of the pattern', required=False, type=int, action='store',default=3)
    parser.add_argument("-l","--lag", help='set the lag of the lattice', required=False, type=int, action='store',default=10)
    parser.add_argument("-v","--value", help='set the value of the lattice', required=False, type=int, action='store',default=1)
    parser.add_argument("-i","--iter", help='set the number of iterations',required=False, type=int, action='store',default=10)
    parser.add_argument("-p","--plot",help='plot data or not',action='store_false')
    parser.add_argument("--ticks",help='display pixel edges',action='store_true')
    parser.add_argument("--method",help='the function type',action='store',type=str, choices=['parallel','sequence'], default='sequence')
    parser.add_argument("--dist",help='distances',action="store",type=str,default="{'dist': [0,5,10]}")

    args = parser.parse_args()
    dim = args.dim
    data = np.zeros((dim,dim)) + np.random.normal(0.5,0.3,size=(dim,dim))
    ydim, xdim = data.shape
    lag = args.lag
    data[::lag] = 1
    data[1::lag] = 1
    data[2::lag] = 1
    print('FRAME: ',(dim,dim))
    print('METHOD: ',args.method.upper())


    if args.method == 'sequence':    
        start_time = time()
        corr = compute(data)
        print(corr.shape)
        # print('CORR\n',corr[:,0,:])
        tot_corr = np.zeros((2*ydim-1,2*xdim-1))
        tot_corr[ydim-1:,xdim-1:] = corr[:,:,0]
        tot_corr[:ydim-1,xdim:] = corr[::-1,:,1][:-1,1:]
        tot_corr[:ydim,:xdim] = corr[::-1,::-1,2]
        tot_corr[ydim:,:xdim-1] = corr[:,::-1,3][1:,:-1]
        end_time = time()
        print('\nCOMPUTATIONAL TIME: ',(end_time-start_time)/60,' m')
        print(tot_corr[ydim-1,xdim-1],np.max(tot_corr))
        # print(data[0])

    ## PARALLEL
    elif args.method == 'parallel':    
        start_time = time()
        corr = parallel_compute(data,processes=args.cores)
        print('After paral:', corr.shape)
        tot_corr = np.zeros((2*ydim-1,2*xdim-1))
        tot_corr[ydim-1:,xdim-1:] = corr[:,:,0]
        tot_corr[:ydim-1,xdim:] = corr[::-1,:,1][:-1,1:]
        tot_corr[:ydim,:xdim] = corr[::-1,::-1,2]
        tot_corr[ydim:,:xdim-1] = corr[:,::-1,3][1:,:-1]
        end_time = time()
        print('\nCOMPUTATIONAL TIME: ',(end_time-start_time)/60,' m')
        print(tot_corr[ydim-1,xdim-1],np.max(tot_corr))
    print('\n'+'======'*10+'\n') 
    
    # plt.figure()
    # plt.imshow(data,origin='lower')
    # plt.figure()
    # plt.imshow(tot_corr,origin='lower')
    # plt.show()
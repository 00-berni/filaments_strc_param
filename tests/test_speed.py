from datetime import datetime
import numpy as np
from time import time
import argparse
from ..filpy import parallel_compute, sequence_compute, combine_results

def make_pattern(data: np.ndarray, pattern: str = 'lattice',**kwargs) -> np.ndarray:
    new_data = np.copy(data)
    xdim, ydim = new_data.shape
    if pattern == 'lattice':
        wd = kwargs['width']
        sp = kwargs['spacing']
        lag = sp+wd
        xpos = np.arange(xdim//lag+1)*lag
        ypos = np.arange(ydim//lag+1)*lag
        xpos = np.concatenate([xpos + i for i in range(wd)])
        ypos = np.concatenate([ypos + i for i in range(wd)])
        xpos = xpos[xpos<xdim]
        # ypos = ypos[ypos<ydim]
        new_data[xpos,:] += 1
        # new_data[:,ypos] = 1
    elif pattern == 'filament':
        wd = kwargs['width']
        sp = kwargs['spacing']
        sequence = [0,1,2,3,3,3,2,2,1,1,0,-1,-2,-1,-1,-2,-3,-3,-4,-4,-3,-2,-2,-1,-1]
        patt = np.array(sequence*(ydim//len(sequence))+sequence[:ydim%len(sequence)])
        pos  = np.arange(ydim)
        
        for i in range(wd):
            xpos = sp+patt + i
            idx = xpos<xdim
            xpos = xpos[idx]
            pos = pos[idx]
            new_data[xpos,pos] += 1
    return new_data


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
    parser.add_argument("--maxlag", help='set the max lag', type=int, action='store',default=10)
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
    max_lag = args.maxlag
    xlim = (max_lag, xdim-max_lag)
    ylim = (max_lag, ydim-max_lag)
    data = make_pattern(data,pattern='filament',width=args.width,spacing=args.maxlag+args.lag)
    print('DATE:\t',datetime.now())
    print('FRAME:\t',(dim,dim))
    print('METHOD:\t',args.method.upper())
    print('MASK:\t',xlim,ylim)
    print('MAXLAG:\t',max_lag)

    ## SEQUENCE
    if args.method == 'sequence':    
        start_time = time()
        corr = sequence_compute(data,mask_ends=(xlim,ylim),mode='tpcf')
        print('CORR SHAPE:\t',corr.shape)
        tot_corr = combine_results(corr)
        end_time = time()
        print('\nCOMPUTATIONAL TIME: ',(end_time-start_time)/60,' m')

    ## PARALLEL
    elif args.method == 'parallel':    
        start_time = time()
        corr = parallel_compute(data,mask_ends=(xlim,ylim),mode='tpcf',processes=args.cores)
        print('CORR SHAPE:', corr.shape)
        tot_corr = combine_results(corr)
        end_time = time()
        print('\nCOMPUTATIONAL TIME: ',(end_time-start_time)/60,' m')
    print('\n'+'======'*10+'\n') 

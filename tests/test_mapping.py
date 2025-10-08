import numpy as np
import matplotlib.pyplot as plt 
import argparse
from time import time
from multiprocessing import Pool, cpu_count
from functools import partial
from numpy.typing import ArrayLike

from .test_func import *
# from .test_func import distance





def compute(data: np.ndarray, mask_ends: tuple[tuple[int,int], tuple[int,int]]):
    tmp_data = data - np.mean(data)
    ydim, xdim = tmp_data.shape
    (xo, xe), (yo, ye) = mask_ends
    m_ydim = ye-yo 
    m_xdim = xe-xo
    max_lag = min(xo,yo,xdim-xe,ydim-ye,int(np.sqrt(m_xdim**2+m_ydim**2)))
    corr_dim = max_lag+1
    xx, yy = np.meshgrid(np.arange(m_xdim),np.arange(m_ydim))
    xx += xo
    yy += yo
    corr = np.zeros((corr_dim,corr_dim,4))
    xsgn = (1,1,-1,-1)
    ysgn = (1,-1,-1,1)
    for j in range(corr_dim):
        for i in range(corr_dim):
            corr[j,i] = [np.sum(tmp_data[yy,xx] * tmp_data[yy+t*j,xx+k*i]) if i**2+j**2 <= max_lag**2 else 0 for k,t in zip(xsgn,ysgn) ]
    return np.asarray(corr)

def step1(step: int):
    xsgn = (1,1,-1,-1)
    ysgn = (1,-1,-1,1)
    j,i = step 
    if i**2+j**2 <= g_max_lag**2:
        return [np.sum(g_tmp_data[g_yy,g_xx] * g_tmp_data[g_yy+t*j,g_xx+k*i]) for k,t in zip(xsgn,ysgn)]
    else:
        return [0,0,0,0]


def parallel_compute(data: np.ndarray, mask_ends: tuple[tuple[int,int], tuple[int,int]], processes: int = cpu_count()-1):
    global g_tmp_data, g_max_lag
    g_tmp_data = np.copy(data) - np.mean(data)
    ydim, xdim = data.shape
    (xo, xe), (yo, ye) = mask_ends
    m_ydim = ye-yo 
    m_xdim = xe-xo
    g_max_lag = min(xo,yo,xdim-xe,ydim-ye,int(np.sqrt(m_xdim**2+m_ydim**2)))
    corr_dim = g_max_lag+1
    global g_xx, g_yy
    g_xx, g_yy = np.meshgrid(np.arange(m_xdim),np.arange(m_ydim))
    g_xx += xo
    g_yy += yo
    positions = np.asarray(np.meshgrid(np.arange(corr_dim),np.arange(corr_dim))).T.reshape(corr_dim*corr_dim,2)
    with Pool(processes=processes) as pool:
        corr_i = pool.map(step1, positions)
    return np.asarray(corr_i).reshape(corr_dim,corr_dim,4)


if __name__ == '__main__':
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
    print('FRAME: ',(dim,dim))
    print('METHOD: ',args.method.upper())


    if args.method == 'sequence':    
        start_time = time()
        corr = compute(data,mask_ends=(xlim,ylim))
        print(corr.shape)
        corr_dim = corr.shape[0] 
        tot_corr = np.zeros((2*corr_dim-1,2*corr_dim-1))
        tot_corr[corr_dim-1:,corr_dim-1:] = corr[:,:,0]
        tot_corr[:corr_dim-1,corr_dim:] = corr[::-1,:,1][:-1,1:]
        tot_corr[:corr_dim,:corr_dim] = corr[::-1,::-1,2]
        tot_corr[corr_dim:,:corr_dim-1] = corr[:,::-1,3][1:,:-1]
        end_time = time()
        print('\nCOMPUTATIONAL TIME: ',(end_time-start_time)/60,' m')
        print(tot_corr[corr_dim-1,corr_dim-1],np.max(tot_corr))
        # print(data[0])

    ## PARALLEL
    elif args.method == 'parallel':    
        start_time = time()
        corr = parallel_compute(data,mask_ends=(xlim,ylim),processes=args.cores)
        print('After paral:', corr.shape)
        corr_dim = corr.shape[0] 
        # print('CORR\n',corr[:,0,:])
        tot_corr = np.zeros((2*corr_dim-1,2*corr_dim-1))
        tot_corr[corr_dim-1:,corr_dim-1:] = corr[:,:,0]
        tot_corr[:corr_dim-1,corr_dim:] = corr[::-1,:,1][:-1,1:]
        tot_corr[:corr_dim,:corr_dim] = corr[::-1,::-1,2]
        tot_corr[corr_dim:,:corr_dim-1] = corr[:,::-1,3][1:,:-1]
        end_time = time()
        print('\nCOMPUTATIONAL TIME: ',(end_time-start_time)/60,' m')
        print(tot_corr[corr_dim-1,corr_dim-1],np.max(tot_corr))
    print('\n'+'======'*10+'\n') 
    
    plt.figure()
    plt.imshow(data,origin='lower')
    plt.plot([max_lag,max_lag,xdim-max_lag,xdim-max_lag,max_lag],[max_lag,ydim-max_lag,ydim-max_lag,max_lag,max_lag],color='red')
    plt.figure()
    tot_corr[corr_dim-1,corr_dim-1] = 0
    plt.imshow(tot_corr,origin='lower')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    tot_corr = filpy.asym_tpcf(data,(xlim,ylim),result='cum',zero_cover=True)
    img = ax.imshow(tot_corr,origin='lower')
    fig.colorbar(img,ax=ax)
    circle_num = 5
    # radii = np.arange(5,max_lag+(max_lag%circle_num),(max_lag+1)//circle_num)
    radii = np.arange(25,max_lag,25)
    centre = (max_lag,max_lag)
    for r in radii:
        circle = plt.Circle(centre,r,color='white',fill=False,linestyle='dashed')
        ax.add_patch(circle)
        ax.annotate(f'{r:.0f}',(centre[0],centre[0]),(centre[0]+int(r/np.sqrt(2))+3,centre[0]+int(r/np.sqrt(2))+3),color='white')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sf = filpy.asym_sf(data,(xlim,ylim),result='div')
    tot_sf = filpy.combine_results(sf)
    img = ax.imshow(tot_sf,origin='lower',norm='log',vmin=tot_sf[tot_sf!=0].min())
    fig.colorbar(img,ax=ax,extend='min')
    circle_num = 5
    radii = np.arange(5,max_lag+(max_lag%circle_num),(max_lag+1)//circle_num)
    centre = (max_lag,max_lag)
    for r in radii:
        circle = plt.Circle(centre,r,color='white',fill=False,linestyle='dashed')
        ax.add_patch(circle)
        ax.annotate(f'{r:.0f}',(centre[0],centre[0]),(centre[0]+int(r/np.sqrt(2))+3,centre[0]+int(r/np.sqrt(2))+3),color='white')

    plt.figure()
    dists, iso_stfc = filpy.convolve_result(sf)

    plt.plot(dists,iso_stfc,'.--')

    plt.show()
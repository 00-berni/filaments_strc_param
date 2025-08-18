import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft, fft2, ifft2
from time import time
import argparse
from .test_func import *
from .test_func import distance
import math
import warnings

# build the logger
logger_name = __name__ 
logger = logging.getLogger(logger_name)
logger.setLevel('DEBUG')

## DATA
FILE_NAME = filpy.FileVar(__file__,path=True)   #: path of the file

def single_dist(field: np.ndarray, dist: float, all_pos: np.ndarray, debug: bool = True) -> float:
    positions = lambda index : all_pos[:,index:][:,np.argwhere(distance(all_pos[:,index],all_pos[:,index:]) == dist)]    
    if debug:
        start = time()
    corr = np.sum([np.sum(field[*all_pos[:,N]] * field[*positions(N)])  for N in range(all_pos.shape[1]-1)])
    if debug:
        end = time()
        logger.debug(f'corr : compilation time: {end-start} s')
    return corr

def integer_correlation(field: np.ndarray, distances: ArrayLike, precision=14) -> np.ndarray:
    distances = np.asarray(distances)
    distances = distances.astype(int)
    xdim, ydim = field.shape
    x = np.arange(xdim)
    y = np.arange(ydim)
    yy, xx = np.meshgrid(y,x)
    logger.debug('Compute pxs')
    start = time()
    pxs = np.array([np.sqrt(np.round(d**2,decimals=precision)-np.arange(1,d)**2) for d in distances], dtype='object')
    old_pxs = np.copy(pxs)
    # logger.debug(f'NO good pxs\n{pxs}')
    pxs = np.array([p[np.mod(p,1) == 0].astype(int) for p in pxs],dtype='object')
    end = time()
    logger.debug(f'px : compilation time: {(end-start)} s')
    pos = np.array([len(p)!=0 for p in pxs])
    s_log = ''
    for op, p, d in zip(old_pxs[pos],pxs[pos],distances[pos]):
        if not np.all(np.isclose([d]*len(p),np.sqrt(p**2+p[::-1]**2),rtol=1e-8)):
            s_log = s_log + f'\n{d}:\t{np.sqrt(p**2+p[::-1]**2)}\t{p} ->\n\t{op[-1]} - {np.round(op[-1],decimals=13)}'
    if s_log == '':
        logger.debug('APPROX OK')
    else:
        logger.debug('APPROX BAD')
        logger.debug('All pos'+s_log)

    logger.info('Compute the correlation')
    start = time()
    try:
        correlations = np.array([np.sum([ 
                                        np.sum(field[x[:-d],:] * field[x[:-d]+d,:]) +    
                                        np.sum(field[:,y[:-d]] * field[:,y[:-d]+d]) +
                                        np.sum([np.sum(
                                                field[xx[:-i,:-j],yy[:-i,:-j]]*field[xx[:-i,:-j]+i,yy[:-i,:-j]+j] + 
                                                field[xx[i:,:-j],yy[i:,:-j]]*field[xx[i:,:-j]-i,yy[i:,:-j]+j])
                                                for i,j in zip(pxs[idx],pxs[idx][::-1]) if len(pxs[idx]) != 0 ] )
                                        ]) for d, idx in zip(distances,range(len(distances)))])
    except:
        logger.debug(f'NO good pxs\n{pxs}')
        logger.debug(f'Result -> {len(pxs[0]) != 0}')
        logger.debug(f'Dist {distances}')
        raise
    end = time()
    logger.debug(f'corr : compilation time: {(end-start)/60:.3f} m')
    return correlations

def irrational_correlation(field: np.ndarray, distances: ArrayLike, precision: int = 14) -> np.ndarray:
    distances = np.asarray(distances)
    logger.debug('Run Irrational')
    logger.debug(f'Precision {precision}')
    xdim, ydim = field.shape
    x = np.arange(xdim)
    y = np.arange(ydim)
    yy, xx = np.meshgrid(y,x)
    logger.debug('Compute pxs')
    start = time()
    pxs = np.array([np.sqrt(np.round(d**2,decimals=precision)-np.arange(1,d)**2) for d in distances], dtype='object')
    # logger.debug(f'PXS: {np.round(pxs[3][-1],decimals=14)}')
    # logger.debug(f'PXS: {pxs[3][-1]}')
    s_log = ''
    for p, d in zip(pxs,distances):
        s_log = s_log + f'\n[{d}] - {d**2} - pxs\n\t{p} ---> {p[-1]}'
    # logger.debug('NO good pxs'+s_log)
    old_pxs = np.copy(pxs)
    pxs = np.array([p[np.mod(p,1) == 0].astype(int) for p in pxs],dtype='object')
    # logger.debug(f'NO good pxs\n{pxs}')
    end = time()
    logger.debug(f'px : compilation time: {(end-start)} s')
    logger.info('Compute the correlation')
    correlations = np.zeros(len(distances))
    pos = np.array([len(p)!=0 for p in pxs])

    s_log = ''
    for op, p, d in zip(old_pxs[pos],pxs[pos],distances[pos]):
        if not np.all(np.isclose([d]*len(p),np.sqrt(p**2+p[::-1]**2),rtol=1e-8)):
            s_log = s_log + f'\n{d}:\t{np.sqrt(p**2+p[::-1]**2)}\t{p} ->\n\t{op[-1]} - {np.round(op[-1],decimals=13)}'
    if s_log == '':
        logger.debug('APPROX OK')
    else:
        logger.debug('APPROX BAD')
        logger.debug('All pos'+s_log)

    if np.any(pos):
        try:
            correlations[pos] = np.array([np.sum([
                                                   np.sum(
                                                          field[xx[:-i,:-j],yy[:-i,:-j]] * field[xx[:-i,:-j]+i,yy[:-i,:-j]+j] + 
                                                          field[xx[i:,:-j] ,yy[i:,:-j]]  * field[xx[i:,:-j]-i,yy[i:,:-j]+j]
                                                          )
                                                  for i,j in zip(p,p[::-1]) 
                                                 ]) 
                                          for p in pxs[pos]
                                         ]) 
        except:
            logger.debug(f'Dist : {distances}')
            logger.debug(f'Positions :\n{pxs[pos]}')
            for i,j in zip(pxs[pos],pxs[pos][::-1]):
                logger.debug(f'P: ({i},{j})')
            raise
    return correlations

def test(field: np.ndarray, bins: int | float | np.ndarray | None = None, no_zero: bool = False, precision: int = 10, display_plot: bool = True) -> None:
    usage_start = ram_usage()
    tracemalloc.start()
    logger.info('Call the function `TEST`')
    logger.debug('Copy the data and remove the mean')
    field = np.copy(field) - field.mean()
    logger.debug('Compute all positions in the grid')
    if isinstance(bins,(float,int)):
        if bins == 0:
            corr = (field**2).sum()
        elif bins.is_integer():
            corr = integer_correlation(field,[bins],precision=precision)[0]
        else:
            corr = irrational_correlation(field,[bins],precision=precision)[0]
    else:
        bins = np.asarray(bins)
        bins = bins[bins != 0]
        corr = np.zeros(bins.size)
        int_pos = np.mod(bins,1) == 0
        logger.info('Check integers')
        start = time()
        if np.any(int_pos):
            logger.debug('Interger are present')
            corr[int_pos] = integer_correlation(field,bins[int_pos],precision=precision)
        if np.any(~int_pos):
            logger.info('Run the routine')
            corr[~int_pos] = irrational_correlation(field=field,distances=bins[~int_pos],precision=precision)
        end = time()
        logger.info(f'corr : compilation time: {(end-start)/60} m')
        if not no_zero:
            corr = np.append([(field**2).sum()],corr)
            bins = np.append([0],bins)
        # logger.debug(f'Positions:\n{pos}')
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot,logger=logger)
    usage_end = ram_usage()
    logger.info(f'Ram Usage {(usage_end -usage_start)/1024**3} Gb')
    logger.info('END')
    if display_plot and not isinstance(bins,(float,int)):
        filpy.quickplot((bins,corr),fmt='--.')
        plt.axhline(0,linestyle='dashed',color='black',alpha=0.5)
        div = int(field.shape[0]//PARAM)
        if field.shape[0]%PARAM != 0: div += 1 
        for i in range(div):
            for j in range(i,div):
                d = distance((0,0),(i,j))*PARAM
                plt.axvline(d,color='red',linestyle='dotted')
                plt.annotate(f'({i},{j})',(d,corr.max()),(d+0.02,corr.max()))
        plt.show()
    return corr

def factorization(n):
    """Source: https://stackoverflow.com/questions/32871539/can-this-integer-factorization-in-python-be-improved

    Parameters
    ----------
    n : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    factors = []

    def get_factor(n):
        x_fixed = 2            
        cycle_size = 2
        x = 2
        factor = 1
        while factor == 1:
            for _ in range(cycle_size):
                if factor > 1: break
                x = (x * x + 1) % n
                factor = math.gcd(x - x_fixed, n)

            cycle_size *= 2
            x_fixed = x
        return factor

    while n > 1:
        next = get_factor(n)
        factors.append(next)
        n //= next

    return factors

def compute_correlation(field: np.ndarray, diagonal_dist: bool = True, display_plot: bool = True) -> tuple[np.ndarray, np.ndarray]:
    usage_start = ram_usage()
    tracemalloc.start()
    logger.info('Call the function `compute_correlation`')
    logger.info(f'`diagonal_dist` parameter set to {diagonal_dist}')
    logger.debug('Copy the data and remove the mean')
    field = np.copy(field) - field.mean()
    logger.debug('Compute all positions in the grid')
    xdim, ydim = field.shape
    start = time()
    xx, yy = np.meshgrid(np.arange(xdim),np.arange(ydim))
    all_pos = np.array([xx.flatten(),yy.flatten()])
    del xx,yy
    end = time()
    logger.debug(f'pos : compilation time: {end-start} s')

    if diagonal_dist:

        logger.info('Compute where')
        

        # snap_dist = tracemalloc.take_snapshot()
        logger.debug('Compute all distances in the grid')
        start = time()
        prova = np.array([distance(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])], dtype='object')
        end = time()
        logger.debug(f'prova : compilation time: {end-start} s')

        start = time()
        all_dist = np.concatenate([distance(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])])
        end = time()
        logger.debug(f'dist : compilation time: {end-start} s')
        logger.info('Compute the array with the unique distances')
        unq_dist = np.unique(all_dist) if diagonal_dist else np.unique(all_dist.astype(int))
        # snap_dist1 = tracemalloc.take_snapshot()
        # top_stats = snap_dist1.compare_to(snap_dist, 'lineno')
        # logger.debug("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     logger.debug(stat)


        logger.debug('Compute each element')
        start = time()
        prova1 = np.array([field[*all_pos[:,N]] * field[*all_pos[:,N:]] for N in range(all_pos.shape[1])],dtype='object')
        end = time()
        logger.debug(f'prova1 : compilation time: {end-start} s')
        logger.debug(f'prova1_shape = {prova1.shape}')
        start = time()
        all_elem = np.concatenate([field[*all_pos[:,N]] * field[*all_pos[:,N:]] for N in range(all_pos.shape[1])])
        end = time()
        logger.debug(f'elem : compilation time: {end-start} s')
        logger.debug(f'all_elm_shape = {all_elem.shape}')

        # snapshot1 = tracemalloc.take_snapshot()

        logger.info('Compute the correlation')
        start = time()
        correlations = np.array([np.sum(all_elem[all_dist == d]) for d in unq_dist])
        end = time()
        logger.info(f'corr : compilation time: {(end-start)/60:.3f} m')
    
    else:
        xdim, ydim = field.shape
        logger.info('Compute the array with the unique distances')
        unq_dist = np.arange(max(*field.shape))

        logger.info('Compute the correlation')
        start = time()
        correlations = np.array([np.sum([ 
                                         np.sum(field[nx,:] * field[nx+d,:]) + 
                                         np.sum(field[:,ny] * field[:,ny+d]) 
                                         for nx,ny in zip(range(int(xdim-d)),range(int(ydim-d)))]) for d in unq_dist[1:]])
        correlations = np.append([(field**2).sum()],correlations)
        end = time()
        logger.info(f'corr : compilation time: {(end-start)/60:.3f} m')
    logger.info(f'corr size = {correlations.size}')
    snapshot2 = tracemalloc.take_snapshot()
    display_top(snapshot2,logger=logger)
    # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    # logger.debug("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     logger.debug(stat)
    if display_plot:
        filpy.quickplot((unq_dist,correlations),fmt='.-')
        plt.axhline(0,linestyle='dashed',color='black',alpha=0.5)
        div = int(field.shape[0]//PARAM)
        if field.shape[0]%PARAM != 0: div += 1 
        if diagonal_dist:
            for i in range(div):
                for j in range(i,div):
                    d = distance((0,0),(i,j))*PARAM
                    plt.axvline(d,color='red',linestyle='dotted')
                    plt.annotate(f'({i},{j})',(d,correlations.max()),(d+0.02,correlations.max()))
        else:
            for i in range(div):
                d = i*PARAM
                plt.axvline(d,color='red',linestyle='dotted')
        plt.show()
    usage_end = ram_usage()
    logger.info(f'Ram Usage {(usage_end -usage_start)/1024**3} Gb')
    logger.info('END')
    return unq_dist, correlations


def diagonal_values(field: np.ndarray, dist: float, n_px: tuple[int,int], logger: logging.Logger, precision: int = 15) -> float:
    xdim, ydim = field.shape
    x, y = n_px
    logger.debug(f'Pos ({x},{y})')
    # logger.debug('Compute (i,j)')
    # start = time()
    logger.debug(f'sq dist = {dist**2}')
    edges = (np.ceil(dist)-1, np.floor(dist))
    i = np.arange(-min(x,edges[0]),min(xdim-x, edges[1]))
    i = i[i!=0]
    j = np.round(np.sqrt(dist**2 - i**2),decimals=precision)
    logger.debug(f'initial elements\n{i}\t{j}')
    pos = np.logical_and(np.mod(j,1) == 0, j <= ydim-1 - y)
    # logger.debug(f'i cond {np.mod(j,1) == 0}')
    # logger.debug(f'ii cond {j <= ydim-1 - y}')
    # logger.debug(f'pos\n{pos}')
    # end = time()
    # logger.debug(f'i,j : computational time {end-start} s')
    if np.any(pos):        
        i = i[pos].astype(int) 
        j = j[pos].astype(int)
        logger.debug(f'Post elements\n{i}\t{j}')
        corr_xy = np.sum(field[x,y]*field[x+i,y+j])
    else:
        corr_xy = 0
    return corr_xy


    
def new_corr(field: np.ndarray, dist: float, logger: logging.Logger,precision: int = 15) -> float:
    """Compute correlation for a certain lag

    Parameters
    ----------
    field : np.ndarray
        data from which mean was subtracted
    dist : float
        lag between pixels
    logger : logging.Logger
        logger
    precision : int, optional
        set the precision of each real number computation, by default 15

    Returns
    -------
    corr : float
        computed tpcf
    """
    logger.debug(f'Run new_corr function for dist {dist}')
    xdim, ydim = field.shape    #: size of the field
    if (xdim**2+ydim**2) < dist: 
        warnings.warn('Out')
        corr = 0
    elif dist == 0:
        corr = (field**2).sum()
    elif dist.is_integer():
        dist = int(dist)
        logger.debug('Compute the array with the unique distances')
        logger.debug('Compute the correlation')
        start = time()
        corr = 0
        if dist < min(xdim,ydim):
            logger.debug('Interger computation')
            logger.debug(f'{xdim-dist} - {ydim-dist}')
            corr += np.sum([ 
                        np.sum(field[nx,:] * field[nx+dist,:]) + 
                        np.sum(field[:,ny] * field[:,ny+dist]) 
                        for nx,ny in zip(range(xdim-dist),range(ydim-dist))])                
        if dist > 3:
            elems, counts = np.unique(factorization(dist), return_counts=True)
            square_cond = np.logical_and(elems%4 == 3, counts%2 != 0)
            if not np.all(square_cond):
                logger.debug('Perfect squares')
                x = np.arange(xdim)
                y = np.arange(ydim)
                logger.debug('Compute all the positions')
                x, y = np.meshgrid(x,y)
                logger.debug('Start the routine to compute correlation')
                corr += np.sum([ diagonal_values(field,dist,(i,j),logger,precision=precision) for i,j in zip(x.flatten(),y.flatten())])
        end = time()
        logger.debug(f'corr : compilation time: {(end-start)/60:.3f} m')
    else:
        xdim, ydim = field.shape
        x = np.arange(xdim)
        y = np.arange(ydim)
        logger.debug('Compute all the positions')
        x, y = np.meshgrid(x,y)
        logger.debug('Start the routine to compute correlation')
        # start = time()
        corr = np.sum([ diagonal_values(field,dist,(i,j),logger,precision=precision) for i,j in zip(x.flatten(),y.flatten())])
        # end = time()
        # logger.info(f'Computational time {(end-start)/60} m')
    logger.debug('END')
    return corr

## PIPELINE
if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",help='set log',nargs='*',type=str,action="store",choices=['file','bash','all','DEBUG', 'INFO'],default=None)
    parser.add_argument("test",help='selected test',type=str,choices=['lattice','random','compare','test'],default='lattice')
    parser.add_argument("--no-diag", help='compute horizontal and vertical only', action='store_false')
    parser.add_argument("-m","--mode", help='mode of the log',type=str, action='store',default='w')
    parser.add_argument("-d","--dim", help='field size',type=int, action='store',default=32)
    parser.add_argument("-s","--seed", help='set seed', type=int, action='store',default=10)
    parser.add_argument("-e","--edges", help='set max and min of noise', type=float, required=False, nargs=2, action='store',default=[0,2])
    parser.add_argument("-l","--lag", help='set the lag of the lattice', required=False, type=int, action='store',default=5)
    parser.add_argument("-v","--value", help='set the value of the lattice', required=False, type=int, action='store',default=1)
    parser.add_argument("-i","--iter", help='set the number of iterations',required=False, type=int, action='store',default=10)
    parser.add_argument("-p","--plot",help='plot data or not',action='store_false')
    parser.add_argument("--method",help='the function type',action='store',type=str, choices=['old','new'], default='new')
    parser.add_argument("--dist",help='distances',action="store",type=str,default="{'dist': [0,5,10]}")

    args = parser.parse_args()
    if args.log is not None:
        log = args.log[0] if len(args.log) != 0 else 'all'
        if log in ['all','file']:
            ch_f = logging.FileHandler(filename=filpy.log_path(FILE_NAME), mode=args.mode)
            frm_f = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
            if len(args.log) == 2:
                ch_f.setLevel(args.log[1])
            ch_f.setFormatter(frm_f)
            logger.addHandler(ch_f)
        if log in ['all','bash']:
            ch_e = logging.StreamHandler()
            frm_e = logging.Formatter('%(levelname)s: %(message)s')
            ch_e.setFormatter(frm_e)
            if len(args.log) == 2:
                ch_e.setLevel(args.log[1])
            else:
                ch_e.setLevel('INFO')
            logger.addHandler(ch_e)

    dim = args.dim    #: size of the field 
    logger.info(f'size = ({dim},{dim})')
    if args.test == 'lattice':
        # build the field
        np.random.seed(args.seed)
        data = np.random.uniform(*args.edges,size=(dim,dim))    #: random signal
        PARAM = args.lag                                        #: lag of the lattice
        data[::PARAM,::PARAM] += args.value
        # display the field
        filpy.show_image(data,cmap='viridis')
        print(args.method)
        if args.method == 'old':
            _ = compute_correlation(data,args.no_diag,args.plot)
        elif args.method == 'new':
             
            d_dict = eval(args.dist)
            if 'dist' in d_dict.keys():
                dists = np.sort(d_dict['dist'])
            else:
                if not 'down' in d_dict.keys():
                    d_dict['down'] = 0
                if not 'top' in d_dict.keys():
                    d_dict['top'] = 30
                if not 'len' in d_dict.keys():
                    d_dict['len'] = 50
                values = (d_dict['down'],d_dict['top'],d_dict['len'])           
                dists = np.linspace(*values)
            tracemalloc.start()
            field = np.copy(data) - data.mean()
            start = time()
            corr = np.array([new_corr(field,d,logger) for d in dists] )
            end = time()
            logger.info(f'Computational time {(end-start)/60} m')
            if args.plot:
                filpy.quickplot((dists,corr),fmt='.--')
                plt.axhline(0,linestyle='dashed',color='black',alpha=0.5)
                div = int(field.shape[0]//PARAM)
                if field.shape[0]%PARAM != 0: div += 1 
                for i in range(div):
                    d = i*PARAM
                    plt.axvline(d,color='red',linestyle='dotted')
                plt.show()
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot,logger=logger)
            usage_end = ram_usage()

    ## COMPARE TWO METHODS
    elif args.test == 'compare':
        # build the field
        np.random.seed(args.seed)
        data = np.random.uniform(*args.edges,size=(dim,dim))  #: random signal
        PARAM = args.lag                           #: lag of the lattice
        data[::PARAM,::PARAM] += args.value
        # display the field
        filpy.show_image(data,cmap='viridis')
        old_dist, old_tpcf = compute_correlation(data,args.no_diag,display_plot=False)
        field = np.copy(data) - data.mean()
        tracemalloc.start()
        start = time()
        usage_start = ram_usage()
        new_tpcf = np.array([new_corr(field,d,logger) for d in np.arange(dim)])
        end = time()
        logger.info(f'Computational time {(end-start)/60} m')
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot,logger=logger)
        usage_end = ram_usage()
        logger.info(f'Ram Usage {(usage_end -usage_start)/1024**3} Gb')
        fig, ax = plt.subplots(1,1)
        ax.plot(old_dist,old_tpcf,'--.',color='blue') 
        ax.plot(np.arange(dim),new_tpcf,'--+',color='red' ) 
        plt.show() 

    ## TEST
    elif args.test == 'test':
        # build the field
        np.random.seed(args.seed)
        data = np.random.uniform(*args.edges,size=(dim,dim))*0    #: random signal
        PARAM = args.lag                                        #: lag of the lattice
        data[::PARAM,::PARAM] += args.value
        # display the field
        filpy.show_image(data,cmap='viridis')
        # start = time()
        # old_dist, old_tpcf = compute_correlation(data,args.no_diag,display_plot=False)
        # end = time()
        # logger.info(f'Computational time {(end-start)/60} m')
        # new_dist = old_dist
        new_dist = np.unique(np.concatenate([np.sqrt(np.arange(i,dim)**2+i**2) for i in range(dim)]))
        start = time()
        new_tpcf = test(data,bins=new_dist,display_plot=args.plot)
        end = time()
        logger.info(f'Computational time {(end-start)/60} m')
        # diff = ~np.isclose(new_tpcf-old_tpcf,0)
        # if np.any(diff):
        #     logger.info('Problems')
        #     logger.debug(f'The problem\n{[*old_dist[diff]]}\n{[i for i in old_tpcf[diff] - new_tpcf[diff]]}')
        # else:
        #     logger.info('Ok!')
        # fig, ax = plt.subplots(1,1)
        # ax.plot(old_dist,old_tpcf,'--.',color='blue') 
        # ax.plot(new_dist,new_tpcf,'--+',color='red' ) 
        # plt.show() 


    ## RANDOM SIGNAL
    elif args.test == 'random':
        corr = 0
        ITER = args.iter
        logger.info('START the ROUTINE')
        if args.method == 'old':
            routine_start = time()
            for i in range(ITER):
                logger.debug(f'ITERATION n. {i:00d}')
                start = time()
                u_d, tmp_c = compute_correlation(np.random.uniform(-1,1,size=(dim,dim)),False,display_plot=False)
                corr += tmp_c
                if i != 0:
                    logger.warning(f'CORR mean = {(corr[1:]/i).mean()}')
                end = time()
                logger.debug(f'Iteration : computational time {end-start} s')
                logger.debug(f'END')
        elif args.method == 'new':
            u_d = np.sort([np.sqrt(i**2+j**2) for i in range(10) for j in range(10)])
            routine_start = time()
            for i in range(ITER):
                logger.info(f'ITERATION n. {i:02d}')
                start = time()
                tmp_c = np.array([new_corr(np.random.uniform(-1,1,size=(dim,dim)),dist=d,logger=logger) for d in u_d])
                corr += tmp_c
                if i != 0:
                    logger.warning(f'CORR mean = {(corr[1:]/i).mean()}')
                end = time()
                logger.debug(f'Iteration : computational time {end-start} s')
                logger.debug(f'END')
        routine_end = time()
        logger.info(f'Computational time routine : {(routine_end-routine_start)/60} m')
        logger.info('END ROUTINE')
        corr /= ITER
        filpy.quickplot((u_d,corr),fmt='.-')
        plt.axhline(0,linestyle='dashed',color='black',alpha=0.5)
        plt.show()

## FFT
# logger.info('FFT PIPELINE')
# logger.debug('Copy the data and remove the mean')
# field = np.copy(data) - data.mean()

# logger.debug('Compute the fft2')
# field_fft = fft2(field)
# logger.debug('Compute the product')
# tmp = field_fft * np.conjugate(field_fft)
# logger.debug('Compute the ifft2')
# tmp_img = np.abs(ifft2(np.abs(tmp)))
# logger.debug('Delete the tmp variable')
# del tmp

# logger.info('Rearrange the field')
# MID = dim // 2
# logger.debug(f'MID value = {MID}')
# logger.debug(f'Image shape = {tmp_img.shape}{MID}')
# corr_img = np.copy(tmp_img)
# corr_img[None:MID  , None:MID ] = tmp_img[MID :None , MID :None]
# corr_img[MID :None , MID :None] = tmp_img[None:MID  , None:MID ]
# corr_img[None:MID  , MID :None] = tmp_img[MID :None , None:MID ]
# corr_img[MID :None , None:MID ] = tmp_img[None:MID  , MID :None]
# filpy.show_image(tmp_img)
# filpy.show_image(corr_img)

# logger.debug('Compute all distances in the grid')
# start = time()
# distances = np.array([ [distance((i,j),(MID,MID)) for i in range(dim)] for j in range(dim)])
# end = time()
# logger.debug(f'dist : compilation time: {end-start} s')
# logger.info('Compute the array with the unique distances')
# unq_dist = np.unique(distances)

# logger.info('Compute the correlation')
# start = time()
# correlations = np.array([np.sum(corr_img[distances == d]) for d in unq_dist])
# end = time()
# logger.info(f'corr : compilation time: {end-start} s')

# filpy.quickplot((np.arange(corr_img.shape[1])-MID,corr_img[MID]),fmt='.--')
# filpy.quickplot((unq_dist,correlations),fmt='.--')
# div = int(field.shape[0]//PARAM)+1
# for i in range(div):
#     for j in range(i,div):
#         d = distance((0,0),(i,j))*PARAM
#         # if d > np.max(unq_dist):
#         #     break
#         plt.axvline(d,color='red',linestyle='dotted')
#         plt.annotate(f'({i},{j})',(d,correlations.max()),(d+0.02,correlations.max()))
# plt.show()

# # #


# all_pos = [ (i,j) for i in range(data.shape[0]) for j in range(data.shape[1])]
# all_pos = np.asarray(all_pos).T

# start = time()
# res_dist = np.concatenate([distance(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])])
# # res_dist = np.array([list(distance(all_pos[:,N],all_pos[:,N:])) for N in range(all_pos.shape[1])],dtype='object').sum()
# end = time()
# print('Compilation time:', end-start,'s')
# print(type(res_dist),len(res_dist))
# # del res_dist
# # start = time()
# # res_corr = [calc_corr_ij(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])]
# # end = time()
# start1 = time()
# res_corr = np.concatenate([calc_corr_ij(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])])
# end1 = time()
# # print('Compilation time:', end-start,'s')
# print('Compilation time:', end1-start1,'s')


# # exit()
# # distances = np.ravel([r[0] for r in res])
# # correlations = np.ravel([r[1] for r in res])
# print('dist')
# unq_dist = np.unique(res_dist)
# print('dist end')
# correlations1 = np.array([np.sum(res_corr[res_dist == d]) for d in unq_dist])
# print('Compilation time:', end1-start1,'s')

# correlations1 /= correlations1.max()

# filpy.quickplot((unq_dist,correlations1-correlations),fmt='.--')

# plt.show()


# corr = correlate2d(data,data)
# lags = correlation_lags(len(data),len(data))

# filpy.show_image(corr,cmap='viridis',show=True)
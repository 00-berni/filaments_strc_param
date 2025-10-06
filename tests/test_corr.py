import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft, fft2, ifft2
from time import time
import argparse
from .test_func import *
from .test_func import distance

# build the logger
logger_name = __name__ 
logger = logging.getLogger(logger_name)
logger.setLevel('DEBUG')

## DATA
FILE_NAME = filpy.FileVar(__file__,path=True)   #: path of the file


def compute_pxs(distances: np.ndarray, precision: int = 15) -> np.ndarray:
    start = time()
    logger.info('Run `compute_pxs`')
    pxs = np.array([np.sqrt(np.round(d**2,decimals=precision)-np.arange(1,d)**2) for d in distances], dtype='object')
    # tmp_pxs = pxs[pos]
    tmp_dst = distances
    tmp_pxs = np.array([p[np.mod(p,1) == 0].astype(int) for p in pxs],dtype='object')
    rtol = 10**(-precision)
    check_pos = np.array([ ~np.all(np.isclose(d-np.sqrt(p**2+p[::-1]**2),np.zeros(len(p)),rtol=rtol)) for p,d in zip(tmp_pxs,tmp_dst)],dtype=bool)
    s_deb = ''
    for p,d in zip(tmp_pxs,tmp_dst):
        s_deb = s_deb + f'\n\t[{d}] - {d-np.sqrt(p**2+p[::-1]**2)}\n{p}'
    logger.debug(f'PXS'+s_deb)
    logger.debug(f'Precision -> {precision}')
    logger.debug(f'CHECK POS\n{check_pos}')
    logger.info('Check distances')
    while np.any(check_pos):
        logger.debug('Start routine')
        logger.debug(f'CHECK LEN : {len(check_pos[check_pos])}')
        logger.debug(f'CHECK SHAPE : {check_pos.shape}')
        logger.debug(f'\n\tCHECK POS\n{check_pos}')
        precision -= 1
        if precision < 0:
            raise ValueError('Precision cannot be negative')
        tmp_pxs = np.array([np.sqrt(np.round(d**2,decimals=precision)-np.arange(1,d)**2) for d in tmp_dst], dtype='object')
        logger.debug(f'TMP -> {tmp_pxs[check_pos].shape}')
        logger.debug(f'TMP\n{tmp_pxs[check_pos]}')
        prova = np.array([p[np.mod(p,1) == 0].astype(int) for p in tmp_pxs],dtype='object')
        logger.debug(f'PROVA {prova.shape}\n{prova}')
        tmp_pxs = prova
        rtol = 10**(-precision)
        check_pos[check_pos] = np.array([ ~np.all(np.isclose(d-np.sqrt(p**2+p[::-1]**2),np.zeros(len(p)),rtol=rtol)) for p,d in zip(tmp_pxs[check_pos],tmp_dst[check_pos])],dtype=bool)
    logger.debug('Check end')
    logger.debug(f'CHECK POS\n{check_pos}')
    logger.info(f'Precision -> {precision}')
    pxs = tmp_pxs
    end = time()
    logger.debug(f'px : compilation time: {(end-start)/60} m')
    return pxs        

def integer_correlation(field: np.ndarray, distances: ArrayLike, precision=14) -> np.ndarray:
    distances = np.asarray(distances)
    distances = distances.astype(int)
    xdim, ydim = field.shape
    x = np.arange(xdim)
    y = np.arange(ydim)
    yy, xx = np.meshgrid(y,x)
    logger.debug('Compute pxs')
    # start = time()
    # pxs = np.array([np.sqrt(np.round(d**2,decimals=precision)-np.arange(1,d)**2) for d in distances], dtype='object')
    # old_pxs = np.copy(pxs)
    # # logger.debug(f'NO good pxs\n{pxs}')
    # pxs = np.array([p[np.mod(p,1) == 0].astype(int) for p in pxs],dtype='object')
    # end = time()
    # logger.debug(f'px : compilation time: {(end-start)} s')
    # pos = np.array([len(p)!=0 for p in pxs])
    # s_log = ''
    # for op, p, d in zip(old_pxs[pos],pxs[pos],distances[pos]):
    #     if not np.all(np.isclose([d]*len(p),np.sqrt(p**2+p[::-1]**2),rtol=1e-8)):
    #         s_log = s_log + f'\n{d}:\t{np.sqrt(p**2+p[::-1]**2)}\t{p} ->\n\t{op[-1]} - {np.round(op[-1],decimals=13)}'
    # if s_log == '':
    #     logger.debug('APPROX OK')
    # else:
    #     logger.debug('APPROX BAD')
    #     logger.debug('All pos'+s_log)
    pxs = compute_pxs(distances=distances,precision=precision)
    logger.info('Compute the correlation')
    start = time()
    try:
        correlations = np.array([
                                 np.sum(field[x[:-d],:] * field[x[:-d]+d,:]) +    
                                 np.sum(field[:,y[:-d]] * field[:,y[:-d]+d]) +
                                 np.sum([np.sum(
                                                field[xx[:-i,:-j],yy[:-i,:-j]]*field[xx[:-i,:-j]+i,yy[:-i,:-j]+j] + 
                                                field[xx[i:,:-j],yy[i:,:-j]]*field[xx[i:,:-j]-i,yy[i:,:-j]+j]
                                                )
                                         for i,j in zip(pxs[idx],pxs[idx][::-1]) if len(pxs[idx]) != 0 
                                        ])
                                 for d, idx in zip(distances,range(len(distances)))
                                ])
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
    # start = time()
    # pxs = np.array([np.sqrt(np.round(d**2,decimals=precision)-np.arange(1,d)**2) for d in distances], dtype='object')
    # # logger.debug(f'PXS: {np.round(pxs[3][-1],decimals=14)}')
    # # logger.debug(f'PXS: {pxs[3][-1]}')
    # s_log = ''
    # for p, d in zip(pxs,distances):
    #     s_log = s_log + f'\n[{d}] - {d**2} - pxs\n\t{p} ---> {p[-1]}'
    # # logger.debug('NO good pxs'+s_log)
    # old_pxs = np.copy(pxs)
    # pxs = np.array([p[np.mod(p,1) == 0].astype(int) for p in pxs],dtype='object')
    # # logger.debug(f'NO good pxs\n{pxs}')
    # end = time()
    # logger.debug(f'px : compilation time: {(end-start)} s')
    pxs = compute_pxs(distances=distances,precision=precision)
    logger.info('Compute the correlation')
    correlations = np.zeros(len(distances))
    pos = np.array([len(p)!=0 for p in pxs])

    # s_log = ''
    # for op, p, d in zip(old_pxs[pos],pxs[pos],distances[pos]):
    #     if not np.all(np.isclose([d]*len(p),np.sqrt(p**2+p[::-1]**2),rtol=1e-8)):
    #         s_log = s_log + f'\n{d}:\t{np.sqrt(p**2+p[::-1]**2)}\t{p} ->\n\t{op[-1]} - {np.round(op[-1],decimals=13)}'
    # if s_log == '':
    #     logger.debug('APPROX OK')
    # else:
    #     logger.debug('APPROX BAD')
    #     logger.debug('All pos'+s_log)

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

def test(field: np.ndarray, bins: int | float | np.ndarray | None = None, no_zero: bool = False, precision: int = 13, param: int = 5, display_plot: bool = True) -> None:
    usage_start = ram_usage()
    tracemalloc.start()
    logger.info('Call the function `TEST`')
    logger.debug('Copy the data and remove the mean')
    field = np.copy(field) - field.mean()
    logger.debug('Compute all positions in the grid')
    xdim, ydim = field.shape
    if isinstance(bins,(float,int)):
        if bins == 0:
            corr = (field**2).sum()
        elif bins**2 > xdim**2+ydim**2:
            corr = 0
        elif bins.is_integer():
            corr = integer_correlation(field,[bins],precision=precision)[0]
        else:
            corr = irrational_correlation(field,[bins],precision=precision)[0]
    else:
        bins = np.asarray(bins)
        bins = bins[bins != 0]
        corr = np.zeros(bins.size)
        pos = bins**2 <= xdim**2 + ydim**2 
        tmp_bins = bins[pos]
        tmp_corr = corr[pos]
        int_pos = np.mod(tmp_bins,1) == 0
        logger.info('Check integers')
        start = time()
        if np.any(int_pos):
            logger.debug('Interger are present')
            tmp_corr[int_pos] = integer_correlation(field,tmp_bins[int_pos],precision=precision)
        if np.any(~int_pos):
            logger.info('Run the routine')
            tmp_corr[~int_pos] = irrational_correlation(field=field,distances=tmp_bins[~int_pos],precision=precision)
        end = time()
        logger.info(f'corr : compilation time: {(end-start)/60} m')
        bins[pos] = tmp_bins
        corr[pos] = tmp_corr 
        if not np.all(pos): logger.debug(f'{len(pos[~pos])} distances greater than the size of the frame')          
        if not no_zero:
            corr = np.append([(field**2).sum()],corr)
            bins = np.append([0],bins)
            # corr /= corr[0]
        # logger.debug(f'Positions:\n{pos}')
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot,logger=logger)
    usage_end = ram_usage()
    logger.info(f'Ram Usage {(usage_end -usage_start)/1024**3} Gb')
    logger.info('END')
    if display_plot and not isinstance(bins,(float,int)):
        filpy.quickplot((bins,corr),fmt='--.')
        plt.axhline(0,linestyle='dashed',color='black',alpha=0.5)
        div = int(field.shape[0]//param)
        if field.shape[0]%param != 0: div += 1 
        for i in range(div):
            for j in range(i,div):
                d = distance((0,0),(i,j))*param
                plt.axvline(d,color='red',linestyle='dotted')
                plt.annotate(f'({i},{j})',(d,corr.max()),(d+0.02,corr.max()))
        plt.show()
    return corr


def compute_correlation(field: np.ndarray, diagonal_dist: bool = True, param: int = 5, display_plot: bool = True) -> tuple[np.ndarray, np.ndarray]:
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
        div = int(field.shape[0]//param)
        if field.shape[0]%param != 0: div += 1 
        if diagonal_dist:
            for i in range(div):
                for j in range(i,div):
                    d = distance((0,0),(i,j))*param
                    plt.axvline(d,color='red',linestyle='dotted')
                    plt.annotate(f'({i},{j})',(d,correlations.max()),(d+0.02,correlations.max()))
        else:
            for i in range(div):
                d = i*param
                plt.axvline(d,color='red',linestyle='dotted')
        plt.show()
    usage_end = ram_usage()
    logger.info(f'Ram Usage {(usage_end -usage_start)/1024**3} Gb')
    logger.info('END')
    return unq_dist, correlations


def __from_rth_to_xy(radius: ArrayLike, theta: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    x = np.round(radius*np.cos(theta),0).astype(int)
    y = np.round(radius*np.sin(theta),0).astype(int)
    return x,y

def __compute_pos(data_pos: tuple[np.ndarray,np.ndarray], coord: tuple[float,float], mode: Literal['polar','cartesian'] = 'polar') -> tuple[tuple[np.ndarray,np.ndarray],tuple[np.ndarray,np.ndarray]]:
    if mode == 'polar':
        coord = __from_rth_to_xy(*coord)
    i,j = coord
    xx,yy = data_pos
    xdim = np.max(xx)+1
    ydim = np.max(yy)+1
    xpos = np.logical_and(xx+i >=0, xx+i<xdim)
    ypos = np.logical_and(yy+j >=0, yy+j<ydim)
    pos = np.logical_and(xpos,ypos)
    return pos

def tpcf(field: np.ndarray, distances: np.ndarray, mode: Literal['polar','cartesian'] = 'polar') -> ArrayLike:
    usage_start = ram_usage()
    tracemalloc.start()
    logger.info('Call the function `TPCF`')
    logger.debug('Copy the data and remove the mean')
    field = np.copy(field) - field.mean()
    if mode == 'polar':
        distances = __from_rth_to_xy(*distances)
    x, y = distances
    no_zero = (x!=0) + (y!=0)
    x = x[no_zero].astype(int)
    y = y[no_zero].astype(int)
    logger.debug(f'LEN POS : {no_zero.shape}')
    logger.debug(f'LEN X : {x.shape}')
    logger.debug(f'TYPE X : {x.dtype}')
    correlations = np.zeros(x.size)
    logger.debug(f'LEN CORR : {correlations.shape}')
    xdim, ydim = field.shape
    yy, xx = np.meshgrid(np.arange(ydim),np.arange(xdim))
    operation = lambda pos, shift : np.sum(field[xx[pos],yy[pos]] * field[xx[pos]+shift[0],yy[pos]+shift[1]]) if np.any(pos) else 0
    start_time = time()
    correlations = np.array([ operation(__compute_pos((xx,yy),(i,j),mode='cartesian'),(i,j)) for i,j in zip(x,y)]) 
    logger.info(f'time correlation: {(time()-start_time)/60} m')
    correlations = np.append([np.sum(field**2)],correlations)
    snapshot2 = tracemalloc.take_snapshot()
    display_top(snapshot2,logger=logger)
    usage_end = ram_usage()
    logger.info(f'Ram Usage {(usage_end - usage_start)/1024**3} Gb')
    logger.info('END')
    return correlations

def sf(field: np.ndarray, distances: np.ndarray, mode: Literal['polar','cartesian'] = 'polar',order: int = 1) -> ArrayLike:
    usage_start = ram_usage()
    tracemalloc.start()
    logger.info('Call the function `SF`')
    logger.debug('Copy the data and remove the mean')
    if mode == 'polar':
        distances = __from_rth_to_xy(*distances)
    x, y = distances
    no_zero = (x!=0) + (y!=0)
    x = x[no_zero]
    y = y[no_zero]
    logger.debug(f'LEN POS : {no_zero.shape}')
    logger.debug(f'LEN X : {x.shape}')
    structure = np.zeros(x.size)
    logger.debug(f'LEN CORR : {structure.shape}')
    xdim, ydim = field.shape
    yy, xx = np.meshgrid(np.arange(ydim),np.arange(xdim))
    operation = lambda pos, shift : np.sum(np.abs(field[xx[pos],yy[pos]] - field[xx[pos]+shift[0],yy[pos]+shift[1]])**order)
    structure = np.array([ operation(__compute_pos((xx,yy),(i,j),mode='cartesian'),(i,j)) for i,j in zip(x,y)]) 
    structure = np.append([0],structure)
    snapshot2 = tracemalloc.take_snapshot()
    display_top(snapshot2,logger=logger)
    usage_end = ram_usage()
    logger.info(f'Ram Usage {(usage_end - usage_start)/1024**3} Gb')
    logger.info('END')
    return structure


def another_tpcf(field: np.ndarray, distances: np.ndarray, mode: Literal['polar','cartesian'] = 'polar') -> ArrayLike:
    usage_start = ram_usage()
    tracemalloc.start()
    logger.info('Call the function `TPCF`')
    logger.debug('Copy the data and remove the mean')
    field = np.copy(field) - field.mean()
    if mode == 'polar':
        distances = __from_rth_to_xy(*distances)
    x, y = distances
    no_zero = (x!=0) + (y!=0)
    x = x[no_zero].astype(int)
    y = y[no_zero].astype(int)
    logger.debug(f'LEN POS : {no_zero.shape}')
    logger.debug(f'LEN X : {x.shape}')
    logger.debug(f'TYPE X : {x.dtype}')
    correlations = np.zeros(x.size)
    logger.debug(f'LEN CORR : {correlations.shape}')
    xdim, ydim = field.shape
    yy, xx = np.meshgrid(np.arange(ydim),np.arange(xdim))
    start_time = time()
    for k in range(x.size):
        i,j = x[k],y[k]
        pos = __compute_pos((xx,yy),(i,j),mode='cartesian')
        correlations[k] = np.sum(field[xx[pos],yy[pos]] * field[xx[pos]+i,yy[pos]+j]) 
    logger.info(f'time correlation: {(time()-start_time)/60} m')
    correlations = np.append([np.sum(field**2)],correlations)
    snapshot2 = tracemalloc.take_snapshot()
    display_top(snapshot2,logger=logger)
    usage_end = ram_usage()
    logger.info(f'Ram Usage {(usage_end - usage_start)/1024**3} Gb')
    logger.info('END')
    return correlations

def __mapping(data_pos: tuple[np.ndarray,np.ndarray], couples: list[tuple[int,int]]) -> np.ndarray:
    xx, yy = data_pos
    xdim = xx.max()+1
    ydim = yy.max()+1
    pos = np.array([ (xx+i>=0)*(xx+i<xdim)*(yy+j>=0)*(yy+j<ydim) for (i,j) in couples ], dtype='object')
    return pos

def another_tpcf2(field: np.ndarray, distances: np.ndarray, mode: Literal['polar','cartesian'] = 'polar') -> ArrayLike:
    usage_start = ram_usage()
    tracemalloc.start()
    logger.info('Call the function `TPCF`')
    logger.debug('Copy the data and remove the mean')
    field = np.copy(field) - field.mean()
    if mode == 'polar':
        distances = __from_rth_to_xy(*distances)
    x, y = distances
    correlations = np.zeros((*x.shape,4))
    logger.debug(f'LEN CORR : {correlations.shape}')
    xdim, ydim = field.shape
    yy, xx = np.meshgrid(np.arange(ydim),np.arange(xdim))
    start_time = time()
    for ki,kj in np.transpose(np.unravel_index(np.arange(x.size),x.shape)):
        i,j = x[ki,kj],y[ki,kj]
        couples = [(i,j),(i,-j),(-i,-j),(-i,j)]
        pos = __mapping((xx,yy),couples)
        correlations[ki,kj] = [np.sum(field[xx[p.astype(bool)],yy[p.astype(bool)]] * field[xx[p.astype(bool)]+ii,yy[p.astype(bool)]+jj]) for p, (ii,jj) in zip(pos,couples) ] 
    logger.info(f'time correlation: {(time()-start_time)/60} m')
    # correlations = np.append([np.sum(field**2)],correlations)
    snapshot2 = tracemalloc.take_snapshot()
    display_top(snapshot2,logger=logger)
    usage_end = ram_usage()
    logger.info(f'Ram Usage {(usage_end - usage_start)/1024**3} Gb')
    logger.info('END')
    return correlations


## PIPELINE
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
            sequence = [0,1,2,3,3,3,2,2,1,1,0,-1,-2,-1]
            patt = np.array(sequence*(ydim//len(sequence))+sequence[:ydim%len(sequence)])
            pos  = np.arange(ydim)
            
            for i in range(wd):
                xpos = sp*2+patt + i
                idx = xpos<xdim
                xpos = xpos[idx]
                pos = pos[idx]
                new_data[xpos,pos] += 1
        return new_data

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",help='set log',nargs='*',type=str,action="store",choices=['file','bash','all','DEBUG', 'INFO'],default=None)
    parser.add_argument("test",help='selected test',type=str,choices=['lattice','random','compare','test','pattern','time','speed'],default='lattice')
    parser.add_argument("--no-diag", help='compute horizontal and vertical only', action='store_false')
    parser.add_argument("-m","--mode", help='mode of the log',type=str, action='store',default='w')
    parser.add_argument("-o","--order", help='field size',type=int, action='store',default=1)
    parser.add_argument("-d","--dim", help='field size',type=int, action='store',default=32)
    parser.add_argument("-s","--seed", help='set seed', type=int, action='store',default=10)
    parser.add_argument("-e","--edges", help='set max and min of noise', type=float, required=False, nargs=2, action='store',default=[0,2])
    parser.add_argument("-w","--width", help='set the width of the pattern', required=False, type=int, action='store',default=3)
    parser.add_argument("-l","--lag", help='set the lag of the lattice', required=False, type=int, action='store',default=5)
    parser.add_argument("-v","--value", help='set the value of the lattice', required=False, type=int, action='store',default=1)
    parser.add_argument("-i","--iter", help='set the number of iterations',required=False, type=int, action='store',default=10)
    parser.add_argument("-p","--plot",help='plot data or not',action='store_false')
    parser.add_argument("--ticks",help='display pixel edges',action='store_true')
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
    PARAM = args.lag                                        #: lag of the lattice
    if args.test == 'lattice':
        # build the field
        np.random.seed(args.seed)
        data = np.random.uniform(*args.edges,size=(dim,dim))    #: random signal
        data[::PARAM,::PARAM] += args.value
        # display the field
        filpy.show_image(data,cmap='viridis')
        print(args.method)
        if args.method == 'old':
            _ = compute_correlation(data,args.no_diag,PARAM,args.plot)
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
            corr = test(data,dists,param=PARAM)
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
        start = time()
        old_dist, old_tpcf = compute_correlation(data,args.no_diag,PARAM,display_plot=False)
        end = time()
        logger.info(f'Computational time {(end-start)/60} m')
        new_dist = old_dist
        start = time()
        new_tpcf = test(data,bins=new_dist,param=PARAM,display_plot=False)
        end = time()
        logger.info(f'Computational time {(end-start)/60} m')
        diff = ~np.isclose(new_tpcf-old_tpcf,0,rtol=1e-10)
        if np.any(diff):
            logger.info('Problems')
            logger.debug(f'The problem\n{[*old_dist[diff]]}\n{[i for i in old_tpcf[diff] - new_tpcf[diff]]}')
        else:
            logger.info('Ok!')
        start = time()
        xdim,ydim = data.shape
        yy,xx = np.meshgrid(np.arange(-ydim+1,ydim),np.arange(-xdim+1,xdim))
        
        
        # logger.info(f'{xx}\n{yy}')
        new_tpcf2 = tpcf(data,(xx,yy), mode='cartesian')
        new_dist2 = np.sqrt(xx**2+yy**2).flatten()
        directions = np.arctan2(yy,xx).flatten()
        pos = np.where(new_dist2 == 0.)[0][0]
        logger.info(f'{pos}')
        new_dist2[:pos+1]  = np.append(0,new_dist2[:pos])
        directions[:pos+1] = np.append(0,directions[:pos])
        end = time()
        logger.info(f'Computational time {(end-start)/60} m')
        fig, ax = plt.subplots(1,1)
        ax.plot(old_dist,old_tpcf,'--.',color='blue') 
        ax.plot(new_dist,new_tpcf,'--+',color='red' ) 
        ax.plot(new_dist2,new_tpcf2,'x',color='green' ) 
        ax.axhline(0,linestyle='dotted',color='k',alpha=0.7)
        fig2, ax2 = plt.subplots(1,1,subplot_kw={'projection':'polar'})
        img = ax2.scatter(directions[1:],new_dist2[1:],c=new_tpcf2[1:],cmap='seismic')
        fig2.colorbar(img,ax=ax2)
        fig3, ax3 = plt.subplots(1,1)
        pos = (xx.flatten()!=0)+(yy.flatten()!=0)
        img = ax3.scatter(xx.flatten()[pos],yy.flatten()[pos],c=new_tpcf2[1:],cmap='seismic')
        fig3.colorbar(img,ax=ax3)
        plt.show() 

    ## TEST
    elif args.test == 'test':
        # build the field
        np.random.seed(args.seed)
        data = np.random.uniform(*args.edges,size=(dim,dim))*2    #: random signal
        PARAM = args.lag                                        #: lag of the lattice
        data[::PARAM,::PARAM] += args.value
        # display the field
        filpy.show_image(data,cmap='viridis')
        # start = time()
        # old_dist, old_tpcf = compute_correlation(data,args.no_diag,display_plot=False)
        # end = time()
        # logger.info(f'Computational time {(end-start)/60} m')
        # new_dist = old_dist
        lim = dim
        new_dist = np.unique(np.concatenate([np.sqrt(np.arange(i,lim)**2+i**2) for i in range(lim)]))
        start = time()
        new_tpcf = test(data,bins=new_dist,param=PARAM,display_plot=args.plot)
        end = time()
        logger.info(f'Computational time {(end-start)/60} m')
        
#        start = time()
#        strc_fun = filpy.compute_sf(data,bins=new_dist,order=10,display_plot=args.plot)
#        end = time()
#        logger.info(f'Computational time {(end-start)/60} m')
        start = time()
        xdim,ydim = data.shape
        yy,xx = np.meshgrid(np.arange(-ydim+1,ydim),np.arange(-xdim+1,xdim))
        
        
        # logger.info(f'{xx}\n{yy}')
        new_tpcf2 = tpcf(data,(xx,yy), mode='cartesian')
        start_ram = ram_usage()
        tracemalloc.start()
        new_dist2 = np.sqrt(xx**2+yy**2).flatten()
        directions = np.arctan2(yy,xx).flatten()
        pos = np.where(new_dist2 == 0.)[0][0]
        logger.info(f'{pos}')
        new_dist2[:pos+1]  = np.append(0,new_dist2[:pos])
        directions[:pos+1] = np.append(0,directions[:pos])

        dist2 = np.unique(new_dist2)
        corr2 = np.array([np.sum(new_tpcf2[new_dist2==d]) for d in dist2])
        end = time()
        logger.info(f'Computational time {(end-start)/60} m')
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot,logger=logger)
        end_ram = ram_usage()
        logger.info(f'Ram usage {(end_ram-start_ram)/1024**3} Gb')
        if args.plot:
            fig, ax = plt.subplots(1,1)
            ax.plot(new_dist,new_tpcf,'--.',color='red' ) 
            ax.plot(new_dist2,new_tpcf2,'x',color='green',alpha=0.2) 
            ax.plot(dist2,corr2,'+',color='blue' ) 
            ax.axhline(0,linestyle='dotted',color='k',alpha=0.7)
            fig2, ax2 = plt.subplots(1,1,subplot_kw={'projection':'polar'})
            img = ax2.scatter(directions[1:],new_dist2[1:],c=new_tpcf2[1:],cmap='seismic',marker='.')
            fig2.colorbar(img,ax=ax2)
            ax2.set_thetamin(45)
            ax2.set_thetamax(135)
            # fig3, ax3 = plt.subplots(1,1)
            # pos = (xx.flatten()!=0)+(yy.flatten()!=0)
            # img = ax3.scatter(xx.flatten()[pos],yy.flatten()[pos],c=new_tpcf2[1:],cmap='seismic')
            # fig3.colorbar(img,ax=ax3)
            plt.show() 


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
                u_d, tmp_c = compute_correlation(np.random.uniform(-1,1,size=(dim,dim)),False,PARAM,display_plot=False)
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
                tmp_c = test(np.random.uniform(-1,1,size=(dim,dim)),u_d,param=PARAM,display_plot=False)
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

    elif args.test == 'pattern':

        dim = args.dim
        np.random.seed(args.seed)
        data = np.zeros((dim,dim)) + np.random.random((dim,dim))*2       
        width = args.width
        spacing = args.lag
        data = make_pattern(data,'filament',width=width,spacing=spacing)
        # data = make_pattern(data,'lattice',width=width,spacing=spacing)
        fig, ax = filpy.show_image(data,colorbar=False)
        if args.ticks:
            axp = fig.gca()
            axp.set_xticks(np.arange(0,dim,1))
            axp.set_yticks(np.arange(0,dim,1))
            axp.set_xticks(np.arange(-0.5,dim,1),minor=True)
            axp.set_yticks(np.arange(-0.5,dim,1),minor=True)
            axp.grid(which='minor',color='r', linestyle='-', linewidth=2) 
        if args.no_diag:
            dist = np.unique(np.concatenate([np.sqrt(np.arange(i,dim)**2+i**2) for i in range(data.shape[0])]))
        else:
            dist = np.arange(data.shape[0])
        corr = test(data,dist,param=PARAM,display_plot=False)


        ## NEW
        start = time()
        xdim,ydim = data.shape
        yy,xx = np.meshgrid(np.arange(-ydim+1,ydim),np.arange(-xdim+1,xdim))
        
        
        # logger.info(f'{xx}\n{yy}')
        new_tpcf2 = another_tpcf(data,(xx,yy), mode='cartesian')[1:]
        start_ram = ram_usage()
        tracemalloc.start()
        new_dist2 = np.sqrt(xx**2+yy**2).flatten()
        directions = np.arctan2(xx,yy).flatten()
        pos = np.where(new_dist2 == 0.)[0][0]
        logger.info(f'{pos}')
        new_dist2[:pos+1]  = np.append(0,new_dist2[:pos])
        directions[:pos+1] = np.append(0,directions[:pos])
        new_dist2=new_dist2[1:]
        directions=directions[1:]

        plt.figure()
        ccorr = filpy.asym_tpcf(data,result='cum')
        ydim, xdim = ccorr.shape
        xdim = (xdim+1) // 2
        ydim = (ydim+1) // 2
        # ccorr[ydim-1,xdim-1] = 0
        plt.imshow(ccorr,origin='lower')

        plt.figure()
        plt.imshow(filpy.asym_sf(data,result='cum'))


        # stfunc = sf(data,(xx,yy), mode='cartesian',order=args.order)


        dist2 = np.unique(new_dist2)
        corr2 = np.array([np.sum(new_tpcf2[new_dist2==d]) for d in dist2])
        end = time()
        logger.info(f'Computational time {(end-start)/60} m')
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot,logger=logger)
        end_ram = ram_usage()
        logger.info(f'Ram usage {(end_ram-start_ram)/1024**3} Gb')

        plt.figure()
        flat_corr = np.sum(filpy.asym_tpcf(data),axis=2)/4
        ydim, xdim = flat_corr.shape
        xx, yy = np.meshgrid(np.arange(xdim),np.arange(ydim))
        dd = np.sqrt(xx**2+yy**2)
        flat_corr = np.array([np.sum(flat_corr[yy[dd==d],xx[dd==d]])/yy[dd==d].size for d in dist2])
        print(dist2[0],flat_corr[0])
        pos = dist2 <= min(xdim,ydim)//2
        plt.subplot(211)
        plt.plot(dist,corr,'.--')
        plt.subplot(212)
        plt.plot(dist2[pos],flat_corr[pos],'.--')
        plt.figure()
        plt.subplot(211)
        plt.plot(dist2[pos],flat_corr[pos],'x--')
        plt.axhline(0,color='k')
        plt.subplot(212)
        plt.plot(dist2[~pos],flat_corr[~pos],'x--')
        plt.axhline(0,color='k')




        ## NEW 2
        # logger.info('FOR LOOP')
        # start_time = time()
        # tracemalloc.start()
        # xdim,ydim = data.shape
        # yy,xx = np.meshgrid(np.arange(1,ydim),np.arange(1,xdim))     

        # # logger.info(f'{xx}\n{yy}')
        # snapshot = tracemalloc.take_snapshot()
        # display_top(snapshot,logger=logger)
        # start_ram = ram_usage()
        # new_tpcf2 = another_tpcf2(data,(xx,yy), mode='cartesian')
        # end_ram = ram_usage()
        # start_ram = ram_usage()
        # tracemalloc.start()
        # new_dist2 = np.repeat(np.sqrt(xx**2+yy**2).flatten()[1:],4,axis=0)
        # directions = np.array([np.arctan2(i*xx,j*yy).flatten()[1:] for (i,j) in [(1,1),(1,-1),(-1,-1),(-1,1)]])
        # logger.info(f'Ram usage {(end_ram-start_ram)/1024**3} Gb')
        # end_time = time()
        # logger.info(f'Computational time {(end_time-start_time)/60} m')
        # if args.plot:
        #     # fig0, ax0 = plt.subplots(1,1)
        #     # ax0.plot(new_dist2,stfunc,'+',color='orange',alpha=0.2) 
        #     # fig, ax = plt.subplots(1,1)
        #     # ax.plot(dist,corr,'--.',color='red' ) 
        #     # ax.plot(new_dist2,new_tpcf2,'x',color='green',alpha=0.2) 
        #     # ax.plot(dist2,corr2,'+',color='blue' ) 
        #     # ax.axhline(0,linestyle='dotted',color='k',alpha=0.7)
        fig2, ax2 = plt.subplots(1,1,subplot_kw={'projection':'polar'})
        img = ax2.scatter(directions[1:],new_dist2[1:],c=new_tpcf2[1:],cmap='seismic',marker='.')
        # img = ax2.scatter(directions.flatten(),new_dist2.flatten(),c=new_tpcf2.flatten(),cmap='seismic',marker='.')
        fig2.colorbar(img,ax=ax2)
        # ax2.set_theta_zero_location("N")
        #     fig4, ax4 = plt.subplots(1,1,subplot_kw={'projection':'polar'})
        #     img = ax4.scatter(directions[1:],new_dist2[1:],c=stfunc[1:],cmap='seismic',marker='.')
        #     fig4.colorbar(img,ax=ax4)
        #     # ax4.set_theta_zero_location("N")
        #     # ax2.set_thetamin(45)
        #     # ax2.set_thetamax(135)
        #     # fig3, ax3 = plt.subplots(1,1)
        #     # pos = (xx.flatten()!=0)+(yy.flatten()!=0)
        #     # img = ax3.scatter(xx.flatten()[pos],yy.flatten()[pos],c=new_tpcf2[1:],cmap='seismic')
        #     # fig3.colorbar(img,ax=ax3)
        #     # fig  = plt.figure()
        #     # ax01 = fig.add_subplot(1,3,1)
        #     # ax01.set_title('Data')
        #     # img01 = ax01.imshow(data,cmap='gray',origin='lower')
        #     # fig.colorbar(img01,ax=ax01,location='bottom')
        #     # ax02 = fig.add_subplot(1,3,2,projection='polar')
        #     # ax02.set_title('Two-Point Corr. Func.')
        #     # img02 = ax02.scatter(directions[1:],new_dist2[1:],c=new_tpcf2[1:],cmap='seismic',marker='.')
        #     # fig.colorbar(img02,ax=ax02,location='bottom')
        #     # ax03 = fig.add_subplot(1,3,3,projection='polar')
        #     # ax03.set_title('Structure Func.')
        #     # img03 = ax03.scatter(directions[1:],new_dist2[1:],c=stfunc[1:],cmap='seismic',marker='.')
        #     # fig.colorbar(img03,ax=ax03,location='bottom')
        plt.show() 


        # plt.figure()
        # plt.plot(dist,corr,'--.',color='blue')
        # plt.axhline(0,color='black',linestyle='dotted',alpha=0.5)
        # lag = spacing + width
        # for i in np.arange(1,dim//lag):
        #     sub_term = min(width,spacing)
        #     ac_pos = i*lag - sub_term
        #     if ac_pos <= dist.max():
        #         plt.axvline(ac_pos,color='orange',linestyle='dashed')            
        #     c_pos = ac_pos + sub_term
        #     if c_pos <= dist.max():
        #         plt.axvline(c_pos,color='green',linestyle='dashed')

        # plt.grid(color='grey',alpha=0.6,linestyle='dotted')
        # plt.show()
        
        # sf = filpy.compute_sf(data,dist,3)


    elif args.test == 'time':
        dim = args.dim
        np.random.seed(args.seed)
        data = np.zeros((dim,dim)) + np.random.random((dim,dim))*2       
        width = args.width
        spacing = args.lag
        data = make_pattern(data,'lattice',width=width,spacing=spacing)
        
        start_time = time()
        tracemalloc.start()
        xdim,ydim = data.shape
        yy,xx = np.meshgrid(np.arange(-ydim+1,ydim),np.arange(-xdim+1,xdim))     

        # logger.info(f'{xx}\n{yy}')
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot,logger=logger)
        new_tpcf2 = tpcf(data,(xx,yy), mode='cartesian')
        start_ram = ram_usage()
        tracemalloc.start()
        new_dist2 = np.sqrt(xx**2+yy**2).flatten()
        directions = np.arctan2(xx,yy).flatten()
        pos = np.where(new_dist2 == 0.)[0][0]
        logger.info(f'{pos}')
        new_dist2[:pos+1]  = np.append(0,new_dist2[:pos])
        directions[:pos+1] = np.append(0,directions[:pos])

        dist2 = np.unique(new_dist2)
        corr2 = np.array([np.sum(new_tpcf2[new_dist2==d]) for d in dist2])
        end_time = time()
        logger.info(f'Computational time {(end_time-start_time)/60} m')
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot,logger=logger)
        end_ram = ram_usage()
        logger.info(f'Ram usage {(end_ram-start_ram)/1024**3} Gb')

        ## NEW
        logger.info('FOR LOOP')
        start_time = time()
        tracemalloc.start()
        xdim,ydim = data.shape
        yy,xx = np.meshgrid(np.arange(-ydim+1,ydim),np.arange(-xdim+1,xdim))     

        # logger.info(f'{xx}\n{yy}')
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot,logger=logger)
        new_tpcf2 = another_tpcf(data,(xx,yy), mode='cartesian')
        start_ram = ram_usage()
        tracemalloc.start()
        new_dist2 = np.sqrt(xx**2+yy**2).flatten()
        directions = np.arctan2(xx,yy).flatten()
        pos = np.where(new_dist2 == 0.)[0][0]
        logger.info(f'{pos}')
        new_dist2[:pos+1]  = np.append(0,new_dist2[:pos])
        directions[:pos+1] = np.append(0,directions[:pos])

        dist2 = np.unique(new_dist2)
        corr2 = np.array([np.sum(new_tpcf2[new_dist2==d]) for d in dist2])
        end_time = time()
        logger.info(f'Computational time {(end_time-start_time)/60} m')
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot,logger=logger)
        end_ram = ram_usage()
        logger.info(f'Ram usage {(end_ram-start_ram)/1024**3} Gb')

        logger.info('ONLY DIAGONAL')
        start_time = time()
        tracemalloc.start()
        xx = np.delete(np.arange(-xdim+1,xdim),xdim-1)
        yy = np.delete(np.arange(-ydim+1,ydim),ydim-1)
        xx = np.append(xx,np.zeros(ydim*2-2))
        yy = np.append(np.zeros(xdim*2-2),yy)
        logger.debug(f'{xx}')
        logger.debug(f'{yy}')
        # logger.info(f'{xx}\n{yy}')
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot,logger=logger)
        new_tpcf2 = tpcf(data,(xx,yy), mode='cartesian')
        start_ram = ram_usage()
        tracemalloc.start()
        new_dist2 = np.sqrt(xx**2+yy**2).flatten()
        directions = np.arctan2(xx,yy).flatten()
        new_dist2  = np.append(0,new_dist2)
        directions = np.append(0,directions)

        dist2 = np.unique(new_dist2)
        corr2 = np.array([np.sum(new_tpcf2[new_dist2==d]) for d in dist2])
        end_time = time()
        logger.info(f'Computational time {(end_time-start_time)/60} m')
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot,logger=logger)
        end_ram = ram_usage()
        logger.info(f'Ram usage {(end_ram-start_ram)/1024**3} Gb')

        if args.plot:
            fig, ax = plt.subplots(1,1)
            ax.plot(new_dist2,new_tpcf2,'x',color='green',alpha=0.2) 
            ax.plot(dist2,corr2,'+',color='blue' ) 
            ax.axhline(0,linestyle='dotted',color='k',alpha=0.7)
            fig2, ax2 = plt.subplots(1,1,subplot_kw={'projection':'polar'})
            img = ax2.scatter(directions[1:],new_dist2[1:],c=new_tpcf2[1:],cmap='seismic',marker='.')
            fig2.colorbar(img,ax=ax2)
            plt.show() 

    elif args.test == 'speed':
        logger.info('!! SPEED TEST !!')
        times1 = []
        times2 = []
        times3 = []
        rams1 = []
        rams2 = []
        rams3 = []
        dims = [10,50,80]#,120,200,300,500,600]
        for dim in dims:
            logger.info(f'dim .: {dim}')
            np.random.seed(args.seed)
            data = np.zeros((dim,dim)) + np.random.random((dim,dim))*2       
            width = args.width
            spacing = args.lag
            data = make_pattern(data,'lattice',width=width,spacing=spacing)
            
            start_time = time()
            tracemalloc.start()
            xdim,ydim = data.shape
            yy,xx = np.meshgrid(np.arange(-ydim+1,ydim),np.arange(-xdim+1,xdim))     

            # logger.info(f'{xx}\n{yy}')
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot,logger=logger)
            start_ram = ram_usage()
            new_tpcf2 = tpcf(data,(xx,yy), mode='cartesian')
            end_ram = ram_usage()
            rams1 += [end_ram-start_ram]
            start_ram = ram_usage()
            tracemalloc.start()
            new_dist2 = np.sqrt(xx**2+yy**2).flatten()
            directions = np.arctan2(xx,yy).flatten()
            pos = np.where(new_dist2 == 0.)[0][0]
            logger.info(f'{pos}')
            new_dist2[:pos+1]  = np.append(0,new_dist2[:pos])
            directions[:pos+1] = np.append(0,directions[:pos])

            dist2 = np.unique(new_dist2)
            corr2 = np.array([np.sum(new_tpcf2[new_dist2==d]) for d in dist2])
            end_time = time()
            times1 += [(end_time-start_time)/60]
            logger.info(f'Computational time {(end_time-start_time)/60} m')
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot,logger=logger)
            end_ram = ram_usage()
            logger.info(f'Ram usage {(end_ram-start_ram)/1024**3} Gb')

            ## NEW
            logger.info('FOR LOOP')
            start_time = time()
            tracemalloc.start()
            xdim,ydim = data.shape
            yy,xx = np.meshgrid(np.arange(-ydim+1,ydim),np.arange(-xdim+1,xdim))     

            # logger.info(f'{xx}\n{yy}')
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot,logger=logger)
            start_ram = ram_usage()
            new_tpcf2 = another_tpcf(data,(xx,yy), mode='cartesian')
            end_ram = ram_usage()
            rams2 += [end_ram-start_ram]
            start_ram = ram_usage()
            tracemalloc.start()
            new_dist2 = np.sqrt(xx**2+yy**2).flatten()
            directions = np.arctan2(xx,yy).flatten()
            pos = np.where(new_dist2 == 0.)[0][0]
            logger.info(f'{pos}')
            new_dist2[:pos+1]  = np.append(0,new_dist2[:pos])
            directions[:pos+1] = np.append(0,directions[:pos])

            dist2 = np.unique(new_dist2)
            corr2 = np.array([np.sum(new_tpcf2[new_dist2==d]) for d in dist2])
            end_time = time()
            times2 += [(end_time-start_time)/60]
            logger.info(f'Computational time {(end_time-start_time)/60} m')
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot,logger=logger)
            end_ram = ram_usage()
            logger.info(f'Ram usage {(end_ram-start_ram)/1024**3} Gb')

            ## NEW 2
            logger.info('FOR LOOP')
            start_time = time()
            tracemalloc.start()
            xdim,ydim = data.shape
            yy,xx = np.meshgrid(np.arange(ydim),np.arange(xdim))     

            # logger.info(f'{xx}\n{yy}')
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot,logger=logger)
            start_ram = ram_usage()
            new_tpcf2 = another_tpcf2(data,(xx,yy), mode='cartesian')
            end_ram = ram_usage()
            rams3 += [end_ram-start_ram]
            start_ram = ram_usage()
            tracemalloc.start()
            new_dist2 = np.repeat(np.sqrt(xx**2+yy**2).flatten()[1:],4,axis=0)
            directions = np.array([np.arctan2(i*xx,j*yy).flatten()[1:] for (i,j) in [(1,1,1,-1,-1,-1,-1,1)]])

            dist2 = np.unique(new_dist2)
            corr2 = np.array([np.sum(new_tpcf2[new_dist2==d]) for d in dist2])
            end_time = time()
            times3 += [(end_time-start_time)/60]
            logger.info(f'Computational time {(end_time-start_time)/60} m')
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot,logger=logger)
            end_ram = ram_usage()
            logger.info(f'Ram usage {(end_ram-start_ram)/1024**3} Gb')
        plt.figure()
        plt.plot(dims,times1,'--.',color='blue')
        plt.plot(dims,times2,'--x',color='orange')
        plt.figure()
        plt.plot(dims,rams1,'--.',color='blue')
        plt.plot(dims,rams2,'--x',color='orange')
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft, fft2, ifft2
from time import time
import argparse
from .test_func import *

# build the logger
logger_name = __name__ 
logger = logging.getLogger(logger_name)
logger.setLevel('DEBUG')

## DATA
FILE_NAME = filpy.FileVar(__file__,path=True)   #: path of the file


def compute_correlation(field: np.ndarray, diagonal_dist: bool = True, display_plot: bool = True) -> tuple[np.ndarray, np.ndarray]:
    tracemalloc.start()
    logger.info('Call the function `compute_correlation`')
    logger.info(f'`diagonal_dist` parameter set to {diagonal_dist}')
    logger.debug('Copy the data and remove the mean')
    field = np.copy(field) - field.mean()
    logger.debug('Compute all positions in the grid')
    start = time()
    all_pos = np.array([ (i,j) for i in range(field.shape[0]) for j in range(field.shape[1])]).T
    end = time()
    logger.debug(f'pos : compilation time: {end-start} s')

    if diagonal_dist:
        # snap_dist = tracemalloc.take_snapshot()
        logger.debug('Compute all distances in the grid')
        start = time()
        all_dist = np.concatenate([filpy.distance(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])])
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
        logger.debug('')
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
                    d = filpy.distance((0,0),(i,j))*PARAM
                    plt.axvline(d,color='red',linestyle='dotted')
                    plt.annotate(f'({i},{j})',(d,correlations.max()),(d+0.02,correlations.max()))
        else:
            for i in range(div):
                d = i*PARAM
                plt.axvline(d,color='red',linestyle='dotted')
        plt.show()

    logger.info('END')
    return unq_dist, correlations

## PIPELINE
if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",help='set log',nargs='*',type=str,action="store",choices=['file','bash','all'],default=None)
    parser.add_argument("test",help='selected test',type=str,choices=['lattice','random'],default='lattice')
    parser.add_argument("--no-diag", help='compute horizontal and vertical only', action='store_false')
    parser.add_argument("-m","--mode", help='mode of the log',type=str, action='store',default='w')
    parser.add_argument("-d","--dim", help='field size',type=int, action='store',default=32)
    parser.add_argument("-s","--seed", help='set seed', type=int, action='store',default=10)
    parser.add_argument("-e","--edges", help='set max and min of noise', required=False, nargs=2, action='store',default=[0,2])
    parser.add_argument("-l","--lag", help='set the lag of the lattice', required=False, type=int, action='store',default=5)
    parser.add_argument("-v","--value", help='set the value of the lattice', required=False, type=int, action='store',default=1)
    parser.add_argument("-i","--iter", help='set the number of iterations',required=False, type=int, action='store',default=10)
    parser.add_argument("-p","--plot",help='plot data or not',action='store_false')
    
    args = parser.parse_args()
    if args.log is not None:
        log = args.log[0] if len(args.log) != 0 else 'all'
        if log in ['all','file']:
            ch_f = logging.FileHandler(filename=filpy.log_path(FILE_NAME), mode=args.mode)
            frm_f = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
            ch_f.setFormatter(frm_f)
            logger.addHandler(ch_f)
        if log in ['all','bash']:
            ch_e = logging.StreamHandler()
            frm_e = logging.Formatter('%(levelname)s: %(message)s')
            ch_e.setFormatter(frm_e)
            ch_e.setLevel('INFO')
            logger.addHandler(ch_e)

    dim = args.dim    #: size of the field 
    if args.test == 'lattice':
        # build the field
        np.random.seed(args.seed)
        data = np.random.uniform(*args.edges,size=(dim,dim))  #: random signal
        PARAM = 5                           #: lag of the lattice
        data[::PARAM,::PARAM] += args.value
        # display the field
        filpy.show_image(data,cmap='viridis')
        _ = compute_correlation(data,args.no_diag,args.plot)

    ## RANDOM SIGNAL
    elif args.test == 'random':
        corr = 0
        ITER = args.iter
        logger.info('START the ROUTINE')
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
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.axes import Axes
from datetime import datetime
from time import time
from PIL import Image
from .test_func import TEST_DIR,logging, filpy


def plot_mask(xmask: tuple[int,int], ymask: tuple[int,int], ax: Axes | None = None):
    (xo,xe) = xmask 
    (yo,ye) = ymask
    if ax is None:
        plt.plot([xo,xo,xe,xe,xo],[yo,ye,ye,yo,yo],color='red')
    else:
        ax.plot([xo,xo,xe,xe,xo],[yo,ye,ye,yo,yo],color='red')


LOGGER_NAME = __name__ 
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel('DEBUG')

IMG_NAME = 'Pisa_test-case_IMG_5371.jpeg'
PIC_DIR = filpy.PROJECT_DIR -1 + 'pictures'
PIC_PATH = filpy.FileVar(IMG_NAME,PIC_DIR)
 
RES_DIR = TEST_DIR + 'test_results' + IMG_NAME.split('.')[0] + 'random_sampling'
RES_DIR.make_dir()

FILE_NAME = filpy.FileVar(__file__, path=True)

if __name__ == '__main__':

    ## PARSER SETTING
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',help='Mode',action='store', type=str, choices=['run','analysis'], default='run')
    parser.add_argument('--loglv',help='Log level',action='store', type=str, choices=['DEBUG','INFO'], default='INFO')
    parser.add_argument('--logmode',help='Log mode',action='store', choices=['w','a'], type=str, default='w')
    parser.add_argument('-x','--xsize',help='x size of the mask',action='store',type=int, default=355)
    parser.add_argument('-y','--ysize',help='y size of the mask',action='store',type=int, default=175)
    parser.add_argument('-n','--num',help='Number of masks',action='store', type=int, default=10)
    parser.add_argument('-o','--sford',help='Order of the structure function',action='store', type=int, default=2)

    parser.add_argument('--sample',help='A particular center mask',action='store', type=int,nargs=2, default=[])

    ARGS = parser.parse_args()

    ## LOG STUFF
    LOG_MODE = ARGS.logmode
    ch_f  = logging.FileHandler(filename=filpy.log_path(FILE_NAME), mode=LOG_MODE)
    frm_f = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    ch_f.setLevel(ARGS.loglv)
    ch_f.setFormatter(frm_f)
    logger.addHandler(ch_f)
    if LOG_MODE == 'a':
        logger.info('='*15)
        logger.info('='*15)
    logger.info('RUN RANDOM SAMPLING SCRIPT')
    logger.info(f'DATE:\t{datetime.now()}')
    logger.info(f'Mask size:\t({ARGS.xsize},{ARGS.ysize})')
    
    ## MAIN
    picture = Image.open(PIC_PATH.path())
    data = np.asarray(picture)
    data = data[:,:3250]
    avg_data = np.average(data,axis=2)
    ydim, xdim = avg_data.shape
    logger.info(f'Image size:\t({xdim},{ydim})')
    m_xsize = ARGS.xsize
    m_ysize = ARGS.ysize
    MAXLAG = int(np.sqrt(m_xsize**2 + m_ysize**2))
    XREGION = ( MAXLAG+m_xsize//2, xdim-(MAXLAG+m_xsize//2)+1 )
    YREGION = ( MAXLAG+m_ysize//2, ydim-(MAXLAG+m_ysize//2)+1 )
    XRDIM = XREGION[1]-XREGION[0]
    YRDIM = YREGION[1]-YREGION[0]
    logger.info(f'Region of the sampling:\t({XRDIM},{YRDIM})')
    del picture

    plt.figure()
    plt.imshow(avg_data,cmap='gray',origin='lower')
    plot_mask(XREGION,YREGION)
    plt.show()

    if len(ARGS.sample) == 0:
        SAMPLE = np.random.choice(XRDIM*YRDIM,size=ARGS.num,replace=False)
        YPOS, XPOS = np.unravel_index(SAMPLE,(YRDIM,XRDIM))
        XSAMPLE = np.arange(XRDIM)[XPOS]
        YSAMPLE = np.arange(YRDIM)[YPOS]
        del SAMPLE, XPOS, YPOS
    else:
        XSAMPLE = [ARGS.sample[0]]
        YSAMPLE = [ARGS.sample[1]]

    HEADER = 'xx,yy,tpcf0,stfc0,tpcf1,stfc1,tpcf2,stfc2,tpcf3,stfc3'
    file_paths_list = filpy.FileVar([],RES_DIR)
    for xo,yo in zip(XSAMPLE,YSAMPLE):
        start_routine = time()
        log_str = f'Centre:\t({xo},{yo})\n'
        xedges = (xo - m_xsize//2, xo + m_xsize//2+1)
        yedges = (yo - m_ysize//2, yo + m_ysize//2+1)
        tpcf, stfc = filpy.asym_tpcf_n_sf(avg_data,mask_ends=(xedges,yedges),order=ARGS.sford)
        f_ydim, f_xdim = tpcf.shape[:2]
        xx, yy = np.meshgrid(np.arange(f_xdim),np.arange(f_ydim))
        data_filename = f'tpfc-sf_{xo}-{yo}.csv'
        file_paths_list = file_paths_list + data_filename
        log_str = log_str + '\tFilename: ' + data_filename +'\n'
        np.savetxt(file_paths_list.path()[-1],np.transpose([xx.flatten(),yy.flatten(),
                                               tpcf[:,:,0].flatten(),stfc[:,:,0].flatten(),
                                               tpcf[:,:,1].flatten(),stfc[:,:,1].flatten(),
                                               tpcf[:,:,2].flatten(),stfc[:,:,2].flatten(),
                                               tpcf[:,:,3].flatten(),stfc[:,:,3].flatten()]), delimiter=',', header=HEADER,fmt=['%d','%d','%f','%f','%f','%f','%f','%f','%f','%f'])
        end_routine = time()
        log_str = log_str + f'\tCOMPUTATIONAL TIME: {(end_routine-start_routine)/60} m'
        logger.info(log_str)
    logger.info(f'END FOR ROUTINE')

    del ARGS
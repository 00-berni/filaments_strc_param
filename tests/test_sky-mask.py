import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from PIL import Image
from .test_func import TEST_DIR,logging, filpy

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
    parser.add_argument('-x','--xsize',help='x size of the mask',action='store',type=int, default=350)
    parser.add_argument('-y','--ysize',help='y size of the mask',action='store',type=int, default=175)

    parser.add_argument('-b','--bsize',help='bin size',action='store', type=int, default=3)
    parser.add_argument('-n','--bnum',help='bin number',action='store', type=int, default=None)
    # parser.add_argument('--xcut',help='edges along x',action='store',type=int,nARGS=2, default=[2130,2800])
    # parser.add_argument('--ycut',help='edges along y',action='store',type=int,nARGS=2, default=[1300,2147])

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
    logger.info('RUN RANDOM SAMPLING')
    logger.info(f'run:\t{datetime.now()}')
    logger.info(f'Mask size:\t({ARGS.xsize},{ARGS.ysize})')
    
    ## MAIN
    picture = Image.open(PIC_PATH.path())
    data = np.asarray(picture)
    data = data[:,:3250]
    avg_data = np.average(data,axis=2)
    ydim, xdim = avg_data.shape
    m_xsize = ARGS.xsize
    m_ysize = ARGS.ysize
    MAXLAG = int(np.sqrt(m_xsize**2 + m_ysize**2))

    plt.figure()
    plt.imshow(avg_data,cmap='gray',origin='lower')
    plt.axvline(MAXLAG+m_xsize//2,color='red')
    plt.axvline(xdim-(MAXLAG+m_xsize//2),color='red')
    plt.axhline(MAXLAG+m_ysize//2,color='red')
    plt.axhline(ydim-(MAXLAG+m_ysize//2),color='red')
    plt.show()
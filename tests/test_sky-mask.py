import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
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
    parser = argparse.ArgumentParser(description="Data are saved as:\nmaxlag\txedges\tyedges\txx\tyy\ttpcf\tstfc")
    parser.add_argument('mode',help='Mode',action='store', type=str, choices=['run','analysis','list'])
    parser.add_argument('--dir',help='Directory',dest='chdir',action='store',default=None)
    parser.add_argument('-x','--xsize',help='x size of the mask',action='store',type=int, default=355)
    parser.add_argument('-y','--ysize',help='y size of the mask',action='store',type=int, default=175)
    parser.add_argument('-n','--num',help='Number of masks',action='store', type=int, default=10)
    parser.add_argument('-o','--order',help='Order of the structure function',action='store', type=int, default=2)
    parser.add_argument('-s','--selection',help='Selected file',action='store', type=int,nargs='*')

    parser.add_argument('--loglv',help='Log level',action='store', type=str, choices=['DEBUG','INFO'], default='INFO')
    parser.add_argument('--logmode',help='Log mode',action='store', choices=['w','a'], type=str, default='a')
    parser.add_argument('--seed',help='Seed of the random sampling',action='store', type=int, default=None)
    parser.add_argument('--replace',help='If `True` same position can be chosen',action='store_true')
    parser.add_argument('--sample',help='A particular center mask',action='store', type=int,nargs=2, default=[])


    ARGS = parser.parse_args()
    
    ## LOG STUFF
    LOG_MODE = ARGS.logmode
    ch_f  = logging.FileHandler(filename=filpy.log_path(FILE_NAME), mode=LOG_MODE)
    frm_f = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    ch_f.setLevel(ARGS.loglv)
    ch_f.setFormatter(frm_f)
    logger.addHandler(ch_f)
    DATE = datetime.now()
    if LOG_MODE == 'a':
        logger.info('\n'+'='*60)
        logger.info('\n'+'='*60)
    logger.info('RUN RANDOM SAMPLING SCRIPT')
    logger.info(f'DATE:\t{DATE}')
    logger.info(f'Mask size:\t({ARGS.xsize},{ARGS.ysize})')
    
    MAIN_MODE = ARGS.mode

    ## MAIN
    if MAIN_MODE == 'list':
        logger.info('Print the list of data')
        if ARGS.chdir is None:
            _ = RES_DIR.dir_list(print_res=True)
        else:
            dir_list = RES_DIR.dir_list()
            ch_dir = ARGS.chdir if not ARGS.chdir.isdigit() else dir_list[int(ARGS.chdir)]
            _ = RES_DIR.dir_list(ch_dir,print_res=True)
              


    elif MAIN_MODE == 'run':
        picture = Image.open(PIC_PATH.path())
        data = np.asarray(picture)
        data = data[:,:3250]
        avg_data = np.average(data,axis=2)
        ydim, xdim = avg_data.shape
        logger.info(f'Image size:\t({xdim},{ydim})')
        m_xsize = ARGS.xsize
        m_ysize = ARGS.ysize
        logger.info(f'Mask sizes:\t{m_xsize} - {m_ysize}')
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

        if len(ARGS.sample) == 0:
            if ARGS.seed is not None:
                logger.info(f'Seed:\t{ARGS.seed}')
            np.random.seed(ARGS.seed)
            SAMPLE = np.random.choice(XRDIM*YRDIM,size=ARGS.num,replace=ARGS.replace)
            YPOS, XPOS = np.unravel_index(SAMPLE,(YRDIM,XRDIM))
            XSAMPLE = np.arange(XRDIM)[XPOS]
            YSAMPLE = np.arange(YRDIM)[YPOS]
            del SAMPLE, XPOS, YPOS
        else:
            XSAMPLE = [ARGS.sample[0]]
            YSAMPLE = [ARGS.sample[1]]

        plt.plot(XSAMPLE,YSAMPLE,'.',color='red')
        plt.show()
        plt.close('all')

        FILE_DIR = RES_DIR + (DATE.date().strftime("%Y-%m-%d") + f'_{m_xsize}-{m_ysize}')  
        FILE_DIR.make_dir()      
        file_paths_list = filpy.FileVar([],FILE_DIR)
        for xo,yo in tqdm(zip(XSAMPLE,YSAMPLE)):
            start_routine = time()
            log_str = f'Centre:\t({xo},{yo})\n'
            xedges = (xo - m_xsize//2, xo + m_xsize//2+1)
            yedges = (yo - m_ysize//2, yo + m_ysize//2+1)
            tpcf, stfc = filpy.asym_tpcf_n_sf(avg_data,mask_ends=(xedges,yedges),order=ARGS.order)
            f_ydim, f_xdim = tpcf.shape[:2]
            xx, yy = np.meshgrid(np.arange(f_xdim),np.arange(f_ydim))
            data_filename = f'tpfc-sf_{xo}-{yo}.npz'
            file_paths_list = file_paths_list + data_filename
            log_str = log_str + '\tFilename: ' + data_filename +'\n'
            np.savez_compressed(file_paths_list.path()[-1],
                                maxlag=MAXLAG,
                                xedges=np.asarray(xedges,dtype=int),
                                yedges=np.asarray(yedges,dtype=int),
                                xx=xx.astype(int),
                                yy=yy.astype(int),
                                tpcf=tpcf,
                                stfc=stfc)
            end_routine = time()
            log_str = log_str + f'\tCOMPUTATIONAL TIME: {(end_routine-start_routine)/60} m'
            logger.info(log_str)
        logger.info(f'END FOR ROUTINE')

    elif MAIN_MODE == 'analysis':
        dir_list = RES_DIR.dir_list()
        ch_dir = ARGS.chdir if not ARGS.chdir.isdigit() else dir_list[int(ARGS.chdir)]
        FILE_DIR = RES_DIR + ch_dir
        obj_list = FILE_DIR.dir_list()
        SELECTION = ARGS.selection
        if len(SELECTION) == 1:
            data_filename = filpy.FileVar(obj_list[SELECTION[0]],FILE_DIR)
        else:
            xo, yo = SELECTION
            data_filename = filpy.FileVar(f'tpcf-sf_{xo}-{yo}.npz',FILE_DIR)
        data = np.load(data_filename.path())
        maxlag = data['maxlag']
        xedges = data['xedges']
        yedges = data['yedges']
        xx = data['xx']
        yy = data['yy']
        tpcf = data['tpcf']
        stfc = data['stfc']
        del data


        xsize = int(np.diff(xedges)[0])
        ysize = int(np.diff(yedges)[0])
        xc = int(np.mean(xedges))
        yc = int(np.mean(yedges))
        # maxlag = int(np.sqrt(xsize**2+ysize**2))

        picture = Image.open(PIC_PATH.path())
        data = np.asarray(picture)
        data = data[:,:3250]
        avg_data = np.average(data,axis=2)
        del picture


        plt.figure()
        plt.imshow(avg_data,cmap='gray',origin='lower')
        plot_mask(xmask=xedges,ymask=yedges)
        plt.plot(xc,yc,'.r')
        
        xedges += np.array([-maxlag,maxlag])
        yedges += np.array([-maxlag,maxlag])
        sel_reg = avg_data[slice(*yedges),slice(*xedges)]
        

        plt.figure()
        plt.imshow(sel_reg,cmap='gray',origin='lower')
        plot_mask(xmask=(maxlag,xedges[1]-xedges[0]-maxlag),ymask=(maxlag,yedges[1]-yedges[0]-maxlag))

        plt.figure()
        plt.imshow(filpy.combine_results(stfc))
        plt.figure()
        plt.imshow(filpy.combine_results(tpcf))
        plt.show()


    del ARGS
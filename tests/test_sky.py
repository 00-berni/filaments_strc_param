import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft, fft2, ifft2
from time import time
import argparse
from .test_func import *
from .test_func import distance
from PIL import Image

logger_name = __name__ 
logger = logging.getLogger(logger_name)
logger.setLevel('DEBUG')

PIC_DIR = filpy.PROJECT_DIR -1 + 'pictures'
FILE_NAME = filpy.FileVar(__file__,path=True)   #: path of the file



if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-diag',help='take vertical and horizontal distances only',action='store_true')
    parser.add_argument("--log",help='set log',nargs='*',type=str,action="store",choices=['file','bash','all','DEBUG', 'INFO'],default=None)
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


    PIC_PATH = filpy.FileVar('Pisa_test-case_IMG_5331.jpeg',PIC_DIR) 
    picture = Image.open(PIC_PATH.path())
    data = np.asarray(picture)
    cut = slice(800,2000), slice(1600,3000)
    data = data[*cut][:,:,0]
    plt.figure()
    plt.imshow(data)
    start = time()
    if args.no_diag:
        dists = np.arange(max(data.shape))
        corrs = filpy.compute_tpcf(data,dists)
    else:
        dists, corrs = filpy.tpcf(data)
    end = time()
    

    plt.show()
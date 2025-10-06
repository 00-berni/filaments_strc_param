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




def plot_images(image: np.ndarray, mask: tuple | None = None,**kwargs):
    plt.figure()
    plt.imshow(image,cmap='gray',**kwargs)
    if mask is not None:
        (xo,xe), (yo,ye) = mask
        plt.plot([xo,xo,xe,xe,xo],[yo,ye,ye,yo,yo],color='red')


PIC_DIR = filpy.PROJECT_DIR -1 + 'pictures'
FILE_NAME = filpy.FileVar(__file__,path=True)   #: path of the file



if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--bsize',help='bin size',action='store', type=int, default=3)
    parser.add_argument('-n','--bnum',help='bin number',action='store', type=int, default=None)
    parser.add_argument('--xcut',help='edges along x',action='store',type=int,nargs=2, default=[2130,2800])
    parser.add_argument('--ycut',help='edges along y',action='store',type=int,nargs=2, default=[1300,2147])
    parser.add_argument('-x','--xstart',help='x start',action='store',type=int, default=0)
    parser.add_argument('-y','--ystart',help='y start',action='store',type=int, default=200)

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


    PIC_PATH = filpy.FileVar('Pisa_test-case_IMG_5371.jpeg',PIC_DIR) 
    picture = Image.open(PIC_PATH.path())
    data = np.asarray(picture)
    avg_data = np.average(data,axis=2)
    xedges = args.xcut
    yedges = args.ycut
    cut = (slice(*yedges), slice(*xedges))
    cut_data = avg_data[*cut]
    print(cut_data.shape)
    mask = (xedges,yedges)
    plot_images(avg_data,mask=mask)
    plot_images(cut_data)
    
    tot_corr = filpy.asym_tpcf(avg_data,mask_ends=mask,result='cum',zero_cover=True)

    plot_images(tot_corr)
    

    plt.show()

    exit()

    xstart = args.xstart
    ystart = args.ystart
    ydim,xdim = cut_data.shape
    min_dim = min(xdim-xstart,ydim-ystart)
    if args.bsize is None:
        bin_num = args.bnum
        bin_width = min_dim // bin_num
    elif args.bnum is None:
        bin_width = args.bsize
        bin_num = min_dim // bin_width
    print('BIN WIDTH:',bin_width)
    print('BIN NUM:',bin_num)
    bin_data = np.array([ [np.average(cut_data[ystart+j*bin_width:ystart+(j+1)*bin_width+1,xstart+i*bin_width:xstart+(i+1)*bin_width+1]) for i in range(bin_num-1)] for j in range(bin_num-1)])

    for i in range(bin_num):
        plt.axhline(ystart+i*bin_width,color='red',alpha=0.5)
        plt.axvline(xstart+i*bin_width,color='red',alpha=0.5)


    # fig, ax = plt.subplots(1,1)
    # ax.imshow(cut_data[ystart:ystart+bin_num*bin_width+1,xstart:xstart+bin_num*bin_width+1],cmap='gray')

    # for j in range(2):
    #     for i in range(2):
    #         ax.plot(i*bin_width+bin_width/2+xstart,j*bin_width+bin_width/2+ystart,'ob')
    #         ax.axvline(xstart+i*bin_width,color='red',linestyle='dotted')
    #         print('Y:',ystart+j*bin_width)
    #         print('X:',xstart+i*bin_width)
    #         plot_images(cut_data[ystart+j*bin_width:ystart+(j+1)*bin_width,xstart+i:xstart+(i+1)*bin_width])
    #     ax.axhline(ystart+j*bin_width,color='red',linestyle='dotted')


    print(bin_data.shape)
    plot_images(bin_data)

    plt.figure()
    plt.subplot(121)
    plt.imshow(cut_data[ystart:ystart+bin_num*bin_width+1,xstart:xstart+bin_num*bin_width+1],cmap='gray')
    plt.subplot(122)
    plt.imshow(bin_data,cmap='gray')

    tstart = time()
    stfc = filpy.asym_sf(bin_data,order=2,result='cum')
    tend = time()
    print('SF TIME:',(tstart-tend)//60,'m')    

    plt.figure()
    plt.imshow(stfc)
    plt.show()

    del stfc

    tstart = time()
    tpcf = filpy.asym_tpcf(bin_data,result='cum')
    tend = time()
    print('TPCF TIME:',(tstart-tend)//60,'m')

    plt.figure()
    plt.imshow(tpcf)
    plt.show()
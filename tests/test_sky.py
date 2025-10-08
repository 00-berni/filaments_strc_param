import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft, fft2, ifft2
from time import time
import argparse
from .test_func import *
from .test_func import distance
from PIL import Image
import os

logger_name = __name__ 
logger = logging.getLogger(logger_name)
logger.setLevel('DEBUG')


IMG_DIR = TEST_DIR + 'test_results'
if not os.path.isdir(IMG_DIR.PATH):
    os.mkdir(IMG_DIR.PATH)


def test_convolve_result(res_matrix: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    flat_res = np.sum(res_matrix,axis=2)
    ydim, xdim = flat_res.shape
    start_ram = ram_usage()
    xx, yy = np.meshgrid(np.arange(xdim),np.arange(ydim))
    end_ram = ram_usage()
    print('XX YY RAM:',(end_ram-start_ram)/(1024)**3,'Gb')
    dist = np.sqrt(xx**2+yy**2)
    start_ram = ram_usage()
    unq_dist = np.unique(dist[dist<=xdim])
    end_ram = ram_usage()
    print('UNIQUE DIST RAM:',(end_ram-start_ram)/(1024)**3,'Gb')
    start_ram = ram_usage()
    pos = [ np.asarray(np.where(dist==d)).astype(int) for d in unq_dist]
    end_ram = ram_usage()
    print('POS RAM:',(end_ram-start_ram)/(1024)**3,'Gb')
    # norm_val = lambda p : 4*len(p) if mode == 'mean' else 1
    start_ram = ram_usage()
    flat_res = np.asarray([np.sum(flat_res[yy[*p],xx[*p]])/(4*p.shape[1]) for p in pos])
    end_ram = ram_usage()
    print('SUM RAM:',(end_ram-start_ram)/(1024)**3,'Gb')
    return unq_dist, flat_res


def plot_images(image: np.ndarray, mask: tuple | None = None,**kwargs):
    plt.figure()
    if 'origin' not in kwargs.keys():
        kwargs['origin'] = 'lower'
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

    tracemalloc.start()
    IMG_NAME = 'Pisa_test-case_IMG_5371.jpeg'
    PIC_PATH = filpy.FileVar(IMG_NAME,PIC_DIR) 
    IMG_DIR = IMG_DIR + IMG_NAME.split('.')[0]
    if not os.path.isdir(IMG_DIR.PATH):
        os.mkdir(IMG_DIR.PATH)

    picture = Image.open(PIC_PATH.path())
    data = np.asarray(picture)
    avg_data = np.average(data,axis=2)
    xedges = args.xcut
    yedges = args.ycut
    cut = (slice(*yedges), slice(*xedges))
    cut_data = avg_data[*cut]
    print(cut_data.shape)
    mask = (xedges,yedges)
    start_ram = ram_usage()
    plot_images(avg_data,mask=mask)
    plt.savefig((IMG_DIR+'field.png').PATH)
    end_ram = ram_usage()
    print('RAM IMAGE:',(end_ram-start_ram)/(1024)**3,'GB')
    plot_images(cut_data)
    # plt.show()
    # plt.close('all')
    
    IMG_DIR = IMG_DIR + f'_{xedges[0]}.{yedges[0]}'
    if not os.path.isdir(IMG_DIR.PATH):
        os.mkdir(IMG_DIR.PATH)

    start = time()
    corr, stfc = filpy.asym_tpcf_n_sf(avg_data,mask_ends=mask)
    tot_corr = filpy.combine_results(corr)
    tot_stfc = filpy.combine_results(stfc)
    end = time()
    print('ALL TIME:',(end-start)/60,'m')


    # start = time()
    # corr = filpy.asym_tpcf(avg_data,mask_ends=mask,result='div',zero_cover=False)
    # tot_corr = filpy.combine_results(corr)
    # end = time()
    # print('CORR TIME:',(end-start)/60,'m')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    img = ax.imshow(tot_corr,origin='lower')
    fig.colorbar(img,ax=ax)
    max_lag = tot_corr.shape[0]//2
    circle_num = 5
    radii = np.arange(5,max_lag+(max_lag%circle_num),(max_lag+1)//circle_num)
    centre = (max_lag,max_lag)
    for r in radii:
        circle = plt.Circle(centre,r,color='white',fill=False,linestyle='dashed')
        ax.add_patch(circle)
        ax.annotate(f'{r:.0f}',(centre[0],centre[0]),(centre[0]+int(r/np.sqrt(2))+3,centre[0]+int(r/np.sqrt(2))+3),color='white')
    fig.savefig((IMG_DIR+'tpcf-2d.png').PATH)

    

    plt.figure()
    plt.title('TPCF')
    start_ram = ram_usage()
    dists, iso_corr = test_convolve_result(corr)
    end_ram = ram_usage()
    print('MARG TPCF:', (end_ram-start_ram)/(1024)**3, 'Gb')
    plt.plot(dists,iso_corr,'.--')
    plt.savefig((IMG_DIR+'tpcf-1d.png').PATH)
    

    mini_mask = ((max_lag,max_lag+xedges[1]-xedges[0]),(max_lag,max_lag+yedges[1]-yedges[0]))
    plot_images(avg_data[yedges[0]-max_lag:yedges[1]+max_lag+1,xedges[0]-max_lag:xedges[1]+max_lag+1],mask=mini_mask)
    plt.savefig((IMG_DIR+'frame.png').PATH)


    # start = time()
    # stfc = filpy.asym_sf(avg_data,mask_ends=mask,result='div')
    # tot_stfc = filpy.combine_results(stfc)
    # end = time()
    # print('STFC TIME:',(end-start)/60,'m')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    img = ax.imshow(tot_stfc,origin='lower',norm='log',vmin=tot_stfc[tot_stfc!=0].min())
    fig.colorbar(img,ax=ax,extend='min')
    max_lag = tot_stfc.shape[0]//2
    circle_num = 5
    radii = np.arange(5,max_lag+(max_lag%circle_num),(max_lag+1)//circle_num)
    centre = (max_lag,max_lag)
    for r in radii:
        circle = plt.Circle(centre,r,color='white',fill=False,linestyle='dashed')
        ax.add_patch(circle)
        ax.annotate(f'{r:.0f}',(centre[0],centre[0]),(centre[0]+int(r/np.sqrt(2))+3,centre[0]+int(r/np.sqrt(2))+3),color='white')
    fig.savefig((IMG_DIR+'stfc-2d.png').PATH)



    plt.figure()
    plt.title('SF')
    start_ram = ram_usage()
    dists, iso_sf = test_convolve_result(stfc)
    end_ram = ram_usage()
    print('MARG SF:', (end_ram-start_ram)/(1024)**3, 'Gb')
    plt.plot(dists,iso_sf,'.--')
    plt.savefig((IMG_DIR+'stfc-1d.png').PATH)

    plt.figure()
    norm_sf = iso_sf/iso_sf.sum()
    bin_num = 100
    bin_dist = np.arange(0,dists[-1]+2)
    bin_data = [ np.mean(norm_sf[(dists >= bin_dist[i])*(dists < bin_dist[i+1])]) if np.any((dists >= bin_dist[i])*(dists < bin_dist[i+1])) else 0 for i in range(len(bin_dist)-1)]

    plt.stairs(bin_data,bin_dist)
    plt.savefig((IMG_DIR+'stfc-hist.png').PATH)

    plt.figure()
    counts, sf_bins = np.histogram(bin_data,100)

    max_pos = counts.argmax()
    values = (sf_bins[max_pos-1],sf_bins[max_pos])
    poss = np.where((bin_data>=values[0])*(bin_data<=values[1]))[0]
    # ch_lag = bin_dist[poss].mean()
    # print('CH_LAG', ch_lag)
    # ch_lag = (ch_lag + bin_dist[poss+1].mean())/2
    # print('CH_LAG2',ch_lag)


    plt.hist(bin_data,100)
    plt.savefig((IMG_DIR+'stfc-hist-distr.png').PATH)




    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot, logger=logger)

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
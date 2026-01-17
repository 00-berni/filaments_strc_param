import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from scipy.ndimage import gaussian_filter

import filpy
from filpy import mydisperse as myd
from filpy.typing import *
from filpy import cube_stuff as cb

LOG_PATH = '.'.join(__file__.split('.')[:-1] + ['log'])
# build the logger
logger_name = __name__ 
logger = logging.getLogger(logger_name)
logger.setLevel('DEBUG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TestIR_PSF',
                                     description='Make simulation to see distortion after Sobel filtering',
                                    )
    # parameters
    field_parser = parser.add_argument_group('field parameter')
    field_parser.add_argument('-d','--dim',help='set the size of the image. By default `11`',type=int,action='store',nargs='*',default=[11])
    field_parser.add_argument('-v','--maxval',help='set a value of the point-like source. By default `1` perc',type=float,action='store',default=1)
    field_parser.add_argument('-b','--bkg',help='set a value of the background as percent of the maximum value. By default `30` perc',type=float,action='store',default=30)
    field_parser.add_argument('-B','--bkg-mode',help='choose the mode of background. Default by `"constant"`',type=str,choices=['constant','uniform','normal'],default='constant')
    parser.add_argument('-k','--kernel',help='set the kernel of the PSF. By default `"Gaussian"`',type=str,action='store',choices=['Gaussian'],default='Gaussian')
    parser.add_argument('-p','--kernel-param',help='set the parameters for the kernel. By default `[]`',type=str,action='store',nargs='*',default=['sigma'])
    # log stuff
    log_parser = parser.add_argument_group('logging')
    log_parser.add_argument('--log-out',help='set log output. By default `"file"`',nargs='*',type=str,action='store',choices=['file','bash','all'],default=['file'])
    log_parser.add_argument('--log-lv',help='set level of log output. By default `"DEBUG"`',nargs='*',type=str,action='store',choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],default=['DEBUG'])
    log_parser.add_argument('--log-mode', help='mode of the log. By default `"w"`',type=str, action='store',choices=['w','a'],default='w')
    log_parser.add_argument('--no-log',help='disable the log',action='store_false')
    # stuff
    stuff_parser = parser.add_argument_group('stuff')
    stuff_parser.add_argument('-l','--list',help='list the kernel options',action='store_true')
    stuff_parser.add_argument('--no-display', action='store_false', help='pictures are not plotted')

    args = parser.parse_args()

    # parameters
    dim = args.dim
    source = args.maxval
    bkg = args.bkg*source/100
    kernel = args.kernel
    # log stuff
    log_on = args.no_log
    log_out = args.log_out
    log_lv = args.log_lv
    # stuff
    display_plots = args.no_display


    if log_on:
        if 'all' in log_out:
            log_out = ['file','bash']
        
        if len(log_out) > len(log_lv):
            log_lv += [log_lv[-1]]*(len(log_out) - len(log_lv))

        for out, lv in zip(log_out,log_lv):
            if out == 'bash':
                ch_e = logging.StreamHandler()  
            else:
                ch_e = logging.FileHandler(filename=LOG_PATH, mode=args.log_mode)
            frm_e = logging.Formatter('%(levelname)s: %(message)s')
            ch_e.setFormatter(frm_e)
            ch_e.setLevel(lv)
            logger.addHandler(ch_e)

    if args.list:
        logger.info(f'Print the info about the {kernel} kernel')
        print('SELECTED KERNEL:', kernel.upper())
        if kernel == 'Gaussian':
            names_list = 'Variable names:\n\t- ' + '\n\t- '.join(gaussian_filter.__code__.co_varnames[1:])
            print(names_list)
    else:
        if len(dim) == 1: dim = dim*2

        if any(np.asarray(dim)%2 == 0):
            valerr = 'Only odd numbers are allowed for the size'
            logger.error('ValueError: ' + valerr) 
            raise ValueError(valerr) 
        
        dim = tuple(dim[::-1])
        bkg_val = bkg
        logger.debug('Compute the bkg')
        logger.info('BKG choice:',args.bkg_mode)
        if args.bkg_mode == 'uniform':
            bkg = np.random.uniform(0,bkg,size=dim)
        elif args.bkg_mode == 'normal':
            bkg = np.random.normal(loc=bkg,scale=0.2*bkg,size=dim)
        
        logger.info('Generate the field and add the background')
        field = np.zeros(dim) + bkg
        logger.debug('Find the center')
        ycen, xcen = np.asarray(dim) // 2
        logger.debug('Add the source')
        field[ycen,xcen] += source
        logger.info('Field generated')

        if display_plots:
            filpy.show_image(field,title='Source field',show=True)

        logger.info('Convolve the kernel')
        logger.info('KERNEL choice:',kernel)
        if kernel == 'Gaussian':
            kernel = gaussian_filter()




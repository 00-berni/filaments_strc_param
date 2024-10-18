import numpy as np
import filpy as flp
from astropy.io import fits

data_info = ['AR_FCRAO.fits','DEC_FCRAO.fits','VEL_FCRAO_CO.fits','MBM40_FCRAO_CO_CLEANED_upheader.fits']
DATA_DIR = flp.PROJECT_DIR + 'MBM40' + 'CO'

paths = flp.FileVar(data_info,DATA_DIR)

# collect units for ra, dec and vel of 3
_, ra  = flp.get_data_fit(paths[0], display_plots=False)     
_, dec = flp.get_data_fit(paths[1], display_plots=False)     
_, vel = flp.get_data_fit(paths[2], display_plots=False)     
ra = ra[0]
dec = dec[0]
vel = vel[0]
print(ra.shape)
print(dec.shape)
print(vel.shape)

def prepare_data(data):
    step = np.diff(data)[0]
    hpx = len(data)//2
    hval = data[hpx]
    cpx = hpx + 1.5 if len(data)%2 == 0 else len(data)/2
    cval = hval + step/2
    return cpx, cval, step

ra_cpx, ra_cval, ra_step = prepare_data(ra)
dec_cpx, dec_cval, dec_step = prepare_data(dec)
vel_cpx, vel_cval, vel_step = prepare_data(vel)

# DO NOT USED
hdul, co_data = flp.get_data_fit(paths[3],display_plots=False)   
print(co_data.shape)

def add_cart(name,value,comment=''):
    hdul[0].header[name] = value
    hdul[0].header.comments[name] = comment


hdul[0].header['OBJECT'] = 'FCRAO MBM40 CO cleaned map'
hdul[0].header.comments['OBJECT'] = 'Object Name'
add_cart('CTYPE1','RA      ','1st axis type')
add_cart('CRVAL1',ra_cval,'Reference pixel value')
add_cart('CRPIX1',ra_cpx,'Reference pixel')
add_cart('CDELT1',np.round(ra_step,decimals=5),'Pixel size in world coordinate units')
add_cart('CROTA1',0.0,'Axis rotation in degrees')
add_cart('CTYPE2','DEC     ','2st axis type')
add_cart('CRVAL2',dec_cval,'Reference pixel value')
add_cart('CRPIX2',dec_cpx,'Reference pixel')
add_cart('CDELT2',np.round(dec_step,decimals=5),'Pixel size in world coordinate units')
add_cart('CROTA2',0.0,'Axis rotation in degrees')
add_cart('CTYPE3','VELO-LSR','3st axis type')
add_cart('CRVAL3',vel_cval*1000,'Reference pixel value')
add_cart('CRPIX3',vel_cpx,'Reference pixel')
add_cart('CDELT3',np.round(vel_step,decimals=2)*1000,'Pixel size in world coordinate units')
add_cart('CROTA3',0.0,'Axis rotation in degrees')
add_cart('EQUINOX',2000.00,'Equinox of coordinates (if any)')
add_cart('BUNIT','K','Units of pixel data values')
hdul[0].header['OBSERVER'] = 'FCRAO : Five College Radio Astronomy Observatory'

up_header = hdul[0].header

new_file = flp.FileVar('MBM40_FCRAO_CO_CLEANED_upheader.fits',DATA_DIR)
fits.writeto(new_file.path(), co_data, up_header, overwrite=True)

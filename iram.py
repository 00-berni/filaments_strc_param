import numpy as np
import matplotlib.pyplot as plt
import filpy
from filpy import u
from filpy import cube_stuff as cb
from filpy import IR_PATHS

IR60_HDUL, IR60_DATA   = filpy.get_data_fit(IR_PATHS[0],display_plots=False)
IR100_HDUL, IR100_DATA = filpy.get_data_fit(IR_PATHS[1],display_plots=False) 
INT_UNIT = 'MJy / sr'

ir60_wcs = cb.WCS(IR60_HDUL[0].header)
filpy.show_image(IR60_DATA,projection=ir60_wcs,title='IRAM 60 $\\mu$m',show=True,vmax=5,barlabel=INT_UNIT)

ir100_wcs = cb.WCS(IR100_HDUL[0].header)
filpy.show_image(IR100_DATA,projection=ir100_wcs,title='IRAM 100 $\\mu$m',show=True,barlabel=INT_UNIT)

from scipy.ndimage import sobel, gaussian_filter

ir100_gfilt = gaussian_filter(IR100_DATA,sigma=2)
# ir100_gfilt *= IR100_DATA.max()/ir100_gfilt.max()

ir100_x = sobel(ir100_gfilt,axis=0)
ir100_y = sobel(ir100_gfilt,axis=1)

ir100_filt = np.sqrt(ir100_x**2 + ir100_y**2)
# ir100_filt *= IR100_DATA.max()/ir100_filt.max()


filpy.show_image(ir100_gfilt,projection=ir100_wcs,title='Gaussian smoothed IRAM 100',show=True,barlabel=INT_UNIT)
filpy.show_image(ir100_x,projection=ir100_wcs,title='Sobel IRAM 100 : horizontal gradient',show=True,barlabel=INT_UNIT)
filpy.show_image(ir100_y,projection=ir100_wcs,title='Sobel IRAM 100 : vertical gradient',show=True,barlabel=INT_UNIT)
filpy.show_image(ir100_filt,projection=ir100_wcs,title='Sobel filter IRAM 100',show=True,barlabel=INT_UNIT)

ir100 = ir100_gfilt-ir100_filt
filpy.show_image(ir100_gfilt-ir100_filt,projection=ir100_wcs,title='IRAM 100 : subtraction of Sobel filtered image',show=True,barlabel=INT_UNIT)
filpy.show_image([ir100,IR100_DATA-ir100_filt],num_plots=(1,2),projection=ir100_wcs,title='Comparison IRAM 100 after subtraction of Sobel filter',subtitles=('From Gaussian smoothed','From Original'),show=True,colorbar=False)
filpy.show_image([IR100_DATA,ir100_gfilt,ir100],num_plots=(1,3),projection=ir100_wcs,title='Comparison IRAM 100',subtitles=('Original','Gaussian filter','Sobel subtraction'),show=True,colorbar=False)


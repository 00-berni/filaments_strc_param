import numpy as np
import matplotlib.pyplot as plt
from astropy.io.fits import writeto
from spectral_cube import SpectralCube, Slice
import filpy as flp
from pandas import read_csv
from filpy import u, u_vel
u.add_enabled_units(u.def_unit(['K (Tb)'], represents=u.K))


DATA_DIR = flp.PROJECT_DIR + 'MBM40'
co_data, file_names, _ = read_csv(flp.FileVar('data.csv',DATA_DIR).path()).to_numpy().transpose()

co_data = flp.FileVar(co_data,DATA_DIR+'CO')
file_names = flp.FileVar(file_names, DATA_DIR + 'HI')


co_cube : SpectralCube = SpectralCube.read(co_data[0]).with_spectral_unit(u_vel)

vel, dec, ra = co_cube.world[:]

ra  =  ra[0,0,:].value
dec = dec[0,:,0].value
vel = vel[:,0,0].value

del co_cube

# # #

vel_range = (vel[0]*u_vel, vel[-1]*u_vel)
ra_range  = (248*u.deg,238*u.deg)
dec_range = (17*u.deg,24*u.deg)


cube1 : SpectralCube = SpectralCube.read(file_names[0]).with_spectral_unit(u_vel).subcube(*ra_range,*dec_range,*vel_range)


data1 = cube1.hdu.data
head1 = cube1.header

_,dec1,_ = cube1.world[:]
print(dec1[0,[0,-3,-2,-1],0])

dec1 = dec1[0,:,0]
del cube1

# # #

dec_range = (22.6*u.deg,27*u.deg)

cube2 : SpectralCube = SpectralCube.read(file_names[1]).with_spectral_unit(u_vel).subcube(*ra_range,*dec_range,*vel_range)


data2 = cube2.hdu.data
head2 = cube2.header

_,dec2,_ = cube2.world[:]
print(dec2[0,[0,1,2,-1],0])

dec2 = dec2[0,:,0]
new_dec = np.append(dec1[:-1],dec2)
del cube2, dec1, dec2, vel_range,ra_range,dec_range

# # #

new_header = head1.copy()
new_header.remove('SLICE')
new_data = np.append(data1[:,:-1,:],data2,axis=1)
step = float(head1['CDELT2'])

## AXIS
naxis1 = int(head1['NAXIS2'])
naxis2 = int(head2['NAXIS2'])
new_axis = naxis1 + naxis2 - 1
new_header['NAXIS2'] = new_axis
print('New Axis', new_axis)

## CRPIX
half_px = new_axis//2
hval = new_dec[half_px].value
new_cpix = new_axis / 2 if new_axis % 2 != 0 else half_px + 1.5
new_cval = hval + step/2

new_header['CRPIX2'] = new_cpix
new_header['CRVAL2'] = new_cval

print('New Crpix',new_cpix)
print('New Crval',new_cval)

new_file = flp.FileVar('GALFA_HI_sticked.fits',DATA_DIR+'HI')
writeto(new_file.path(), new_data, new_header, overwrite=True)

cube : SpectralCube = SpectralCube.read(new_file.path()).with_spectral_unit(u_vel)

_,dec,_ = cube.world[:]
print(dec[0,:,0])
print(new_header.tostring(sep='\n'))

del cube, dec

import numpy as np
import matplotlib.pyplot as plt
import filpy
from scipy.fft import fft,ifft, fft2, ifft2
from filpy import distance
from time import time
import logging
logging.getLogger(__name__)

## DATA
FILE_NAME = filpy.FileVar(__file__,path=True)   #: path of the file
logger = logging.basicConfig(filename=filpy.log_path(FILE_NAME),filemode='w', encoding='utf-8',level=logging.ERROR)
dim = 31                                        #: size of the field 
# build the field
xx, yy = np.meshgrid(np.arange(dim),np.arange(dim))
np.random.seed(10)
data = np.random.random((dim,dim)) 
PARAM = 5
data = np.zeros((dim,dim))
data[::PARAM,::PARAM] = 1
filpy.show_image(data,cmap='viridis')

# log = filpy.Log_File(FILE_NAME,print_cond=True)


## PIPELINE
data_mean = data.mean()
field = np.copy(data) - data_mean


all_pos = np.array([ (i,j) for i in range(field.shape[0]) for j in range(field.shape[1])]).T

logger.info('Compute all the distances')
start = time()
res_dist = np.concatenate([distance(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])])
end = time()
logger.info(f'Compilation time: {end-start} s')
logger.info('Compute all the distances')
start1 = time()
res_corr = np.concatenate([field[*all_pos[:,N]] * field[*all_pos[:,N:]] for N in range(all_pos.shape[1])])
end1 = time()
logger.info(f'Compilation time: {end1-start1} s')

logger.info('Compute the correlation')
start1 = time()
unq_dist = np.unique(res_dist)
correlations = np.array([np.sum(res_corr[res_dist == d]) for d in unq_dist])
end1 = time()
logger.info(f'Compilation time: {end1-start1} s')

filpy.quickplot((unq_dist,correlations),fmt='.--')
for i in range(10):
    for j in range(10):
        d = distance((0,0),(i,j))*PARAM
        if d > np.max(unq_dist):
            break
        plt.axvline(d,color='red',linestyle='dotted')
plt.show()


## FFT
# data_fft = fft2(data)
# tmp = data_fft * np.conjugate(data_fft)
# tmp_img = np.abs(ifft2(np.abs(tmp)))
# del tmp
# MID = dim // 2
# print(tmp_img.shape,MID)
# corr_img = np.copy(tmp_img)
# corr_img[ : MID, : MID] = tmp_img[MID : , MID : ]
# corr_img[MID : , MID : ] = tmp_img[ : MID, : MID]
# corr_img[ : MID, MID : ] = tmp_img[MID : , : MID]
# corr_img[MID : , : MID] = tmp_img[ : MID, MID : ]
# filpy.show_image(tmp_img)
# filpy.show_image(corr_img)

# distances = np.array([ [distance((i,j),(MID+0.5,MID+0.5)) for i in range(dim)] for j in range(dim)])
# unq_dist = np.unique(distances)

# correlations = np.array([np.sum(corr_img[distances == d]) for d in unq_dist])

# filpy.quickplot((unq_dist,correlations),fmt='.--')
# for i in range(10):
#     for j in range(10):
#         d = distance((0,0),(i,j))*PARAM
#         if d > np.max(unq_dist):
#             break
#         plt.axvline(d,color='red',linestyle='dotted')
# plt.show()

# # #


# all_pos = [ (i,j) for i in range(data.shape[0]) for j in range(data.shape[1])]
# all_pos = np.asarray(all_pos).T

# start = time()
# res_dist = np.concatenate([distance(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])])
# # res_dist = np.array([list(distance(all_pos[:,N],all_pos[:,N:])) for N in range(all_pos.shape[1])],dtype='object').sum()
# end = time()
# print('Compilation time:', end-start,'s')
# print(type(res_dist),len(res_dist))
# # del res_dist
# # start = time()
# # res_corr = [calc_corr_ij(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])]
# # end = time()
# start1 = time()
# res_corr = np.concatenate([calc_corr_ij(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])])
# end1 = time()
# # print('Compilation time:', end-start,'s')
# print('Compilation time:', end1-start1,'s')


# # exit()
# # distances = np.ravel([r[0] for r in res])
# # correlations = np.ravel([r[1] for r in res])
# print('dist')
# unq_dist = np.unique(res_dist)
# print('dist end')
# correlations1 = np.array([np.sum(res_corr[res_dist == d]) for d in unq_dist])
# print('Compilation time:', end1-start1,'s')

# correlations1 /= correlations1.max()

# filpy.quickplot((unq_dist,correlations1-correlations),fmt='.--')

# plt.show()


# corr = correlate2d(data,data)
# lags = correlation_lags(len(data),len(data))

# filpy.show_image(corr,cmap='viridis',show=True)
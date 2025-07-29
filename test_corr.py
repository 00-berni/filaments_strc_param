import numpy as np
import matplotlib.pyplot as plt
import filpy
from scipy.fft import fft,ifft, fft2, ifft2
from filpy import distance
from scipy.signal import correlate2d, correlation_lags
from scipy.spatial import KDTree

from time import time

dim = 80
xx, yy = np.meshgrid(np.arange(dim),np.arange(dim))
np.random.seed(10)
data = np.random.random((dim,dim)) 

def mydist(*args):
    return distance(*args)

filpy.show_image(data,cmap='viridis')
all_pos = [ (i,j) for i in range(data.shape[0]) for j in range(data.shape[1])]
all_pos = np.asarray(all_pos).T
# start = time()
# distances = [distance(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])]
# end = time()
# print('Parallel: ', end-start, 's')
# del distances
# start = time()
# distances = [] 
# for N in range(all_pos.shape[1]):
#     distances += [mydist(all_pos[:,N],all_pos[:,N:])]
# end = time()
# print('Sequential: ', end-start, 's')




# exit()
def corr_r(lag:float):
    for N in range(len(all_pos)):
        val0 = data[tuple(all_pos[:,N])]
# dist = np.linspace(0,150,300)
# signal = KDTree(data)
# random = KDTree(np.random.random((dim,dim)))
# DD = signal.count_neighbors(signal,dist)
# DR = signal.count_neighbors(random,dist)
# RR = random.count_neighbors(random,dist)
# # f = len(signal)/len(random)
# corr = (DD - 2*DR  + RR)/RR
# print(corr)
# filpy.quickplot((dist,corr),fmt='.--')
# plt.show()


# corr = ifft2(fft2(data)**2)
# filpy.show_image(np.real(corr),cmap='viridis')
# filpy.show_image(np.abs(corr),show=True,cmap='viridis')
# filpy.quickplot(np.real(corr).diagonal(),fmt='.--')
# filpy.quickplot(np.abs(corr).diagonal(),fmt='.--')
# plt.show()

# @filpy.timeit
def calc_corr_ij(pos: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    x,y = pos
    val0 = data[x,y]
    # dists = distance(pos,positions)
    print(positions.shape,end='\r')
    corr_ij = val0*data[*positions]
    return corr_ij

all_pos = [ (i,j) for i in range(data.shape[0]) for j in range(data.shape[1])]
all_pos = np.asarray(all_pos).T

start = time()
res_dist = np.concatenate([distance(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])])
# res_dist = np.array([list(distance(all_pos[:,N],all_pos[:,N:])) for N in range(all_pos.shape[1])],dtype='object').sum()
end = time()
print('Compilation time:', end-start,'s')
print(type(res_dist),len(res_dist))
# del res_dist
# start = time()
# res_corr = [calc_corr_ij(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])]
# end = time()
start1 = time()
res_corr = np.concatenate([calc_corr_ij(all_pos[:,N],all_pos[:,N:]) for N in range(all_pos.shape[1])])
end1 = time()
# print('Compilation time:', end-start,'s')
print('Compilation time:', end1-start1,'s')


# exit()
# distances = np.ravel([r[0] for r in res])
# correlations = np.ravel([r[1] for r in res])
print('dist')
unq_dist = np.unique(res_dist)
print('dist end')
correlations = np.array([np.sum(res_corr[res_dist == d]) for d in unq_dist])

filpy.quickplot((unq_dist,correlations),fmt='.--')

plt.show()

# corr = correlate2d(data,data)
# lags = correlation_lags(len(data),len(data))

# filpy.show_image(corr,cmap='viridis',show=True)
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates
from astropy.io import fits as ffits
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt


def make_only_one_block(data, mode='volume', minimize_dim=False):
    from amunra.utils import find_detached_surfaces
    mask = ~np.isnan(data)
    volumes = find_detached_surfaces(mask)
    max_len = 0
    if mode == 'volume':
        lens = np.array([len(vol) for vol in volumes])
        for vol, length in zip(volumes, lens):
            if length != lens.max():
                x, y = zip(*list(vol))
                data[x, y] = np.nan

    elif mode == 'mass':
        mass = []
        ll = 0
        nn = 0
        for vol in volumes:
            x, y, z = zip(*list(vol))
            ll+=1
            mass.append(data[x, y, z].sum())
            if ll>len(volumes)/100 * nn:
                nn+=10
                print('{}%'.format(nn), end=' ')
        print('\n')
        mass = np.array(mass)
        index = np.where(mass==mass.max())[0][0]

        for i, vol in enumerate(volumes):
            if i!=index:
                x, y, z = zip(*list(vol))
                data[x, y, z] = np.nan


    elif mode == 'density_peak':
        density = []
        for vol in volumes:
            x, y, z = zip(*list(vol))
            density.append(data[x, y, z].max())
        density = np.array(density)
        index = np.where(density == density.max())[0][0]
        for i, vol in enumerate(volumes):
            if i != index:
                x, y, z = zip(*list(vol))
                data[x, y, z] = np.nan
    else:
        raise(NameError, 'mode "{}" not found'.format(mode))

    if minimize_dim:
        mask = ~np.isnan(data)
        volumes = find_detached_volumes(mask)
        # Only one volume, should be fast
        assert len(volumes)==1
        xmax = 0
        ymax = 0
        zmax = 0
        xmin = data.shape[0]
        ymin = data.shape[1]
        zmin = data.shape[2]
        x, y, z = zip(*list(volumes[0]))
        xmax = xmax if np.array(x).max() < xmax else np.array(x).max()
        ymax = ymax if np.array(y).max() < ymax else np.array(y).max()
        zmax = zmax if np.array(z).max() < zmax else np.array(z).max()
        xmin = xmin if np.array(x).min() > xmin else np.array(x).min()
        ymin = ymin if np.array(y).min() > ymin else np.array(y).min()
        zmin = zmin if np.array(z).min() > zmin else np.array(z).min()
        data = data[xmin: xmax, ymin: ymax, zmin: zmax]
        print('Cutting: new matrix dimensions: ', data.shape)
    return data, '_one_surface'


def remove_isolated(data, min_len, minimize_dim=False):
    from amunra.utils import find_detached_volumes
    mask = ~np.isnan(data)
    volumes = find_detached_volumes(mask)
    lens = np.array([len(vol) for vol in volumes])

    xmax = 0
    ymax = 0
    zmax = 0
    xmin = data.shape[0]
    ymin = data.shape[1]
    zmin = data.shape[2]

    for vol, length in zip(volumes, lens):
        if length < min_len:
            x, y, z = zip(*list(vol))
            data[x, y, z] = np.nan
        if length >= min_len:
            xmax = xmax if np.array(x).max()< xmax else np.array(x).max()
            ymax = ymax if np.array(y).max()< ymax else np.array(y).max()
            zmax = zmax if np.array(z).max()< zmax else np.array(z).max()
            xmin = xmin if np.array(x).min()> xmin else np.array(x).min()
            ymin = ymin if np.array(y).min()> ymin else np.array(y).min()
            zmin = zmin if np.array(z).min()> zmin else np.array(z).min()

    if minimize_dim:
        xmax = xmax if xmax < data.shape[0] else None
        ymax = ymax if ymax < data.shape[1] else None
        zmax = zmax if zmax < data.shape[2] else None

        data = data[xmin: xmax, ymin: ymax, zmin: zmax]
        print('Cutting: new matrix dimensions: ', data.shape)

    return data, '_no_less_{}_{}'.format(min_len, '_MD' if ~minimize_dim else '')

def find_background_thr(data, nsigma):
    # repetitions = number of itereations to remove real data
    # sensitivity = condition to exit iteration on derived threshold
    mean = data.mean()
    std = data.std()

    def gaussian(x, mu, sigma, N):
        return N / (np.sqrt(2. * np.pi) * sigma) * np.exp(- ((x - mu) / sigma) ** 2 / 2)

    values, bins0 = np.histogram(data, bins=int(np.sqrt(data.size)))

    bins = (bins0[1:] + bins0[:-1]) / 2
    errs = values ** 0.5


    mask1 = bins < values.max()

    popt, pcov = curve_fit(gaussian, bins[mask1], values[mask1],
                           p0=[bins[np.where(values == max(values))[0][0]], std, data.size])

    print('Mean + Bk sigma: {:.2f} + {:.2f}'.format(popt[0], popt[1]))

    #plt.stairs(values, bins0)
    #plt.plot(bins, gaussian(bins, *popt), 'r')
    #plt.axvline(popt[0] + nsigma * popt[1], linestyle='--', color='k')
    #plt.show()


    return popt[0] + nsigma * popt[1]

def remove_noise(data, thr):
    data_tmp = data.copy()
    data_tmp[data < thr] = np.nan
    return data_tmp

def add_borders(data):
    data_new = np.random.random(( (data.shape[0]+4),(data.shape[1]+4),(data.shape[2]+4)) )
    data_new = data_new * data.mean()*10**-6
    data_new[2:-2,2:-2,2:-2] = data
    return data_new, '-LC'


if __name__ == '__main__':
    path = '/run/media/psuin/Seagate Basic/PhD/Observation/RCW79/'  # '/run/media/psuin/Seagate Basic/PhD/Observation/NGC6334/'
    data_f_name = 'RCW79_12CO32_20_8_0p5'
    data_f = data_f_name + '.fits' if not data_f_name.endswith('fits') else data_f_name
    add_name = ''
    file = ffits.open(path + data_f )[0]
    data_f = 'Mod--' + data_f
    remove_iso = True
    one_surf = False # not active if remove iso
    mode_onesurf = 'density_peak'
    min_dim = False
    leave_with_noise_at_sigma = -1 # not work if min_dim

    smooth = 3
    nsigma = 5
    min_len = 1000
    header = file.header
    original_data = file.data
    data = file.data

    # ADDED LINES!!
    remove_edges = 10
    original_data = original_data[:, remove_edges:-remove_edges, remove_edges:-remove_edges]
    data = data[:, remove_edges:-remove_edges, remove_edges:-remove_edges]
    #####

    if len(original_data.shape)>3:
        original_data = original_data[0]

    not_nan_data = original_data[~np.isnan(original_data)]
    L00 = len(not_nan_data)

    if smooth > 1:
        import scipy.signal
        smooth_matrix = np.ones((smooth,smooth))/smooth**2
        for i in range(original_data.shape[0]):
            original_data[i,:,:] = scipy.signal.convolve(original_data[i,:,:], smooth_matrix, 'same')

        if leave_with_noise_at_sigma == 0:
            original_data[original_data<0]=1e-10*np.random.random(len(original_data[original_data<0]))
        add_name += '_smothed-{}'.format(smooth)

    thr = find_background_thr(not_nan_data, nsigma=nsigma)
    thr_leave_noise = find_background_thr(not_nan_data, nsigma=leave_with_noise_at_sigma)
    print('Masking below: {:.2f}'.format(thr))
    if nsigma>0:
        add_name += '_CutSigma-{}'.format(nsigma)
    print('Percentage of data removed by first masking: {:.2f}%'.format(original_data[original_data<thr].size/L00*100))
    data = remove_noise(original_data, thr)

    L0 = len(data[~np.isnan(data)])

    if remove_iso:
        data, name = remove_isolated(data, min_len, min_dim)
        print('Percentage of data removed by masking isolated pixels: {:.2f}%'.format((L0-len(data[~np.isnan(data)]))/L0*100))
        add_name += name
    elif one_surf:
        data, name = make_only_one_block(data, mode=mode_onesurf, minimize_dim=min_dim)
        print('Percentage of data removed by masking isolated pixels: {:.2f}%'.format((L0-len(data[~np.isnan(data)]))/L0*100))

        add_name += name

    print('Remaining data: {:.2f}%'.format(len(data[~np.isnan(data)]) / L00 * 100))
    print('Equivalent to: {:.1f}^3'.format(len(data[~np.isnan(data)])**(1/3)))


    if leave_with_noise_at_sigma != 0:
        not_nan_data_pos = np.where(~np.isnan(data))
        slice_data = np.array([
          [not_nan_data_pos[0].min(), None if (not_nan_data_pos[0].max() + 1)==data.shape[0] else not_nan_data_pos[0].max() + 1],
          [not_nan_data_pos[1].min(), None if (not_nan_data_pos[1].max() + 1)==data.shape[1] else not_nan_data_pos[1].max() + 1],
          [not_nan_data_pos[2].min(), None if (not_nan_data_pos[2].max() + 1)==data.shape[2] else not_nan_data_pos[2].max() + 1],
                      ])
        # AD HOC FIX
        slice_data[1, 0] = 0
        slice_data[2, 0] = 0
        slice_data[1, 1] = original_data.shape[1]
        slice_data[2, 1] = original_data.shape[2]
        original_data = original_data[slice_data[0, 0]:slice_data[0, 1],
                                      slice_data[1, 0]:slice_data[1, 1],
                                      slice_data[2, 0]:slice_data[2, 1],]

        shape_new = original_data.shape
        for i in range(3):
            slice_data[i, 1] = slice_data[i, 1] if slice_data[i, 1] is not None else not_nan_data_pos[i].max()
            middle_pix = (slice_data[i, 0] + slice_data[i, 1])/2 + remove_edges
            # AD HOC FIX!!
            #if i == 2 and 'CII' in data_f_name: middle_pix += 1
            #if i == 1 and '12CO' in data_f_name: middle_pix += 1
            #if i == 2 and '13CO' in data_f_name: middle_pix -= 12
            ##
            header_num = 3 - i
            print(header['CRPIX{}'.format(header_num)]*2-header['NAXIS{}'.format(header_num)])

            diff_centers = middle_pix - header['CRPIX{}'.format(header_num)]
            header['CRVAL{}'.format(header_num)] = header['CRVAL{}'.format(header_num)] + diff_centers * header['CDELT{}'.format(header_num)]
            header['CRPIX{}'.format(header_num)] = shape_new[i]/2
            header['NAXIS{}'.format(header_num)] = shape_new[i]



        print('new_data_shape: ', original_data.shape)
        add_name = add_name + '_MD_WN'
        if leave_with_noise_at_sigma > 0:
            original_data[original_data < thr_leave_noise] = np.nan
            if one_surf:
                make_only_one_block(original_data, mode='volume', minimize_dim=False)
            elif remove_isolated:
                remove_isolated(original_data, min_len, min_dim)
            add_name += '_THR-{}'.format(leave_with_noise_at_sigma)
    else:
        original_data = data





    ffits.writeto(path + data_f + add_name + '.fits', original_data, overwrite=True, header=header)

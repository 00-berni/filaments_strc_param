from astropy.io import fits as astrofits
from tvtk.api import tvtk, write_data
from amunra.data.PostProcessSim import Sim_PP_class
from amunra.data.simulationData import Simulation
from amunra.utils import find_sim_path
import numpy as np
from unyt.array import unyt_array as ytarr
from pydisperse.fits2vtk import fits_to_vtk
from astropy.io.fits.header import Header


def triD_to_2D(filepath):
    data_fits = astrofits.open(filepath)
    data_cube = data_fits[0].data
    header = data_fits[0].header
    # removes the third axis information
    remove_keys = []
    for key in header.cards:
        if key[0].endswith('3'):
            remove_keys.append(key[0])
    for key in remove_keys:
        header.remove(key)

    data_cube = data_cube.astype(float)
    data_cube[np.isnan(data_cube)] = 10**-5
    data2d = data_cube.sum(axis=0)
    results = {'data2d':data2d, 'header':header}
    return results


if __name__ == '__main__':
    Folds = ['12CO', '13CO', 'CII']
    Name = {'12CO': '12CO32', '13CO': '13CO32', 'CII': 'CII'}
    for Fold in Folds:
        file_path = '/run/media/psuin/Seagate Basic/PhD/Observation/RCW79/{}/'.format(Fold)
        file2d_path = '/run/media/psuin/Seagate Basic/PhD/Observation/RCW79/2D/{}/'.format(Fold)
        name_path = file_path + ('Mod--RCW79_{}_20_8_0p5.fits_smothed-3_CutSigma-5_no_less_1000__MD_MD_WN'.format(Name[Fold]))
        finale_name_path = file2d_path + ('2D_Mod--RCW79_{}_20_8_0p5.fits_smothed-3_CutSigma-5_no_less_1000__MD_MD_WN'.format(Name[Fold]))
        results = triD_to_2D(name_path + '.fits')
        astrofits.writeto(finale_name_path + '.fits', results['data2d'], overwrite=True, header=results['header'])

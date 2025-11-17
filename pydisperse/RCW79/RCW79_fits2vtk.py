from astropy.io import fits as astrofits
from tvtk.api import tvtk, write_data
from amunra.data.PostProcessSim import Sim_PP_class
from amunra.data.simulationData import Simulation
from amunra.utils import find_sim_path
import numpy as np
from unyt.array import unyt_array as ytarr
from pydisperse.fits2vtk import fits_to_vtk



if __name__ == '__main__':
    Folds = [ 'Col_dens']
    Name = {'12CO': '12CO32', '13CO': '13CO32', 'CII': 'CII'}
    two_dim=False
    for Fold in Folds:
        file_path = '/run/media/psuin/Seagate Basic/PhD/Observation/RCW79/{}/{}/'.format('2D' if two_dim else '', Fold)
        if Fold=='Col_dens':
            name_path = file_path + 'rcw79_coldens_high'
        else:
            name_path = file_path + ('{}Mod--RCW79_{}_20_8_0p5.fits_smothed-3_CutSigma-5_no_less_1000__MD_MD_WN'.format('2D_' if two_dim else '', Name[Fold]))
        vtk_grid = fits_to_vtk(name_path + '.fits')
        write_data(vtk_grid, name_path + '.vtk')

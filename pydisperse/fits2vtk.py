from astropy.io import fits as astrofits
from tvtk.api import tvtk, write_data
import numpy as np
from unyt.array import unyt_array as ytarr

def fits_to_vtk(filepath, remove_nans=True):
    data_fits = astrofits.open(filepath)
    data_cube = data_fits[0].data
    if len(data_cube.shape) == 2:
        data_cube = np.array([data_cube])
    data_cube = data_cube.T
    data_cube = data_cube.astype(float)

    if remove_nans:
        data_cube[np.isnan(data_cube)] = -10**5
    grid = tvtk.ImageData(spacing=(1,1,1), origin=(0, 0, 0),
                      dimensions=data_cube.shape)
    grid.point_data.scalars = data_cube.ravel(order='F')
    grid.point_data.scalars.name = 'Test Data'
    return grid



if __name__ == '__main__':
    from amunra.data.PostProcessSim import Sim_PP_class
    from amunra.data.simulationData import Simulation
    from amunra.utils import find_sim_path

    SimType = 'HJ'
    SimPathFold = find_sim_path(SimType)
    Sim0 = Simulation(SimPathFold)
    Sim = Sim_PP_class(Sim0)



    time_snap = 0 #Myr
    PPV = False
    Already_done = False
    resolution = 128
    edges = [ytarr([0, 0, 0], 'pc'),
             ytarr([30, 30, 30], 'pc')]

    base_fold = '{}__{}Myr__[{},{},{}]-[{},{},{}]__{}'.format(SimType, time_snap,
                                                            int(edges[0][0].to_value() * 10),
                                                            int(edges[0][1].to_value() * 10),
                                                            int(edges[0][2].to_value() * 10),
                                                            int(edges[1][0].to_value() * 10),
                                                            int(edges[1][1].to_value() * 10),
                                                            int(edges[1][2].to_value() * 10),
                                                            resolution)
    import os
    base_fold = '/disperse_tmp/'
    if not os.path.exists(SimPathFold + base_fold):
        os.mkdir(file_path + base_fold)
    base_name = '{}__{}Myr__{}-{}-{}---{}-{}-{}__{}'.format(SimType, time_snap,
                                                            int(edges[0][0].to_value() * 10),
                                                            int(edges[0][1].to_value() * 10),
                                                            int(edges[0][2].to_value() * 10),
                                                            int(edges[1][0].to_value() * 10),
                                                            int(edges[1][1].to_value() * 10),
                                                            int(edges[1][2].to_value() * 10),
                                                            resolution)

    name_path = SimPathFold + '/{}{}'.format('PPV_' if PPV else '' ,base_name)

    if not Already_done and not PPV:
        fits_grid = Sim.get_fits_cube(time=time_snap, edges=edges, resolution=resolution)
        fits_grid.writeto(name_path + '.fits')
    vtk_grid = fits_to_vtk(name_path + '.fits')
    write_data(vtk_grid, name_path + '.vtk')

    #file_path = '/run/media/psuin/Seagate Basic/PhD/Observation/Loris Cubes'
    #file_name = 'COHRS_43p00_0p00_CUBE_3T2_R2.fits_smothed-5_CutSigma-3_one_surface'
    #img = fits_to_vtk(file_path + '/' + file_name + '.fits')

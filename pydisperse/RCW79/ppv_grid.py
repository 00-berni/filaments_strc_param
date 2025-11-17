import matplotlib.pyplot as plt
from astropy.io import fits as astrofits
from tvtk.api import tvtk, write_data
from amunra.data.PostProcessSim import Sim_PP_class
from amunra.data.simulationData import Simulation
from amunra.utils import find_sim_path
import numpy as np
from unyt.array import unyt_array as ytarr
from unyt.array import unyt_quantity as ytqty
import yt.units
from apollon.visualisation import return_axes_from_proj


def restructure_grid(region, resolution,
                     resolutionV, edges, Vedges, axes_l):


    velocity = region[('ramses', axes_l[2]+'-velocity')].to('km/s')
    mass = region[('gas', 'mass')].to('g')
    axis_1 = region[('ramses', axes_l[0])].to('cm')
    axis_2 = region[('ramses', axes_l[1])].to('cm')
    cell_size = region[('ramses', 'd'+axes_l[1])].to('cm')
    edges_grid = [edges[0].to('cm'), edges[1].to('cm')]

    mass = mass.value

    # rescaling velocity to new ppv
    VV_axis = (velocity-Vedges[0]) / (Vedges[1] - Vedges[0]) * resolutionV
    VV_axis = VV_axis.value
    VV_axis = np.floor(VV_axis).astype(int)

    dV_axis = (Vedges[1] - Vedges[0]).value / resolutionV


    empty_grid = np.zeros((resolution, resolution, resolutionV))

    #testing VV_axis = np.random.rand(len(axis_1))

    # rescaling to new grid
    cell_size_1 = cell_size / (edges[1][0] - edges[0][0]) * resolution / 2
    cell_size_2 = cell_size / (edges[1][1] - edges[0][1]) * resolution / 2
    axis_1 = (axis_1 - edges[0][0]) / (edges[1][0] - edges[0][0]) * resolution
    axis_2 = (axis_2 - edges[0][1]) / (edges[1][1] - edges[0][1]) * resolution

    axis_1 = axis_1.value
    axis_2 = axis_2.value
    cell_size_1 = cell_size_1.value
    cell_size_2 = cell_size_2.value

    # TODO WARNING does not take into account borders
    cell_crop = np.array([
        ## left - right
        # percentage of volume on the left side
        (
            (1 - (axis_1 - cell_size_1) % 1)
        ),  # left side percentage

        # percentage of volume on the right side
        (
            ((axis_1 + cell_size_1) % 1)
        ),  # right side percentage

        ## up - down
        # percentage of volume on the down side
        (
            (1 - (axis_2 - cell_size_2) % 1)
        ),  # down side percentage

        # percentage of volume on the up side
        (
            ((axis_2 + cell_size_2) % 1)
        ),  # up side percentage
    ])
    print(cell_crop)
    extents = np.array([
        # left right
        np.maximum(0, np.floor(axis_1 - cell_size_1 + 1)),  # +1 to take only full cells
        np.minimum(resolution - 1, np.floor(axis_1 + cell_size_1 - 1) + 1),
        # -1 to take only full cells, +1 to take into account for slices
        # up down
        np.maximum(0, np.floor(axis_2 - cell_size_2 + 1)),
        np.minimum(resolution - 1, np.floor(axis_2 + cell_size_2 - 1) + 1)
    ], dtype=int)

    percentage = 5
    print('Restructuring grid:')
    # TODO at the moment there is no spread in V
    dens = (mass / (cell_size_1 * cell_size_2 * 4)) / dV_axis
    for i in range(len(cell_size_1)):
        if i/len(cell_size_1) * 100 > percentage:
            print(str(percentage) + '%', end=' ')
            percentage += 5
        if 0 < VV_axis[i] < resolutionV:
            # full
            empty_grid[
                slice(extents[0, i],
                      extents[1, i] if extents[1, i] < resolution - 1 else None),
                slice(extents[2, i],
                      extents[3, i] if extents[3, i] < resolution - 1 else None),
                VV_axis[i]
            ] += dens[i]

            # CASE: cell smaller than grid
            if extents[0, i] > extents[1, i] and extents[2, i] > extents[3, i]:
                    empty_grid[
                        int(np.floor(axis_1[i])),
                        int(np.floor(axis_2[i])),
                        VV_axis[i]
                    ] += dens[i] * cell_size_1[i] * cell_size_2[i] * 4
            elif extents[0, i] > extents[1, i]:
                # down
                if extents[2, i] > 0:
                    empty_grid[
                        int(np.floor(axis_1[i])),
                        extents[2, i] - 1,
                        VV_axis[i]
                    ] += dens[i] * cell_crop[2, i] * cell_size_1[i] * 2

                # up
                if extents[3, i] < resolution - 1:
                    empty_grid[
                        int(np.floor(axis_1[i])),
                        extents[3, i],
                        VV_axis[i]
                    ] += dens[i] * cell_crop[3, i] * cell_size_1[i] * 2

            elif extents[2, i] > extents[3, i]:
                # left
                if extents[0, i] > 0:
                    empty_grid[
                        extents[0, i] - 1,
                        int(np.floor(axis_2[i])),
                        VV_axis[i]
                    ] += dens[i] * cell_crop[0, i] * cell_size_2[i] * 2
                # right
                if extents[1, i] < resolution - 1:
                    empty_grid[
                        extents[1, i],
                        int(np.floor(axis_2[i])),
                        VV_axis[i]
                    ] += dens[i] * cell_crop[1, i] * cell_size_2[i] * 2

            else:
                # Normal case
                # left
                if extents[0, i] > 0:
                    empty_grid[
                        extents[0, i] - 1,
                        slice(extents[2, i],
                              extents[3, i] if extents[3, i] < resolution - 1 else None),
                        VV_axis[i]
                    ] += dens[i] * cell_crop[0, i]
                # right
                if extents[1, i] < resolution - 1:
                    empty_grid[
                        extents[1, i],
                        slice(extents[2, i],
                              extents[3, i] if extents[3, i] < resolution - 1 else None),
                        VV_axis[i]
                    ] += dens[i] * cell_crop[1, i]

                # down
                if extents[2, i] > 0:
                    empty_grid[
                        slice(extents[0, i],
                              extents[1, i] if extents[1, i] < resolution - 1 else None),
                        extents[2, i] - 1,
                        VV_axis[i]
                    ] += dens[i] * cell_crop[2, i]

                # up
                if extents[3, i] < resolution - 1:
                    empty_grid[
                        slice(extents[0, i],
                              extents[1, i] if extents[1, i] < resolution - 1 else None),
                        extents[3, i],
                        VV_axis[i]
                    ] += dens[i] * cell_crop[3, i]

                # four angles
                # lower left
                if extents[2, i] > 0:
                    if extents[0, i] > 0:
                        empty_grid[
                            extents[0, i] - 1,
                            extents[2, i] - 1,
                            VV_axis[i]
                        ] += dens[i] * cell_crop[0, i] * cell_crop[2, i]

                    # lower right
                    if extents[1, i] < resolution - 1:
                        empty_grid[
                            extents[1, i],
                            extents[2, i] - 1,
                            VV_axis[i]
                        ] += dens[i] * cell_crop[1, i] * cell_crop[2, i]

                # upper left
                if extents[3, i] < resolution - 1:
                    if extents[0, i] > 0:
                        empty_grid[
                            extents[0, i] - 1,
                            extents[3, i],
                            VV_axis[i]
                        ] += dens[i] * cell_crop[0, i] * cell_crop[3, i]

                    # upper right
                    if extents[1, i] < resolution - 1:
                        empty_grid[
                            extents[1, i],
                            extents[3, i],
                            VV_axis[i]
                        ] += dens[i] * cell_crop[1, i] * cell_crop[3, i]

    return empty_grid

def ppp2ppv(Snap, edges, Vedges, resolution, resolutionV, proj_ax='z', only_H2=False):
    if Snap.yt_grid_data is None:
        Snap.load_yt_grid_data()
    gas = Snap.yt_grid_data

    axes_n, axes_l = return_axes_from_proj(proj_ax)
    velocity_axis = axes_l[2]

    if edges is None:
        code_length = ytqty(gas.units.code_length.get_conversion_factor(yt.units.pc)[0], 'pc')
        left_corner = gas.domain_left_edge.value * code_length
        right_corner = gas.domain_right_edge.value * code_length
        edges = [left_corner, right_corner]
    #Ncell_HR = np.array([((edges[1][0] - edges[0][0])/ytqty(0.0073,'pc')).value,
    #            ((edges[1][1] - edges[0][1])/ytqty(0.0073,'pc')).value,
    #            ((edges[1][2] - edges[0][2])/ytqty(0.0073,'pc')).value]).astype(int)

    region = gas.r[
             edges[0][axes_n[0]]: edges[1][axes_n[0]], #: Ncell_HR[0]*1j,
             edges[0][axes_n[1]]: edges[1][axes_n[1]],# : Ncell_HR[1]*1j,
             edges[0][axes_n[2]]: edges[1][axes_n[2]],# : Ncell_HR[2]*1j,
             ]

    grid = restructure_grid(region,
                         resolution,
                         resolutionV, edges, Vedges, axes_l)
    return grid




if __name__ == '__main__':
    SimType = 'HJ'
    SimPathFold = find_sim_path(SimType)
    Sim0 = Simulation(SimPathFold)
    Sim = Sim_PP_class(Sim0)

    file_path = '/run/media/psuin/Seagate Basic/PhD/Observation/Disperse/'

    time_snap = 3 #Myr
    resolution = 128
    edges = [ytarr([12, 14, 12], 'pc'),
             ytarr([16, 18, 16], 'pc')]
    Vedges = [ytqty(-5, 'km/s'), ytqty(5, 'km/s')]
    resolutionV = 50

    Snap = Sim.Sim.get_snap_at_time(3)

    ppv_grid = ppp2ppv(Snap, edges, Vedges, resolution, resolutionV)
    astrofits.writeto(file_path + 'PPV_{}__{}Myr__{}-{}-{}---{}-{}-{}__{}.fits'.format(SimType, time_snap,
                      int(edges[0][0].to_value()*10), int(edges[0][1].to_value()*10), int(edges[0][2].to_value()*10),
                      int(edges[1][0].to_value()*10), int(edges[1][1].to_value()*10), int(edges[1][2].to_value()*10),
                      resolution), ppv_grid.T)
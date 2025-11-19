from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from amunra.data.simulationData import Simulation
from amunra.data.PostProcessSim import Sim_PP_class
from amunra.utils import find_sim_path
from yt.units.yt_array import unyt_array as ytarr

from mydisperse.skl_hdf5_comparison import skl_hdf5_comparison
import os
import re
import h5py
import numpy as np

# TODO
# AT THE MOMENT ASSUMES HJ SIMULATION
def check_if_fits_folder_exists(output_num, edges_control, res_control:int):
    fits_storage = '/run/media/psuin/Seagate Basic/PhD/FITS_STORAGE/Test_Disperse/HJ/'
    snapshots = os.listdir(fits_storage)

    for file_fits in snapshots:
        p = re.compile(r'\d*\.?\d+|\d+')  # Compile a pattern to capture float values
        floats = [float(i) for i in p.findall(file_fits)]
        num = floats[0]
        edges = [ [floats[1],floats[2],floats[3]],[floats[4],floats[5],floats[6]]]
        res = floats[7]

        if output_num==num and res==res_control and\
            edges==edges_control:
            return fits_storage + file_fits
    return None

def create_hdf5_snapshot(snap, edges, res):
    fits_path = '/run/media/psuin/Seagate Basic/PhD/FITS_STORAGE/Test_Disperse/HJ/'
    name_file = fits_path + '{}_{}_{}_{}-{}_{}_{}_{}'.format(snap.num,
                                                 edges[0][0], edges[0][1], edges[0][2],
                                                 edges[1][0], edges[1][1], edges[1][2],
                                                 res)

    fits_grid_density = SimPP.get_fits_cube(snap.timeMYR,edges, res, quantity='Density')[0].data
    fits_grid_Bx = SimPP.get_fits_cube(snap.timeMYR,edges, res, quantity='magnetic_field_x', code='gas')[0].data
    fits_grid_By = SimPP.get_fits_cube(snap.timeMYR,edges, res, quantity='magnetic_field_y', code='gas')[0].data
    fits_grid_Bz = SimPP.get_fits_cube(snap.timeMYR,edges, res, quantity='magnetic_field_z', code='gas')[0].data
    fits_grid_vx = SimPP.get_fits_cube(snap.timeMYR,edges, res, quantity='velocity_x', code='gas')[0].data
    fits_grid_vy = SimPP.get_fits_cube(snap.timeMYR,edges, res, quantity='velocity_y', code='gas')[0].data
    fits_grid_vz = SimPP.get_fits_cube(snap.timeMYR,edges, res, quantity='velocity_z', code='gas')[0].data
    fits_grid_temp = SimPP.get_fits_cube(snap.timeMYR,edges, res, quantity='temperature', code='gas')[0].data
    tmp_array = [fits_grid_density,
                 fits_grid_Bx, fits_grid_By, fits_grid_Bz,
                 fits_grid_vx, fits_grid_vy, fits_grid_vz, fits_grid_temp]
    name_list = ['Density',
                 'B_x', "B_y", "B_z",
                 "V_x", "V_y", "V_z", "T"]

    f = h5py.File(name_file, 'w')
    for grid, name in zip(tmp_array, name_list):
        dset_d = f.create_dataset(name, data=grid)
        dset_d.attrs['units'] = str(grid.units)

    f.close()
    return name_file

path = find_sim_path('HJ')
Sim = Simulation(path)
SimPP = Sim_PP_class(Sim)
px_res = 128
grid_res = 512

disp_param = {'edges':[[10,10,10],[20,20,20]], 'res':10/px_res, 'cut':1e-19}
#disp_param = {'edges':[[12.5,12.5,12.5],[17.5,17.5,17.5]], 'res':5/px_res, 'cut':1e-19}
dt_time = 0.3
t0 = 1.9
t_max = 3.5

for snap in Sim.snapshots:
    if snap.timeMYR < t0 or snap.timeMYR > t_max:
        continue
    t0 = t0 + dt_time 
    print('Snap {} -- {:.2f}'.format(snap.num, snap.timeMYR))

    snap.load_filament_data()
    snap.filament_data.load_skl(disp_param, autorun=True, nsig=False)
    #snap.filament_data.convert_fits_2_vtp(disp_param)

    #lens = fils.skl.fil_data.get_property_array('len')
    #fils.skl.dump_persistence_diagram()
    #plt.show()

    #plt.hist(lens, bins=int(np.sqrt(len(lens))))
    #plt.show()


    file_h5 = check_if_fits_folder_exists(output_num=snap.num,
                                            edges_control=disp_param['edges'],
                                            res_control=grid_res)
    if file_h5 is None:
        file_h5 = create_hdf5_snapshot(snap, disp_param['edges'], grid_res)
    import time
    obj = skl_hdf5_comparison(file_h5, snap.filament_data.skl, skl_res=px_res)
    obj.keep_just_skeleton()

    fils = obj.filaments
    obj.trim('V',1e6, '<')
    obj.trim('Density', 1.67*2/0.7 * 2e3 * 1e-24, '>')
    j=0
    for fil in fils.fils:
        path_dir = '/home/psuin/Desktop/Plot_to_show/Disperse_Simulation/fil{}'.format(j)
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)
        j+=1

        results = obj.get_radial_density_cut(fil, zero_intensity_at_border=False,
                                                  normalise_to_max_intensity=False)
        radial_density, distances, ok = (results['radial_density'],
                                     results['distances'],
                                     results['ok'])
        if results['ok']<0:
            continue
        distances *= (obj.edges[1][0] - obj.edges[0][0])/grid_res
        n_ave=4
        i=0
        rd_ave = np.zeros_like(distances)
        plt.figure()
        for r in radial_density:
            if  i<n_ave:
                rd_ave += r/n_ave
                i+=1
            else:
                plt.plot(distances, rd_ave, 'k', alpha=0.2)
                rd_ave = np.zeros_like(distances)
                i=0

        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$R$ [pc]')
        plt.ylabel(r'$\langle\rho\rangle/\rho_\mathrm{max}$')
        plt.savefig(path_dir +'/00_filament.png')
        plt.close('all')
    #obj.trim_below_robustness_ratio(robustness_ratio=10)
    #obj.create_vtk_skeleton(path=fils.loaded_skl_fold)
    #pass

    ''' for fil in obj.filaments:
        plt.plot(fil.points[:,0], fil.points[:,1])
    plt.xlim(0,128)
    plt.ylim(0,128)
    plt.show()'''
    for fil in obj.filaments:
        plt.plot(fil.points[:,0], fil.points[:,1])
    plt.xlim(0,grid_res)
    plt.ylim(0,grid_res)
    plt.show()

    obj.trim('V', 10**6, '<')
    obj.plot_angle_density_phase_diagram('B')
    obj.plot_angle_density_phase_diagram('V', qty_ref='B')
    obj.plot_angle_density_phase_diagram('B', qty_ref='V')
    break
    #fig = snap.filament_data.skl.dump_persistence_diagram()
    #plt.show()

    
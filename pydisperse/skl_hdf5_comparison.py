import os.path

from .skel import Skel
import h5py
from amunra import cool_plot_specifics
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from apophis.misc import constants as cst
from yt.units.yt_array import unyt_array as ytarr
import re, os
from pydisperse.skel import Filament, CriticalPoint, Filaments
from matplotlib import gridspec
import matplotlib as mpl
import vtk
from copy import deepcopy

class skl_hdf5_comparison:
    def __init__(self, h5_path, skl, skl_res, purge_borders=False):
        # This flag determines whether we can still use pydisperse tools
        self.trimmed = False
        # skl_path can be the path or directly the skeleton
        h5file = h5py.File(h5_path)
        self.quantities = {}
        for key in list(h5file.keys()):
            # standart quantities are B_x, B_y, B_z, Density, V_x, V_y, V_z, T
            self.quantities[key] = ytarr(h5file[key][:].T, h5file[key].attrs['units']) # convert to array, transposing

        self.recover_parameters_from_name(h5_path)

        if type(skl)==str:
            self.skl_path = skl
            self.skl = Skel(skl)
        else:
            self.skl = skl

        self.filaments = self.skl.fil_data

        self.grid_data_dim = self.quantities['Density'].shape[0]
        self.disperse_res = skl_res
        # checking grid and skeleton have same resolution
        if skl_res != self.grid_data_dim :
            print('converting skeleton')
            self._convert_skeleton_at_HR_grid()

        self.get_property_filaments()
        if purge_borders:
            self.purge_borders()


    def recover_parameters_from_name(self, path):
        dirs = path.split('/')
        name = dirs[-1]
        p = re.compile(r'\d*\.?\d+|\d+')  # Compile a pattern to capture float values
        floats = [float(i) for i in p.findall(name)]
        self.num = floats[0]
        self.edges = [ [floats[1],floats[2],floats[3]],[floats[4],floats[5],floats[6]]]
        self.res = floats[7]
        self.dx = (self.edges[1][0] - self.edges[0][0])/self.res
        self.dy = (self.edges[1][1] - self.edges[0][1])/self.res
        self.dz = (self.edges[1][2] - self.edges[0][2])/self.res

    def get_property_filaments(self):
        for fil in self.filaments:
            self.get_quantity_along(fil)

    def _compare_different_resolutions(self):
        pass

    def get_quantity_along(self, fil, purge_edges=True):
        # TODO
        #  right now it doens't take into account shifts from the true positions
        xp, yp, zp = fil.points[:, 0].astype(int), fil.points[:, 1].astype(int), fil.points[:, 2].astype(int)

        # TODO
        #  Purging points at borders
        #  !! MIGHT GIVE PROBLEMS WITH PYDISPERSE
        #  Selection

        fil.quantities = {}
        for key in self.quantities.keys():
            data = self.quantities[key][:]
            # Maybe because of numpy arrays?
            fil.quantities[key] = data[xp, yp, zp]
        for qty in ['V', 'B']:
            # For the moment assumes this is V or B
            fil.quantities[qty] = (fil.quantities[qty + '_x'] ** 2 +
                                   fil.quantities[qty + '_y'] ** 2 +
                                   fil.quantities[qty + '_z'] ** 2) ** 0.5

    def get_angle_quantity_direction(self, fil, quantity):
        if quantity in ['B', 'magnetic_field']:
            quantity = 'B'
        if quantity in ['V', 'velocity', 'v']:
            quantity = 'V'
        if not hasattr(fil, 'direction'):
            ok = self._get_direction_filament(fil)
            if ok < 0:
                print('Something went wrong, perhaps short filament: {} points'.format(len(fil.points)))
                return ok

        fil.quantities['angle_with_' + quantity] = []
        for i in range(len(fil.points)):
            # scalar product and then dividing by intensity
            q_x = fil.quantities[quantity + '_x'][i].value
            q_y = fil.quantities[quantity + '_y'][i].value
            q_z = fil.quantities[quantity + '_z'][i].value
            angle_rad = self._get_angle_from_vectors(fil.direction[i], np.array([q_x, q_y, q_z]))
            fil.quantities['angle_with_' + quantity].append(angle_rad)
        fil.quantities['angle_with_' + quantity] = np.array(fil.quantities['angle_with_' + quantity])
        return 0

    def plot_angle_density_phase_diagram(self, qty, qty_ref='Density'):
        if qty in ['B', 'magnetic_field']:
            qty = 'B'
        if qty in ['V', 'velocity', 'v']:
            qty = 'V'
        if qty_ref in ['B', 'magnetic_field']:
            qty_ref = 'B'
        if qty_ref in ['V', 'velocity', 'v']:
            qty_ref = 'V'
        points_density = []
        points_degree = []

        # Necessary to avoid redundancies!! (dicts are faster)
        analysed_points = []

        for i, fil in enumerate(self.filaments):
            #points_density = [1e-20]
            #points_degree = [0]
            if qty_ref=='Density':
                if 'angle_with_' + qty not in fil.quantities.keys():
                    ok = self.get_angle_quantity_direction(fil, qty)
                if ok < 0 :
                    print('Skipping filament {}: {} points'.format(i, len(fil.points)))
                    continue
                angle_list = fil.quantities['angle_with_' + qty]
            else:
                angle_list = []

                for j in range(len(fil.points)):
                    # Compute the array of angles
                    v1 = np.array([fil.quantities[qty_ref+'_x'][j], fil.quantities[qty_ref+'_y'][j], fil.quantities[qty_ref+'_z'][j]])
                    v2 = np.array([fil.quantities[qty+'_x'][j], fil.quantities[qty+'_y'][j], fil.quantities[qty+'_z'][j]])

                    angle_list.append(self._get_angle_from_vectors(v1, v2))

                angle_list=np.array(angle_list)
            mask = self._get_non_matching_mask(points=fil.points, ref_points=analysed_points)

            # Here add additional masks if needed
            #mask_B = (fil.quantities['B_x']**2 + fil.quantities['B_y']**2 + fil.quantities['B_z']**2)**0.5 > 2e-5
            #mask = mask_B & mask
            tmp = fil.quantities[qty_ref][mask]
            if qty_ref == 'B':
                tmp = tmp * 10**6 # from G to muG
            elif qty_ref == 'V':
                tmp = tmp / 10**5 # from cm/s to km/s
            points_density.extend(list(tmp))
            points_degree.extend(list(angle_list[mask]))
            analysed_points.extend(fil.points[mask])

        # plotting
        x = np.array(points_density)
        y = np.array(points_degree)

        fig = plt.figure(figsize=(12, 12))
        mpl.rcParams['xtick.direction'] = 'out'
        mpl.rcParams['ytick.direction'] = 'out'
        mpl.rcParams['xtick.top'] = False
        mpl.rcParams['ytick.right'] = False

        gs = gridspec.GridSpec(4, 4, figure=fig)
        ax_joint = fig.add_subplot(gs[1:4, 0:3])
        ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

        nbin=int(len(x)**0.5 * 0.7)
        x_bins = np.logspace(start=np.log10(min(x)), stop=np.log10(max(x)), num=nbin)
        y_bins = np.linspace(0, np.pi/2, nbin)
        import matplotlib.colors as mcolors
        from scipy.ndimage import gaussian_filter
        cmap = plt.cm.plasma
        cmap.set_under('black')

        # 2D histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
        X, Y = np.meshgrid(xedges, yedges)
        sigma = 1  # Adjust the sigma value to change the smoothness
        smoothed_hist = gaussian_filter(hist, sigma=sigma)
        h = ax_joint.pcolormesh(X, Y, smoothed_hist.T, shading='auto', cmap=cmap, norm=mcolors.Normalize(vmin=1))
        #fig.colorbar(h, ax=ax_joint)

        #ax_joint.scatter(x,y)
        y = np.array(y)
        y_pi = y / np.pi
        unit = 0.125
        y_tick = np.arange(0, 0.5 + unit, unit)

        y_label = [r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3}{8}\pi$", r"$\frac{\pi}{2}$"]
        ax_joint.set_yticks(y_tick * np.pi)
        ax_joint.set_yticklabels(y_label)
        for i, tick in enumerate(ax_joint.get_yticklabels()):
            tick.set_fontsize(20 if i!=0 else 16)

        ax_joint.set_ylim(0, np.pi/2)
        ax_joint.set_xscale('log')
        if qty_ref == 'Density':
            ax_joint.set_xlabel(r'$\rho$ [g\,cm$^{-3}$]')
        elif qty_ref == 'B':
            ax_joint.set_xlabel(r'$|B|$ [$\mu G$]')
        elif qty_ref == 'V':
            ax_joint.set_xlabel(r'$|V|$ [$km/s$]')

        ax_joint.set_ylabel(r'$\theta_{{{}}}$'.format(qty))
        # Angle histogram
        ax_marg_y.hist(y, bins=y_bins, orientation='horizontal', density=True)
        ax_marg_y.set_xlabel(r'P($\theta_{{{}}}$)'.format(qty))

        # Plotting random distribution
        y_rand = np.linspace(0, np.pi/2)
        ax_marg_y.plot(np.sin(y_rand), y_rand )

        # Density histogram
        ax_marg_x.hist(x, bins=x_bins)
        ax_marg_x.set_ylabel('Counts')
        ax_marg_x.set_xscale('log')

        # Turn off tick labels on marginals
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        plt.show()

    def get_radial_density_cut(self, fil, space=12, points_per_distance=1, max_distance=40,
                                  normalise_to_max_intensity=False, zero_intensity_at_border=False,
                                  central_point=None
                                  ):
        # given a filament, it divides it into sets of 'space' points.
        # For each set:
        # it evaluates the radial profile in the perpedicular plane/line (ppp, ppv)
        # The difference with a projection is that you take only the field data around the filaments
        result_dict={'ok':1, 'radial_density':[], 'distances':[]}
        ok = self._get_direction_filament(fil, space//2)
        if ok < 0:
            print('Something went wrong, perhaps short filament: {} points'.format(len(fil.points)))
            result_dict['ok']=ok
            return result_dict

        # for the moment no using split array but doing a moving average with space points
        #segs = split_array(fil.points, space)
        #data_seg = self.fits_field

        data_grid = self.quantities['Density']
        coord_grid = np.indices(data_grid.shape)


        def mask_between_planes(points, direction, x0, x1):
            # perpendicular plane equation passing through x0:
            #   sum(ai*xi) = sum(ai*x0)
            plane0 = (x0*direction).sum()
            plane1 = (x1*direction).sum()
            tmp_matrix = points[:,0]*direction[0] +\
                        points[:,1]*direction[1] +\
                        points[:,2]*direction[2]
            mask0 = min(plane0, plane1) < tmp_matrix
            mask1 = max(plane0, plane1) > tmp_matrix

            return mask0 & mask1

        points_array_perp_quantity=[]
        distances_bin = np.linspace(0, max_distance, max_distance // points_per_distance)
        import time
        tt = time.time()
        print('Evaluating the radial profile')
        for i, point in enumerate(fil.points):
            x0 = fil.points[max(0, i-space//2)]
            x1 = fil.points[min(len(fil.points)-1, i+space//2+1)]
            #I'll make the line pass from the average point, to avoid errors due to angles
            x_ave = fil.points[max(0, i-space//2): min(len(fil.points), i+space//2)+2].mean(axis=0)
            direction = fil.direction[i]
            direction[2]=0
            conservative_box = max_distance + space
            interesting_grid = coord_grid[:,
                max(int(x_ave[0] - conservative_box), 0):min(int(x_ave[0] + conservative_box), coord_grid.shape[1]),
                max(int(x_ave[1] - conservative_box), 0):min(int(x_ave[1] + conservative_box), coord_grid.shape[2]),
                max(int(x_ave[2] - conservative_box), 0):min(int(x_ave[2] + conservative_box), coord_grid.shape[3]),
                ]

            coord_points = interesting_grid.transpose(1, 2, 3, 0).reshape(-1, 3)
            # I don't need points more distant than max_distance**2+space**2/4
            # but I don't even need a mask to calculate this, I can manually remove true slices
            mask_distance = ((coord_points-x_ave)**2).sum(axis=1) < 1.2*(max_distance**2+space**2)
            coord_points = coord_points[mask_distance]

            mask_planes = mask_between_planes(coord_points, direction, x0, x1)
            coord_points = coord_points[mask_planes]

            d = (np.cross(direction, x_ave - coord_points) ** 2).sum(axis=1)/(direction**2).sum()


            perp_qty = []
            for j in range(len(distances_bin)):
                if j == len(distances_bin) -1:
                    break
                dist0 = distances_bin[j]
                dist1 = distances_bin[j+1]
                mask0 = d > dist0
                mask1 = d < dist1
                mask = mask0 & mask1
                radial_data = data_grid[coord_points[mask][:,0], coord_points[mask][:,1], coord_points[mask][:,2]].mean()

                perp_qty.append(radial_data)
            perp_qty = np.array(perp_qty)
            if zero_intensity_at_border:
                perp_qty -= perp_qty.min()
            if normalise_to_max_intensity:
                perp_qty /= perp_qty.max()
            points_array_perp_quantity.append(perp_qty)
            if np.mod(i,5)==0:
                fig = plt.figure(1, figsize=(6, 12))
                ax_image = plt.subplot(2, 1, 1)

                from matplotlib.colors import LogNorm
                ax_image.imshow(self.quantities['Density']
                      [max(int(x_ave[0] - conservative_box), 0):min(int(x_ave[0] + conservative_box), coord_grid.shape[1]),
                       max(int(x_ave[1] - conservative_box), 0):min(int(x_ave[1] + conservative_box), coord_grid.shape[2]),
                       max(int(x_ave[2] - conservative_box), 0):min(int(x_ave[2] + conservative_box), coord_grid.shape[3]),
                      ].sum(axis=2).T,
                      origin='lower',
                      norm=LogNorm(vmin=5e-20, vmax=5e-17), cmap='inferno',
                      extent=np.array([max(int(x_ave[0] - conservative_box), 0),
                                         min(int(x_ave[0] + conservative_box), coord_grid.shape[1]),
                                         max(int(x_ave[1] - conservative_box), 0),
                                         min(int(x_ave[1] + conservative_box), coord_grid.shape[2])]) * \
                               (self.edges[1][0] - self.edges[0][0]) / self.grid_data_dim)

                ax_image.plot(fil.points[max(0, i - space // 2): min(len(fil.points), i + space // 2) + 2][:, 0] * (
                            self.edges[1][0] - self.edges[0][0]) / self.grid_data_dim,
                              fil.points[max(0, i - space // 2): min(len(fil.points), i + space // 2) + 2][:, 1] * (
                                          self.edges[1][0] - self.edges[0][0]) / self.grid_data_dim, 'b')


                x_ave *= (self.edges[1][0] - self.edges[0][0]) / self.grid_data_dim
                L = 0.5
                ll = np.linspace(-L,L,10000) + x_ave[0]
                if direction[1] == 0:
                    direction[1] = 1e-3
                yy = linear_fit(ll, -direction[0] / direction[1], x_ave[1] + direction[0] / direction[1] * x_ave[0])
                mask_line = (yy-x_ave[1])**2 + (ll-x_ave[0])**2 < 0.25
                ax_image.plot(ll[mask_line],
                              yy[mask_line],'k' )


                ax_graph = plt.subplot(2, 1, 2)
                ax_graph.plot(3/4* (distances_bin[:-1]**4-distances_bin[1:]**4)/
                         (distances_bin[:-1]**3-distances_bin[1:]**3) *
                         (self.edges[1][0] - self.edges[0][0])/self.grid_data_dim,
                         perp_qty, 'k')

                ax_graph.set_yscale('log')
                ax_graph.set_xscale('log')
                ax_graph.set_xlabel(r'$R$ [pc]')
                ax_graph.set_ylabel(r'$\langle\rho\rangle$ [g\,cm$^{-3}$]')

                plt.tight_layout()

                plt.savefig('/home/psuin/Desktop/Plot_to_show/Disperse_Simulation/fil{}/fil_seg{}'.format(
                    np.where(np.array(self.filaments.fils)==fil)[0][0],i))
                plt.close('all')

        result_dict['radial_density']=np.array(points_array_perp_quantity)
        result_dict['distances']= 3/4* ((distances_bin[:-1]**4-distances_bin[1:]**4)/
                                        (distances_bin[:-1]**3-distances_bin[1:]**3))
        return result_dict





    def trim(self, qty, threshold, condition , min_len=5):
        '''
        Given a thrshold, split the filaments retaining only the portions above the threshold.
        It retains only the filaments longer than min_len

        Note: the function only mimics the behaviour of the Critical points, so some functionalities may not be manteined
        '''
        self.trimmed = True
        new_filaments = []
        fil_to_remove = []
        for fil in self.filaments:
            # Conditions: gtr less
            if condition == 'gtr' or condition=='>':
                mask = fil.quantities[qty] > threshold
            elif condition == 'less' or condition=='<':
                mask = fil.quantities[qty] < threshold
            # Splits into segments above threshold
            indeces = self._split_mask_in_segments(mask)
            for i0, i1 in zip(indeces[0], indeces[1]):
                if len(fil.points[i0:i1]) >= min_len:
                    new_fil = self._create_new_filament(fil, i0, i1)
                    new_filaments.append(new_fil)

            # Remove old filament
            fil_to_remove.append(fil)
        self.filaments += new_filaments
        self.filaments -= fil_to_remove

        # Reinitialising new filaments
        self.get_property_filaments()
    def make_filaments_unique(self):
        '''
        eliminates overimposition of filaments.
        '''
    def keep_just_skeleton(self):
        '''
        The function keeps the main skeleton retrieved by disperse,
        '''
        self.trimmed = True
        self.filaments.get_property_array('_cp1')
        self.filaments.get_property_array('_cp2')
        c1_appearences = {'cp':[], 'N_appear':[], 'index_appear':[]}
        c2_appearences = {'cp':[], 'N_appear':[], 'index_appear':[]}

        def check_appeareances(cp_list, cp_app):
            for i, cc in enumerate(cp_list):
                if cc in cp_app['cp']:
                    cp_app['N_appear'][np.where(np.array(cp_app['cp']) == cc)[0][0]] += 1
                    cp_app['index_appear'][np.where(np.array(cp_app['cp']) == cc)[0][0]].append(i)
                else:
                    cp_app['cp'].append(cc)
                    cp_app['N_appear'].append(1)
                    cp_app['index_appear'].append([i])

        check_appeareances(self.filaments._cp2, c2_appearences)
        check_appeareances(self.filaments._cp1, c1_appearences)
        for fil in self.filaments:
            jj = np.where(np.array(c2_appearences['cp']) == fil._cp2)[0][0]
            fil.connections = c2_appearences['N_appear'][jj]

        self.filaments.get_property_array('connections')

        # recursive function that check the bifurcation points, and follow them until a new bifurcation is reached
        def find_fil_with_same_cp(point, filaments, typ):
            cp_list_tot = getattr(self.filaments, '_cp{}'.format(typ - 1))
            indexes = np.where(np.array(cp_list_tot) == point)[0]
            if len(indexes) != 2:
                # edge of the skeleton or connection point
                return None
            cp_list = getattr(filaments, '_cp{}'.format(typ- 1))
            indexes = np.where(np.array(cp_list) == point)[0]
            return indexes

        def find_cp(fil, typ):
            return getattr(fil, '_cp{}'.format(typ - 1))
        def find_other_cp(fil, typ):
            return getattr(fil, '_cp{}'.format(4 - typ))

        def connect_filaments(fil1, fil2, typ):
            if fil1 is not None:
                assert np.all(fil1.points[0] == fil2.points[2 - typ])
                # TODO
                #  Notice that there can be redundancies
                #  Might cause a problem when redundancy is not taken into account
                #  A possible solution is to separate the filaments above a certain fraction/number of points

                # Maximum is the second extreme -> fil1 comes after
                # the connection point is always the first point of fil1, last of fil2
                # if the connection is a maximum ok, otherwise we need to reverse the filament
                points = list(fil2.points)
                if typ == 2:
                    points.reverse()
                points.extend(list(fil1.points))

                # The new filament connects the maxima
                # cp2 remains fixed by contruction
                fil_tot = Filament(find_cp(fil2, typ), fil1._cp2, np.array(points))
                return fil_tot
            else:
                # first filament
                return fil2


        def follow_fil_crumbles(fil, fils_to_check, crit_point, connected_fil, typ):
            """
            recursively follows the filaments creatting the connected_fil.

            :param fil: current filament
            :param fils_to_check: list of remaining filaments
            :param crit_point: last cp analysed
            :param connected_fil: current connected filament
            :return: connected_fil
            """
            fils_to_check = fils_to_check - fil
            connected_fil = connect_filaments(connected_fil, fil, typ)
            # change the critical point
            crit_point = find_other_cp(fil, typ)
            typ = 3 if typ==2 else 2

            # finding the next filament or terminating
            fils_ind = find_fil_with_same_cp(crit_point, fils_to_check, typ)
            if fils_ind is None or len(fils_ind) > 1:
                # end of filament
                return connected_fil
            elif len(fils_ind) == 1:
                # continue looking for filament
                connected_fil = follow_fil_crumbles(fils_to_check[fils_ind[0]], fils_to_check, crit_point, connected_fil, typ)
                return connected_fil

        # Print statistics
        print('Connecting skeleton')
        print('\t\tN_fil\t\t <l>')
        print('Before\t {}\t\t{:.2f}'.format(len(self.filaments), np.mean(self.filaments.lenghts)))
        # Start from those appearing only once
        # cp2 are maxima, cp1 are saddles

        # create list to keep track of visited filaments
        remaining_fils_to_check = list(self.filaments)
        # create list to store new filaments
        new_fils = []
        remaining_fils_to_check = Filaments(remaining_fils_to_check)
        starting_fils  = remaining_fils_to_check[remaining_fils_to_check.connections != 2]

        while ( len(remaining_fils_to_check) > 0 and
                len(starting_fils) > 0
        ):

            current_filament = starting_fils[0]
            new_fil = follow_fil_crumbles(fil=current_filament, fils_to_check=remaining_fils_to_check,
                                crit_point=current_filament._cp2, # always starting with the maximum
                                connected_fil=None,typ=3)
            new_fils.append(new_fil)
            # Filaments have been updated
            remaining_fils_to_check.connections = remaining_fils_to_check.get_property_array('connections') # updating the list
            starting_fils = remaining_fils_to_check[remaining_fils_to_check.connections != 2]

        # TODO remaining_fils_to_check=0 means no isolated loops
        #assert len(remaining_fils_to_check) == 0

        self.filaments = Filaments(new_fils)
        self.get_property_filaments()

        print('After\t {}\t\t{:.2f}'.format(len(self.filaments), np.mean(self.filaments.lenghts)))



    def purge_borders(self, min_len=5):
        '''
        Split filaments touching the border
        It retains only the filaments longer than min_len

        Note: the function only mimics the behaviour of the Critical points, so some functionalities may not be manteined
        '''

        new_filaments = []
        fil_to_remove = []
        for fil in self.filaments:
            xp, yp, zp = fil.points[:, 0].astype(int), fil.points[:, 1].astype(int), fil.points[:, 2].astype(int)

            (bd0x, bd0y, bd0z,
             bd1x, bd1y, bd1z) = (xp==0, yp==0, zp==0,
                                  xp==self.res-1, yp==self.res-1, zp==self.res-1)
            mask = bd0x | bd0y | bd0z | bd1x | bd1y | bd1z
            # Splits into segments touching the border
            indeces = self._split_mask_in_segments(~mask)
            for i0, i1 in zip(indeces[0], indeces[1]):
                if len(fil.points[i0:i1]) >= min_len:
                    new_fil = self._create_new_filament(fil, i0, i1)
                    new_filaments.append(new_fil)

            # Remove old filament
            fil_to_remove.append(fil)
        self.filaments += new_filaments
        self.filaments -= fil_to_remove

        # Reinitialising new filaments
        self.get_property_filaments()
        self.trimmed=True

    def create_vtk_skeleton(self, path=None):

        if hasattr(self, 'skl_path'):
            path = os.path.dirname(self.skl_path)
        elif path is None:
            ValueError('Skeleton Folder path not provided')

        points = vtk.vtkPoints()
        indeces = []
        attributes = {}
        for attr in self.filaments[0].quantities.keys():
            attributes[attr] = (vtk.VTK_FLOAT, [])
        attributes['index'] = (vtk.VTK_INT, [])

        for i, fil in enumerate(self.filaments):
            xp, yp, zp = fil.points[:, 0].astype(int), fil.points[:, 1].astype(int), fil.points[:, 2].astype(int)

            for attr in self.filaments[0].quantities.keys():
                attributes[attr][1].extend( list( fil.quantities[attr].value ) )
            attributes['index'][1].extend(list(np.ones_like(xp) * i))

            for x, y, z in zip(xp, yp, zp):
                points.InsertNextPoint(x, y, z)

        # Create a vtkPolyData object
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)

        # Add set_index attribute to polyData
        for key in attributes.keys():
            vtkArray = self._create_vtk_array(attributes[key][1], key, attributes[key][0])
            polyData.GetPointData().AddArray(vtkArray)

        # Create a vertex for each point. VTK requires cells to visualize the points.
        vertices = vtk.vtkCellArray()
        for i in range(points.GetNumberOfPoints()):
            vertex = vtk.vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            vertices.InsertNextCell(vertex)

        polyData.SetVerts(vertices)


        # Write the .vtp file
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(path + "/modified_skeleton.vtp")
        writer.SetInputData(polyData)
        writer.Write()

    def _get_angle_from_vectors(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_1 = (v1[0]**2 + v1[1]**2 + v1[2]**2) ** 0.5
        norm_2 = (v2[0]**2 + v2[1]**2 + v2[2]**2) ** 0.5
        cos_angle = dot_product / (norm_1 * norm_2)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_rad = angle_rad if angle_rad < np.pi / 2 else np.pi - angle_rad
        return angle_rad

    def _get_non_matching_mask(self, points, ref_points):
        """
        Returns a mask for the points array indicating True for points not present in ref_points.
        """
        # Ensure the input arrays are numpy arrays
        if not len(ref_points):
            return np.ones(len(points), dtype=np.bool_)
        points = np.asarray(points)
        ref_points = np.asarray(ref_points)

        # Create an array of shape (len(points), 1) for broadcasting to work with np.all and np.any
        points_expanded = np.expand_dims(points, axis=1)

        # Compare each point against all reference points
        # The result is an array of shape (len(points), len(ref_points)) where each element [i, j]
        # is True if points[i] matches ref_points[j], otherwise False
        matches = np.all(points_expanded == ref_points, axis=2)

        # Use np.any to check if each point in points matches at least one point in ref_points
        # The result is negated to get points NOT in ref_points
        non_matching_mask = ~np.any(matches, axis=1)
        return non_matching_mask

    def _create_new_filament(self, fil, i0, i1):
        # create a new filament
        points = fil.points[i0: i1]
        vals = fil.quantities['Density'][i0: i1]

        cp1 = CriticalPoint(typ=fil._cp1.typ if i0 == 0 else 2,
                            pos=points[0],
                            val=vals[0],
                            pair=[],
                            boundary=self._check_if_point_at_bounday(points[0]),
                            destCritId=[],
                            filId=[])
        cp2 = CriticalPoint(typ=fil._cp2.typ if i1 is None else 2,
                            pos=points[-1],
                            val=vals[-1],
                            pair=[],
                            boundary=self._check_if_point_at_bounday(points[-1]),
                            destCritId=[],
                            filId=[])
        return Filament(cp1, cp2, points)

    def _check_if_point_at_bounday(self, point):
        boolean = point[0]==0 or point[1]==0 or point[2]==0 or \
                  point[0]==self.res-1 or point[1]==self.res-1 or point[2]==self.res-1
        return boolean


    def _create_vtk_array(self, data, name, array_type=vtk.VTK_FLOAT):
        """Creates a VTK array from a Python list."""
        vtk_array = vtk.vtkFloatArray() if array_type == vtk.VTK_FLOAT else vtk.vtkIntArray()
        vtk_array.SetName(name)
        for value in data:
            vtk_array.InsertNextValue(value)
        return vtk_array

    def _split_mask_in_segments(self, arr):
        # Given an boolean array, returns the starting and ending indeces of the sequence, ready to be sliced
        # [True, True, True, False, True, False, True, True] -> [[0, 6][3, None]
        pad_arr = np.pad(arr, (1, 1), mode='constant', constant_values=False) # Add False at the edges
        index_0 = 0
        index_1 = 0
        indeces = [[],[]] # low indeces, high indences
        for i in range(len(pad_arr)):
            if pad_arr[i]:  # value is True, equivalent to 1 in a boolean array
                index_1 = i
            else:  # value is False, sequence of ones has been interrupted
                if index_0!=index_1:
                    indeces[0].append(index_0 - 1) # -1 due to pad array
                    indeces[1].append(index_1)
                index_0 = i + 1
                index_1 = i + 1

        if len(indeces[1]) ==0:
            # no cell satisfies the condition
            return indeces
        if indeces[1][-1] == len(arr):
            indeces[1][-1] = None # making it a slice

        return indeces


    def _get_direction_filament(self, fil, space=20):
        # with a moving mask of 'space * 2' points, create the array that associate
        # the 3D normalised vector parallel to the filament to each point
        fil.direction = []
        if len(fil.points) < space*2:
            print('Filament shorter than 2 Spaces -- {} points.    Skipping'.format(space*2))
            return -1
        for i in range(len(fil.points)):
            # Dealing with borders
            min_ind = max(0, i - space)
            max_ind = min(i + space +1, len(fil.points))
            if max_ind == len(fil.points):
                max_ind = None
            # selecting the portion to average
            mask = slice(min_ind, max_ind)
            segment = fil.points[mask]
            # Calculate the mean of the points, i.e. the 'center' of the cloud
            datamean = segment.mean(axis=0)

            # Do an SVD on the mean-centered data.
            uu, dd, vv = np.linalg.svd(segment - datamean)
            # vv[0] is the vector of the data
            fil.direction.append(vv[0])
        return 0

    def _find_max_among_cells(self, x0, conversion):
        '''give the cell x0 in native resolution, looks for the conversion**3 cells nearby and returns the index of the maximum'''
        index = x0 * conversion + conversion//2
        mat = np.mgrid[
              max(index[0] - conversion,0):min(index[0] + (conversion+1),self.grid_data_dim),
              max(index[1] - conversion,0):min(index[1] + (conversion+1),self.grid_data_dim),
              max(index[2] - conversion,0):min(index[2] + (conversion+1),self.grid_data_dim),
               ].astype(int)
        dens = self.quantities['Density'][mat[0], mat[1], mat[2]]
        max_pos_index = np.where(dens==dens.max())
        point = np.array([mat[0][max_pos_index][0], mat[1][max_pos_index][0], mat[2][max_pos_index][0]])
        return point




    def _convert_skeleton_at_HR_grid(self):
        # for each pair of points in the filament
        def find_next_HR_cell(x0, x1, cells):
            # computes the "distance" (without a |x1-x0|^-1 factor) of a set of points and returns the lowest result
            direction = x1 - x0
            d = (np.cross(direction, x1 - cells) ** 2).sum(axis=1)
            return cells[np.where(d == d.min())[0][0]]


        new_filaments = []
        # test to verify the correct correspondence frid - skeleton
        '''self.quantities['Density'][156, 276, 158] *= 1000
        plt.imshow(self.quantities['Density'].sum(axis=2).T, origin='lower')
        plt.show()'''
        for fil in self.filaments:
            # Step 1:
            # convert skel data into high-res grid position
            # this assumes the grid covers the same region (of course)

            #converts the points
            new_points = []
            conversion = self.grid_data_dim / self.disperse_res
            #from matplotlib.colors import LogNorm
            #plt.imshow(self.quantities['Density'].sum(axis=2).T, origin='lower',
            #          norm=LogNorm(vmin=5e-20, vmax=5e-17), cmap='inferno')

            #plt.plot(fil.points[:,0]*conversion+conversion/2,
            #         fil.points[:,1]*conversion+conversion/2,'k')

            for point in fil.points:
                pp = self._find_max_among_cells(point, conversion)
                new_points.append(pp)
            new_points = np.array(new_points)
            # remove occurrences
            new_points = list(map(tuple, new_points))
            new_points = np.array([np.array(coord) for coord in new_points])

            fil.points = new_points
            fil._cp1.pos = new_points[0]
            fil._cp2.pos = new_points[-1]

            #plt.scatter(fil.points[:,0], fil.points[:,1], color='c')
            new_points = []

            for i in range(len(fil.points)):
                # end of filament
                if i == len(fil.points) - 1:
                    new_points.append(fil.points[i])
                    break

                # Step 2:
                # construct the new segment, connecting HR cells

                x0 = fil.points[i]
                x1 = fil.points[i + 1]

                direction = (x1 - x0).astype(int)

                current_point = x0

                '''All this is cool but unnecessary if the line is oriented along the movements in the disperse grid, 
                becuase the exact solution allows to follow the line'''

                while (current_point != x1).any():
                    new_points.append(current_point)

                    # 8 cases, always selecting 16 cells depending on the direction
                    movements = np.array([ [1,0,0], [0,1,0], [0,0,1],
                                [1,1,0], [1,0,1], [0,1,1],
                                [1,1,1]])
                    if direction[0] < 0:
                        movements[:,0] *= -1 	# negative x movement

                    if direction[1] < 0:
                        movements[:,1] *= -1 	# negative y movement

                    if direction[2] < 0:
                        movements[:,2] *= -1 	# negative z movement

                    candidates = movements + current_point
                    current_point = find_next_HR_cell(x0, x1, candidates)
                    if (current_point< 0 ).any() or (current_point>self.grid_data_dim).any():
                        ValueError('missed the point')

                '''else:
                    while (current_point != x1).any():
                        new_points.append(current_point)
                        # when current_point==x1 exits without appending. Either appends at the previous line or at the break
                        current_point = current_point + direction/max(direction)'''

            new_points = np.array(new_points)
            new_filaments.append(Filament(fil._cp1, fil._cp2, new_points))
            #plt.scatter(new_points[:,0], new_points[:,1], color='k', s=0.5)
            #plt.show()



        self.filaments = Filaments(new_filaments)
    def _debug_filaments(self, fils=None):
        if fils is None:
            fils = self.filaments
        for fil in fils:
            plt.plot(fil.points[:, 0], fil.points[:, 1])
        plt.xlim(0, self.grid_data_dim)
        plt.ylim(0, self.grid_data_dim)
        plt.show()
def linear_fit(x, a, b):
    return a*x+b

def b_from_min_distance_from_zero(slope, distance):
    # takes into account the sign of b
    return distance * np.sqrt(slope**2 + 1)

def min_distance_from_zero_from_b(a, b):
    # takes into account the sign of b
    return b * np.sqrt( 1 - a**2/(1 + a**2) )

from .typing import *
from .skel import Skel, Filament
from astropy.io import fits as astrofits
# from amunra import p
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
# from apophis.misc import constants as cst
import astropy.units as u
class skl_fits_comparison():
    def __init__(self, fits_path: str, skl_path: str, grid_type: Literal['ppp','ppv'] = 'ppv', distance: float = 3900):
        tmp = astrofits.open(fits_path)
        self.fits_field = tmp[0].data.T # Transposing right now to avoid inconsistencies with SKELDATA
        self.fits_header = tmp[0].header
        self.skl = Skel(skl_path)
        self.filaments = self.skl.fil_data
        self.grid_type = grid_type
        self.get_density_filaments()
        self.distance = distance

    def convert_to_standard(self, array: FloatArray, num_axis: int) -> FloatArray:
        central_v = self.fits_header['CRVAL{}'.format(num_axis)]
        len_axis  = self.fits_header['NAXIS{}'.format(num_axis)]
        delta     = self.fits_header['CDELT{}'.format(num_axis)]

        min_val = central_v - len_axis/2 * delta
        array_converted = array * delta + min_val
        return array_converted

    # REMEMBER FOR PPV: CUBES HAVE DIMENSION
    # MASS/(dx^2 * dv)
    # MULTIPLY FOR dv to get local column density of that velocity bin
    def get_density_filaments(self) -> None:
        for fil in self.filaments:
            self.get_density_along(fil)

    def get_density_along(self, fil: Filament) -> None:
        # TODO
        #  right now it doens't take into account shifts from the true positions
        xp = fil.points[:, 0].astype(int)
        yp = fil.points[:, 1].astype(int)
        zp = fil.points[:, 2].astype(int)
        # remember in ppv this is not density!!!
        if self.grid_type == 'ppp':
            fil.density = self.fits_field[xp, yp, zp]
        if self.grid_type == 'ppv':
            DV_step = self.fits_header['CDELT3']
            fil.col_density = self.fits_field[xp, yp, zp] * DV_step
            fil.intensity = self.fits_field[xp, yp, zp]
        else:
            raise ValueError('Unknown field type')

    def plot_filaments_in_dens_vs_velocity_space(self) -> None:
        # Assumes that the filaments are extracted from a ppv datacube
        if self.grid_type == 'ppp':
            raise TypeError("This is a ppp skeleton, you don't have such informtion")
        print(len(self.filaments))
        for fil in self.filaments:
            # converting in units
            Vrad_fils = self.convert_to_standard(fil.points[:, 2], num_axis=3)/10**3
            plt.plot(fil.intensity, Vrad_fils)

        plt.xscale('log')
        plt.xlabel(r'Intensity\\,[K (km/s)$^{-1}$]')
        plt.ylabel(r'V$_\\mathrm{rad}$ [km\\,s$^{-1}$]')
        plt.show()

    def plot_filaments_in_phys_length_vs_velocity_space(self) -> None:
        # Assumes that the filaments are extracted from a ppv datacube
        if self.grid_type == 'ppp':
            raise TypeError("This is a ppp skeleton, you don't have such informtion")
        print(len(self.filaments))
        for fil in self.filaments:
            proj_pos = fil.points[:, :2]
            vel = fil.points[:, 2]
            proj_len = (np.roll(proj_pos, 1, axis=0) - proj_pos)[1:]
            vel_displacement = (np.roll(vel, 1, axis=0) - vel)[1:]

            # Converting units
            proj_len[:, 0] = proj_len[:, 0] * self.fits_header['CDELT1'] # RA array
            proj_len[:, 1] = proj_len[:, 1] * self.fits_header['CDELT2'] # DEC array
            vel_displacement = vel_displacement * self.fits_header['CDELT3']/10**3

            segs_len = np.cumsum(np.sqrt(np.sum(proj_len ** 2, axis=1)))
            vel_len = np.cumsum(vel_displacement)

            plt.plot(segs_len, vel_len)

        plt.xlabel(r'Length\\,[degrees]')
        plt.ylabel(r'V$_\\mathrm{rad}$ [km\\,s$^{-1}$]')
        plt.show()


    def get_density_cut_histogram(self, fil: Filament, space: int = 10, points_per_distance: int = 1, 
                                  max_distance: float = 30, normalise_to_max_intensity: bool = False, zero_intensity_at_border: bool = False, central_point: Optional[Sequence] = None
                                  ) -> list[FloatArray]:
        """        
            given a filament, it divides it into sets of 'space' points.
            For each set:
            it evaluates the radial profile in the perpedicular plane/line (ppp, ppv)
            The difference with a projection is that you take only the field data around the filaments
        """
        #########
        # Remember that one is the velocity space. Can be used to select the extent of the filament but not for the distance
        def split_array(arr, size):
            arrs = []
            while len(arr) > size:
                piece = arr[:size]
                arrs.append(piece)
                arr = arr[size:]
            arrs.append(arr)
            return arrs

        if space < 3:
            ValueError('Cannot compute perpendicular line with less than 3 points')
        segs = split_array(fil.points, space)
        data_seg = self.fits_field
        grid = np.indices(data_seg.shape)

        if self.grid_type == 'ppv':

            def mask_between_lines(indices_matrix, a, b1, b2):
                b_min = min(b1, b2)
                b_max = max(b1, b2)
                mask_grid_up = indices_matrix[1] > indices_matrix[0] * a + b_min
                mask_grid_low = indices_matrix[1] < indices_matrix[0] * a + b_max


                return mask_grid_up & mask_grid_low
            x = fil.points[:, 0].astype(int)
            y = fil.points[:, 1].astype(int)
            v = fil.points[:, 2].astype(int)
            segs_plot=[]
            for ana_seg in segs:
                x = ana_seg[:, 0].astype(int)
                y = ana_seg[:, 1].astype(int)
                v = ana_seg[:, 2].astype(int)

                if all(x == x[0]) and all(y == y[0]):
                    continue
                # compute perpendicular line
                popt, pcov = curve_fit(linear_fit, x, y)
                # creates the lines that contain the interesting data
                perp_slope = -1/popt[0]
                perp_b_min = (popt[0] + 1/popt[0]) * x.min() + popt[1]
                perp_b_max = (popt[0] + 1/popt[0]) * x.max() + popt[1]

                # compute region around the filament, costrained by the velocity
                mask_grid_pos = mask_between_lines(grid, perp_slope, perp_b_min, perp_b_max)
                mask_grid_v = (grid[2] > v.min()) & (grid[2] <= v.max()+1)
                combined_mask = mask_grid_pos & mask_grid_v
                #interesting_data = data_seg.copy()
                #interesting_data[~combined_mask] = 0
                #interesting_data_2d = interesting_data.sum(axis=2)

                # compute the distance from 0, so that we can divide the remaining data depending on distance from fil
                d_0 = min_distance_from_zero_from_b(popt[0], popt[1])
                d_lines = np.linspace(-max_distance + d_0, max_distance + d_0, max_distance*2//points_per_distance)
                b_lines = b_from_min_distance_from_zero(popt[0], d_lines)

                '''plt.plot(x, y)
                xx = np.linspace(x.min() - 10, x.max() + 10, 100)
                plt.plot(xx, linear_fit(xx, *popt))
                plt.plot(xx, linear_fit(xx, perp_slope, perp_b_min))
                plt.plot(xx, linear_fit(xx, perp_slope, perp_b_max))
                plt.axis('equal')
                plt.show()
                '''
                '''interesting_data = data_seg.copy()
                interesting_data[~combined_mask] = 0
                interesting_data_2d = interesting_data.sum(axis=2)
                plt.imshow(data_seg.sum(axis=2).T, origin='lower')

                #plt.imshow(interesting_data_2d.T, origin='lower')
                plt.plot(x, y, 'r')
                plt.show()'''

                # computes the points within a certain distance from the filament
                longitudinal_intensity = []
                i = -1
                for b_min, b_max in zip(b_lines[:-1],b_lines[1:]):
                    i += 1
                    mask_distance = mask_between_lines(grid, popt[0], b_min, b_max)
                    distance_points_mask = combined_mask & mask_distance
                    '''
                    if i == len(d_lines)//2:
                    mask_distance = mask_between_lines(grid, popt[0], b_lines[0], b_lines[-1])
                    distance_points_mask_tmp = combined_mask & mask_distance
                    interesting_data = data_seg.copy()
                    interesting_data[~distance_points_mask_tmp] = 0
                    interesting_data_2d = interesting_data.sum(axis=2)
                    plt.imshow(interesting_data_2d.T, origin='lower')
                    plt.plot(x, y)
                    plt.show()'''

                    value_of_the_field_at_distance = data_seg[distance_points_mask].mean()
                    longitudinal_intensity.append(value_of_the_field_at_distance)
                dist_pc = self.convert_pxs_in_pc( (d_lines[:-1]+d_lines[1:])/2 - d_0)
                # Change sign according to central_point position
                if central_point is not None:
                     # b of central point less than b line
                     b_point = central_point[1] - popt[0]*central_point[0]
                     if b_point < popt[1]:
                         dist_pc = -dist_pc

                longitudinal_intensity = np.array(longitudinal_intensity)

                if zero_intensity_at_border:
                    longitudinal_intensity -= longitudinal_intensity.min()
                if normalise_to_max_intensity:
                    longitudinal_intensity /= longitudinal_intensity.max()
                segs_plot.append([dist_pc, longitudinal_intensity])

        return segs_plot

    def convert_pxs_in_pc(self, dist_pxs: FloatArrayLike) -> FloatArrayLike:
        res = abs(self.fits_header['CDELT1']) * 3600 # from degrees to arcseconds
        phys_dist = res * self.distance * u.au/u.pc * dist_pxs
        return phys_dist

def linear_fit(x, a: float, b: float) -> FloatArrayLike:
    return a*x+b

def b_from_min_distance_from_zero(slope: float, distance: FloatArrayLike) -> FloatArrayLike:
    # takes into account the sign of b
    return distance * np.sqrt(slope**2 + 1)

def min_distance_from_zero_from_b(a, b):
    # takes into account the sign of b
    return b * np.sqrt( 1 - a**2/(1 + a**2) )
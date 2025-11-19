from skel import Skel
from skl_fits_comparison import skl_fits_comparison
import matplotlib.pyplot as plt
import numpy as np
from amunra import cool_plot_specifics
file_path = '/run/media/psuin/Seagate Basic/PhD/Observation/RCW79'
cut = 0.5
folder = 'CII/'.format(str(cut))
file_name_skl = file_path + '/' + folder + '/' + 'cut_{}/skl_c{}.up.NDskl.a.NDskl'.format(str(cut), str(cut))
file_name_fits = file_path + '/' + folder + '/' + 'Mod--RCW79_CII_20_8_0p5.fits_smothed-3_CutSigma-5_no_less_1000__MD_MD_WN.fits'

skeleton = Skel(file_name_skl)
#skeleton.fil_data._cp1
skeleton.persistence_diagram()
#skeleton.persistence_histogram()
#plt.show()
#plt.show()

PP_obj = skl_fits_comparison(file_name_fits, file_name_skl)
#PP_obj.plot_filaments_in_dens_vs_velocity_space()
fig, ax = plt.subplots()
filaments_plot = []
for fil in PP_obj.filaments:
    fil_plot = PP_obj.get_density_cut_histogram(fil,
                                     zero_intensity_at_border=True,
                                     normalise_to_max_intensity=True, central_point=[PP_obj.fits_field.shape[0]/2, PP_obj.fits_field.shape[1]/2]
                                                )

    for fil_seg in fil_plot:
        plt.plot(fil_seg[0], fil_seg[1], color='k', alpha=0.05, linewidth=1)
        #plt.show()
        filaments_plot.append(fil_seg)



n_points = 60
d_lines = np.linspace(-30,30, n_points)
d_lines = PP_obj.convert_pxs_in_pc(d_lines)
repetitions = np.zeros_like(d_lines)
I_unique = np.zeros_like(d_lines)

print('Entering for loop')
percentage = 1

for fil in filaments_plot:
    for i in range(len(fil[0])):
        distance = fil[0][i]
        intensity = fil[1][i]

        if np.isnan(intensity):
            continue
        try:
            index = np.where(d_lines > distance)[0][0]
        except:
            index = n_points - 1

        repetitions[index] += 1

        I_unique[index] += intensity

I_unique = np.array(I_unique)
repetitions = np.array(repetitions)

I_unique[repetitions!=0] = I_unique[repetitions!=0]/repetitions[repetitions!=0]
def lorentian(x, a, d, c, k):
    return a * 1 / (1 + ((x-c)/d)**2) + k
def lorentian_and_gaussian(x, sigma, A, a, d, c, k):
    return A*np.e**(-x**2/(2*sigma**2)) + k + a * 1 / (1 + ((x-c)/d)**2)
def gaussian(x, sigma, A, k):
    return A*np.e**(-x**2/(2*sigma**2)) + k
from scipy.optimize import curve_fit

plt.plot(d_lines, I_unique, 'k')



'''
xx = np.linspace(-30,30, 10000)
mask = (d_lines < 1.5) & (d_lines > -1.5)
popt_lor, pcov_lor = curve_fit(lorentian, d_lines[mask], I_unique[mask])
popt, pcov = curve_fit(lorentian_and_gaussian, d_lines, I_unique, p0=[1,0.3, *popt_lor],bounds=([0,0,0,0,-1,-1], [100,1,1,10,1,1]))
print(popt)
plt.plot(xx, lorentian(xx, *popt_lor),'r', label='Lorentzian fit')

plt.plot(xx, lorentian_and_gaussian(xx, *popt),'b', label='Lorentzian+Gaussian fit')
plt.plot(xx, gaussian(xx, popt[0], popt[1], popt[-1]), 'g--', label='Gaussian component')
'''
plt.xlabel('Distance [pc]')
plt.ylabel('Normalised intensity')
plt.legend()
plt.show()
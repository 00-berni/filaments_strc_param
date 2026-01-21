import numpy as np
import matplotlib.pyplot as plt
import argparse

import filpy
from filpy import mydisperse as myd
from filpy.typing import *
from filpy import cube_stuff as cb


class Target():
    def __init__(self, filepath: str, hotpx: bool = True, verbose: bool = True):
        from os import path
        self.name = path.split(filepath)[-1]
        self._path = filepath
        self.read_data(hotpx=hotpx,verbose=verbose)

    def read_data(self, hotpx: bool, verbose: bool) -> None:
        hdul, data = filpy.get_data_fit(self._path,hotpx=hotpx,print_header=verbose, display_plots=False)
        wcs = cb.WCS(hdul[0].header)

        self.hdul = hdul.copy()
        self.header = hdul[0].header
        self.data = data.copy()
        self.wcs  = wcs.copy()

    @property
    def path(self):
        return self._path

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def nickname(self) -> str:
        name_list = self.name.split('_')
        name = name_list[0]
        if 'row' in name_list:
            name = name + '_row'
        return name

    def px_to_coord(self, *sel: IntArrayLike) -> FloatArray:
        coord = self.wcs.pixel_to_world_values(*sel)
        return np.asarray(coord)
    
    def coord_to_px(self, *sel: FloatArrayLike) -> IntArray:
        coord = self.wcs.world_to_pixel_values(*sel)
        return np.asarray(coord)

    def plot(self, **figargs):
        if 'barlabel' not in figargs.keys():
            figargs['barlabel'] = INT_UNIT
        if 'title' not in figargs.keys():
            figargs['title'] = self.name
        return filpy.show_image(self.data, projection=self.wcs,**figargs)
    
    def disperse(self, nsig: float, ncut: Optional[float], nsmooth: int, nthreads: int, patches: bool = True, outdir: str = '', skl_sel: str = 'skl_brk') -> None:
        if len(self.shape) == 3:
            dim = '3D'
        elif len(self.shape) == 2:
            dim = '2D'
        else:
            raise ValueError('Only 3D or 2D data are allowed')  
        self.skl_name = myd.run_disperse(self.path,nsig=nsig,cutp=ncut,nsmooth=nsmooth,patches=patches,nthreads=nthreads,dim=dim,outdir=outdir)
        self.skel(skl_sel=skl_sel)

    def skel(self, skl_sel: str = 'skl_brk') -> None:
        self.skl = myd.Skel(self.skl_name[skl_sel])
        self.fil_pos = np.empty((2,0))
        for fil in self.skl.fil:
            self.fil_pos = np.append(self.fil_pos,fil.points.T,axis=1)

    def plot_network(self,**figargs):
        figargs['colorbar'] = False
        figargs['show'] = False
        try:        
            fig, ax = self.plot(**figargs)
            ax.plot(*self.fil_pos,'.b',alpha=0.7)
            return fig, ax
        except AttributeError:
            Warning('To run self.plot_network, self.disperse() or self.skel() is required')
            pass

    def __getitem__(self, sel: Union[int,slice,list,tuple]) -> FloatArrayLike:
        return self.data[sel]

    def __setitem__(self, sel: Union[int,slice,list,tuple], item: FloatArrayLike) -> None:
        self.data[sel] = item

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        output_str = 'File' + self._path + '\n' + self.header.tostring(sep='\n')
        return output_str 
    
    def __repr__(self) -> str:
        return self.header.tostring(sep='\n')

class TargetIterator():
    def __init__(self, targets: list[Target]):
        self._targets = targets
        self._index = 0

    def __next__(self) -> Target:
        ''''Returns the next value from team object's lists '''
        if self._index < len(self._targets):
            self._index += 1
            return self._targets[self._index - 1]
        raise StopIteration


class TargetList():
    def __init__(self, file_names: list[str], hotpx: Union[bool,list[bool]] = True, verbose: Union[bool,list[bool]] = True):
        if isinstance(hotpx,bool):
            hotpx = [hotpx]*len(file_names)
        if isinstance(verbose,bool):
            verbose = [verbose]*len(file_names)

        targets = []
        for name, hpx, verb in zip(file_names,hotpx,verbose):
            targets += [Target(filepath=name,hotpx=hpx,verbose=verb)]
        
        self.paths = file_names
        self._hotpx = hotpx
        self._verbose = verbose
        self.targets: list[Target] = targets

    @property
    def names(self) -> list[str]:
        return [trg.name for trg in self.targets]
    @property
    def hdul(self):
        return [trg.hdul for trg in self.targets]
    @property
    def data(self) -> list[FloatArray]:
        return [trg.data for trg in self.targets]
    @property
    def header(self):
        return [trg.header for trg in self.targets]
    @property
    def wcs(self):
        return [trg.wcs for trg in self.targets]

    def plot(self, show: bool = True, *args: dict):
        diff = len(self.targets) - len(args)
        if diff > 0:
            args = [*args] + [{}]*diff
        
        if not show: 
            figs = []
            axes = [] 
        for kwargs, trg in zip(args, self.targets):
            kwargs['show'] = False
            fig, ax = trg.plot(**kwargs)
            if not show:
                figs += [fig]
                axes += [ax]
        if not show:
            return figs, axes
        else:
            plt.show()
            return

    def disperse(self, nsig: Union[float, list[float]], ncut: Union[Optional[float], list[Optional[float]]], nsmooth: Union[int, list[int]], nthreads: int, patches: bool = True, outdir: str = '', skl_sel: str = 'skl_brk') -> None:
        self.skl_names = []
        self.skls = []
        self.fils = []

        if not isinstance(nsig,Sequence):
            nsig = [nsig] * len(self)
        if not isinstance(ncut,Sequence):
            ncut = [ncut] * len(self)
        if not isinstance(nsmooth,Sequence):
            nsmooth = [nsmooth] * len(self)

        
        for trg, sig, cut, smooth in zip(self.targets, nsig, ncut, nsmooth):
            trg.disperse(nsig=sig,
                         ncut=cut,
                         nsmooth=smooth,
                         nthreads=nthreads,
                         patches=patches,
                         outdir=outdir,
                         skl_sel=skl_sel)
            print('TARGET',trg.skl)
            self.skl_names += [trg.skl_name]
            self.skls += [trg.skl]
            self.fils += [trg.fil_pos]

    def plot_network(self, show: bool = True, *args: dict):
        diff = len(self.targets) - len(args)
        if diff > 0:
            args = [*args] + [{}]*diff
        
        if not show: 
            figs = []
            axes = [] 
        for kwargs, trg in zip(args, self.targets):
            fig, ax = trg.plot_network(**kwargs)
            if not show:
                figs += [fig]
                axes += [ax]
        if not show:
            return figs, axes
        else:
            plt.show()
            return 


    def __getitem__(self, sel: Union[int,slice]) -> Union[Target,'TargetList']:
        if isinstance(sel,int):
            return self.targets[sel]
        else:
            return TargetList(self.paths[sel],hotpx=self._hotpx[sel],verbose=self._verbose[sel])

    def __len__(self) -> int:
        return len(self.targets)

    def __iter__(self) -> TargetIterator:
        return TargetIterator(self.targets)
    
    def __str__(self) -> str:
        intro = 'List of Target'
        if len(self) > 1: intro = intro + 's'
        intro = intro +':\n' + '\n'.join(self.names)
        return intro

    def __repr__(self) -> str:
        return 'TargetList object\n'+self.__str__()

def sobel_filter(data: FloatArray, remove_neg: bool = True) -> FloatArray:
    from scipy.ndimage import sobel
    sobel_filt_x = sobel(data,axis=0)
    sobel_filt_y = sobel(data,axis=1)
    sobel_filt = np.sqrt(sobel_filt_x**2 + sobel_filt_y**2)
    fdata = data - sobel_filt
    if remove_neg:
        fdata[fdata < 0] = 0
    return fdata


INT_UNIT = 'MJy / sr'

IR_FILES = filpy.IR_PATHS

MY_DIR = filpy.DataDir(path=(filpy.PROJECT_DIR -1).join('mytest'),mkdir=True)


if __name__ == '__main__':

    def mask_values(data, data_s, data_gs, cp_field, int_sigma,shift,**pltkwargs):
        ymax, xmax = filpy.find_argmax(data_gs)
        max_obj = data[ymax-shift:ymax+shift+1,xmax-shift:xmax+shift+1].copy()
        cp_obj  = data_s[ymax-shift:ymax+shift+1,xmax-shift:xmax+shift+1].copy()
        cpg_obj = data_gs[ymax-shift:ymax+shift+1,xmax-shift:xmax+shift+1].copy()

        figg, axx = filpy.show_image([data,max_obj,cp_obj,cpg_obj],num_plots=(2,2),
                                     subtitles=['Field','Field obj','Sobel','Gauss+Sobel'],
                                     cmap='viridis',
                                     colorbar=False,
                                     **pltkwargs)
        axx[0,0].plot(xmax,ymax,'xr')
        axx[1,0].plot(*np.where(cp_obj <= 0)[::-1],'.r')
        plt.show()

        ypos, xpos = np.where(cp_obj <= 0)
        xpos = xpos[abs(xpos-shift) <= 2*int_sigma]
        ypos = ypos[abs(ypos-shift) <= 2*int_sigma]
        min_len = min(len(xpos),len(ypos))
        maxdist = max(np.sqrt((xpos[:min_len]-shift)**2+(ypos[:min_len]-shift)**2))

        bkg = np.median(cp_obj[cp_obj>0])
        print('BKG:',bkg)
        from matplotlib.colors import LogNorm
        fig0, ax0 = filpy.show_image(max_obj, 
                                     title='Original',
                                     norm=LogNorm(),
                                     cmap='viridis')
        ax0.plot(shift,shift,'xr')
        fig2, ax2 = filpy.show_image(cpg_obj, 
                                     title='Gaussian + Sobel',
                                     norm=LogNorm(),
                                     cmap='viridis')
        ax2.plot(shift,shift,'xr')

        yy, xx = np.meshgrid(np.arange(max_obj.shape[0])-shift,
                             np.arange(max_obj.shape[1])-shift
                            )
        dist_mat = np.sqrt(xx**2+yy**2)
        dists = np.sort(np.unique(dist_mat))
        avg_profile = np.empty(0)
        from scipy.optimize import curve_fit
        fig = plt.figure()
        ax = fig.add_subplot()
        for d in dists:
            value = max_obj[dist_mat == d]
            avg_profile = np.append(avg_profile,np.mean(value))
            ax.plot([d]*len(value),value,'x')
        ax.plot(dists,avg_profile,'.--',color='black')
        avg_val = np.mean(avg_profile[dists > maxdist])
        filpy.h_lines(ax,
                      [bkg,np.median(data),avg_val],
                      ['green','orange','red'],
                      linestyles='dashed')
        filpy.v_lines(ax,
                      [maxdist,int_sigma*2,shift],
                      ['violet','blue'],
                      linestyles='dotted')

        grad1 = np.diff(avg_profile)/np.diff(dists)
        print('AVG VAL:',avg_val)
        filpy.quickplot(grad1,
                        title='Grad 1',
                        fmt='.--')
        filpy.quickplot([(dists[1:]+dists[:-1])/2, avg_profile[1:]/avg_profile[:-1]],
                        title='Ratio',
                        fmt='.--')
        plt.show()

        bkg = avg_val
        ypos, xpos = np.where(cp_obj<=0)
        cp_obj[cp_obj<=0] = bkg

        from matplotlib.patches import Circle
        fig1, ax1 = filpy.show_image(np.where(cp_obj<0,bkg,cp_obj),
                                     title='Sobel',
                                     norm=LogNorm(),
                                     cmap='viridis')
        ax1.plot(shift,shift,'xr')
        ax1.add_patch(Circle((shift,shift),maxdist,fill=False))

        cp_field[ypos+ymax-shift,xpos+xmax-shift] = bkg
        fig3, ax3 = filpy.show_image(cp_field,
                                     title='Masked field',
                                     **pltkwargs)
        ax3.plot(xmax,ymax,'xr')
        maxdist = int(maxdist)
        data_gs[ymax-maxdist:ymax+maxdist+1,xmax-maxdist:xmax+maxdist+1] = 0
        _ = filpy.show_image(data_gs,title='Removal',**pltkwargs)
        plt.show()
        return data_gs, cp_field



    parser = argparse.ArgumentParser(prog='TestIR',
                                     description='Read and analyze IR data',
                                    )
    ## Commands
    parser.add_argument('-l','--list', action='store_true', help='print the list of file in the data directory')
    parser.add_argument('-s','--selections', action='store', type=int, nargs='*', default=[], help='index(ces) of the selected object(s)')
    parser.add_argument('-o','--open-data',action='store_true')
    # filtering commands
    parser.add_argument('-f','--filter', action='store_true', help='sobel filter')
    parser.add_argument('--sigma', action='store', type=float, nargs='*', default=[2], help='set the sigma for the Gaussian filter. By default `2`')
    # masking commands
    parser.add_argument('-m','--mask', action='store_true', help='mask point sources')
    parser.add_argument('--nshift', action='store', type=int, nargs='*', default=[5], help='set the number of sigma from the point source. By default `5`')
    # DisPerSe commands
    parser.add_argument('-d','--disperse',action='store_true', help='run disperse')
    parser.add_argument('--nsig', action='store', type=float, nargs='*', default=[0.3], help='set the value of nsig. By default `0.3`')
    parser.add_argument('--nsmooth', action='store', type=int, nargs='*', default=[4], help='set the number of smoothing routines. By default `4`')
    parser.add_argument('--ncut', action='store', type=float, nargs='*', default=[], help='set the value of ncut')
    parser.add_argument('--ncores', action='store', type=int, default=11, help='set the number of cores to use')
    # stuff
    parser.add_argument('-u','--update', action='store_true', help='remove all the files in output directory')
    parser.add_argument('--vmax', action='store',nargs='*',type=float,default=[],help='set the max value to display in images')
    parser.add_argument('--vmin', action='store',nargs='*',type=float,default=[],help='set the min value to display in images')
    # turn off property
    parser.add_argument('--no-display', action='store_false', help='pictures are not plotted')
    parser.add_argument('--no-hotpx', action='store_false', help='prevent the hot pixels removal')
    parser.add_argument('--no-log',action='store_false', help='print the output on the bash')
    parser.add_argument('--no-store',action='store_false', help='no save for array data')
    parser.add_argument('--no-verbose', action='store_false', help='light the output information')

    args = parser.parse_args()

    selections = args.selections
    data_filter = args.filter
    data_mask = args.mask

    verbose = args.no_verbose
    log = args.no_log if not args.list else False
    hotpx = args.no_hotpx
    display_plots = args.no_display
    disperse = args.disperse
    store = args.no_store

    vmaxs = args.vmax
    vmins = args.vmin

    if log:
        import sys
        org = sys.stdout
        f = open(filpy.PROJECT_DIR.join('test_ir.log'),"w")
        sys.stdout = f

    if args.list and not data_mask:
        IR_FILES.tree()
    elif args.list and data_mask:
        MY_DIR.tree()
    elif selections is not None:
        if verbose or args.update:
            MY_DIR.tree()
        if args.update:
            MY_DIR.clear(exceptions='.npz')
            MY_DIR.tree()

        if verbose:
            init_string = '\n'+'-'*50+'\nSELECTED TARGET'
            if len(selections) != 1: 
                init_string = init_string + 'S'
            print(init_string+':')

        # if display_plots:
        if len(vmaxs) == 0:
            vmaxs = [None]*len(selections)
        elif len(vmaxs) < len(selections):
            vmaxs += [vmaxs[-1]]*(len(selections)-len(vmaxs))
        if len(vmins) == 0:
            vmins = [None]*len(selections)
        elif len(vmaxs) < len(selections):
            vmins += [vmins[-1]]*(len(selections)-len(vmins))

        displ_kwargs = [{'vmax': vmax, 
                         'vmin': vmin} for vmax, vmin in zip(vmaxs, vmins)]

        ########

        if data_filter:
            targets = TargetList(file_names=[IR_FILES[i] for i in selections],hotpx=hotpx,verbose=verbose)
            if display_plots:
                targets.plot(True,*displ_kwargs)

            from scipy.ndimage import gaussian_filter
            sigma = args.sigma
            if not isinstance(sigma, Sequence):
                sigma = [sigma]*len(targets)

            filtered_data = [] 
            for id, sel in enumerate(selections):
                trg = targets[id]
                s = sigma[id]
                pltkwargs = displ_kwargs[id]

                data = trg.data

                data_file = (IR_FILES.dir + 'II_125').join('main.dat')
                data_table = filpy.read_iras_data(data_file,store_data=True)
                data_table_r = filpy.read_iras_data((IR_FILES.dir + 'AJ_109_2318').join('table1.dat'),store_data=True,selection='radio')

                ra = data_table['ra'] * filpy.u.hour
                dec = data_table['dec'] * filpy.u.degree
                ra_r = data_table_r['ra'] * filpy.u.hour
                dec_r = data_table_r['dec'] * filpy.u.degree

                print('RA',ra[0])

                from astropy.coordinates import SkyCoord, ICRS, Angle, FK5
                from astropy.time import Time
                
                test = Angle(trg.px_to_coord(10,10) * filpy.u.deg)
                edges = Angle(trg.px_to_coord([[0,trg.shape[0]],[0,trg.shape[0]]],[[0,0],[trg.shape[1],trg.shape[1]]]) * filpy.u.deg)
                ra_edg = edges[0]
                dec_edg = edges[1]

                min_ra  = ra_edg.min()
                min_dec = dec_edg.min()
                max_ra  = ra_edg.max()
                max_dec = dec_edg.max()


                print('MIN POS:',min_ra.to_string('hour',sep='hms'),min_dec.to_string('deg',sep='dms'))
                print('MAX POS:',max_ra.to_string('hour',sep='hms'),max_dec.to_string('deg',sep='dms'))

                sky = SkyCoord(ra=[min_ra,max_ra],dec=[min_dec,max_dec],equinox='J2000').transform_to(FK5(equinox='B1950.0'))
                print('B MIN POS:',Angle(sky.to_table()['ra'][0]).to_string('hour',sep='hms'),Angle(sky.to_table()['dec'][0]).to_string('deg',sep='dms'))
                print('B MAX POS:',Angle(sky.to_table()['ra'][1]).to_string('hour',sep='hms'),Angle(sky.to_table()['dec'][1]).to_string('deg',sep='dms'))


                database = SkyCoord(ra=ra, dec=dec,equinox='B1950.0',obstime=Time('B1983.5')).transform_to(FK5(equinox='J2000.0'))
                database_r = SkyCoord(ra=ra_r, dec=dec_r,equinox='B1950.0').transform_to(FK5(equinox='J2000.0'))
                if verbose:
                    print(trg.wcs)
                    print(data_table)
                    print(database)
                    print('RA SHAPE',ra.shape)
                    print('RA SkyCoord',database.shape)
                    print('RA J2000\n',Angle(database.to_table()['ra']).hour)
                    print('DEC J2000\n',Angle(database.to_table()['dec']).deg)
                    print('POS:',test[0].hms, test[1].dms)
                    print('EDGES:')
                    for r, d in zip(ra_edg.flatten(),dec_edg.flatten()):
                        print(r.to_string('hour',sep='hms'), d.to_string('deg',sep='dms'))
                    print('EDG RA:',min_ra.to_string('hour',sep='hms'),max_ra.to_string('hour',sep='hms'))
                    print('EDG DEC:',min_dec.to_string('deg',sep='deg'),max_dec.to_string('deg',sep='deg'))
                    print('TEST',trg.coord_to_px(test[0].deg,test[1].deg))

                def filter_data(coords: SkyCoord, ra_ext: tuple[float,float], dec_ext: tuple[float,float]) -> tuple[FloatArray,FloatArray]:
                    table = coords.to_table()
                    rr = table['ra'].value
                    dd = table['dec'].value
                    min_rr, max_rr = ra_ext
                    min_dd, max_dd = dec_ext
                    rr_pos = np.logical_and(rr < max_rr,rr > min_rr)
                    dd_pos = np.logical_and(dd < max_dd,dd > min_dd)
                    pos = np.logical_and(rr_pos,dd_pos)
                    rr = rr[pos] 
                    dd = dd[pos] 
                    return rr, dd

                min_ra  = min_ra.value
                min_dec = min_dec.value
                max_ra  = max_ra.value
                max_dec = max_dec.value

                ra = database.to_table()['ra'].value
                dec = database.to_table()['dec'].value

                ra_sel, dec_sel = filter_data(database,(min_ra,max_ra),(min_dec,max_dec))
                ra_sel_r, dec_sel_r = filter_data(database_r,(min_ra,max_ra),(min_dec,max_dec))

                print('RA SEL:\n',ra_sel)

                ra_sel_ang = Angle(ra_sel*filpy.u.deg)
                dec_sel_ang = Angle(dec_sel*filpy.u.deg)
                
                if verbose:
                    print('SOURCES')
                    for r, d in zip(ra_sel_ang,dec_sel_ang):
                        print(r.to_string('hour',sep='hms'),d.to_string('deg',sep='dms'))

                ra_sel_px, dec_sel_px = trg.coord_to_px(ra_sel,dec_sel)
                ra_sel_r_px, dec_sel_r_px = trg.coord_to_px(ra_sel_r,dec_sel_r)

                _ , ax = filpy.show_image(data,projection=trg.wcs,**pltkwargs)
                ax.plot(10,10,'or')
                ax.plot(ra_sel_px,dec_sel_px,'xr')
                ax.plot(ra_sel_r_px,dec_sel_r_px,'+g')
                plt.show()
                

                exit()

                # # #

                new_name = trg.name.split('.')[:-1] + ['fits']
                # gaussian filter
                gauss_filt = gaussian_filter(data, sigma=s) 
                new_name[-2] = new_name[-2] + '_gauss'
                if display_plots:
                    filpy.show_image(gauss_filt,show=True, projection=trg.wcs)

                fdata = sobel_filter(data, remove_neg=False)
                fgdata = sobel_filter(gauss_filt)
                filtered_data += [fgdata]


                from scipy.ndimage import maximum_filter
                from photutils.detection import find_peaks
                maxf_data = maximum_filter(data,size=(3,3))
                fmaxf_data = sobel_filter(maxf_data)
                peaks = find_peaks(data,np.median(data),box_size=20)
                print(peaks['x_peak'].quantity)
                xpeaks = peaks['x_peak'].quantity
                ypeaks = peaks['y_peak'].quantity
                _, axs = filpy.show_image([data,maxf_data,fmaxf_data],num_plots=(1,3),
                                 subtitles=['data','Max filter','Sobel'],
                                 colorbar=False,
                                 **pltkwargs)
                axs[0].plot(xpeaks,ypeaks,'xr')
                axs[2].plot(xpeaks,ypeaks,'xr')
                plt.show()
                exit()
                if store:
                    filt_filename = trg.nickname + f'_{sel:02d}_s{s}_' + 'filtered-data'
                    filt_filename = MY_DIR.join(filt_filename)
                    np.savez(filt_filename,
                             path = trg.path,
                             data = data,
                             data_s = fdata,
                             data_g = gauss_filt,
                             data_gs = fgdata
                            )
                    MY_DIR.update_database()    
                    if verbose:
                        print('\nINFO: Store array data in ' + filt_filename + '.npz')
                        print('\t> path   \t path of the input data')
                        print('\t> data   \t unfiltered data')
                        print('\t> data_s \t data after sobel removal')
                        print('\t> data_g \t gaussian filtered data')
                        print('\t> data_gs\t gaussian filtered data and sobel removal')

                if display_plots:
                    fig, ax = filpy.show_image([data, fdata, gauss_filt, fgdata],
                                               (2,2),
                                               subtitles=[trg.nickname,'filtered','gaussian','gaussian+sobel'],
                                               projection=trg.wcs,
                                               colorbar=False,
                                               **pltkwargs
                                              )
                    # ax[1].plot(*filpy.find_argmax(fdata)[::-1],'xr')
                    plt.show()

                from astropy.io import fits
                new_name[-2] = new_name[-2] + '_sobel-filter'
                new_name = '.'.join(new_name)
                hdu = fits.PrimaryHDU(data=fdata,header=trg.header)
                hdu.writeto(IR_FILES.dir.join(new_name),overwrite=True)
                if verbose:
                    print('INFO: save filtered data as : ' + new_name)
                
        if data_mask:
            nshifts = args.nshift
            if len(nshifts) < len(selections):
                nshifts += [nshifts[-1]]*(len(selections)-len(nshifts))
            for id, mask_sel in enumerate(selections):
                nshift = nshifts[id]
                pltkwargs = displ_kwargs[id]

                # load data
                outputs = MY_DIR.files
                print(outputs[mask_sel])
                with np.load(outputs[mask_sel]) as filt_data:
                    filepath = filt_data['path']
                    data     = filt_data['data']
                    data_s   = filt_data['data_s']
                    data_g   = filt_data['data_g']
                    data_gs  = filt_data['data_gs']
                
                cnts, bins = np.histogram(data.flatten(),data.shape[0])
                plt.figure()
                plt.hist(data.flatten(),data.shape[0])
                plt.show()

                maxpos = np.argmax(cnts)
                mean_val = (bins[maxpos] + bins[maxpos+1])/2
                print('HIST VALUE:',mean_val)
                print('MEDIAN:',np.median(data))

                sel, s = outputs.file[mask_sel].split('_')[-3:-1]
                sel = int(sel)
                s = float(s[1:])

                # # #


                # # #

                from astropy.stats import sigma_clipped_stats
                from photutils.detection import DAOStarFinder
                mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                print('ASTRO:',np.array((mean, median, std)))
                daofind = DAOStarFinder(fwhm=3.0, threshold=2.*std)
                sources = daofind(data - median)
                print(sources.colnames)
                print('\t'.join(['x', 'y', 'sharpness', 'roundness1', 'roundness2']))
                for source in sources:
                    print('\t'.join(['{:.0f}','{:.0f}','{:f}','{:f}','{:f}']).format(source['xcentroid'], source['ycentroid'], source['sharpness'], source['roundness1'], source['roundness2']))
                    
                from astropy.visualization import SqrtStretch
                from astropy.visualization.mpl_normalize import ImageNormalize
                from photutils.aperture import CircularAperture
                positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
                apertures = CircularAperture(positions, r=4.0)
                plt.figure()
                ax1 = plt.subplot(121)
                plt.imshow(data, cmap='Greys_r', origin='lower',**pltkwargs)
                apertures.plot(color='blue', lw=1.5, alpha=0.5)
                plt.xlim(-0.5,data.shape[1]-0.5)
                plt.ylim(-0.5,data.shape[0]-0.5)
                del_pos = sources['roundness1'] > 0
                # plt.plot(sources['xcentroid'][del_pos],sources['ycentroid'][del_pos],'+g')

                plt.subplot(122,sharey=ax1,sharex=ax1)
                plt.title('Sobel')
                plt.imshow(data_s,cmap='Greys_r',origin='lower',**pltkwargs)
                apertures.plot(color='blue', lw=1.5, alpha=0.5)
                plt.plot(sources['xcentroid'], sources['ycentroid'],'xr')

                plt.figure()
                plt.imshow(data_s,cmap='Greys_r',origin='lower',**pltkwargs)
                plt.plot(sources['xcentroid'], sources['ycentroid'],'xr')
                plt.show()

                print('\nSOBEL TRIAL')
                tmp_data = abs(data+abs(data_s))
                tmp_kwargs = {key: val for key, val in pltkwargs.items()}
                tmp_kwargs.pop('vmax')
                filpy.show_image(data=[data,data_s,abs(data_s),tmp_data],num_plots=(2,2),
                                 subtitles=['Data','Sobel','Abs Sobel','Summed'],
                                 show=True,
                                 sharex=0,
                                 sharey=0,
                                 vmax=[4,4,4,4],
                                 aspect='auto',
                                 colorbar_pos = 'right',
                                 barlabel = '',
                                 **tmp_kwargs)
                del tmp_kwargs

                mean, median, std = sigma_clipped_stats(tmp_data, sigma=3.0)
                print('ASTRO:',np.array((mean, median, std)))
                daofind = DAOStarFinder(fwhm=3., threshold=5.*std)
                sources = daofind(tmp_data-median)
                print('\t'.join(['x', 'y', 'sharpness', 'roundness1', 'roundness2']))
                for source in sources:
                    print('\t'.join(['{:.0f}','{:.0f}','{:f}','{:f}','{:f}']).format(source['xcentroid'], source['ycentroid'], source['sharpness'], source['roundness1'], source['roundness2']))
                xpos, ypos = sources['xcentroid'], sources['ycentroid']
                new_pos = np.empty((2,0))
                near_pnts = np.zeros(xpos.shape).astype(bool)
                for x,y in zip(xpos,ypos):
                    if (x not in xpos[near_pnts]) or (y not in ypos[near_pnts]):
                        dist = np.sqrt((x-xpos)**2 + (y-ypos)**2)
                        near_pnts = dist <= 5
                        px = (ypos[near_pnts].astype(int),xpos[near_pnts].astype(int))
                        mean_y = np.average(ypos[near_pnts],weights=data[px[0],px[1]])
                        mean_x = np.average(xpos[near_pnts],weights=data[px[0],px[1]])
                        new_pos = np.append(new_pos,[[mean_x],[mean_y]],axis=1)
                xpos, ypos = new_pos
                positions = np.transpose((xpos,ypos))
                apertures = CircularAperture(positions, r=4.0)
                plt.figure()
                ax1 = plt.subplot(121)
                plt.imshow(tmp_data - median, cmap='Greys_r', origin='lower',**pltkwargs)
                apertures.plot(color='blue', lw=1.5, alpha=0.5)
                plt.plot(xpos,ypos,'.g')
                plt.xlim(-0.5,data.shape[1]-0.5)
                plt.ylim(-0.5,data.shape[0]-0.5)

                plt.subplot(122,sharey=ax1,sharex=ax1)
                plt.title('Sobel')
                plt.imshow(data,cmap='Greys_r',origin='lower',**pltkwargs)
                apertures.plot(color='blue', lw=1.5, alpha=0.5)
                plt.plot(sources['xcentroid'], sources['ycentroid'],'xr')
                plt.plot(xpos,ypos,'.g')

                plt.show()
                exit()
                tmp_field = data.copy()
                length = 7
                for x, y in zip(xpos.astype(int),ypos.astype(int)):
                    print('CEN VAL:',x,y,'\t',data[y,x],tmp_data[y,x])
                    mini_obj = tmp_data[y-length:y+length+1,x-length:x+length+1]
                    _, ax = filpy.show_image(mini_obj)
                    ax.plot(length,length,'or')
                    yarr = np.arange(mini_obj.shape[0]) - length
                    xarr = np.arange(mini_obj.shape[1]) - length
                    yy, xx = np.meshgrid(yarr,xarr)
                    dist = np.sqrt(xx**2+yy**2)
                    unq_dist = np.sort(np.unique(dist))
                    d_pos = [dist == d for d in unq_dist]
                    avg_profile = np.array([ np.mean(data[yy[d],xx[d]]) for d in d_pos])
                    filpy.quickplot(avg_profile,fmt='.--')
                    plt.show()


                exit()

                ymin, xmin = filpy.find_argmin(data_s)
                ymax, xmax = filpy.find_argmax(data_s)

                max_pos = [[xmax],[ymax]]
                min_pos = [[xmin],[ymin]]
                _, axs = filpy.show_image([data,data_s],num_plots=(1,2),
                                         subtitles=['Data','Sobel'],
                                         colorbar = False,
                                         **pltkwargs)
                for ax in axs:
                    ax.plot(xmin,ymin,'xr')
                    ax.plot(xmax,ymax,'xg')
                plt.show()

                cp_data = data_s.copy()
                print('MIN:',cp_data[ymin,xmin])
                print('MAX:',cp_data[ymax,xmax])

                mv_i = np.arange(2,8)
                lowdx = np.where(cp_data[ymin,xmin+mv_i]>0)[0].min()+1
                lowsx = np.where(cp_data[ymin,xmin-mv_i]>0)[0].min()+1
                lownt = np.where(cp_data[ymin+mv_i,xmin]>0)[0].min()+1
                lowst = np.where(cp_data[ymin-mv_i,xmin]>0)[0].min()+1
                updx = np.where(cp_data[ymax,xmax+mv_i]>0)[0].min()+1
                upsx = np.where(cp_data[ymax,xmax-mv_i]>0)[0].min()+1
                upnt = np.where(cp_data[ymax+mv_i,xmax]>0)[0].min()+1
                upst = np.where(cp_data[ymax-mv_i,xmax]>0)[0].min()+1

                low_shift = max(lowdx,lowsx,lownt,lowst)
                up_shift = max(updx,upsx,upnt,upst)

                print('LOWS:\n\t',lowdx,lowsx,lownt,lowst)
                print('LOW:\t',low_shift)
                print('UPS:\n\t',updx,upsx,upnt,upst)
                print('UP:\t',up_shift)

                miniobj_max = cp_data[ymax-up_shift:ymax+up_shift+1,xmax-up_shift:xmax+up_shift+1]
                miniobj_min = cp_data[ymin-low_shift:ymin+low_shift+1,xmin-low_shift:xmin+low_shift+1]

                _, axs = filpy.show_image([miniobj_max,miniobj_min],
                                  num_plots=(1,2),
                                  colorbar=False,
                                  **pltkwargs)
                axs[0].plot(*np.where(miniobj_max<=0)[::-1],'.r')
                axs[0].plot([up_shift+updx,up_shift-upsx,up_shift,up_shift],
                            [up_shift,up_shift,up_shift+upnt,up_shift-upst],'xg')
                axs[1].plot(*np.where(miniobj_min<=0)[::-1],'.r')
                axs[1].plot([low_shift+lowdx,low_shift-lowsx,low_shift,low_shift],
                            [low_shift,low_shift,low_shift+lownt,low_shift-lowst],'xg')
                plt.show()

                quad = slice(up_shift-1,up_shift+2)
                miniobj_max[quad,quad] = -1

                low_ypos, low_xpos = np.where(miniobj_min<=0)
                up_ypos, up_xpos = np.where(miniobj_max<=0)


                print('LOW MEAN:',np.mean(miniobj_min[miniobj_min>0]))
                cp_data[up_ypos+ymax-up_shift,up_xpos+xmax-up_shift] = np.mean(miniobj_max[miniobj_max>0])
                cp_data[low_ypos+ymin-low_shift,low_xpos+xmin-low_shift] = np.mean(miniobj_min[miniobj_min>0])

                ymin, xmin = filpy.find_argmin(cp_data)
                ymax, xmax = filpy.find_argmax(cp_data)
                print('MIN:',cp_data[ymin,xmin])
                print('MAX:',cp_data[ymax,xmax])
                _, axs = filpy.show_image([data,cp_data],num_plots=(1,2),
                                          subtitles=['Data','Sobel'],
                                          colorbar = False,
                                          **pltkwargs)
                for ax in axs:
                    ax.plot(*max_pos,'.r')
                    ax.plot(*min_pos,'.g')
                    ax.plot(xmin,ymin,'xr')
                    ax.plot(xmax,ymax,'xg')
                plt.show()

                exit()

                int_sigma = int(s) if s > 0 else 1
                shift = int_sigma*nshift
                cp_field = data.copy()
                cp_filter = data_gs.copy()

                cp_filter[170:211,176:224] = 0
                cp_filter[160:170,220:249] = 0

                for i in range(3):
                    print('\n\nRUN',i,'~~~~~~~'*10)
                    cp_filte, cp_field = mask_values(data,data_s,cp_filter,cp_field,int_sigma,shift,**pltkwargs)

        ########

        if disperse:
            targets = TargetList(file_names=[IR_FILES[i] for i in selections],hotpx=hotpx,verbose=verbose)
            if display_plots:
                targets.plot(show=True)

            nsig = args.nsig
            ncut = args.ncut
            nsmooth = args.nsmooth
            nthreads = args.ncores

            targets.disperse(nsig=nsig,
                             ncut=ncut,
                             nsmooth=nsmooth,
                             nthreads=nthreads,
                             outdir=MY_DIR.path
                            )
            
            print('HEY',targets.skls)
            
            if verbose: MY_DIR.tree()

            targets.plot_network(*displ_kwargs)

        if log:
            sys.stdout = org
            f.close()

        

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
    
    def px_to_coord(self, *sel: IntArrayLike) -> FloatArray:
        coord = self.wcs.pixel_to_world_values(*sel)
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
        if 'skl' not in self.__dict__.keys():
            raise AttributeError('To run self.disperse() or self.skel() is required')

        figargs['colorbar'] = False
        figargs['show'] = False
        
        fig, ax = self.plot(**figargs)
        ax.plot(*self.fil_pos,'.b',alpha=0.7)
        return fig, ax


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

MY_DIR = filpy.PROJECT_DIR -1 + 'mytest'
MY_DIR.make_dir()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TestIR',
                                     description='read and analyze IR data',
                                    )
    parser.add_argument('-l','--list', action='store_true', help='print the list of file in the data directory')
    parser.add_argument('-s','--selection', action='store', type=int, nargs='*', default=None, help='index(ces) of the selected object')
    parser.add_argument('-f','--filter', action='store_true', help='sobel filter')
    parser.add_argument('--sigma', action='store', type=float, nargs='*', default=2, help='set the sigma for the Gaussian filter')

    parser.add_argument('--nsig', action='store', type=float, nargs='*', default=0.3, help='set the value of nsig')
    parser.add_argument('--nsmooth', action='store', type=int, nargs='*', default=4, help='set the number of smoothing routines')
    parser.add_argument('--ncut', action='store', type=float, nargs='*', default=None, help='set the value of ncut')
    parser.add_argument('--ncores', action='store', type=int, default=11, help='set the number of cores to use')

    parser.add_argument('-u','--update', action='store_true', help='remove all the files in output directory')
    
    parser.add_argument('--no-log',action='store_false', help='print the output on the bash')
    parser.add_argument('--no-disperse',action='store_false', help='run the script without disperse')
    parser.add_argument('--no-hotpx', action='store_false', help='prevent the hot pixels removal')
    parser.add_argument('--no-display', action='store_false', help='pictures are not plotted')
    parser.add_argument('--no-verbose', action='store_false', help='light the output information')

    args = parser.parse_args()

    selection = args.selection
    data_filter = args.filter

    verbose = args.no_verbose
    log = args.no_log if not args.list else False
    hotpx = args.no_hotpx
    display_plots = args.no_display
    disperse = args.no_disperse

    if log:
        import sys
        org = sys.stdout
        f = open(filpy.FileVar('test_ir.log',dirpath=filpy.PROJECT_DIR-1).path,"w")
        sys.stdout = f

    if args.list:
        IR_FILES.tree()
    elif selection is not None:
        if verbose or args.update:
            MY_DIR.tree()
        if args.update:
            MY_DIR.clear()
            MY_DIR.tree()

        if verbose:
            init_string = '\n'+'-'*50+'\nSELECTED TARGET'
            if len(selection) != 1: init_string = init_string + 'S'
            print(init_string+':')

        targets = TargetList(file_names=[IR_FILES[i] for i in selection],hotpx=hotpx,verbose=verbose)


        if display_plots:
            targets.plot(show=True)


        ########

        if data_filter:
            from scipy.ndimage import gaussian_filter
            sigma = args.sigma
            if not isinstance(sigma, Sequence):
                sigma = [sigma]*len(targets)

            filtered_data = [] 
            for trg, s in zip(targets,sigma):
                data = trg.data
                new_name = trg.name.split('.')[:-1] + ['fits']
                # gaussian filter
                gauss_filt = gaussian_filter(data, sigma=s) 
                new_name[-2] = new_name[-2] + '_gauss'
                if display_plots:
                    filpy.show_image(gauss_filt,show=True, projection=trg.wcs)

                fdata = sobel_filter(data, remove_neg=False)
                fgdata = sobel_filter(gauss_filt)
                filtered_data += [fgdata]

                if display_plots:
                    fig, ax = filpy.show_image([data, fdata, gauss_filt, fgdata],
                                               (2,2),
                                               subtitles=[trg.name,'filtered','gaussian','gaussian+sobel'],
                                               projection=trg.wcs,
                                               colorbar=False,
                                               vmax=3,
                                               vmin=0
                                              )
                    # ax[1].plot(*filpy.find_argmax(fdata)[::-1],'xr')
                    plt.show()

                from astropy.io import fits
                new_name[-2] = new_name[-2] + '_sobel-filter'
                new_name = '.'.join(new_name)
                hdu = fits.PrimaryHDU(data=fdata,header=trg.header)
                hdu.writeto(IR_FILES.dir.join(new_name),overwrite=True)
                if verbose:
                    print('INFO: save filtered data as : '+new_name)
                
                # y, x = np.where(fdata < 0)
                # fig, ax = filpy.show_image(data,vmax=3)
                # # ax.plot(x,y,'xr',alpha=0.3)
                # ax.scatter(x,y,c=abs(fdata[y,x]),marker='x')
                # fig, ax = filpy.show_image(fdata,vmin=0,vmax=3)
                # # ax.plot(x,y,'xr')
                # ax.scatter(x,y,c=abs(fdata[y,x]),marker='x')
                # ax.plot(*filpy.find_argmax(fdata)[::-1],'xr')
                # plt.show()

                ymax, xmax = filpy.find_argmax(fgdata)
                int_sigma = int(s) if s > 0 else 1
                shift = int_sigma*4
                max_obj = data[ymax-shift:ymax+shift+1,xmax-shift:xmax+shift+1].copy()
                cp_obj  = fdata[ymax-shift:ymax+shift+1,xmax-shift:xmax+shift+1].copy()
                cpg_obj = fgdata[ymax-shift:ymax+shift+1,xmax-shift:xmax+shift+1].copy()
                # y,x = np.where(cp_obj < 0)
                # obj_dist = np.sqrt((x-shift)**2 + (y-shift)**2)
                # max_dist = np.max(obj_dist)
                # max_pos = obj_dist==max_dist
                # xedg = [x[max_pos],cp_obj.shape[0]-x[max_pos]] if x[max_pos] < shift else [cp_obj.shape[0]-x[max_pos],x[max_pos]]
                # yedg = [y[max_pos],cp_obj.shape[1]-y[max_pos]] if y[max_pos] < shift else [cp_obj.shape[1]-y[max_pos],y[max_pos]]
                bkg = np.mean(cp_obj[cp_obj>0])
                print(cp_obj[cp_obj>0])
                print('BKG:',bkg)
                cp_obj[cp_obj<=0] = bkg
                from matplotlib.colors import LogNorm
                plt.figure()
                plt.imshow(max_obj,origin='lower',norm=LogNorm())
                plt.plot(shift,shift,'xr')
                plt.figure()
                plt.imshow(np.where(cp_obj<0,0,cp_obj),origin='lower')
                plt.plot(shift,shift,'xr')
                plt.figure()
                plt.imshow(cpg_obj,origin='lower',norm=LogNorm())
                plt.plot(shift,shift,'xr')
                plt.show()

                yy, xx = np.meshgrid(np.arange(max_obj.shape[0])-shift,
                                     np.arange(max_obj.shape[1])-shift
                                    )
                dist_mat = np.sqrt(xx**2+yy**2)
                dists = np.sort(np.unique(dist_mat))
                avg_profile = np.empty(0)
                plt.figure()
                for d in dists:
                    value = max_obj[dist_mat == d]
                    avg_profile = np.append(avg_profile,np.mean(value))
                    plt.plot([d]*len(value),value,'x')
                plt.plot(dists,avg_profile,'.--')
                plt.show()
                






        ########

        if disperse:
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
            
            if verbose: MY_DIR.tree()

            targets.plot_network({'vmax':3})

    if log:
        sys.stdout = org
        f.close()

        

from __future__ import print_function

from .typing import *
# from numpy.typing import NDArray
from subprocess import check_call, call
import numpy as np

__all__ = [
           'run_delaunay',
           'skl_names',
           'run_disperse',
           'write_NDfield_ascii',
           'change_field_value_with_file',
           'change_field_value'
          ]


def run(cmd: list[str], executable: list[str] = ['/bin/bash','-i','-c'], last: list[str] = ['&&']) -> None:
    """Run the code

    Parameters
    ----------
    cmd : str
        The command to execute via bash
    """
    print()
    print(' '.join(cmd))
    print()
    # if executable or last:
    #     cmd[0]  = '"' + cmd[0]
    #     cmd[-1] = cmd[-1] + '"'
    #     cmd = [' '.join(cmd)]
    check_call(cmd)

opt_default = {}
   
def run_delaunay(filename: str, nsmooth: float, density_file: Optional[str] = None, btype: Literal['mirror','periodic','smooth','void'] = 'smooth', dim: Literal['2D','3D'] = '3D', blocks: Optional[list[int]] = None, outdir: str = '.',**runkwargs) -> str:
    """Delaunay 2D and 3D command of DisPerSE

    Parameters
    ----------
    filename : str
        The name of a file containing the discrete particle coordinates in a `field` format.
    nsmooth : float
        Smooth the `'field_value'` data field associated with vertices by averaging its value with that of its direct neighbors in the network `nsmooth` times
    density_file : None | str, optional
        Density distribution, by default `None`
    btype : Literal['mirror','periodic','smooth','void'], optional
        This option is used to set how the distribution should be extrapolated outside the bounding box (an estimation of the distribution outside the bounding box is needed to correctly estimate the topology and density of the distribution close to its boundaries). Possible boundary types are:
            * `'mirror'` : 
                        the distribution outside the bounding box is a mirrored copy of that inside.
            * `'periodic'` : 
                        use periodic boundary condition (i.e. space is paved with copies of the distribution in the bounding box). Note that this option does **NOT** enforce periodic boundary conditions as it does not tell delaunay_nD to reconnect the Delaunay cells that cross the bounding box (this is achieved with -periodic).
            * `'smooth'` : 
                        a surface of guard particles is added outside the bounding box and new particles are added by interpolating the estimated density computed on the boundary of the distribution. This boundary type is useful when the actual boundaries of the sample are complex (i.e. not a cube), such as for a 3D galaxy catalog limited to a portion of the sky.
            * `'void'` : 
                        the distribution is supposed to be void outside the bounding box., 
            
            By default `'smooth'`
    dim : Literal['2D', '3D'], optional
        Set the dimension of the tasselation, by default '3D'
    blocks : None | list[int], optional
        `blocks = [NChunks, NThreads]`
        If not `None`, instead of computing the delaunay tesselation of the full distribution, divide it into `NChunks` overlapping sub blocks and process them `NThreads` at a time. The subblocks are then automatically reassembled into the full delaunay tesselation. This option can either be used to increase speed by parallelizing the process (for high values of `NThreads`) or decrease the memory consumption (when `NChunks` >> `NThreads`). By default `None`

    Returns
    -------
    SDndnet_fname : str
        The name of the output unstructured network in NDnet format, with density computed for each vertex using DTFE (i.e. density at a given vertex is proportional to the total volume of the surrounding cells).
    """
    if dim not in ['2D','3D']:
        raise ValueError("The dim variable can assume only the values '2D' and '3D'")
    
    if density_file is not None:
        # set the distribution to zero out of the boundaries
        btype='void'

    # set the -blocks option
    if blocks is None:
        blocks_opt = []
    else:
        blocks_opt = ["-blocks", format(blocks[0], 'd'), format(blocks[1], 'd')]

    from os import path
    input_dir, input_name = path.split(filename)
    if input_dir == outdir:
        outname = filename
    else:
        outname = path.join(outdir, input_name)

    # run the command for tasselation
    run( ["delaunay_" + dim, filename,
          "-outName", outname,
          "-btype", btype] +
          blocks_opt,
          **runkwargs )  

    ndnet_fname = outname + ".NDnet"       #: NDnet file
    # rename the output for the -blocks option
    if blocks is not None:        
        temp_ndnet_fname = outname + "_G.NDnet"
        print(" rename {0} into {1} \n".format(temp_ndnet_fname, ndnet_fname))
        call(["mv", temp_ndnet_fname, ndnet_fname])
        
    # remove test_smooth.dat
    call(["rm", "test_smooth.dat"])
    
    #  add separated density field 
    if  density_file is not None:
        change_field_value_with_file(ndnet_fname, density_file)

    #  smooth density field
    if nsmooth > 0:
        run( ["netconv", ndnet_fname,
              "-outName", ndnet_fname,
              "-smoothData", "field_value", format(nsmooth,'g')],
              **runkwargs  )
        # update the output file
        old_SDndnet_fname = ndnet_fname + ".SD.NDnet"
        SDndnet_fname = outname + ".SD{0:d}.NDnet".format(nsmooth)
        print(" rename {0} into {1} \n".format(old_SDndnet_fname, SDndnet_fname))
        call(["mv", old_SDndnet_fname, SDndnet_fname])
    else:
        SDndnet_fname = ndnet_fname
        
    # convert to vtu
    run( ["netconv", SDndnet_fname,
          "-outName", SDndnet_fname,
          "-to", "vtu"],
          **runkwargs )

    # remove the previous NDnet file
    if SDndnet_fname != ndnet_fname:
        call(["rm", ndnet_fname])
    
    return SDndnet_fname

   
def skl_names(skl_fname: str, walls: bool = False, patches: bool = False) -> dict[str,str]:
    """Collects the names of the outputs

    Parameters
    ----------
    skl_fname : str
        File name of the output .NDskl
    walls : bool, optional
        If `True` voids and nodes filenames are collected, by default `False`
    patches : bool, optional
        If `True` walls filenames are collected, by default `False`

    Returns
    -------
    outnames : dict[str,str]
        Dictionary with the names of each output file. The keys are:
        * `'skl'`     : `skl_fname`
        * `'skl_brk'` : the .BRK file
        * `'skl_vtp'` : the NDskl file in .vtp format
        * `'segs'`    : the segments file
        * `'voids'`   : the manifolds J0a file in .vtu format, optional
        * `'nodes'`   : the manifolds J3d file in .vtu format, optional
        * `'walls'`   : the manifolds J1a file in .vtu format, optional

    Notes
    -----
    The `skl_fname` consists of `basename + '.' + '.up.NDskl' + ('.S{nsmooth}') + '.a.NDskl'`
    ```python
    split = skl_fname.split('.')
    split = [basename, 'up', 'NDskl', 'S{nsmooth}', 'a', 'NDskl']
    ```
    """
    # initialize the dictionary
    outnames = {'skl' : skl_fname}
    # extract information from the name
    split = skl_fname.split('.')
    # check skeleton smoothnig
    if split[-3][0] == 'S':
        smooth_ext = "." + split[-3] 
        basename = ".".join(split[:-6])
        persist_ext = "." + split[-6]
    else:
        smooth_ext = ""
        basename = ".".join(split[:-5])
        persist_ext = "." + split[-5]

    outnames['skl_brk'] = basename + persist_ext + ".up.NDskl" + ".BRK" + smooth_ext + ".a.NDskl"
    outnames['skl_vtp'] = basename + persist_ext + ".up.NDskl" + smooth_ext + ".vtp"
    # outnames['crits'] = basename + persist_ext + ".up.NDskl.BRK" + smooth_ext + ".a.crits"
    outnames['segs'] = basename + persist_ext +".up.NDskl.BRK" + smooth_ext + ".a.segs"        
    if patches:
        outnames['voids'] = basename + persist_ext + "_manifolds_J0a.NDnet" + smooth_ext + ".vtu"
        outnames['nodes'] = basename + persist_ext + "_manifolds_J3d.NDnet" + smooth_ext + ".vtu"
    if walls:
        outnames['walls'] = basename + persist_ext + "_manifolds_J1a.NDnet" + smooth_ext + ".vtu"
    
    return outnames
       
                            
def run_disperse(filename: str, nsig: int, nsmooth: int, cutp: Optional[float] = None, 
                 walls: bool = False, patches: bool = False, mask: Optional[str] = None, 
                 robustness: bool = False, nthreads: Optional[int] = None, 
                 dim: Literal['2D','3D'] = '3D', force_loops: bool = True, outdir: str = '.',**runkwargs) -> dict[str, str]:
    """DisPerSE pipeline
    
    Parameters
    ----------
    filename : str
        The cell complex defining the topology of the space and optionally the function discretely sampled over it. The file may be a Healpix FITS file, a regular grid or an unstructured network readable by netconv or fieldconv (run netconv or fieldconv without argument for a list of supported file formats). The value of the function from which the Morse-smale complex will be computed should be given for each vertex (or pixel in case of a regular grid)
    nsig : int
        The persistence ratio threshold in terms of _"number of sigmas"_. Any persistence pair with a persistence ratio (i.e. the ratio of the values of the points in the pair) that has a probability less than `nsig` to appear in a random field will be cancelled. This may only be used for discretely sampled density fields (such as N-body simulations or discrete objects catalogs) whose density was estimated through the DTFE of a delaunay tesselation. This option is typically used when the input network was produced using `run_delaunay()`, in any other case, use `cutp` instead
    nsmooth : int
        Smooth the `'field_value'` data field associated with vertices by averaging its value with that of its direct neighbors in the network `nsmooth` times
    cutp : float | None, optional
        Any persistence pair with persistence lower than the given threshold will be cancelled. Use `nsig` instead of this when the input network was produced with `run_delaunay()`. The cut value should be typically set to the estimated amplitude of the noise. 
        By default `None`
    walls : bool, optional
        If `True` the function dumps walls, that is ascending 1-manifolds (required `dim == '3D'`). By default `False`
    patches : bool, optional
        If `True` the function dumps voids and nodes' region, that is ascending 1-manifolds and descending n-manifold respectively, where n = 2 for `dim == '2D'` or n = 3 for `dim == '3D'`. 
        By default `False`
    mask : str | None, optional
        The file must be a 1D array of values of size the number of vertices (or pixels) in the network, in a readable grid format. A value of 0 corresponds to a visible vertex/pixel while any other value masks the vertex/pixel. Adding a trailing `~` to the filename (without space) reverses this behavior, a value of 0 masking the corresponding pixels/vertices. 
        By default `None`
    robustness : bool, optional
        If `True` it enables the computation of robustness and robustness ratio: a robustness value is tagged for each segments and node of the output skeleton files. Robustness is a local measure of how contrasted the critical points and filaments are with respect to their background. In a sense, robustness is a continuous version of persistence which extends to filaments (persistence applies to pairs of critical points only). It can be costly for very large data sets. 
        By default `False`
    nthreads : int | None, optional
        The number of threads. If it is `None` it is usually set to the total number of cores available by openMP. 
        By default `None`
    dim : Literal['2D', '3D'], optional
        Set the dimension of the tasselation, 
        by default '3D'
    force_loops : bool, optional
        If `True` it forces the simplification of non-cancellable persistence pairs (saddle-saddle pairs in 3D or more that are linked by at least 2 different arcs). When two critical points of critical index difference 1 are linked by 2 or more arcs, they may not be cancelled as this would result in a discrete gradient loop. This is not a problem in 2D as such pairs cannot form persistence pairs but in 3D, saddle-saddle persistence pairs may be linked by 2 or more arcs even though their persistence is low. By default those pairs are skipped in order to preserve the properties of the Morse-smale complex but as a result few non persistent features may remain (such as spurious filaments). Fortunately, there are usually none or only very few of those pairs, and their number is shown in the output of mse, in the Simplifying complex section. If you are only interested in identifying structures (as opposed to doing topology), you should probably use '-forceLoops' to remove those spurious structures (such as small non significant filaments). 
        By default `True`
    outdir : str, optional
        The directory where outputs will be stored, 
        by default `'.'`

    Returns
    -------
    outnames : dict[str,str]
        Dictionary with the names of each output file. See output of `skl_names()`

    Notes
    -----
    The pipeline consists in:
        1. compute the Morse-Smale complex through the command `mse`
        2. 
    """   
    if cutp:
        nsigstr = "_c{0:.3g}".format(cutp)
        nsigopt = ["-cut", format(cutp)]
    elif nsig != 0.:
        nsigstr = "_s{0:g}".format(nsig)
        nsigopt = ["-nsig", format(nsig)]
    else:
        nsigstr = ""
        nsigopt = []
        
    if nsmooth > 0:
        conv_smooth = ["-smooth", format(nsmooth,'d')]
        smooth_ext = ".S{0:03d}".format(nsmooth)
    else:
        conv_smooth = []
        smooth_ext = ""

    if mask is None:
        mask_opt = []
    else:
        mask_opt = ["-mask", mask]

    if robustness:
        robustness_opt = ["-robustness"]
    else:
        robustness_opt = []
        
    if nthreads is None:
        nthreads_opt = []
    else:
        nthreads_opt = ["-nthreads", format(nthreads)] 
    
    if force_loops:
        forceloops = ["-forceLoops"]
    else:
        forceloops = []

    from os import path
    input_dir, input_name = path.split(filename)
    if input_dir == outdir:
        outname = filename
    else:
        outname = path.join(outdir, input_name)

    # run mse
    run( ["mse", filename,
          "-outName", outname,
          "-upSkl",                 # save arcs leading to maxima
          "-manifolds"] +           # compute and store all a./d. manifolds
          nsigopt +             
          mask_opt +
          nthreads_opt +  
          robustness_opt +
          forceloops,               # remove spurious structures
         **runkwargs )

    # set the name of NDskl file
    base_name = "{0}{1}".format(outname, nsigstr)
    skl_name = base_name + ".up.NDskl"

    if patches:        
        # dump Voids for tagging the galaxies
        run( ["mse", filename,
              "-outName", outname,
              "-loadMSC", outname + ".MSC"] +
              nsigopt + 
             ["-dumpManifolds", "J0a"] +    # store in .NDnet file voids (a. 0-manifolds)
              forceloops,
              **runkwargs )

        # converting voids NDnet file to vtu
        voids_name = base_name + "_manifolds_J0a.NDnet"
        netconv_cmd = ["netconv", voids_name,
                       "-outName", voids_name,
                       "-to", "vtu"]
        netconv_cmd.extend(conv_smooth)
        run(netconv_cmd,**runkwargs)
   
        # dump Nodes' region for tagging the galaxies
        if dim == '3D':
            node_manifold = 'J3d'
        elif dim == '2D':
            node_manifold = 'J2d'
        else:
            raise ValueError("The dim variable can assume only the values '2D' and '3D'")              
        run( ["mse", filename,
              "-outName", outname,
              "-loadMSC", outname + ".MSC"] +
              nsigopt +
             ["-dumpManifolds", node_manifold,
              "-forceLoops"],
              **runkwargs )
    
        # converting nodes ndnet file to vtu
        nodes_name =  base_name + "_manifolds_" + node_manifold + ".NDnet"
        netconv_cmd = ["netconv", nodes_name,
                       "-outName", nodes_name,
                       "-to", "vtu"]
        netconv_cmd.extend(conv_smooth)              
        run(netconv_cmd,**runkwargs)
            
    if walls and dim=='3D':
        # dump Walls
        run( ["mse", filename,
              "-outName", outname,
              "-loadMSC", outname + ".MSC"] +
              nsigopt +
             ["-dumpManifolds", "J1a"] +
              forceloops,
              **runkwargs )
    
        # converting and smoothing walls ndnet file to vtu
        walls_name =  base_name + "_manifolds_J1a.NDnet"
        netconv_cmd = ["netconv", walls_name,
                       "-outName", walls_name,
                       "-to", "vtu"]
        netconv_cmd.extend(conv_smooth)         
        run(netconv_cmd, **runkwargs)              
    
    # converting and smoothing NDskl to ascii format
    skelconv_cmd = ["skelconv", skl_name,
                    "-outName", skl_name,
                    "-to","NDskl_ascii"]
    skelconv_cmd.extend(conv_smooth)
    run(skelconv_cmd, **runkwargs)          

   # converting and smoothing NDskl to ascii format with breakdown
    skelconv_cmd = ["skelconv", skl_name,
                    "-outName", skl_name,
                    "-breakdown",
                    "-to","NDskl_ascii"]
    skelconv_cmd.extend(conv_smooth)
    run(skelconv_cmd, **runkwargs)          
    
    # converting and smoothing NDskl to vtp format to open in paraview 
    # no breakdown to keep source index the same
    skelconv_cmd = ["skelconv", skl_name,
                    "-outName", skl_name,
                    "-to", "vtp"]
    skelconv_cmd.extend(conv_smooth)
    run(skelconv_cmd, **runkwargs)              
    
    ## converting NDskl to ascii critical points 
#     skelconv_cmd=["skelconv",skl_name,
#                   "-outName",skl_name,
#                   "-breakdown",
#                   "-to","crits_ascii"]
#     skelconv_cmd.extend(conv_smooth)
#     run(skelconv_cmd, **runkwargs)
#         
    # converting NDskl to ascii segments
    skelconv_cmd=["skelconv",skl_name,
                  "-outName",skl_name,
                  "-breakdown",
                  "-to","segs_ascii"]
    skelconv_cmd.extend(conv_smooth)           
    run(skelconv_cmd, **runkwargs)
    
    # remove temp files    
    call(["rm", outname+".MSC", skl_name])
    if patches:
        call(['rm', voids_name, nodes_name])
    if walls:
        call(['rm', walls_name ])
    # return the names of each file
    outnames = skl_names(skl_name + smooth_ext + ".a.NDskl", walls, patches)
    return outnames


def write_NDfield_ascii(filename: str, field: FloatArray) -> None:
    """Save the field in ASCII

    Parameters
    ----------
    filename : str
        Density field file name
    field : NDArray[np.float64]
        A regular grid to be interpolated 
    """
    header = "ANDFIELD\n[{0}]".format(field.size)
    np.savetxt(filename, field, header=header, fmt="%.12g", comments="")


def change_field_value_with_file(ndnet_fname: str, density_file: str, field_name: str = "field_value", outdir: str = '', **runkwargs) -> None:
    """Add the density file to a NDnet file

    Parameters
    ----------
    ndnet_fname : str
        The name of a file containing an unstructured network (for instance, persistence pairs or manifolds as output by mse) in a readable network file format
    density_file : str
        The name of a readable regular grid field format file containing the grid to be interpolated
    field_name : str, optional
        The name of the additional field in the output file, by default `"field_value"`
    """
    from os import path
    input_dir, input_name = path.split(ndnet_fname)
    if input_dir == outdir:
        outname = ndnet_fname
    else:
        outname = path.join(outdir, input_name)

    # add the field to `ndnet_fname`
    ncv_cmd = ["netconv", ndnet_fname,
               "-outName", outname,
               "-addField", density_file, field_name, 
              ]
    run(ncv_cmd, **runkwargs)

    # update the output file
    old_ndnet_fname = outname + ".NDnet"
    print(" rename {0} into {1} \n".format(old_ndnet_fname, outname))
    call(["mv", old_ndnet_fname, outname])


def change_field_value(ndnet_fname: str, field_value: NDArray[np.float64], field_name: str = "field_value") -> None:
    """Add the density field

    Parameters
    ----------
    ndnet_fname : str
        The name of a file containing an unstructured network (for instance, persistence pairs or manifolds as output by mse) in a readable network file format
    field_value : NDArray[np.float64]
        A regular grid to be interpolated
    field_name : str, optional
        Tag for the field file, by default `"field_value"`
    """
    # save the density field in ASCII
    density_fname = ndnet_fname + "_" + field_name + ".a.NDfield"
    write_NDfield_ascii(density_fname, field_value)
    # add the density field to `ndnet_fname`    
    change_field_value_with_file(ndnet_fname, density_fname, field_name)
    # remove the ASCII file
    call(['rm', density_fname])       

        

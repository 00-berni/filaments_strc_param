from typing import Literal
from __future__ import print_function

from subprocess import check_call, call
import numpy as np


def run(cmd: str) -> None:
    """Run the code

    Parameters
    ----------
    cmd : str
        command
    """
    print()
    print(' '.join(cmd))
    print()
    check_call(cmd)

opt_default = {}

    
def run_delaunay(filename: str, nsmooth: float, density_file: None | str = None, btype: Literal['mirror','periodic','smooth','void'] = 'smooth', dim: Literal['2D','3D'] = '3D', blocks: None | list[int] = None) -> str:
    """Delaunay 2D and 3D command of DisPerSE

    Parameters
    ----------
    filename : str
        The name of a file containing the discrete particle coordinates in a `field` format.
    nsmooth : float
        smooth the vertex_field_name data field associated with vertices by averaging its value with that of its direct neighbors in the network Ntimes times
    density_file : None | str, optional
        _description_, by default None
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
        set the dimension of the tasselation, by default '3D'
    blocks : None | list[int], optional
        `blocks = [NChunks, NThreads]`
        If not `None`, instead of computing the delaunay tesselation of the full distribution, divide it into `NChunks` overlapping sub blocks and process them `NThreads` at a time. The subblocks are then automatically reassembled into the full delaunay tesselation. This option can either be used to increase speed by parallelizing the process (for high values of `NThreads`) or decrease the memory consumption (when `NChunks` >> `NThreads`). By default `None`

    Returns
    -------
    SDndnet_fname : str
        the name of an unstructured network in NDnet format, with density computed for each vertex using DTFE (i.e. density at a given vertex is proportional to the total volume of the surrounding cells).
    """

    if density_file is not None:
        btype='void'

    # set the -blocks option
    if blocks is None:
        blocks_opt = []
    else:
        blocks_opt = ["-blocks", format(blocks[0], 'd'), format(blocks[1], 'd')]

    # run the command for tasselation
    run( ["delaunay_" + dim, filename,
          "-outName", filename,
          "-btype", btype] +
          blocks_opt )  

    ndnet_fname = filename + ".NDnet"       #: NDnet file
    # rename the output for the -blocks option
    if blocks is not None:        
        temp_ndnet_fname = filename + "_G.NDnet"
        print(" rename {0} into {1} \n".format(temp_ndnet_fname, ndnet_fname))
        call(["mv", temp_ndnet_fname, ndnet_fname])
        
    # remove test_smooth.dat
    call(["rm", "test_smooth.dat"])
    
    ##  add separated density field 
    if  density_file is not None:
        change_field_value_with_file(ndnet_fname, density_file)

    ##  smooth density field
    if nsmooth > 0:
        run( ["netconv", ndnet_fname,
              "-outName", ndnet_fname,
              "-smoothData", "field_value", format(nsmooth,'g')]  )
        old_SDndnet_fname = ndnet_fname + ".SD.NDnet"
        SDndnet_fname = filename + ".SD{0:d}.NDnet".format(nsmooth)
        print(" rename {0} into {1} \n".format(old_SDndnet_fname, SDndnet_fname))
        call(["mv", old_SDndnet_fname, SDndnet_fname])
    else:
        SDndnet_fname = ndnet_fname
        
    ## convert to vtu    
    run( ["netconv", SDndnet_fname,
          "-outName", SDndnet_fname,
          "-to", "vtu"] )

    # remove NDnet file
    if SDndnet_fname != ndnet_fname:
        call(["rm", ndnet_fname])
    
    return SDndnet_fname

   
def skl_names(skl_fname, walls=False, patches=False):
    outnames = {'skl':skl_fname}
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
#    outnames['crits'] = basename + persist_ext + ".up.NDskl.BRK" + smooth_ext + ".a.crits"
    outnames['segs'] = basename + persist_ext +".up.NDskl.BRK" + smooth_ext + ".a.segs"        
    if patches:
        outnames['voids'] = basename + persist_ext + "_manifolds_J0a.NDnet" + smooth_ext + ".vtu"
        outnames['nodes'] = basename + persist_ext + "_manifolds_J3d.NDnet" + smooth_ext + ".vtu"
    if walls:
        outnames['walls'] = basename + persist_ext + "_manifolds_J1a.NDnet" + smooth_ext + ".vtu"
    
    return outnames
       
              
                            
def run_disperse(filename, nsig, nsmooth, cutp=None, walls=False, patches=False, 
                 mask=None, robustness=False, nthreads=None, dim='3D'):    
    
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
        
        ## run mse
    run( ["mse", filename,
        "-outName", filename,
        "-upSkl", 
        "-manifolds"] +
        nsigopt +
        mask_opt +
        nthreads_opt +  
        robustness_opt +
        ["-forceLoops"] )

    base_name = "{0}{1}".format(filename, nsigstr)
    skl_name = base_name + ".up.NDskl"

    if patches:        
        ## Dump Voids for taging the galaxies
        run( ["mse", filename,
            "-outName", filename,
            "-loadMSC",filename + ".MSC"] +
            nsigopt + 
            ["-dumpManifolds", "J0a",
            "-forceLoops"] )

        ## converting voids ndnet file to vtu
        voids_name = base_name + "_manifolds_J0a.NDnet"
        netconv_cmd = ["netconv", voids_name,
                    "-outName", voids_name,
                    "-to", "vtu"]
        netconv_cmd.extend(conv_smooth)
        run(netconv_cmd)
   
        ## Dump Nodes'region for taging the galaxies
        if dim=='3D':
            node_manifold = 'J3d'
        elif dim=='2D':
            node_manifold = 'J2d'                
        run( ["mse", filename,
            "-outName", filename,
            "-loadMSC", filename + ".MSC"] +
            nsigopt +
            ["-dumpManifolds",node_manifold,
            "-forceLoops"] )
    
        ## converting nodes ndnet file to vtu
        nodes_name =  base_name + "_manifolds_" + node_manifold + ".NDnet"
        netconv_cmd = ["netconv", nodes_name,
                    "-outName", nodes_name,
                    "-to", "vtu"]
        netconv_cmd.extend(conv_smooth)              
        run(netconv_cmd)
            
    if walls and dim=='3D':
        ## Dump Walls
        run( ["mse", filename,
            "-outName", filename,
            "-loadMSC", filename + ".MSC"] +
            nsigopt +
            ["-dumpManifolds", "J1a",
            "-forceLoops"] )
    
        ## converting and smoothing walls ndnet file to vtu
        walls_name =  base_name + "_manifolds_J1a.NDnet"
        netconv_cmd = ["netconv", walls_name,
                    "-outName", walls_name,
                    "-to", "vtu"]
        netconv_cmd.extend(conv_smooth)         
        run(netconv_cmd)              
    
    ## converting and smoothing NDskl to ascii format
    skelconv_cmd = ["skelconv", skl_name,
                    "-outName", skl_name,
                    "-to","NDskl_ascii"]
    skelconv_cmd.extend(conv_smooth)
    run(skelconv_cmd)          

   ## converting and smoothing NDskl to ascii format with breakdown
    skelconv_cmd = ["skelconv", skl_name,
                    "-outName", skl_name,
                    "-breakdown",
                    "-to","NDskl_ascii"]
    skelconv_cmd.extend(conv_smooth)
    run(skelconv_cmd)          
    
    ## converting and smoothing NDskl to vtp format to open in paraview 
    # no breakdown to keep source index the same
    skelconv_cmd = ["skelconv", skl_name,
                    "-outName", skl_name,
                    "-to", "vtp"]
    skelconv_cmd.extend(conv_smooth)
    run(skelconv_cmd)              
    
    ## converting NDskl to ascii critical points 
#     skelconv_cmd=["skelconv",skl_name,
#                   "-outName",skl_name,
#                   "-breakdown",
#                   "-to","crits_ascii"]
#     skelconv_cmd.extend(conv_smooth)
#     run(skelconv_cmd)
#         
    ## converting NDskl to ascii segments
    skelconv_cmd=["skelconv",skl_name,
                  "-outName",skl_name,
                  "-breakdown",
                  "-to","segs_ascii"]
    skelconv_cmd.extend(conv_smooth)           
    run(skelconv_cmd)
    
    # remove temp files    
    call(["rm", filename+".MSC", skl_name])
    if patches:
        call(['rm', voids_name, nodes_name])
    if walls:
        call(['rm', walls_name ])

    return skl_names(skl_name + smooth_ext + ".a.NDskl", walls, patches)


def write_NDfield_ascii(filename, field):
    header="ANDFIELD\n[{0}]".format(field.size)
    np.savetxt(filename, field, header=header, fmt="%.12g", comments="")


def change_field_value_with_file(ndnet_fname: str, density_file, field_name: str = "field_value"):
    ncv_cmd = ["netconv", ndnet_fname,
               "-outName", ndnet_fname,
               "-addField", density_file, field_name, 
              ]
    run(ncv_cmd)
    old_ndnet_fname = ndnet_fname + ".NDnet"
    print(" rename {0} into {1} \n".format(old_ndnet_fname, ndnet_fname))
    call(["mv", old_ndnet_fname, ndnet_fname])


def change_field_value(ndnet_fname, field_value, field_name="field_value"):
    density_fname = ndnet_fname + "_" + field_name + ".a.NDfield"
    write_NDfield_ascii(density_fname, field_value)    
    change_field_value_with_file(ndnet_fname, density_fname, field_name)
    call(['rm', density_fname])       

        

from __future__ import print_function

from .typing import *
from subprocess import check_call, call
import numpy as np

__all__ = [
          ]


FIELD_FORMATS = Literal['NDfield', 
                        'NDfield_ascii', 
                        'FITS', 
                        'survey_ascii', 
                        'SDL-image',
                        'vtk',
                        'vtk_ascii',
                        'vti',
                        'vti_ascii',
                        'NDnet'
                       ]
SKELETON_FORMATS = Literal['NDskl', 
                           'NDskl_ascii', 
                           'segs_ascii', 
                           'crits_ascii', 
                           'SDL-image',
                           'vtk',
                           'vtk_ascii',
                           'vtp',
                           'vtp_ascii',
                           'NDnet'
                          ]
NETWORK_FORMATS = Literal['NDnet', 
                          'NDnet_ascii', 
                          'PLY', 
                          'PLY_ascii', 
                          'vtk',
                          'vtk_ascii',
                          'vtu',
                          'vtu_ascii'
                         ]

def _run(cmd: list[str]) -> None:
    """Run the code

    Parameters
    ----------
    cmd : str
        The command to execute via bash
    """
    print()
    print(' '.join(cmd))
    print()
    check_call(cmd)

def py_mse(filename: str, nsig: Union[float, Sequence[float]] = 0, cut: Optional[Union[float, Sequence[float]]] = None, *mse_args, **mse_kwargs) -> None:
    """Run `mse` function of DisPerSE

    Parameters
    ----------
    filename : str
        The name of a file containing the discrete particle coordinates in a `field` format.
    nsig : float | Sequence[float], optional
        sets persistence ratio threshold for DTFE type densities.
        This sets a limit on persistence ratios in terms of 'n-sigmas'
        Use this for DTFE densities in delaunay tesselations.
    cut : float | Sequence[float], optional
        sets persistence threshold.
        There will be distinct outputs for each given threshold
    
    Args
    ----
    `'noTags'`
        Prevents mse from adding trailing extensions to the output filename such as the persistence cut levels... 
    `'forceLoops'` 
        Forces the simplification of non-cancellable persistence pairs (saddle-saddle pairs in 3D or more that are linked by at least 2 different arcs). When two critical points of critical index difference 1 are linked by 2 or more arcs, they may not be cancelled as this would result in a discrete gradient loop. This is not a problem in 2D as such pairs cannot form persistence pairs but in 3D, saddle-saddle persistence pairs may be linked by 2 or more arcs even though their persistence is low. By default those pairs are skipped in order to preserve the properties of the Morse-smale complex but as a result few non persistent features may remain (such as spurious filaments). Fortunately, there are usually none or only very few of those pairs, and their number is shown in the output of mse, in the Simplifying complex section. If you are only interested in identifying structures (as opposed to doing topology), you should probably use '-forceLoops' to remove those spurious structures (such as small non significant filaments).
    `'robustness'`  
        Enables the computation of robustness and robustness ratio. By default, robustness is not computed as it can be costly for very large data sets. When enabled, a robustness value is tagged for each segments and node of the output skeleton files. See also options -trimBelow of skelconv and the tutorial section for applications.
    `'no_robustness'` 
        The opposite of `robustness`
    `'manifolds'` 
        Forces the computation and storage of all ascending and descending manifolds (i.e. walls, voids, ...). By default, mse only stores manifolds geometry if required (for instance, when the option -dumpManifolds is used), and the resulting backup MSC file will therefore not contain this information and may not be used later to compute manifolds.
    `'interArcsGeom'` 
        This is similar to option -manifolds, but for the arcs linking different types of saddle-points together (in 3D or more). By default, unless option -interSkl is specified, only the geometry of arcs linking extrema to saddle points is computed, so the backup .MSC file may not be used later to retrieve other types of arcs geometry. Specifying -interArcsGeom forces the computation and storage of all types of arcs in the .MSC backup file.
    `'no_arcsGeom'` 
        By default, the geometry of arcs linking extrema and saddle point is always computed even when not directly needed, so that it is stored in the backup .MSC file for later use. Specifying -no_arcsGeom may be used to lower memory usage when only the critical points and/or their persistence pairings are needed.
        Exemple: mse filename -ppairs -no_arcsGeom will compute the critical points and persistence pairs using less memory than mse filename -ppairs, but the resulting .MSC file may not be later used to retrieve arcs geometry.
    `'ppairs'` 
        Dumps the persistence pairs as a NDnet network type file. The resulting file is a 1D network, where the critical points are the vertices and 1-cells (segments) represent pairs. Additional information such as the type, persistence, cell in the initial complex, ... of each critical points and pairs is also stored as additional information. Try running netconv filename.ppairs.NDnet -info for a list of available additional data (see also additional data in the network file format section).
    `'upSkl'` 
        Dumps the "up" skeleton (i.e. arcs linking maxima to saddle-points, which trace the filamentary structures) as a NDskl type skeleton file. This command is an alias of -dumpArcs U.
    `'downSkl'` 
        Dumps the "down" skeleton (i.e. arcs linking minima to saddle-points, which trace the anti-filaments, or void filaments) as a NDskl type skeleton file. This command is an alias of -dumpArcs D.
    `'interSkl'` 
        Dumps the "inter" skeleton (i.e. arcs linking different types of saddle-points together) as a NDskl type skeleton file. This will only work in 3D or more, and be careful that those type of arcs may have a very complex structure. This command is an alias of '-dumpArcs I''.
    `'vertexAsMinima'`
        As mse uses discrete Morse theory, each type of critical point corresponds to a particular cell type. By defaults, 0-cells (vertices) are maxima, 1-cells are saddle points, .... and d-cells are minima (d is the number of dimensions). Using -vertexAsMinima, the association is reversed and 0-cells are associated to minima ... while d-cells are associated to maxima. This option can be useful to identify manifolds or arcs as lists of a particular type of cell.
        Example: mse input_filename -dumpManifolds J0a -vertexAsMinima can be used to identify voids (ascending 0-manifolds) as sets of 0-cells (vertices). As a result, voids are identified as sets of pixels or particles (if input_filename is a grid or the delaunay tesselation of an N-body simulation respectiveley), which is easy to relate to the pixels/particles of the input file. If the command mse input_filename -dumpManifolds J0a had been issued instead, each void would have been described by a set of n-cells (in nD).
    `'descendingFiltration'`
        By default, mse uses an ascending filtration to compute the discrete gradient. This option forces the program to use a descending filtration instead. Not that if mse works correctly (and it should hopefully be the case) the general properties of the Morse-smale complex are not affected, but simply critical points and manifolds geometry will slightly differ.
    `'no_saveMSC'`
        Use this if you do not want mse to write a backup .MSC file.
    `'no_gFilter'`
        Prevents the filtration of null-persistence pairs in the discrete gradient. There is no point in using this option except for debugging ...
    `'debug'`
        Outputs some debug information and files.

    Kwargs
    ------
    field : str, optional
        may be used to set/replace the function value
    outName : str, optional
        Specifies the base name of the output file 
        (extensions are added to this base name depending on the output type).
    mask : str, optional
        Mask must be and array.
        By default, non-null values correspond to masked pixels.
        Adding a trailing `'~'` reverses the convention.
        Note that the last extension correponding to the file format is still added.
    periodicity : int, optional
        sets periodic boundary conditions (PBC) when appropriate.
        parameter is a serie of 0/1 enabling periodic boundary conditions
        along the corresponding direction.
        Example: '-periodicity 0' sets non periodic boundaries
                 '-periodicity 101' sets PBC along dims 1 and 3 (x and z)
    nthreads : int, optional    
        Specifies the number of threads.
    dumpArcs : str, optional
        saves arcs geometry (may be called several times).
          * U(p): arcs leading to maxima.
          * D(own): arcs leading to minima.
          * I(nter): other arcs.
          * C(onnect): keeps at least the connectivity information for all arcs.
        Ex: CU dumps geometry of arcs from maxima, only connectivity for others.
    dumpManifolds : str, optional
        saves manifolds geometry (may be called several times).
          * J(oin): join all the the manifolds in one single file
          * E(xtended): compute extended manifolds
          * P(reserve): do not merge infinitely close submanifolds.
          * D(ual): compute dual cells geometry when appropriate.
          * 0123456789: specifies the critical index (for opt. a/d)
          * a/d: compute the Ascending and/or Descending manifold.
        
        Ex: JD0a2ad3d dumps to a single file the ascending manifolds
         of critical index 0 and 2, and the descending ones for critical
         index 2 and 3. Cells are replaced by their dual where appropriate.
    compactify : Literal['natural', 'sphere', 'torus'], optional
          * `'natural'` is the default value, and almost always the best choice.
          * `'torus'` creates reflective (i.e periodic) boundaries.
          * `'sphere'` links boundaries to a cell at infinity.
    loadMSC : str, optional
        Loads a given backup .MSC file. This will basically skip the computation of the Morse-smale complex, therefore gaining a lot of time. By default, mse always write a backup .MSC after the MS-complex is computed. See options -manifolds and -interArcsGeom for information on the limitations.
    """
    cmd = ['mse', filename]
    if cut:
        if isinstance(cut,Sequence):
            cutopt = list(cut)
        else:
            cutopt = [cut]
        cmd += ['-cut'] + cutopt
    elif nsig != 0:
        if isinstance(nsig,Sequence):
            nsigopt = list(nsig)
        else:
            nsigopt = [nsigopt]
        cmd += ['-nsig'] + nsigopt

    for val in mse_args:
        val = '-' + val
        cmd += [val]
    for key, val in mse_kwargs.items():
        key = ['-' + key]
        cmd += key + [val]
    
    _run(cmd)

def py_delaunay(filename: str, dim: Literal['2D', '3D'], *del_args, **del_kwargs) -> None:
    cmd = ['delaunay_' + dim, filename]
    for val in del_args:
        val = '-' + val
        cmd += [val]
    for key, val in del_kwargs.items():
        key = ['-' + key]
        cmd += key + [val]
    
    _run(cmd)

def py_netconv(filename: str, output: NETWORK_FORMATS, *net_args, **net_kwargs) -> None:
    cmd = ['netconv', filename]
    for val in net_args:
        val = '-' + val
        cmd += [val]
    for key, val in net_kwargs.items():
        key = ['-' + key]
        cmd += key + [val]
    cmd += ['-to', output]
    
    _run(cmd)

def py_skelconv(filename: str, output: SKELETON_FORMATS, *skl_args, **skl_kwargs) -> None:
    cmd = ['skelconv', filename]
    for val in skl_args:
        val = '-' + val
        cmd += [val]
    for key, val in skl_kwargs.items():
        key = ['-' + key]
        cmd += key + [val]  
    cmd += ['-to', output]
    
    _run(cmd)

def py_fieldconv(filename: str, output: FIELD_FORMATS, info: bool = False, outname: Optional[str] = None) -> None:
    """Display information about field format files encoding regular 
       grid or point set coordinates, and convert them to other formats

    Parameters
    ----------
    filename : str
        The name of a file containing a regular grid or point set coordinates in a readable field format.
    output : FIELD_FORMATS
        Outputs a file in the selected writable field format
    info : bool, optional
        Prints information on the input file, 
        by default `False`
    outname : Optional[str], optional
        Specifies the base name of the output file, 
        by default `None`
    """
    cmd = ['fieldconv', filename]
    if info:
        cmd += ['-info']
    if outname:
        cmd += ['-outname', outname]
    cmd += ['-to',output]

    _run(cmd)


from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range

import numpy as np
from numpy.typing import NDArray
from tvtk.api import tvtk
from traits.api import Any
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from typing import Optional

class CatalogVtk(tvtk.UnstructuredGrid):

    __ = Any
    
    def __init__(self, vtkfilename: str = '', obj: Optional[NDArray[np.float64]] = None, update: bool = True, **traits):
        tvtk.UnstructuredGrid.__init__(self, obj=obj, update=update, **traits)
        if vtkfilename:
            self.read_vtu(vtkfilename)
        if np.any(self.cell_types_array == np.array(tvtk.Tetra().cell_type)):
            self._dim = 3
            self._cell_type = tvtk.Tetra().cell_type
        elif np.any(self.cell_types_array == np.array(tvtk.Triangle().cell_type)):
            self._dim = 2
            self._cell_type = tvtk.Triangle().cell_type
        else:
            raise Exception('Invalid Delaunay vtu tessealation')
            
    def read_vtu(self, filename: str) -> None:
        """read vtu delaunay network file"""
        print("Reading vtu file {0} \n".format(filename))
        v = tvtk.XMLUnstructuredGridReader(file_name=filename)
        v.update()
        ug = v.output       #: Unstructured Grid
        self.__dict__.update(ug.__dict__)
   
    def write_vtu(self, filename: str) -> None:
        print("Writing vtu file {0} \n".format(filename))
        w = tvtk.XMLUnstructuredGridWriter(file_name=filename)
        w.set_input_data(self)
        w.write()

    def add_point_array(self, array: NDArray, array_name: str) -> None:
        nb_arr = self.point_data.number_of_arrays
#        if array.dtype.type is np.string_ :
#            tmp = tvtk.StringArray()
#            for s in array:
#                tmp.insert_next_value(s)
#            array = tmp
        if not (array.dtype.type is np.string_  or array.dtype.type is np.unicode_):
            self.point_data.add_array(array)
            self.point_data.get_array(nb_arr).name = array_name
    
    def get_cells(self) -> NDArray[np.int_]:
        cells = tvtk.UnstructuredGrid.get_cells(self).to_array().astype(int)
        #cells.shape = (cells.size // (self._dim+2), self._dim+2)
        cells = cells.reshape((-1, self._dim+2))
        cells = cells[:, 1:] # remove type of cells column
        return cells

    def getPointArray(self) -> NDArray:
        return self.points.to_array()[:,:self._dim]
            
    def remove_guards(self, nbgal: Optional[int] = None) -> None:
        """remove guard particles on the boundary
        (added by option "-btype smooth" of delaunay_3D)
        """ 
        if self.nbguard:
            # lets assume all guards are at the end ... (faster to remove)
            firstguard = np.flatnonzero(self.guardmask)[0]
            if nbgal and firstguard != nbgal:
                raise Exception("guard particles are not beginning after end of catalog")
            # remove points
            self.points.number_of_points = firstguard
            self.points.resize(firstguard)
            # remove cells
            cells = self.get_cells()
            cells = np.delete(cells,np.where(cells >= firstguard)[0], axis=0)
            
            self.set_cells(self._cell_type, cells)
            # remove elements in point arrays
            #self.point_data.number_of_tuples = firstguard
            for i in range(self.point_data.number_of_arrays):
                tmp = tvtk.DoubleArray(name=self.point_data.get_array(i).name)
                tmp.from_array(self.point_data.get_array(i).to_array()[:firstguard])
                self.point_data.add_array(tmp)

    @property
    def guardmask(self) -> NDArray[np.int_] | NDArray[np.bool_]:
        true_index = self.point_data.get_array('true_index')
        if true_index:
            mask = true_index == np.array([-1])
            if not any(mask):
                true_index = true_index.to_array().astype(int)
                true_index[true_index.max()+1:] = -1
                mask = true_index == np.array([-1])
            return mask
        else:
            return np.array([], dtype=bool)
        
    @property
    def nbguard(self) -> float:
        return self.guardmask.sum()

    @property
    def galmask(self) -> NDArray[np.int_] | NDArray[np.bool_]:
        true_index = self.point_data.get_array('true_index')
        if true_index:
            mask = true_index != np.array([-1])
            if np.all(mask):
                true_index = true_index.to_array().astype(int)
                true_index[true_index.max()+1:] = -1
                mask = true_index != np.array([-1])
            return mask
        else: 
            return np.ones(self.points.number_of_points, dtype=bool)
       
    @property
    def nbgal(self) -> float:
        return self.galmask.sum()
                     
    @property
    def volume(self) -> float:
        if self._dim !=3:
            raise Exception('Volume not defined for 2D tesselation')
        vol = 0.
        for icell in range(self.number_of_cells):
            cell = self.get_cell(icell)
            vol += cell.compute_volume(*cell.points)              
        return vol
    
    @property
    def area(self) -> float:
        if self._dim !=2:
            raise Exception('Area not defined for 3D tesselation')
        area = 0.
        for icell in range(self.number_of_cells):
            cell = self.get_cell(icell)
            area += cell.compute_area()              
        return area
    
    
    def interpolate_data(self, points: NDArray):
        probe_data = tvtk.PolyData(points=points)
        probe = tvtk.ProbeFilter()
        probe.set_input_data(probe_data)
        probe.set_source_data(self)
        probe.update()
        return probe.output.point_data

    def interpolate_data2(self, points: NDArray, fieldname: str) -> NDArray[np.float64]:
        if self._delaunay is None:
            self._delaunay = Delaunay(self.getPointArray())
        interp = LinearNDInterpolator(self._delaunay, self.point_data.get_array(fieldname))
        
        return interp(points)

    def _determine_principal_axis(self) -> None:
        p = self.getPointArray()
        p = p - p.mean(axis=0)
        _, _, v = np.linalg.svd(p, full_matrices=False)
        self._PA_rot = v
        
    def orient_principal_axis(self) -> None:
        if not self._PA_oriented:
            if self._PA_rot is None:
                self._determine_principal_axis()
            p = self.getPointArray()
            p = p - p.mean()
            p = self._PA_rot.dot(p.T).T
            self.points = p
            self._PA_oriented = True
        
    def orient_back_original(self) -> None:
        if self._PA_oriented:
            p = self.getPointArray()
            p = self._PA_rot.T.dot(p.T).T
            self.points = p
            
             
    def uniform_grid(self, array_name: str, pixsize: int, average: float = 0, orientPA: bool = True) -> tuple[NDArray[np.float64], float]:
        
        print("Interpolating data array on a uniform grid")
        if self.dim !=3:
            raise Exception('Not implemented in 2D yet...')
            
        if orientPA:
            self.orient_principal_axis()
        
        # set coarse grid (to limit the sampled volume)
        p = self.getPointArray()
        pmin = p.min(axis=0)
        pmax = p.max(axis=0)
        coarse_size = (pmax - pmin).min() / 5. 
        coarse_ratio = coarse_size // pixsize
        coarse_size =  coarse_ratio * pixsize       
        gridsizes = np.ceil((pmax - pmin) / coarse_size)
        pmax_rnd = pmin + (gridsizes + 1) * coarse_size

        s = [slice(t[0], t[1], coarse_size) for t in zip(pmin, pmax_rnd)]
        grid_coords = np.array(np.mgrid[s]).reshape(3,-1).T        
        npix = grid_coords.shape[0]
                
        # filter out coarse cells outside tesselation volume
        cl = tvtk.CellLocator() 
        cl._set_data_set(self)
        cl.build_locator()
        keep = np.zeros(npix) 
        for i in range(npix):
            keep[i] = cl.find_cell(grid_coords[i,:]) 

        # should keep cells where one of the 8 corners is inside the volume
        keep = keep.reshape(gridsizes + 1) != -1
        keep = as_strided(keep, shape=tuple(gridsizes)+(2,2,2), strides=2*keep.strides)
        keep = keep.any(axis=(3,4,5))
        
        # go to final grid
        fine_shape = np.insert(keep.shape, np.arange(3)+1, coarse_ratio)
        fine_strides = np.insert(keep.strides, np.arange(3)+1, 0.)
        keep = as_strided(keep, shape=fine_shape , strides=fine_strides)
        keep = keep.reshape(gridsizes * coarse_ratio)
        pmax_rnd -= coarse_size
        if not average:
            pmin += pixsize / 2.
            pmax_rnd += pixsize / 2.
        s = [slice(t[0], t[1], pixsize) for t in zip(pmin, pmax_rnd)]            
        gridsizes *= coarse_ratio         
        grid_coords = np.broadcast_arrays(*np.ogrid[s])        
        gckeep = np.array([c[keep] for c in grid_coords]).reshape(3,-1).T
        
        # interpolate on fine grid 
        if average > 1:
            # monte carlo sampling
            nkeep = gckeep.shape[0]
            r = np.random.random(size=(nkeep * int(average), 3))
            r = r.reshape(average, nkeep, 3)
            r = gckeep + r * pixsize
            interp = self.interpolate_data(r.reshape(-1,3))
            data = interp.get_array(array_name).to_array()
            valid = interp.get_array('vtkValidPointMask').to_array()
            #undef = density[valid == 1].mean() # for undefined values
            undef = data[valid == 1].min() # for undefined values
            data[valid == 0] = undef            
            data = data.reshape(average, nkeep).mean(axis=0)
            
        else:
            # simple resampling interpolation
            interp = self.interpolate_data(gckeep)
            data = interp.get_array(array_name).to_array()
            valid = interp.get_array('vtkValidPointMask').to_array()
            #undef = data[valid == 1].mean() # for undefined values
            undef = data[valid == 1].min() # for undefined values
            data[valid == 0] = undef
            
        grid = np.zeros(gridsizes)
        grid[keep] = data
        grid[~keep] = undef
        
        if average: pmin += pixsize / 2.
        
        return grid, pmin
    

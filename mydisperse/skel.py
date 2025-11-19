from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import map
from builtins import range
from builtins import object

from typing import TextIO, Callable
import numpy as np
from scipy.spatial import cKDTree as KDTree 
from tvtk.api import tvtk
from collections import deque
import itertools
from multiprocessing import Pool

import numbers
from mydisperse.catalogvtk import CatalogVtk

class SkelError(Exception):
    pass

import matplotlib.gridspec as gridspec
## needed for pickling instance method in python 2 (used in pool.map)
#import copy_reg, types
#def _reduce_method(m):
#    if m.im_self is None:
#        return getattr, (m.im_class, m.im_func.func_name)
#    else:
#        return getattr, (m.im_self, m.im_func.func_name)
#copy_reg.pickle(types.MethodType, _reduce_method)


# read a line, skipping comment lines    
def _readline(f: TextIO) -> str:
    _COMMENT = '#'
    for line in f:
        if line.find(_COMMENT) != 0: break
    return line    


def _check_p(f: TextIO, pattern: str, optional: bool = False) -> str:
    line = _readline(f)
    if line.find(pattern) != 0:
        if not(optional): 
            raise SkelError('wrong format, missing: {0}'.format(pattern))
    else: 
        return line



class CriticalPoint(object):
    def __init__(self, typ: int, pos: np.ndarray, val: float, pair: int, boundary: int, destCritId: list[int], filId: list[int]):
        """Constructor of the class

        Parameters
        ----------
        typ : int
            critical index
        pos : np.ndarray
            CP position
        val : float
            CP value
        pair : int
            index of the CP in the persistence pair
        boundary : int
            0 if CP is not on the boundary
        destCritId : list[int]
            list of index of the other extremity CP for each filament connected to it
        filId : list[int]
            list of the filaments indeces
        """
        self.typ = typ                  
        self.pos = pos                  
        self.val = val                  
        self.pair = pair                 
        self.boundary = boundary         
        self.destCritId = destCritId     
        self.filId = filId               

    @property
    def nfil(self) -> int:
        return len(self.filId)
    
    def unconnect_fil(self, filidx) -> None:
        del self.filId[filidx]
        del self.destCritId[filidx]
        
    def convert_distance(self, convert: Callable):
        self.pos = convert(self.pos)


class Filaments(object):
    def __init__(self, fils: list['Filament']):
        self.fils = fils
        self.get_lenghts()

    def get_lenghts(self) -> None:
        self.lenghts = []
        for fil in self.fils:
            self.lenghts.append(fil.len)

    def get_property_array(self, item: str) -> np.ndarray:
        data = []
        for fil in self.fils:
            data.append(fil.__getattribute__(item))
        setattr(self, item, np.array(data))
        return np.array(data)

    def __getattr__(self, item: str) -> np.ndarray | None:
        if item not in self.__dict__.keys():
            try:
                setattr(self, item, self.get_property_array(item))
            except:
                raise(AttributeError, 'The attribute is not present')
            return self.__dict__[item]

    def __len__(self) -> int:
        return len(self.fils)

    def __iter__(self) -> 'FilsIterator':
        return FilsIterator(self.fils)

    def __getitem__(self, i: numbers.Integral | slice | np.ndarray | None = None) -> 'Filament' | 'Filaments':
        Fils = self.return_fils(i)
        for attr in self.__dict__.keys():
            try:
                Fils.get_property_array(attr)
            except:
                continue
        return Fils

    def __add__(self, other: 'Filament' | list['Filament']) -> 'Filaments':
        if isinstance(other, Filament):
            self.fils.append(other)
            for attr in self.__dict__.keys():
                try:
                    self.get_property_array(attr)
                except:
                    continue
            return self
        elif isinstance(other, list):
            for fil in other:
                if isinstance(fil, Filament):
                    self.fils.append(fil)
                else:
                    raise TypeError("Unsupported operand type(s) for +: 'Filaments' and list of '{}'".format(type(fil)))
            for attr in self.__dict__.keys():
                try:
                    self.get_property_array(attr)
                except:
                    continue
            return self
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Filaments' and '{}'".format(type(other)))


    def remove(self, other: int | 'Filament') -> None:
        if isinstance(other, int):
            self.fils.pop(other)
        elif isinstance(other, Filament):
            try:
                self.fils.remove(other)
            except ValueError:
                print("Filament {} not found in Filaments".format(other))
        else:
            TypeError("Unsupported type(s) for remove: passed '{}'".format(type(other)))
        for attr in self.__dict__.keys():
            try:
                self.get_property_array(attr)
            except:
                continue

    def __sub__(self, other: 'Filament' | list['Filament']) -> 'Filaments':
        if isinstance(other, Filament):
            try:
                self.fils.remove(other)
            except ValueError:
                print("Filament {} not found in Filaments".format(other))
            for attr in self.__dict__.keys():
                try:
                    self.get_property_array(attr)
                except:
                    continue
            return self
        elif isinstance(other, list):
            for fil in other:
                if isinstance(fil, Filament):
                    try:
                        self.fils.remove(fil)
                    except ValueError:
                        print("Filament {} not found in Filaments".format(other))
                else:
                    TypeError("Unsupported operand type(s) for -: 'Filaments' and list of '{}'".format(type(fil)))
            for attr in self.__dict__.keys():
                try:
                    self.get_property_array(attr)
                except:
                    continue
            return self
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Filaments' and '{}'".format(type(other)))

    def return_fils(self, i: numbers.Integral | slice | np.ndarray | None) -> 'Filament' | 'Filaments':
        if isinstance(i, numbers.Integral):
            if i < 0:
                i += len(self)
            if 0 <= i < len(self):
                return self.fils[i]
        elif isinstance(i, slice):
            return Filaments(self.fils[i])
        elif np.array_equal(i, i.astype(int)) or np.array_equal(i, i.astype(bool)):
            if len(i) != len(self):
                # in this case i is a vector of indeces
                return Filaments(np.array(self.fils)[i])
            else:
                # then i is a mask
                i = i.astype(bool)
                return Filaments(np.array(self.fils)[i])
        else:
            raise 'Invalid indeces format'
class FilsIterator:
    def __init__(self, fils: list['Filament']):
        self._fils = fils
        self._index = 0

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < len(self._fils):
            self._index += 1
            return self._fils[self._index - 1]
        # End of Iteration
        raise StopIteration
class Filament(object):
    def __init__(self, cp1: CriticalPoint, cp2: CriticalPoint, points: np.ndarray):
        """Constructor of the class

        Parameters
        ----------
        cp1 : CriticalPoint
            start CP   
        cp2 : CriticalPoint
            end CP
        points : np.ndarray
            sample points of the filament `points.shape == (numb_points, space_dim)`
        """
        self._cp1 = cp1
        self._cp2 = cp2
        self._points = points
            
    @property
    def cp1(self) -> CriticalPoint:
        return self._cp1
    
    @property
    def cp2(self) -> CriticalPoint:
        return self._cp2

    @property
    def points(self) -> np.ndarray:
        return self._points    
    @points.setter
    def points(self, value: np.ndarray) -> None:
        self._points = value

    @property
    def nsamp(self) -> int:
        return self._points.shape[0]
        
    def mid_segments(self) -> np.ndarray:
        return (np.roll(self._points, 1, axis=0) + self._points)[1:]/2.

    @property
    def segments_len(self) -> np.ndarray:
        try:
            return self._segs_len
        except AttributeError:
            d = (np.roll(self._points, 1, axis=0) - self._points)[1:]
            self._segs_len = np.sqrt(np.sum(d**2, axis=1))
            return self._segs_len
        
    @property
    def len(self) -> float:
        try:
            return self._len
        except AttributeError:
            self._len = np.sum(self.segments_len) 
            return self._len
        
    @property
    def segments_cumlen_from_cp1(self) -> np.ndarray:
        try:
            return self._segs_clen_cp1
        except AttributeError:
            self._segs_clen_cp1 = self._compute_segments_cumlen(reverse=False)
            return self._segs_clen_cp1

    @property
    def segments_cumlen_from_cp2(self) -> np.ndarray:
        try:
            return self._segs_clen_cp2
        except AttributeError:
            self._segs_clen_cp2 = self._compute_segments_cumlen(reverse=True)
            return self._segs_clen_cp2
        
    def _compute_segments_cumlen(self, reverse: bool = False) -> np.ndarray:
        """distance from cp1 to each mid-segments
                    from cp2 if reverse is True 
        """            
        slen = self.segments_len
        if reverse:
            slen = slen[::-1]
            return (np.cumsum(slen) -  slen / 2.)[::-1]
        else: 
            return np.cumsum(slen) -  slen / 2.
                
    def convert_distance(self, convert: Callable) -> None:
        self._points = convert(self._points)    


                            
class Skel(object):
        
    def __init__(self, filename: str = ''):
        self.crit = []      #: list of critical points
        self.fil  = []      #: list of filaments
        if filename:
            self._filename = filename
            self.read_NDskl_ascii()
            self._chain()
            self.isValid()
        self.fil_data = Filaments(self.fil)

            
    @property
    def nfil(self):
        return len(self.fil)           


    @property
    def ncrit(self):
        return len(self.crit)           

    
    def read_NDskl_ascii(self):
        with open(self._filename, 'r') as f:
            if _check_p(f, 'ANDSKEL'):
                self.ndims = int(_readline(f))                  

            line = _check_p(f, 'BBOX', optional=True)           #: bounding box, defined by `x0` and `delta`
            if line: 
                start = line.find('BBOX') + 4
                s1 = line.find('[', start)
                s2 = line.find(']', start)
                sub = line[s1+1:s2]
                self.bbox = np.asfarray(sub.split(','))         #: `[x0_1, ... , x0_ndims]`
                start = s2 + 1
                s1 = line.find('[',start)
                s2 = line.find(']',start)
                sub = line[s1+1:s2]
                self.bbox_delta = np.asfarray(sub.split(','))   #: `[delta_1, ... , delta_ndims]`

            # check the critical points section    
            if _check_p(f, '[CRITICAL POINTS]'):                
                ncrit = int(_readline(f))                       
                print('reading: {0} critical points'.format(ncrit))
                
                for _ in range(ncrit):    
                    # read 1st line: info on the cp
                    data = _readline(f).split()
                    typ = int(data[0])                                                 
                    pos = np.array([float(x) for x in data[1:1 + self.ndims]])   
                    val = float(data[1 + self.ndims])           
                    pair = int(data[2 + self.ndims])            
                    boundary = int(data[3 + self.ndims])        

                    # read 2nd line:  number of filaments that connect to the CP
                    nfil = int(_readline(f))                   
                    
                    # collect each filament information in lists
                    destCritId = []                             
                    filId  = []                                 
                    # read nfil lines: destination and cp of the nfil filaments
                    for _ in range(nfil):
                        data = _readline(f).split()
                        destCritId.append(int(data[0]))
                        filId.append(int(data[1]))
                    
                    # store information of the CP
                    this_crit = CriticalPoint(typ, pos, val, pair,
                                              boundary, destCritId, filId)
                    self.crit.append(this_crit)
            
            # check the filaments section
            if _check_p(f, '[FILAMENTS]'):
                nfil = int(_readline(f))
                print('reading: {0} filaments'.format(nfil))

                for _ in range(nfil):
                    # print('reading filament i:{0}'.format(i))                
                    data = _readline(f).split()
                    # extract info about extrema
                    cp1 = int(data[0])
                    cp2 = int(data[1])
                    nsamp = int(data[2])    #: number of sampling points
                    fil_points = np.zeros([nsamp, self.ndims])
                    # build the filament
                    n_to_read = nsamp * self.ndims
                    index = 0
                    while n_to_read:
                        data = np.asfarray(_readline(f).split())
                        npoints = np.size(data) // self.ndims
                        fil_points[index:index+npoints, :] = data.reshape([npoints, self.ndims])
                        index += npoints
                        n_to_read -= np.size(data)
                    # store the data
                    this_fil = Filament(self.crit[cp1], self.crit[cp2], fil_points)
                    self.fil.append(this_fil)
            
            # check CPs data information
            if _check_p(f, '[CRITICAL POINTS DATA]', optional=True):
                print('reading: critical points data')
                self.ncrit_data = int(_readline(f))     #: number of fields associated to each CP
                # collect the names in a list
                self.crit_data_name = []
                for _ in range(self.ncrit_data):
                    self.crit_data_name.append(_readline(f).strip())
                # read and collect data for each CP
                for cp in self.crit:
                    cp.data = [eval(Str) for Str in _readline(f).split()]

                ######################################################################################  ADDED TO SOLVE MISSING RATIO
                    if 'persistence_ratio' not in self.crit_data_name:
                        p_id = self.crit_data_name.index('persistence')
                        cp.data.append(cp.data[p_id])
                        cp.data.append(cp.data[p_id])
                self.crit_data_name.append('persistence_ratio')
                self.crit_data_name.append('persistence_nsigmas')
                ######################################################################################

            # check filaments data information
            if _check_p(f, '[FILAMENTS DATA]', optional=True):
                print('reading: filaments data')
                self.nfil_data = int(_readline(f))
                # collect the names in a list
                self.fil_data_name = []     #: number of fields associated to each sampling point of each filament.
                for _ in range(self.nfil_data):
                    self.fil_data_name.append(_readline(f).strip()) 
                # read and store data of each filament
                for fil in self.fil:
                    fil.data = [eval(Str) for Str in _readline(f).split()]

    # replace id numbers pointing to FIL or CP by there object reference
    def _chain(self):
        for cp in self.crit:
            try:
                cp.pair = self.crit[cp.pair]
                cp.destCritId = [self.crit[i] for i in cp.destCritId]
                cp.filId = [self.fil[i] for i in cp.filId]
            except:
                Warning('something wrong')
                # connecting to himself
                cp.pair = cp
                cp.destCritId = [self.crit[i] for i in cp.destCritId]
                cp.filId = [self.fil[i] for i in cp.filId]
                 
    def isValid(self):
        """check the consistency of the tree"""
        for crit in self.crit:
            for j in range(crit.nfil):
                fil = crit.filId[j]
                if (fil.cp1 == crit and fil.cp2 != crit.destCritId[j])\
                    or (fil.cp2 == crit and fil.cp1 != crit.destCritId[j])\
                    or (fil.cp1 != crit and fil.cp2 != crit):
                    raise SkelError('wrong critical point {0} and associated filament {1}'\
                    .format(self.crit.index(crit),j))
        bad = good = []
        for i,fil in enumerate(self.fil):
            if not(fil in fil.cp1.filId) or not(fil in fil.cp2.filId):
                # don't raise the error but remove the filament from the list
                print("wrong filament {0}, \
                    not listed in its critical points".format(i))
                bad.append(fil)
                #raise SkelError("wrong filament {0}, \
                #   not listed in its critical points".format(i))
            else: 
                good.append(fil)

        if bad:
            self.fil[:] = good
        return True


    @property
    def is_broken_down(self):
        """Check if the breakdown option was set when running mse 
        """
        try:
            return self._is_broken_down
        except AttributeError:
            typ4 = [c.typ for c in self.crit if c.typ == 4 ]
            if len(typ4) == 0:
                self._is_broken_down = False
            else:
                self._is_broken_down = True
            return self._is_broken_down

            
    def filter_nodes_alone(self, filter_filaments=True):
        """remove the nodes with only one filament connected to it
            and eventually removes the associated filaments
        """
        #alones_idx, alones_cp =zip(*[(i, cp) for i, cp in enumerate(self.crit) if cp.nfil==1 and cp.typ==3])
        alones_idx = [i for i, cp in enumerate(self.crit) if cp.nfil==1 and cp.typ==3]
        
        # mark bad maxima to be removed
        mask = np.ones(self.ncrit, dtype=bool)
        mask[alones_idx] = False
        
        if filter_filaments:
            mask_fil = np.ones(self.nfil, dtype=bool)

        for i in alones_idx:
            cp = self.crit[i]
            # remove persistence pair reference
            cp.pair.pair = None
            
            # remove the connected filament
            if filter_filaments:
                # mark the filament to be removed
                mask_fil[self.fil.index(cp.filId[0])] = False 
                # remove the connection in the saddle
                saddle = cp.destCritId[0]
                Id_in_saddle = saddle.destCritId.index(cp)
                saddle.unconnect_fil(Id_in_saddle)
                # if saddle is alone remove it
                if saddle.nfil == 0:
                    mask[self.crit.index(saddle)]  = False
                    
        # remove the maxima
        self.crit[:] = np.array(self.crit)[mask]
        
        #remove all marked filaments
        if filter_filaments: 
            self.fil[:] = np.array(self.fil)[mask_fil]    


    def filter_spurious_saddles(self):
        """remove the spurious saddles on the border superposed to maxima 
        plus the filament of null-length that connect them.
        """
        mask = np.ones(self.ncrit, dtype=bool)
        mask_fil = np.ones(self.nfil, dtype=bool)
        for i,f in enumerate(self.fil):
            if not np.linalg.norm(f.cp1.pos - f.cp2.pos):
                mask_fil[i] = False
                saddle = f.cp1
                saddleId = self.crit.index(saddle)
                #del self.crit[saddleId]
                mask[saddleId] = False
                # remove persistence pair ref
                f.cp1.pair.pair = None
                filId_in_max = f.cp2.filId.index(f)
                f.cp2.unconnect_fil(filId_in_max)
                # if saddle is connected to another max, remove the connection
                if f.cp1.nfil == 2:
                    f2id = 1 
                    if saddle.filId[0] != f:
                        f2id = 0
                    f2 = saddle.filId[f2id]
                    mask_fil[self.fil.index(f2)] = False
                    f2.cp1.pair.pair = None
                    filId_in_max = f2.cp2.filId.index(f2)
                    f2.cp2.unconnect_fil(filId_in_max)
                    
        self.fil[:] = np.array(self.fil)[mask_fil]
        self.crit[:] = np.array(self.crit)[mask]        
    
    
    def generate_Ids(self):
        self.ncrit_data += 1
        self.crit_data_name.append('OriginalId')
        for i,cp in enumerate(self.crit):
            cp.data.append(i)
        
            
    def distance_to_nearest_node(self, points):
        """compute the distance of a given point to 
        nearest node (critical point 3)"""
        # nodes = (x for x in self.crit if x.typ==3)
        try:
            tree = self._node_tree
        except  AttributeError:
            nodes_pos = np.array([x.pos for x in self.crit if x.typ==self.ndims])
            self._node_tree = KDTree(nodes_pos)
            tree = self._node_tree
        d, idx = tree.query(points)
        crits_id = np.array([i for i,x in enumerate(self.crit) if x.typ==self.ndims])
        return d, crits_id[idx]


    def distance_to_nearest_saddle(self, points):
        """compute the distance of a given point to 
        nearest saddle-2 (critical point 2)"""
        # nodes = (x for x in self.crit if x.typ==3)
        try:
            tree = self._saddle_tree
        except  AttributeError:
            saddles_pos = np.array([x.pos for x in self.crit if x.typ==self.ndims-1])
            tree = self._saddle_tree = KDTree(saddles_pos)
        d, idx = tree.query(points)
        saddles_id = np.array([i for i,x in enumerate(self.crit) if x.typ==self.ndims-1])
        return d, saddles_id[idx]

        
    def distance_to_skel(self, points, big=None):
        """compute the distance of a given point to the nearest filament"""
        try:
            b = self._big_segments_limit
        except AttributeError:
            pass
        else:    
            if b != big:
                del self._seg_tree

        try:
            tree = self._seg_tree
        except AttributeError:    
            segs_pos = np.vstack([x.mid_segments() for x in self.fil])
            self._big_segments_limit = big  
            if big:
                # remove big segments (expect boundary removal)
                segs_len = np.hstack([x.segments_len for x in self.fil])
                segs_keep, = (segs_len < big).nonzero()
                segs_pos = segs_pos[segs_keep]
            #print 'number of segments:',len(segs_pos)
            self._seg_tree = KDTree(segs_pos)
            tree = self._seg_tree
                        
        distances, indexes = tree.query(points)
        if big:
            indexes = segs_keep[indexes]
        
        if not self.is_broken_down:
            # deals with overlapped filaments (no breakdown case)
            points = np.array(points)
            indexes = [tree.query_ball_point(points[i], distances[i]+1e-7) for i in range(points.shape[0])]
            n_grp = np.array(list(map(len, indexes)))
            start_grp = np.concatenate(([0], np.cumsum(n_grp[:-1])))
            indexes = list(itertools.chain.from_iterable(indexes)) # flat list
                               
        # get the filament number
        istart_fil = np.cumsum(np.array([x.nsamp-1 for x in self.fil[:-1]]))
        istart_fil = np.concatenate(([0], istart_fil))
        fil_indexes = np.searchsorted(istart_fil, indexes, side='right') - 1
        seg_indexes = indexes - istart_fil[fil_indexes]
        
        if not self.is_broken_down:
            # regroup the flat lists
            fil_indexes = [fil_indexes[s:s+n] for s, n in zip(start_grp, n_grp)]
            seg_indexes = [seg_indexes[s:s+n] for s, n in zip(start_grp, n_grp)]
            
        return distances, fil_indexes, seg_indexes

                           
    def distance_along_filament_to_node(self, fil_index, seg_index):
        '''Compute the distance inside a filament (starting at seg_index) up to node.
        For nodes there is no ambiguity: going up certainly lead to one unique node
        '''
        f = self.fil[fil_index]
        fil_lst, cp = self.follow_filament_to_cp(f.cp2, f)
        if fil_lst is None: raise SkelError("could not follow filament to node")
        d = f.segments_cumlen_from_cp2[seg_index]
        fil_lst.popleft()
        while len(fil_lst):
            d += fil_lst.popleft().len
        cp_id = self.crit.index(cp)
        return d, cp_id


    def distance_along_filament_to_saddle(self, fil_index, seg_index):
        '''Compute the distance inside a filament (starting at seg_index) up to saddle.
        This implementation is only valid for non broken skeleton (ie it leads to a unique saddle)
        '''
        f = self.fil[fil_index]
        fil_lst, cp = self.follow_filament_to_cp(f.cp1, f, node=False)
        if fil_lst is None: raise SkelError("could not follow filament to saddle")
        d = f.segments_cumlen_from_cp1[seg_index]
        fil_lst.popleft()
        while len(fil_lst):
            d += fil_lst.popleft().len
        cp_id = self.crit.index(cp)
        return d, cp_id

             
    def filaments_from_saddle(self, crit_index):
        p = self.crit[crit_index]
        if p.typ != self.ndims-1:
            raise SkelError('wrong type, saddle point expected (type {})'.format(self.ndims-1))
        all_fil_list = []
        all_node_list = []
        for i in range(p.nfil):
            #print('tracing filament {0}/{1}'.format(i,p.nfil))
            fil = p.filId[i] 
            dest = p.destCritId[i] 
            res, node = self.follow_filament_to_cp(dest, fil)
            if res != None:
                all_fil_list.append(res)
                all_node_list.append(node) # only one is returned
        return all_fil_list, all_node_list


    def filaments_from_node(self, crit_index):
        '''return the list of filaments and saddles connected to node
        This routine is useful in the case of broken skeleton (option -breakdown of skelconv)
        otherwise it is straightforward.
        Each filament is returned as a list of broken filament 
        (ie with bifurcation point at extremities)
        '''
        p = self.crit[crit_index]
        if p.typ != self.ndims:
            raise SkelError('wrong type, max point expected (type {})'.format(self.ndims))
        all_fil_list = []
        all_saddle_list = []
        for i in range(p.nfil):
            fil = p.filId[i]
            dest = p.destCritId[i] 
            res, saddle = self.follow_filament_to_cp(dest, fil, node=False)
            if res != None:
                all_fil_list.append(res)
                all_saddle_list.append(saddle) # several may be returned
        return all_fil_list, all_saddle_list
        

    def follow_filament_to_cp(self, p, fil, node=True):
        """follow filament chain through bifurcation points up to node
           (if node=True) or up to saddle (if node=False)
        """
        if node:
            cpgood = self.ndims
            cpbad  = self.ndims - 1
        else:
            cpgood = self.ndims - 1
            cpbad = self.ndims
        if p.typ == cpgood: return deque([fil]), p
        if p.typ == cpbad: return None, None
        # then bifurcation
        new_fil_lst = None
        for i in range(p.nfil):
            f = p.filId[i]
            if f == fil: continue
            this_p = p.destCritId[i]
            fil_lst, cp_lst = self.follow_filament_to_cp(this_p, f, node)
            if fil_lst != None: 
                if all([isinstance(x,Filament) for x in fil_lst]):
                    fil_lst.appendleft(fil)
                else:
                    for c in fil_lst:
                        c.appendleft(fil)
                if new_fil_lst is None:                     
                    new_fil_lst = deque([fil_lst])
                    new_cp_lst = [cp_lst]
                else: 
                    new_fil_lst.append(fil_lst)
                    new_cp_lst.extend(cp_lst)
        if len(new_fil_lst) == 1:
            new_fil_lst = new_fil_lst[0]
            new_cp_lst = new_cp_lst[0]
        return new_fil_lst, new_cp_lst


    def fof_arround_max(self, delaunay_cat, fieldname, densfrac=.1, fof_max=30):
        """compute fof, 
                starting from max, 
                stopping at the density fraction densfrac between max and the highest connected saddle.
                (densfrac=0 means stop at the density of the saddle) 
        """
        print("Computing fof arround max")
        typ3Id = np.array([i for i,x in enumerate(self.crit) if x.typ == 3])
        tree = KDTree(delaunay_cat.points)
        pos3 = np.array([self.crit[i].pos for i in typ3Id])
        _, nearest_id = tree.query(pos3)
        field_value = delaunay_cat.point_data.get_array(fieldname).to_array()
        fof_indices = np.empty(delaunay_cat.number_of_points, dtype=int)
        fof_indices.fill(-1)
        fof_size = np.empty(typ3Id.size, dtype=int)
        fof_size.fill(-1)
        for i, Id in enumerate(typ3Id):
            cells = delaunay_cat.get_cells()
            #print "\ncrit point {0}".format(i)
            cp = self.crit[Id]
            # get all the connected saddle and set the threshold to the highest
            _, saddle_lst = self.filaments_from_node(Id)            
            if not saddle_lst: continue # no filaments from this node...
            saddle_dens = np.array([x.val for x in saddle_lst])
            density_thres = saddle_dens.max()
            density_thres += densfrac*(cp.val - density_thres) 
            # follow delaunay graph for nearest points above thres
            #####################################################
            # start the point set with nearest neighbor of node
            point_set = np.array([nearest_id[i]])
            curpoint = 0 
            while curpoint < point_set.size and point_set.size < fof_max:
                #print curpoint,'/',point_set.size
                # find cells containing current point
                cellsId = (np.where(cells == point_set[curpoint]))[0]
                # add in the set the points of the selected cells and above density threshold
                if cellsId.size != 0: 
                    points = np.unique(np.take(cells, cellsId, axis=0))
                    points_val = np.take(field_value, points)
                    points = np.compress(points_val > density_thres, points) # & (points_val < cp.val)
                    point_new = np.setdiff1d(points, point_set, assume_unique=True)
                    point_set = np.append(point_set, point_new)
                    np.delete(cells, cellsId, axis=0)
                curpoint += 1
            if point_set.size >= fof_max:
                print("crit point {0}".format(i), "fof too big, stopping...")
            else:
                fof_indices[point_set] = i
                fof_size[i] = point_set.size
            #print "fof size: ", point_set.size 
        return fof_indices, fof_size

                    
    def convert_distance(self, conversion):
        for fil in self.fil:
            fil.convert_distance(conversion)
        for crit in self.crit:
            crit.convert_distance(conversion)

    @property        
    def len(self):
        return np.sum([f.len for f in self.fil])

    
    @property
    def mean_len(self):
        return np.mean([f.len for f in self.fil])


    def compute_segments_density(self, delaunay_cat, FieldName):
        assert(isinstance(delaunay_cat, CatalogVtk))
        midsegs = np.vstack([fil.mid_segments() for fil in self.fil])
        all_segs_density = delaunay_cat.interpolate_data2(midsegs, FieldName)
        filstart = np.array([fil.nsamp - 1 for fil in self.fil]).cumsum()
        filstart = np.concatenate(([0], filstart))
        for fil_id, fil in enumerate(self.fil):
            fil.segments_density = all_segs_density[filstart[fil_id]:filstart[fil_id+1]]


    def persistence_histogram(self):
        import matplotlib.pyplot as plt

        persistence_ratio_id = self.crit_data_name.index('persistence_ratio')
        persistence_ratio = np.array([x.data[persistence_ratio_id] for x in self.crit if x.typ <= 3])

        persistence_nsig_id = self.crit_data_name.index('persistence_nsigmas')
        persistence_nsig = np.array([x.data[persistence_nsig_id] for x in self.crit if x.typ <= 3])

        persistence_id = self.crit_data_name.index('persistence')
        persistence = np.array([x.data[persistence_id] for x in self.crit if x.typ <= 3])

        ppair_id = self.crit_data_name.index('persistence_pair')
        ppair = np.array([x.data[ppair_id] for x in self.crit if x.typ <= 3])

        field_val_id = self.crit_data_name.index('field_value')
        field_val = np.array([x.data[field_val_id] for x in self.crit if x.typ <= 3])

        typ = np.array([x.typ for x in self.crit if x.typ <= 3])

        ppair_field_val = field_val[ppair]

        good = (np.where(persistence_ratio != -1))[0]
        bad = (np.where(persistence_ratio == -1))[0]
        low = (np.where(ppair_field_val[good] > field_val[good]))[0]
        pair0 = (np.where(typ[good[low]] == 0))[0]
        pair1 = (np.where(typ[good[low]] == 1))[0]
        pair2 = (np.where(typ[good[low]] == 2))[0]

        plt.figure()
        bins0 = max(2, 100)#int(np.sqrt(len(persistence_nsig[good[low[pair0]]]))))
        bins1 = max(2, 100)#, int(np.sqrt(len(persistence_nsig[good[low[pair1]]]))))
        bins2 = max(2, 100)#, int(np.sqrt(len(persistence_nsig[good[low[pair2]]]))))
        plt.hist(np.log10(persistence_nsig[good[low[pair0]]]), bins=bins0, label='min-saddle1')
        plt.hist(np.log10(persistence_nsig[good[low[pair1]]]), bins=bins1, label='saddle1-saddle2', alpha=0.5)
        plt.hist(np.log10(persistence_nsig[good[low[pair2]]]), bins=bins2, label='saddle2-max', alpha=0.2)
        plt.yscale('log')
        plt.ylabel('Counts')
        plt.xlabel('log persistence nsigmas')
        plt.legend()

        plt.figure()
        plt.hist(np.log10(persistence_ratio[good[low[pair0]]]), bins=bins0, label='min-saddle1')
        plt.hist(np.log10(persistence_ratio[good[low[pair1]]]), bins=bins1, label='saddle1-saddle2', alpha=0.5)
        plt.hist(np.log10(persistence_ratio[good[low[pair2]]]), bins=bins2, label='saddle2-max', alpha=0.2)
        plt.yscale('log')
        plt.ylabel('Counts')
        plt.xlabel('log persistence ratio')
        plt.legend()

        plt.figure()
        plt.hist(persistence[good[low[pair0]]], bins=bins0, label='min-saddle1', density=True)
        plt.hist(persistence[good[low[pair1]]], bins=bins1, label='saddle1-saddle2', alpha=0.5, density=True)
        plt.hist(persistence[good[low[pair2]]], bins=bins2, label='saddle2-max', alpha=0.2, density=True)
        plt.xscale('log')
        plt.ylabel('Counts')
        plt.xlabel('persistence')
        plt.legend()

    def persistence_diagram(self):
        import matplotlib.pyplot as plt

        persistence_ratio_id = self.crit_data_name.index('persistence_ratio')
        persistence_ratio = np.array([x.data[persistence_ratio_id] for x in self.crit if x.typ <=3])

        persistence_nsig_id = self.crit_data_name.index('persistence_nsigmas')
        persistence_nsig = np.array([x.data[persistence_nsig_id] for x in self.crit if x.typ <=3])

        persistence_id = self.crit_data_name.index('persistence')
        persistence = np.array([x.data[persistence_id] for x in self.crit if x.typ <=3])
               
        ppair_id = self.crit_data_name.index('persistence_pair')
        ppair =  np.array([x.data[ppair_id] for x in self.crit if x.typ <=3])
        
        field_val_id = self.crit_data_name.index('field_value')
        field_val = np.array([x.data[field_val_id] for x in self.crit if x.typ <=3])
        
        typ = np.array([x.typ for x in self.crit if x.typ <=3])
        
        ppair_field_val = field_val[ppair]
        
        good = (np.where(persistence_ratio != -1))[0]
        low = (np.where(ppair_field_val[good] > field_val[good]))[0]
        pair0 = (np.where(typ[good[low]] == 0))[0]
        pair1 = (np.where(typ[good[low]] == 1))[0]
        pair2 = (np.where(typ[good[low]] == 2))[0]

        plt.figure()
        plt.loglog(field_val[good[low[pair0]]], persistence_nsig[good[low[pair0]]],\
                '.', label='min-saddle1')
        plt.loglog(field_val[good[low[pair1]]], persistence_nsig[good[low[pair1]]],\
                '.', label='saddle1-saddle2')
        plt.loglog(field_val[good[low[pair2]]], persistence_nsig[good[low[pair2]]],\
                '.', label='saddle2-max')
        plt.xlabel('density')
        plt.ylabel('persistence nsigmas')
        plt.legend()

        plt.figure()
        plt.loglog(field_val[good[low[pair0]]], persistence_ratio[good[low[pair0]]],\
                '.', label='min-saddle1')
        plt.loglog(field_val[good[low[pair1]]], persistence_ratio[good[low[pair1]]],\
                '.', label='saddle1-saddle2')
        plt.loglog(field_val[good[low[pair2]]], persistence_ratio[good[low[pair2]]],\
                '.', label='saddle2-max')
        plt.xlabel('density')
        plt.ylabel('persistence ratio')
        plt.legend()

        plt.figure()
        plt.loglog(field_val[good[low[pair0]]], persistence[good[low[pair0]]],\
                '.', label='min-saddle1')
        plt.loglog(field_val[good[low[pair1]]], persistence[good[low[pair1]]],\
                '.', label='saddle1-saddle2')
        plt.loglog(field_val[good[low[pair2]]], persistence[good[low[pair2]]],\
                '.', label='saddle2-max')
        plt.xlabel('density')
        plt.ylabel('persistence')                  
        plt.legend(loc=4)        
        
        plt.show()


    def dump_persistence_diagram(self, mode='nsig'):
        import matplotlib.pyplot as plt
        variables={'nsig':'persistence_nsigmas', 'persistence':'persistence', 'ratio':'persistence_ratio'}

        persistence_id = self.crit_data_name.index('persistence_nsigmas')
        persistence = np.array([x.data[persistence_id] for x in self.crit if x.typ <=3])

        ppair_id = self.crit_data_name.index('persistence_pair')
        ppair =  np.array([x.data[ppair_id] for x in self.crit if x.typ <=3])

        field_val_id = self.crit_data_name.index('field_value')
        field_val = np.array([x.data[field_val_id] for x in self.crit if x.typ <=3])

        typ = np.array([x.typ for x in self.crit if x.typ <=3])

        ppair_field_val = field_val[ppair]

        good = (np.where(persistence != -1))[0]
        low = (np.where(ppair_field_val[good] > field_val[good]))[0]
        pair0 = (np.where(typ[good[low]] == 0))[0]
        pair1 = (np.where(typ[good[low]] == 1))[0]
        pair2 = (np.where(typ[good[low]] == 2))[0]

        flag_min=True
        flag_sad=True
        flag_max=True

        if len( field_val[good[low[pair0]]] ) < 10:
            flag_min=False
        if not len( field_val[good[low[pair1]]] ):
            flag_sad=False
        if not len( field_val[good[low[pair2]]] ):
            flag_max=False


        fig = plt.figure(figsize=(12,12))
        gs = gridspec.GridSpec(4, 4, figure=fig)
        ax_joint = fig.add_subplot(gs[1:4, 0:3])
        ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

        min_val_d = 10**10
        min_val_p = 10**10
        max_val_d = -10**10
        max_val_p = -10**10
        num_bins = 0
        if flag_min:
            min_val_d=min(min_val_d, field_val[good[low[pair0]]].min())
            max_val_d=max(max_val_d, field_val[good[low[pair0]]].max())
            min_val_p=min(min_val_p, persistence[good[low[pair0]]].min())
            max_val_p=max(max_val_p, persistence[good[low[pair0]]].max())
            num_bins = max(num_bins, len(field_val[good[low[pair0]]]))
        if flag_sad:
            min_val_d=min(min_val_d, field_val[good[low[pair1]]].min())
            max_val_d=max(max_val_d, field_val[good[low[pair1]]].max())
            min_val_p=min(min_val_p, persistence[good[low[pair1]]].min())
            max_val_p=max(max_val_p, persistence[good[low[pair1]]].max())
            num_bins = max(num_bins, len(field_val[good[low[pair1]]]))
        if flag_max:
            min_val_d=min(min_val_d, field_val[good[low[pair2]]].min())
            max_val_d=max(max_val_d, field_val[good[low[pair2]]].max())
            min_val_p=min(min_val_p, persistence[good[low[pair2]]].min())
            max_val_p=max(max_val_p, persistence[good[low[pair2]]].max())
            num_bins = max(num_bins, len(field_val[good[low[pair2]]]))

        num_bins = int(1.5*np.sqrt(num_bins))
        bins_d = np.logspace(np.log10(min_val_d), np.log10(max_val_d), num_bins)
        bins_p = np.logspace(np.log10(min_val_p), np.log10(max_val_p), num_bins)

        if flag_min:
            # ADD SHARE_X
            ax_joint.loglog(field_val[good[low[pair0]]], persistence[good[low[pair0]]], \
                       '.', label='saddle1-saddle2', color='cyan')
            ax_marg_y.hist(persistence[good[low[pair0]]], label='saddle1-saddle2', color='cyan',
                           alpha=0.3, density=True, orientation="horizontal", bins=bins_p)
            ax_marg_x.hist(field_val[good[low[pair0]]], label='saddle1-saddle2', color='cyan',
                           alpha=0.3, bins=bins_d, density=True,)

        if flag_sad:
            ax_joint.loglog(field_val[good[low[pair1]]], persistence[good[low[pair1]]], \
                       '.', label='saddle1-saddle2', color='gold')
            ax_marg_y.hist(persistence[good[low[pair1]]], label='saddle1-saddle2', color='gold',
                           alpha=0.3, density=True, orientation="horizontal", bins=bins_p)
            ax_marg_x.hist(field_val[good[low[pair1]]], label='saddle1-saddle2', color='gold',
                           alpha=0.3, bins=bins_d, density=True,)

        if flag_max:
            ax_joint.loglog(field_val[good[low[pair2]]], persistence[good[low[pair2]]], \
                       '.', label='saddle2-max', color='firebrick')
            ax_marg_y.hist(persistence[good[low[pair2]]], label='saddle2-max', color='firebrick',
                           alpha=0.2, density=True,orientation="horizontal", bins=bins_p)
            ax_marg_x.hist(field_val[good[low[pair2]]], label='saddle2-max', color='firebrick',
                           alpha=0.2, bins=bins_d, density=True,)

        ax_marg_x.set_xscale('log')
        ax_marg_y.set_yscale('log')
        ax_marg_x.set_yscale('log')
        ax_marg_y.set_xscale('log')
        # Turn off tick labels on marginals
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        # Set labels on joint
        ax_joint.set_xlabel('Density')
        ax_joint.set_ylabel('Persistence')

        # Set labels on marginals
        ax_marg_y.set_xlabel('Counts')
        ax_marg_x.set_ylabel('Counts')

        ax_joint.legend(loc='upper left', bbox_to_anchor=(1.04, 1.385), fancybox=True, shadow=True, ncol=1)


        return fig



    def write_vtp(self, filename):
        """
        write skeleton to a vtk PolyData file (.vtp format)
        """
        # get nb o f  points
        npoints = self.ncrit
        fil_npoints = np.array([fil.nsamp-2 for fil in self.fil])
        fil_npoints = fil_npoints.sum()
        npoints += fil_npoints
        
        points = np.zeros((npoints, 3)) 
        verts = np.arange(self.ncrit)[:,np.newaxis] 
        lines = []
        for i,crit in enumerate(self.crit):
            points[i, :self.ndims] = crit.pos
        start = self.ncrit
        for fil in self.fil:
            end = start + fil.nsamp-2
            points[start:end, :self.ndims] = fil.points[1:-1] 
            line = [self.crit.index(fil.cp1)]
            line.extend(list(range(start, end)))
            line.append(self.crit.index(fil.cp2))
            lines.append(line)
            start = end        
        vp = tvtk.PolyData(points=points, verts=verts, lines=lines)
        
        # add point data arrays
        for i in range(self.ncrit_data):
            array = np.array([crit.data[i] for crit in self.crit])
            fillminone = np.empty(fil_npoints, dtype=array.dtype)
            fillminone.fill(-1)
            array = np.concatenate((array, fillminone))
            vp.point_data.add_array(array)
            vp.point_data.get_array(i).name = self.crit_data_name[i]
        array = np.array([crit.typ for crit in self.crit])
        fillminone = np.empty(fil_npoints, dtype=array.dtype)
        fillminone.fill(-1)
        array = np.concatenate((array, fillminone))
        vp.point_data.add_array(array)
        vp.point_data.get_array(i+1).name = "critical_index"
        
        # add cell (lines) data array
        array = np.concatenate((np.repeat(0, self.ncrit), np.arange(self.nfil)))
        vp.cell_data.add_array(array)
        vp.cell_data.get_array(0).name = "arc_id"
        
        array = np.concatenate((np.repeat(1, self.ncrit), np.repeat(2,self.nfil)))
        vp.cell_data.add_array(array)
        vp.cell_data.get_array(1).name = "type"
        
        array = np.concatenate((np.repeat(-1, self.ncrit), \
                                np.array([self.crit.index(fil.cp1) for fil in self.fil])))
        vp.cell_data.add_array(array)                       
        vp.cell_data.get_array(2).name = "down_index"

        array = np.concatenate((np.repeat(-1, self.ncrit), \
                                np.array([self.crit.index(fil.cp2) for fil in self.fil])))
        vp.cell_data.add_array(array)                       
        vp.cell_data.get_array(3).name = "up_index"

        array = np.concatenate((np.repeat(-1, self.ncrit), \
                                np.array([fil.nsamp for fil in self.fil])))
        vp.cell_data.add_array(array)                       
        vp.cell_data.get_array(4).name = "length"
        
        array = np.concatenate((np.array([c.boundary for c in self.crit]), \
                                np.array([fil.cp1.boundary | fil.cp2.boundary for fil in self.fil])))
        vp.cell_data.add_array(array)                       
        vp.cell_data.get_array(5).name = "flags"
                
        print("Writing skeleton vtp file {0} \n".format(filename))

        v = tvtk.XMLPolyDataWriter()
        v.set_input_data(vp)
        v.file_name = filename
        v.write()

        
        
    def write_crits(self, filename):
        with open(filename, 'w') as f:
            print("Writing ascii .crits file {0} \n".format(filename))
            f.write("#critical points\n")
            f.write("#X0 X1 X2 value type pair_id boundary persistence persistence_nsigma persistence_ratio\n")
            f.write("#3 {0:d}\n".format(self.ncrit))
            persistence_ratio_id = self.crit_data_name.index('persistence_ratio')
            persistence_nsig_id = self.crit_data_name.index('persistence_nsigmas')
            persistence_id = self.crit_data_name.index('persistence')
            for crit in self.crit:
                values = list(crit.pos)
                try:
                    pair_id = self.crit.index(crit.pair)
                except ValueError:
                    pair_id = -1
                values.extend([crit.val, crit.typ, pair_id, crit.boundary])
                values.append(crit.data[persistence_id])
                values.append(crit.data[persistence_nsig_id])
                values.append(crit.data[persistence_ratio_id])
                f.write(" ".join(map(str, values)) + "\n")
            f.close()


 
    
voids_buf = None # global var for pool.map
def _compute_cell_vol(idcell):
    global voids_buf
    cell = voids_buf.get_cell(idcell)
    return abs(cell.compute_volume(*cell.points))

class VoidsRegion(object):
    
    def __init__(self,filename):
        self._read(filename)

            
    def _read(self,filename):        
        print("Reading: Void manifolds")
        v=tvtk.XMLUnstructuredGridReader(file_name=filename)
        v.update()
        self.voids = v.output 


    def get_cells(self):
        cells = self.voids.get_cells().to_array().astype(int)
        cells.shape = (cells.size // 5, 5)
        cells = cells[:, 1:5] # remove type of cells column
        return cells

        
    def indices_at_points(self):
        index = self.voids.point_data.get_array('index')
        if index:
            index = index.to_array().astype(int)
        else:
            index = np.arange(self.voids.number_of_points, dtype=int)
        true_index = self.voids.point_data.get_array('true_index')
        if true_index:
            true_index = true_index.to_array().astype(int)
            if not all(index == true_index):
                raise Exception("Error still guards in the voids")
        source_index = self.voids.cell_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        voids_index = np.zeros(np.max(index)+1, dtype=int)-1
        ## cell data to point data
        for p in range(self.voids.number_of_points):
            il = tvtk.IdList()
            self.voids.get_point_cells(p, il)
            if all(source_index[il] == np.array(source_index[il[0]])):
                voids_index[index[p]] = source_index[il[0]]        
        return voids_index



    def volumes_and_mean_densities(self):
        source_index = self.voids.cell_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        usi, ri = np.unique(source_index, return_inverse=True)

        # compute all tetra volumes
        global voids_buf
        voids_buf = self.voids

        pool = Pool()
        cells_vol = np.array(pool.map(_compute_cell_vol, range(self.voids.number_of_cells)))
        pool.close()            

        
        # compute voids volume
        nb_voids = usi.size
        voids_vol = np.zeros(nb_voids)        
        for i, v in zip(ri, cells_vol):
            voids_vol[i] += v

        # compute nb of gals in each void
        cells = self.get_cells()
        voids_dens = np.zeros(nb_voids)
        for i in range(nb_voids):
            this_void_cells = cells[ri == i,:]
            voids_dens[i] = np.unique(this_void_cells).size
            
        voids_dens /= voids_vol

        return voids_vol, voids_dens, usi

                 
                 
                 
class NodesRegion(object):

    def __init__(self,filename,vr):
        self._read(filename)
        if isinstance(vr,VoidsRegion):
            self.vr = vr
        else: raise SkelError('A VoidRegion is needed to initialize a NodeRegion')
 
           
    def _read(self,filename):  
        print("Reading: Peak manifolds")
        v=tvtk.XMLUnstructuredGridReader(file_name=filename)
        v.update()
        self.nodes = v.output 
 
   
    def indices_at_points(self):
        index = self.nodes.point_data.get_array('index')
        if index:
            index = index.to_array().astype(int)
        else:
            index = np.arange(self.nodes.number_of_points, dtype=int)
        true_index = self.nodes.point_data.get_array('true_index')
        if true_index:
            true_index = true_index.to_array().astype(int)
            if not all(index == true_index):
                raise Exception("Error still guards in the nodes")
        source_index = self.nodes.point_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        nodes_index = np.zeros(np.max(index)+1, dtype=int)-1
        nodes_index[index] = source_index
        return nodes_index

            
    def volumes_and_mean_densities(self):
        cells = self.vr.get_cells()
        index_vr = self.vr.voids.point_data.get_array('index')
        if index_vr:
            index_vr = index_vr.to_array().astype(int)
        else:
            index_vr = np.arange(self.vr.voids.number_of_points, dtype=int)
        cells = index_vr[cells] 
        source_index = self.nodes.point_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        index = self.nodes.point_data.get_array('index')
        if index:
            index = index.to_array().astype(int)
        else:
            index = np.arange(self.nodes.number_of_points, dtype=int)    
        usi = np.unique(source_index)
        nb_nodes = usi.size
        nodes_dens = np.zeros(nb_nodes)
        nodes_vol = np.zeros(nb_nodes)
        for i,si in enumerate(usi):
            points, = np.where(source_index==np.array(si))
            points = index[points]
            cellsid = np.in1d(cells[:,0],points)
            cellsid &= np.in1d(cells[:,1],points)
            cellsid &= np.in1d(cells[:,2],points)
            cellsid &= np.in1d(cells[:,3],points)      
            for icell in np.flatnonzero(cellsid):    
                cell = self.vr.voids.get_cell(icell)
                vol = cell.compute_volume(*cell.points) 
                nodes_vol[i] += vol
            if nodes_vol[i]:
                nodes_dens[i] = points.size / nodes_vol[i]
        return nodes_vol, nodes_dens, usi
        



class Walls(object):
    
    def __init__(self,filename):
        print("Reading: Wall manifolds")
        v = tvtk.XMLUnstructuredGridReader(file_name=filename)
        v.update()
        self.walls = v.output 

    
    def _compute_centers(self):
        self._centers = np.zeros((self.walls.number_of_cells,3))
        for i in  range(self.walls.number_of_cells):
            cell = self.walls.get_cell(i)
            cell.triangle_center(cell.points[0],cell.points[1],cell.points[2],
                self._centers[i])


    @property        
    def centers(self):
        try:
            return self._centers
        except AttributeError:
            self._compute_centers()
            return self._centers

        
    def distance(self,points):
        try:
            distances, indexes = self._tree.query(points)
        except AttributeError:
            self._tree = KDTree(self.centers)
            distances, indexes = self._tree.query(points)
        
        source_index = self.walls.cell_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        wall_ids = source_index.take(indexes)
        
        return distances, wall_ids, indexes

    
    def _compute_surfaces(self):
        source_index = self.walls.cell_data.get_array('source_index')
        source_index = source_index.to_array().astype(int)
        usi = np.unique(source_index)
        self._surfaces = np.zeros(usi.size)
        for i,si in enumerate(usi):
            cellsid, = np.where(source_index==np.array(si))
            for icell in cellsid:
                cell = self.walls.get_cell(icell)
                self._surfaces[i] += cell.compute_area() 

    @property                                                                  
    def surfaces(self):
        try:
            return self._surfaces
        except AttributeError:
            self._compute_surfaces()
            return self._surfaces

    @property    
    def total_surface(self):
        return np.sum(self.surfaces)

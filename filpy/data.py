import os
from pandas import DataFrame, read_pickle
import astropy.units as u
from .typing import *

def this_dir() -> str:
    """Compute the current directory

    Returns
    -------
    dir_path : str
        current directory path
    """
    return os.path.dirname(os.path.realpath(__file__))

def file_name(file_path: str) -> str:
    """Return the name of the current file

    Parameters
    ----------
    file_path : str
        the path of the file

    Returns
    -------
    filename : str
        the name of the file
    """
    return os.path.split(file_path)[-1]


class PathVar():
    """Handle the paths

    Attributes
    ----------
    _path : str
        directory path

    Properties
    ----------
    path : str
        directory path
    """

    def __init__(self, path: str = '', mkdir: bool = True) -> None:
        """Construct the path variable

        Parameters
        ----------
        path : str, optional
            path of the directory. If `path == ''` the current 
            directory path is set, by default `''`
        mkdir : bool, optional
            if `True` it creates the directory
        """
        self._path = path if path != '' else this_dir()
        if mkdir and path != '':
            self.make_dir()
            
    @property
    def path(self) -> str:
        return self._path
    
    @path.setter
    def path(self, new_path: str) -> None:
        self._path = new_path
    
    @path.deleter
    def path(self) -> None:
        del self._path

    def check_dir(self) -> bool:
        """Check the presence of directories in the path
        
        Returns
        -------
        dir_presence : bool
            `True` if the path is real
        """
        return os.path.isdir(self.path)

    def make_dir(self) -> None:
        """Make the directories required in the path if they are not"""
        if not self.check_dir():
            dirs = []
            tmp_path = self.path
            while not os.path.isdir(tmp_path):
                tmp_path, dir = os.path.split(tmp_path)
                dirs += [dir]
            for dir in dirs[::-1]:                   
                tmp_path = os.path.join(tmp_path,dir)
                os.mkdir(tmp_path)

    def split(self) -> tuple[str,str]:
        """Split the last directory from the global path

        Returns
        -------
        global_path : str
            directory path
        last_dir : str
            directory/file name
        """
        return os.path.split(self.path)
    
    def directories(self) -> list[str]:
        return self.path.split(os.sep)
    
    def join(self, path: Union[str, list[str]]) -> str:
        if isinstance(path,str):
            return os.path.join(self.path,path)
        elif isinstance(path,list):
            return os.sep.join(path)
        else:
            raise TypeError('Only str and list types are allowed')

    def tree(self) -> None:
        """List the files in the folder"""
        files = os.listdir(self.path)
        if len(files) != 0:
            print('\nFiles in "' + self.path +'"')
            for i, f in enumerate(files):
                print(f'{i:2d} - {f}')
        else:
            print('The directory is EMPTY')

    def dir_list(self, dir: str = '', print_res: bool = False) -> list[str]:
        """List the files in the direct

        Parameters
        ----------
        dir : str, optional
            a subdirectory inside `self.Path`, by default `''`
        print_res : bool, optional
            if `True` the list is printed, by default `False`

        Returns
        -------
        files : list[str]
            list of the files
        """
        path = os.path.join(self.path,dir)
        files = os.listdir(path)
        if print_res:
            print('Files in "' + path +'"')
            for i, f in enumerate(files):
                print(f'{i:2d} - {f}')
        return files
    

    @property
    def files(self) -> 'FileVar':
        """Collect the files in the directory

        Parameters
        ----------
        print_res : bool, optional
            if `True` the list is printed, by default `False`

        Returns
        -------
        collection : FileVar
            files collection
        """
        filesnames = self.dir_list(dir='',print_res=False)
        collection = FileVar(filename=filesnames, dirpath=self.copy())
        return collection

    def copy(self) -> 'PathVar':
        """Copy the class variable"""
        return PathVar(path=self.path,mkdir=False)
    
    def clear(self, verbose: bool = True, exceptions: Optional[Union[str, list[str]]] = None) -> None:
        """Delete all the files in the folder"""
        print('Remove all files in',self.path)
        from subprocess import call
        if isinstance(exceptions,str):
            exceptions = [exceptions]
        if exceptions is not None and verbose:
            print('\nINFO: Files with "'+'" "'.join(exceptions)+'" are not removed')
        for f, names in zip(self.files,self.files.file):
            if exceptions is None or all([excp not in names for excp in exceptions]):
                cmd = ['rm',f]
                if verbose:
                    print(' '.join(cmd))
                call(cmd)
        

    def __add__(self, path: str) -> 'PathVar':
        """Join two paths

        Parameters
        ----------
        path : str
            path to be attached

        Returns
        -------
        new_path : PathVar
            joined paths
        """
        new_path = self.join(path)
        return PathVar(path=new_path,mkdir=False)
     
    def __sub__(self, iter: int) -> 'PathVar':
        """Compute the path nth directory back

        Parameters
        ----------
        iter : int
            the number of directory to bring back

        Returns
        -------
        new_path : PathVar
            computed new_path
        """
        new_path = self.join(self.directories()[:-iter])
        # for _ in range(iter):
        #     new_path = os.path.split(new_path)[0]
        return PathVar(path=new_path,mkdir=False)
    
    def __str__(self) -> str:
        return self.path 
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + ': "' + self.path + '"'

class _FileIterator():
    def __init__(self, files: list[str]):
        self._files = files
        self._index = 0

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < len(self._files):
            self._index += 1
            return self._files[self._index - 1]
        raise StopIteration

class FileVar():
    """Handle the file(s) path(s)

    Attributes
    ----------
    DIR : PathVar
        directory path
    FILE : str | list[str]
        file name or list of files names
        
    """

    def __init__(self, filename: Union[str, list[str]], dirpath: Union[str, PathVar] = '', path: bool = False) -> None:
        """Construct the file(s) path(s) variable

        Parameters
        ----------
        filename : str | list[str]
            file name or list of files names
        dirpath : str | PathVar, optional
            directory path, by default `''`
        path : bool, optional
            if `True` the `filename` path is computed, by default `False`
        """
        if path: 
            dirpath  = os.path.dirname(filename)
            filename = os.path.split(filename)[-1]
        if isinstance(dirpath, str): dirpath = PathVar(path=dirpath)
        
        self._dir  = dirpath.copy()
        self._file = filename if isinstance(filename, str) else [*filename]

    @property
    def dir(self) -> 'PathVar':
        return self._dir
    
    @dir.setter
    def dir(self, new_dir: PathVar) -> None:
        self._dir = new_dir.copy()
    
    @dir.deleter
    def dir(self) -> None:
        del self._dir

    @property
    def file(self) -> str:
        return self._file
    
    @file.setter
    def file(self, new_file: Union[str,list[str]]) -> None:
        if isinstance(new_file,str):
            self._file = new_file
        elif isinstance(new_file, list):
            self._file = [*new_file]
        else:
            raise TypeError('string or list of strings are allowed only')

    @file.deleter
    def file(self) -> None:
        del self._file

    @property
    def path(self) -> Union[str, list[str]]:
        """Compute the file(s) path(s)

        Returns
        -------
        path_str : str | list[str]
            the path or a list of paths
        """
        filename = self.file 
        dirname = self.dir.copy()
        if isinstance(filename,str): 
            return dirname.join(filename)
        else:
            return [dirname.join(name) for name in filename]
        
    def update_file(self) -> None:
        """Update the list of files in `self.dir`"""
        obj_list = self.dir.dir_list()
        if isinstance(self.file,str):
            self.file = [self.file] + obj_list
        else:
            self.file += obj_list

    def copy(self) -> 'FileVar':
        """Copy the variable"""
        new_file = FileVar(filename=self.file,dirpath=self.dir,path=False)
        return new_file
    
    def clear(self, verbose: bool = True) -> None:
        """Delete all files in the folder"""
        self.dir.clear(verbose=verbose)
        self.file = []

    def tree(self) -> None:
        self.dir.tree()

    def __len__(self) -> int:
        if isinstance(self.file,str):
            return 0
        else:
            return len(self.file)


    def __add__(self, new_file: Union[str, list[str]]) -> 'FileVar':
        """Add a file or more to the list of files

        Parameters
        ----------
        new_file : Union[str, list[str]]
            a file or a list of files to add to `self.file`

        Returns
        -------
        new_filevar : FileVar
            the `FileVar` with the updated file list
        """
        new_filevar = self.copy()
        filename = new_filevar.file 
        if isinstance(filename,str):
            new_filevar.file = [filename, new_file]
        else:
            new_filevar.file += [new_file] 
        return new_filevar

    def __getitem__(self, item: Union[int, slice, list]) -> str:
        """Select a certain file path from a list of ones

        Parameters
        ----------
        item : int
            the position of the chosen path

        Returns
        -------
        path_i : str
            the chosen path
        """
        path = self.path
        if isinstance(path,str): 
            return TypeError('Variable is not subscriptable')
        else: 
            if isinstance(item, list):
                return [path[i] for i in item]
            else:
                return path[item]
    
    def __setitem__(self, key: Union[int, slice], item: Union[str, list[str]]) -> None:
        """Modify a file name of a list of paths

        Parameters
        ----------
        key : int
            chosen file
        item : str
            new name
        """
        if isinstance(self.file,str): 
            raise TypeError('Variable is not subscriptable')
        else: 
            self.file[key] = item

    def __str__(self) -> str:
        output = ''
        path = self.path
        if isinstance(path,str):
            output = path
        elif isinstance(path,list):
            for i, f in enumerate(path):
                output = output + f'{i} - {f}\n'
        return output
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + ':\n"\n' + self.__str__() + '"'

    def __iter__(self) -> _FileIterator:
        fpath = self.path
        if isinstance(fpath,str): 
            raise TypeError('Required a list of files')
        else:
            return _FileIterator(self.path)

    def __contains__(self, element: str) -> bool:
        output = element in self.file
        if not output:
            output = element in self.path            
        return output

class DataDir(PathVar):

    STD_NAME = 'file_names.txt'
    STD_INC = "# Files Names"

    def __init__(self, path: str = '', mkdir: bool = True):
        super().__init__(path, mkdir)
        self._filepath = os.path.join(self.path,DataDir.STD_NAME) 
        self.mk_database()
    
    def mk_database(self) -> None:        
        try:
            path = self._filepath
            f = open(path,"x")
            f.close()
            dir_list = self.dir_list(print_res=False)
            dir_list.remove(DataDir.STD_NAME)
            self.overwrite(dir_list)
        except FileExistsError:
            pass

    def update_database(self) -> None:
        dir_list = self.dir_list()
        files_list = self.load_filelist()

        for file in files_list:
            if file not in dir_list:
                files_list.remove(file)
        self.overwrite(files_list)

        new_lines = []
        for dirfile in dir_list:
            if dirfile not in files_list and dirfile != DataDir.STD_NAME:
                new_lines += [dirfile]
        if new_lines:
            self.add_lines(new_lines)        


    def load_filelist(self) -> list[str]:
        names = self.read()
        files_list = names.split('\n')[1:]
        return files_list
    
    def read(self) -> str:
        path = self._filepath
        with open(path,"r") as f:
            text = f.read()
        return text
    
    def readlines(self) -> list[str]:
        path = self._filepath
        with open(path,"r") as f:
            lines = f.readlines()
        return lines
    
    def add_lines(self, lines: Union[str,list[str]]) -> None:
        path = self._filepath
        if isinstance(lines, str):
            lines = [lines]
        with open(path,"a") as f:
            f.write('\n')
            f.write('\n'.join(lines))
    
    def overwrite(self, new_text: list[str]) -> None:
        path = self._filepath
        new_text = [DataDir.STD_INC] + new_text
        with open(path,"w") as f:
            f.write('\n'.join(new_text))


    def tree(self):
        self.update_database()
        file_list = self.load_filelist()
        if file_list:
            print('\nFiles in '+self.path)
            for i, file in enumerate(file_list):
                print(f'{i:2d} - {file}')
        else:
            print('The directory is EMPTY')

    def clear(self, verbose = True, exceptions = None) -> None:
        self.update_database()
        super().clear(verbose, exceptions)

    @property
    def files(self) -> 'DataFile':
        return DataFile(dirpath=self)
        
    def copy(self) -> 'DataDir':
        return DataDir(self.path, mkdir=False)

    def __add__(self, path: str) -> 'DataDir':
        new_path = self.join(path)
        return DataDir(path=new_path, mkdir=False)
    
    def __sub__(self, iter: int) -> 'DataDir':
        new_path = self.join(self.directories()[:-iter])
        return DataDir(path=new_path, mkdir=False)
    

class DataFile(FileVar):
    def __init__(self, dirpath: Union[str,DataDir,PathVar]):
        if isinstance(dirpath,str):
            dirpath = DataDir(dirpath)
        elif isinstance(dirpath, PathVar):
            dirpath = DataDir(dirpath.path)
        
        dirpath.update_database()
        self._dir = dirpath.copy()
        self._file = dirpath.load_filelist()

    @property
    def dir(self) -> DataDir:
        return self._dir

    @dir.setter
    def dir(self, new_dir: DataDir) -> None:
        self._dir = new_dir.copy()

    @property
    def file(self) -> list[str]:
        return self._file

    @file.setter
    def file(self, new_list: list[str]) -> None:
        self._file = [*new_list] 

    def update_file(self) -> None:
        self.dir.update_database()
        self.file = self.dir.load_filelist()
    

    def copy(self) -> 'DataFile':
        new_var = DataFile(self.dir)
        if len(new_var) != len(self):
            new_var.file = self.file
        return new_var

## Useful paths for the library
PKG_DIR = PathVar()
PROJECT_DIR = PKG_DIR - 1
MBM40_DIR = PROJECT_DIR + 'MBM40'
MBM40_IR = DataDir(MBM40_DIR.join('IR'), mkdir=False)
MBM40_CO = DataDir(MBM40_DIR.join('CO'), mkdir=False)
MBM40_HI = DataDir(MBM40_DIR.join('HI'), mkdir=False)

def update_database(data_dir: list[PathVar] = [MBM40_IR,MBM40_HI,MBM40_CO], filename: str = 'data', outdir: Union[str, PathVar] = MBM40_DIR) -> None:
    data_file = FileVar(filename=filename+'.csv',dirpath=outdir) 
    collection = {}
    columns = []
    elements = []
    n_elem = []
    for dir in data_dir:
        column = os.path.split(dir.path)[-1]
        files_list = dir.dir_list()
        elements += [files_list]
        columns += [column]
        n_elem += [len(files_list)]
    max_len = max(n_elem)
    for key, elem, num in zip(columns,elements, n_elem):
        collection[key] = elem + ['']*(max_len-num)
    DataFrame(collection,dtype='str').to_csv(data_file.path,index=False,header=True)

update_database()
DATA_FILE = FileVar(filename='data.csv', dirpath=MBM40_DIR)
CO_PATHS = MBM40_CO.files
HI_PATHS = MBM40_HI.files
IR_PATHS = MBM40_IR.files

def read_database(datafile: Union[str,FileVar] = DATA_FILE, print_res: bool = True) -> DataFrame:
    from pandas import read_csv
    if isinstance(datafile,str):
        dataframe = read_csv(datafile)
    elif isinstance(datafile,FileVar):
        dataframe = read_csv(datafile.path)
    else:
        raise TypeError('Only str or FileVare types are allowed')
    if print_res:
        print(dataframe.to_string())
    return dataframe

## Constants
U_VEL = u.km / u.s
u.add_enabled_units(u.def_unit(['K (Tb)'], represents=u.K))


"""
~~~~ IRAS DATA ~~~~
Byte-per-byte Description of file: main.dat
--------------------------------------------------------------------------------
   Bytes Format  Units   Label    Explanations
--------------------------------------------------------------------------------
   1- 11  A11    ---     IRAS     IRAS source name
  12- 13  I2     h       RAh      Hours RA, equinox 1950.0, epoch 1983.5
  14- 15  I2     min     RAm      Minutes RA, equinox 1950.0, epoch 1983.5
  16- 18  I3     ds      RAds     Seconds RA, equinox 1950.0, epoch 1983.5
      19  A1     ---     DE-      Sign Dec, equinox 1950.0, epoch 1983.5
  20- 21  I2     deg     DEd      Degrees Dec, equinox 1950.0, epoch 1983.5
  22- 23  I2     arcmin  DEm      Minutes Dec, equinox 1950.0, epoch 1983.5
  24- 25  I2     arcsec  DEs      Seconds Dec, equinox 1950.0, epoch 1983.5
  26- 28  I3     arcsec  Major    Uncertainty ellipse major axis
  29- 31  I3     arcsec  Minor    Uncertainty ellipse minor axis
  32- 34  I3     deg     PosAng   Uncertainty ellipse position angle (1)
  35- 36  I2     ---     NHcon    Number of times observed
  37- 45  E9.3   Jy      Fnu_12   Average non-color corrected flux density,
                                    12um (5)
  46- 54  E9.3   Jy      Fnu_25   Average non-color corrected flux density,
                                    25um (5)
  55- 63  E9.3   Jy      Fnu_60   Average non-color corrected flux density,
                                    60um (5)
  64- 72  E9.3   Jy      Fnu_100  Average non-color corrected flux density,
                                    100um (5)
      73  I1     ---   q_Fnu_12   [1,3] Flux density quality, 12um (3)
      74  I1     ---   q_Fnu_25   [1,3] Flux density quality, 25um (3)
      75  I1     ---   q_Fnu_60   [1,3] Flux density quality, 60um (3)
      76  I1     ---   q_Fnu_100  [1,3] Flux density quality, 100um (3)
  77- 78  I2     ---     NLRS     Number of significant LRS spectra (4)
  79- 80  A2     ---     LRSChar  Characterization of averaged LRS spectrum (4)
  81- 83  I3     %     e_Fnu_12   Percent relative flux den. uncertainty, 12um
  84- 86  I3     %     e_Fnu_25   Percent relative flux den. uncertainty, 25um
  87- 89  I3     %     e_Fnu_60   Percent relative flux den. uncertainty, 60um
  90- 92  I3     %     e_Fnu_100  Percent relative flux den. uncertainty, 100um
  93- 97  I5     ---     TSNR_12  10x minimum signal-to-noise ratio, 12um
  98-102  I5     ---     TSNR_25  10x minimum signal-to-noise ratio, 25um
 103-107  I5     ---     TSNR_60  10x minimum signal-to-noise ratio, 60um
 108-112  I5     ---     TSNR_100 10x minimum signal-to-noise ratio, 100um
     113  A1     ---     CC_12    Point source correlation coeff., 12um (8)
     114  A1     ---     CC_25    Point source correlation coeff., 25um (8)
     115  A1     ---     CC_60    Point source correlation coeff., 60um (8)
     116  A1     ---     CC_100   Point source correlation coeff., 100um (8)
 117-118  I2     %       Var      Percent likelihood of variability
     119  A1     ---     Disc     Discrepant fluxes flag, 1 per band,
                                    hex encoded (6)
     120  A1     ---     Confuse  Confusion flags, 1 per band, hex encoded (6)
     121  I1     ---     PNearH   Number of nearby hours-confirmed point sources
     122  I1     ---     PNearW   Number of nearby weeks-confirmed point sources
     123  I1     ---     SES1_12  Nearby seconds-confirmed small ext., 12um (7)
     124  I1     ---     SES1_25  Nearby seconds-confirmed small ext., 25um (7)
     125  I1     ---     SES1_60  Nearby seconds-confirmed small ext., 60um (7)
     126  I1     ---     SES1_100 Nearby seconds-confirmed small ext., 100um (7)
     127  I1     ---     SES2_12  Nearby weeks-confirmed small ext., 12um (7)
     128  I1     ---     SES2_25  Nearby weeks-confirmed small ext., 25um (7)
     129  I1     ---     SES2_60  Nearby weeks-confirmed small ext., 60um (7)
     130  I1     ---     SES2_100 Nearby weeks-confirmed small ext., 100um (7)
     131  A1     ---     HSDFlag  High source density bin flag, hex encoded (6)
     132  I1     ---     Cirr1    Number of nearby 100 micron only WSDB sources
     133  I1     ---     Cirr2    100 micron sky brightness ratio to flux den.
                                    (2)
 134-136  I3     MJy/sr  Cirr3    Total 100 micron sky surface brightness
 137-138  I2     ---     NID      Number of positional associations
     139  I1     ---     IDType   [1,4] Type of association (9)
 140-141  I2     ---     MHcon    ? Possible number of HCONs
 142-145  I4     10-3    FCor_12  ? Flux correction factor applied (5)
 146-149  I4     10-3    FCor_25  ? Flux correction factor applied (5)
 150-153  I4     10-3    FCor_60  ? Flux correction factor applied (5)
 154-157  I4     10-3    FCor_100 ? Flux correction factor applied (5)
--------------------------------------------------------------------------------
"""
def _read_iras_row(row: str) -> tuple:
    """Read a row of `II_125/main.dat`

    Parameters
    ----------
    row : str
        a string 157 char long with information
        reported in notes

    Returns
    -------
    outputs : tuple
        tuple with name, ra, dec, uncertainties in 
        positions, avg fluxes in each band and the 
        corresponding quality

    Notes
    -----

    """
    # identifier
    name = row[:11].split(' ')[0]
    # ra in hours
    ra = float(row[11:13]) + \
         float(row[13:15])/60 + \
         float(row[15:18])/10/3600
    # dec in deg
    dec = float(row[19:21]) + \
          float(row[21:23])/60 + \
          float(row[23:25])/3600         
    # check the sign
    if row[18] == '-': 
        dec *= -1
    
    # unc ellipse major axis
    unc_maj = float(row[25:28]) 
    # unc ellipse minor axis 
    unc_min = float(row[28:31])
    # pos angle
    pos_ang = int(row[31:34])
    # numb of times observed
    nh_con = int(row[34:36])
    # avg non-color corrected flux density
    f_nu_12  = float(row[36:45]) #: 12 um
    f_nu_25  = float(row[45:54]) #: 25 um
    f_nu_60  = float(row[54:63]) #: 60 um
    f_nu_100 = float(row[62:72]) #: 100 um
    # flux density quality
    q_fnu_12  = int(row[72]) #: 12 um
    q_fnu_25  = int(row[73]) #: 25 um
    q_fnu_60  = int(row[74]) #: 60 um
    q_fnu_100 = int(row[75]) #: 100 um
    return name, ra, dec, \
           unc_maj, unc_min, pos_ang, \
           nh_con, \
           f_nu_12, f_nu_25, f_nu_60, f_nu_100, \
           q_fnu_12, q_fnu_25, q_fnu_60, q_fnu_100

IRAS_DEFAULT = ['name','ra','dec','unc_maj','unc_min','pos_ang','nh_con', 'f_nu_12', 'f_nu_25', 'f_nu_60', 'f_nu_100','q_fnu_12', 'q_fnu_25', 'q_fnu_60', 'q_fnu_100']
TYPES_DEFAULT = [str,float,float,float,float,int,int,float,float,float,float,int,int,int,int]

"""
~~~~ IRAS DATA ~~~~
Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format  Units    Label      Explanations
--------------------------------------------------------------------------------
   1- 11  A11    ---      IRAS       IRAS FSC2 name FHHMM.M+DDMM
  13- 14  I2     h        RAh        IRAS FSC2 B1950 right ascension
  16- 17  I2     min      RAm        IRAS FSC2 B1950 right ascension
  19- 22  F4.1   s        RAs        IRAS FSC2 B1950 right ascension
      24  A1     ---      DE-        IRAS FSC2 B1950 Declination sign
  25- 26  I2     deg      DEd        IRAS FSC2 B1950 Declination
  28- 29  I2     arcmin   DEm        IRAS FSC2 B1950 Declination
  31- 32  I2     arcsec   DEs        IRAS FSC2 B1950 Declination
      34  A1     ---    l_S12        [< ] S12 upper limit symbol
  35- 41  F7.3   Jy       S12        IRAS 12 um flux density (1)
      43  A1     ---    l_S25        [< ] S25 upper limit symbol
  44- 50  F7.3   Jy       S25        IRAS 25 um flux density (1)
      52  A1     ---    l_S60        [< ] S60 upper limit symbol
  53- 60  F8.3   Jy       S60        IRAS 60 um flux density (1)
      62  A1     ---    l_S100       [< ] S100 upper limit symbol
  63- 70  F8.3   Jy       S100       IRAS 100 um flux density (1)
  72- 73  I2     h        RA2h       ? Radio B1950 right ascension
  75- 76  I2     min      RA2m       ? Radio B1950 right ascension
  78- 81  F4.1   s        RA2s       ? Radio B1950 right ascension
      83  A1     ---      DE2-        Radio B1950 Declination (sign)
  84- 85  I2     deg      DE2d       ? Radio B1950 Declination
  87- 88  I2     arcmin   DE2m       ? Radio B1950 Declination
  90- 91  I2     arcsec   DE2s       ? Radio B1950 Declination
  92- 97  I6     mJy      S4.85GHz   ? 4.85 GHz flux density (2)
--------------------------------------------------------------------------------
"""
def _read_iras_row_radio(row: str) -> tuple:
    name = row[:11].split(' ')[0]
    # ra in hours
    ra = float(row[12:14]) + \
         float(row[15:17])/60 + \
         float(row[18:22])/3600
    # dec in deg
    dec = float(row[24:26]) + \
          float(row[27:29])/60 + \
          float(row[30:32])/3600
    # check the sign
    if row[23] == '-':
        dec *= -1

    # flux density upper limit
    s12  = float(row[34:41]) #: 12 um
    s25  = float(row[43:50]) #: 25 um
    s60  = float(row[52:60]) #: 60 um
    s100 = float(row[62:70]) #: 100 um
    # 04.85 GHz flux density
    s_radio = int(row[91:97]) if '\n' not in row[91:97] else 0
    return name, ra, dec, \
           s12, s25, s60, s100, \
           s_radio

IRAS_RADIO = ['name','ra','dec','s_12','s_25','s_60','s_100','s_radio']
TYPES_RADIO = [str,float,float,float,float,float,float,int]

from pandas import DataFrame    
def read_iras_data(data_file: str,*, store_data: bool = True, selection: Literal['default','radio'] = 'default') -> DataFrame:
    """Read IRAS catalog

    Parameters
    ----------
    data_file : str
        catalog file path
    store_data : bool, optional
        if `True` data will be saved in pickle format, by default True
    selection : Literal['default','radio'], optional
        catalog type, by default 'default'

    Returns
    -------
    data_frame : DataFrame
        catalog dataframe

    Raises
    ------
    ValueError
        `.dat` or `.pkl` only are allowed
    """
    ext = data_file.split('.')[-1]      #: file extention
    # read from `.dat` file
    if ext == 'dat':
        # read the catalog file
        with open(data_file,'r') as file:
            rows = file.readlines()
        # initialize a list for the columns
        columns = []
        # extract the catalog data
        if selection == 'default':
                columns[:] = IRAS_DEFAULT[:]
                rows = [_read_iras_row(row) for row in rows]
                dtypes = {col: typ for col, typ in zip(IRAS_DEFAULT,TYPES_DEFAULT)}
        elif selection == 'radio':
                columns[:] = IRAS_RADIO[:]
                rows = [_read_iras_row_radio(row) for row in rows]
                dtypes = {col: typ for col, typ in zip(IRAS_RADIO,TYPES_RADIO)}
        data_frame = DataFrame(data=rows,columns=columns).astype(dtype=dtypes)
        # store data in pickle mode
        if store_data:
            storing_dir = FileVar(filename=data_file,path=True).dir
            storing_file = storing_dir.split()[-1].lower()
            data_frame.to_pickle(storing_dir.join(storing_file+'.pkl'))
    # read from `.pkl` file
    elif ext == 'pkl':
        data_frame = read_pickle(data_file)
    else: 
        raise ValueError(f'.{ext} is not allowed')
    return data_frame
import os
from pandas import read_csv
import astropy.units as u


class PathVar():
    """Handle the paths

    Attributes
    ----------
    PATH : str
        directory path
    """
    @staticmethod
    def this_dir() -> str:
        """Compute the current directory

        Returns
        -------
        dir_path : str
            current directory path
        """
        return os.path.dirname(os.path.realpath(__file__))

    def __init__(self, path: str = '') -> None:
        """Construct the path variable

        Parameters
        ----------
        path : str, optional
            path of the directory. If `path == ''` the current 
            directory path is set, by default ''
        """
        self.PATH = path  if path != '' else PathVar.this_dir()

    def check_dir(self) -> bool:
        """Check the presence of directories in the path
        
        Returns
        -------
        dir_presence : bool
            `True` if the path is real
        """
        return os.path.isdir(self.PATH)

    def make_dir(self) -> None:
        """Make the directories required in the path"""
        if not self.check_dir():
            dirs = []
            tmp_path = self.PATH
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
        return os.path.split(self.PATH)

    def copy(self) -> 'PathVar':
        """Copy the class variable"""
        return PathVar(path=self.PATH)

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
        new_path = os.path.join(self.PATH,path)
        return PathVar(path=new_path)
     
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
        new_path = self.PATH
        for _ in range(iter):
            new_path = os.path.split(new_path)[0]
        return PathVar(path=new_path)
    
    def __str__(self) -> str:
        return self.PATH 
    
    def __repr__(self) -> str:
        return 'PathVar: "' + self.PATH + '"'
    

class FileVar():
    """Handle the file(s) path(s)

    Attributes
    ----------
    DIR : PathVar
        directory path
    FILE : str | list[str]
        file name or list of files names
        
    """

    def __init__(self, filename: str | list[str], dirpath: str | PathVar = '', path: bool = False) -> None:
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
        
        self.DIR  = dirpath.copy()
        self.FILE = filename if isinstance(filename, str) else [*filename]

    def path(self) -> str | list[str]:
        """Compute the file(s) path(s)

        Returns
        -------
        path_str : str | list[str]
            the path or a list of paths
        """
        filename = self.FILE 
        dirname = self.DIR.copy()
        if isinstance(filename,str): 
            return (dirname + filename).PATH
        else:
            return [(dirname + name).PATH for name in filename]

    def copy(self) -> 'FileVar':
        """Copy the variable"""
        new_file = FileVar(filename=self.FILE,dirpath=self.DIR,path=False)
        return new_file

    def __add__(self, new_file: str | list[str]) -> 'FileVar':
        new_filevar = self.copy()
        filename = new_filevar.FILE 
        if isinstance(filename,str):
            new_filevar.FILE = [filename, new_file]
        else:
            new_filevar.FILE += [new_file] 
        return new_filevar

    def __getitem__(self,item: int) -> str:
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
        path = self.path()
        if isinstance(path,str): 
            return TypeError('Variable is not subscriptable')
        else: 
            return path[item]
    
    def __setitem__(self, key: int, item: str) -> None:
        """Modify a file name of a list of paths

        Parameters
        ----------
        key : int
            chosen file
        item : str
            new name
        """
        if isinstance(self.FILE,str): 
            raise TypeError('Variable is not subscriptable')
        else: 
            self.FILE[key] = item

    def __str__(self) -> str:
        return str(self.path())
    
    def __repr__(self) -> str:
        return 'FileVar: "' + self.__str__() + '"'



## Paths
PKG_DIR = PathVar()
PROJECT_DIR = PKG_DIR - 1
MBM40_DIR = PROJECT_DIR + 'MBM40'
DATA_FILE = FileVar(filename='data.csv', dirpath=MBM40_DIR)
CO_FILES, HI_FILES, IR_FILES = read_csv(DATA_FILE.path()).to_numpy().transpose()
CO_PATHS = FileVar(CO_FILES, MBM40_DIR + 'CO')
HI_PATHS = FileVar(HI_FILES, MBM40_DIR + 'HI')
IR_PATHS = FileVar(IR_FILES, MBM40_DIR + 'IR')

U_VEL = u.km / u.s
u.add_enabled_units(u.def_unit(['K (Tb)'], represents=u.K))

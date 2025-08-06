import os
from pandas import read_csv
import astropy.units as u


class PathVar():
    
    @staticmethod
    def this_dir() -> str:
        return os.path.dirname(os.path.realpath(__file__))

    def __init__(self, path: str = '') -> None:
        self.PATH = path  if path != '' else PathVar.this_dir()

    def split(self) -> tuple[str,str]:
        return os.path.split(self.PATH)

    def copy(self) -> 'PathVar':
        return PathVar(self.PATH)

    def __add__(self, path: str) -> 'PathVar':
        new_path = os.path.join(self.PATH,path)
        return PathVar(path=new_path)
     
    def __sub__(self, iter: int) -> 'PathVar':
        new_path = self.PATH
        for _ in range(iter):
            new_path = os.path.split(new_path)[0]
        return PathVar(path=new_path)
    
    def __str__(self) -> str:
        return self.PATH 
     

class FileVar(PathVar):

    def __init__(self, filename: str | list[str], dirpath: str | PathVar = '', path: bool = False) -> None:
        if path: 
            dirpath  = PathVar(path = os.path.dirname(filename))
            filename = os.path.split(filename)[-1]
        self.PATH = dirpath if isinstance(dirpath, PathVar) else PathVar(path = dirpath)
        self.FILE = filename

    def path(self) -> str | list[str]:
        filename = self.FILE 
        dirname = self.PATH.copy()
        if isinstance(filename,str): 
            return (dirname + filename).PATH
        else:
            return [(dirname + name).PATH for name in filename]

    def copy(self) -> 'FileVar':
        new_file = FileVar(filename=self.FILE,dirpath=self.PATH,path=False)
        return new_file

    def __getitem__(self,item: int) -> str:
        path = self.path()
        if isinstance(path,str): return TypeError('Variable is not subscriptable')
        else: return path[item]
    
    def __setitem__(self,key:int,item:str) -> None:
        if isinstance(self.FILE,str): 
            return TypeError('Variable is not subscriptable')
        else: 
            self.FILE[key] = item

    def __str__(self) -> str:
        return str(self.path())


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

import os
from pandas import read_csv
import astropy.units as u
from .typing import *


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
        self._path = path  if path != '' else PathVar.this_dir()

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
    
    def tree(self) -> None:
        """List the files in the folder"""
        files = os.listdir(self.path)
        if len(files) != 0:
            print('Files in "' + self.path +'"')
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
        return PathVar(path=self.path)
    
    def clear(self, verbose: bool = True) -> None:
        """Delete all the files in the folder"""
        print('Remove all files in',self.path)
        from subprocess import call
        for f in self.files:
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
        new_path = os.path.join(self.path,path)
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
        new_path = self.path
        for _ in range(iter):
            new_path = os.path.split(new_path)[0]
        return PathVar(path=new_path)
    
    def __str__(self) -> str:
        return self.path 
    
    def __repr__(self) -> str:
        return 'PathVar: "' + self.path + '"'

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
            return (dirname + filename).path
        else:
            return [(dirname + name).path for name in filename]
        
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

    def __getitem__(self, item: int) -> str:
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
        if isinstance(self.file,str): 
            raise TypeError('Variable is not subscriptable')
        else: 
            self.file[key] = item

    def __str__(self) -> str:
        return str(self.path())
    
    def __repr__(self) -> str:
        return 'FileVar: "' + self.__str__() + '"'

    def __iter__(self) -> '_FileIterator':
        fpath = self.path()
        if isinstance(fpath,str): 
            raise TypeError('Required a list of files')
        else:
            return _FileIterator(self.path())

    def __contains__(self, element: str) -> bool:
        output = element in self.file
        if not output:
            output = element in self.path()            
        return output

# def dir_files(curr_dir: PathVar, dir: str = '', print_res: bool = False) -> FileVar:
#     """Collect all the file in a directory

#     Parameters
#     ----------
#     curr_dir : PathVar
#         the current directory
#     dir : str, optional
#         a subdirectory of `curr_dir`, by default `''`
#     print_res : bool, optional
#         if `True` the list is printed, by default `False`

#     Returns
#     -------
#     file_list : FileVar
#         the collection of all the file in the directory
#     """
#     obj_list = curr_dir.dir_list(dir=dir,print_res=print_res)
#     return FileVar(obj_list,curr_dir+dir)


## Useful paths for the library
PKG_DIR = PathVar()
PROJECT_DIR = PKG_DIR - 1
MBM40_DIR = PROJECT_DIR + 'MBM40'
DATA_FILE = FileVar(filename='data.csv', dirpath=MBM40_DIR)
CO_FILES, HI_FILES, IR_FILES = read_csv(DATA_FILE.path()).to_numpy().transpose()
CO_PATHS = FileVar(CO_FILES, MBM40_DIR + 'CO')
HI_PATHS = FileVar(HI_FILES, MBM40_DIR + 'HI')
IR_PATHS = FileVar(IR_FILES, MBM40_DIR + 'IR')

## Constants
U_VEL = u.km / u.s
u.add_enabled_units(u.def_unit(['K (Tb)'], represents=u.K))

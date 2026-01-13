import os
from pandas import DataFrame
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
    PATH : str
        directory path
    """

    def __init__(self, path: str = '', mkdir: bool = True) -> None:
        """Construct the path variable

        Parameters
        ----------
        path : str, optional
            path of the directory. If `path == ''` the current 
            directory path is set, by default ''
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
                print(f'{i} - {file}')
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

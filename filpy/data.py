import os

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

    def __init__(self, filename: str | list[str], dirpath: str | PathVar = '') -> None:
        self.DIR  = dirpath if isinstance(dirpath, PathVar) else PathVar(path = dirpath)
        self.FILE = filename

    def path(self) -> str | list[str]:
        filename = self.FILE 
        dirname = self.DIR.copy()
        if isinstance(filename,str): 
            return (dirname + filename).PATH
        else:
            return [(dirname + name).PATH for name in filename]

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


PKG_DIR = PathVar()
print(PKG_DIR)
PROJECT_DIR = PKG_DIR - 1
print(PROJECT_DIR)
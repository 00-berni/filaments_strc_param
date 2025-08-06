import numpy as np
from numpy.typing import ArrayLike, NDArray
import time
from functools import wraps
from .data import FileVar

class Log_File():

    RESET = "\033[0m"
    PREFIX = {"info"        : "INFO: ",
              "warn"       : "WARNING: ",
              "debug"      : "DEBUG: ",
              "process"    : "     | ",
              "error"      : "ERROR:",
              "critical"   : "CRITICAL: ",
              "log_c"      : "\033[96m",
              "warn_c"     : "\033[93m",
              "error_c"    : "\033[91;1m",
              "critical_c" : "\033[41;1m",
              "debug_c"    : "\033[92m",
              "process_c"  : "\033[92m"
              }
    SEP = "- "*33 +'-' + "\n"*2

    def __init__(self, file_name: FileVar, log_name: FileVar | None = None, mode: str = "w", prefix: dict[str,str] = PREFIX, print_cond: bool = False) -> None:
        if log_name is None:
            from datetime import datetime
            log_name = ''.join(file_name.FILE.split('.')[:-1]+['.log'])
            log_name = FileVar(filename=log_name,dirpath=file_name.PATH)
            with open(log_name.path(), mode) as log:
                log.write("."*29+"LOG FILE"+"."*30+"\n")
                log.write(f"\nDATE : {datetime.now()}")
                log.write(f"\nFILE : '{file_name.FILE}'")
                log.write(f"\nPATH : '{file_name.PATH}'")
                log.write("\n\n"+Log_File.SEP)
                log.write(prefix['info']+"Run the code as main")
                if print_cond: print(prefix['log_c']+prefix['info']+Log_File.RESET+"Run the code as main")
        self.file       = file_name.copy()
        self.file_log   = log_name.copy()
        self.prefix     = prefix.copy()
        self.print_cond = print_cond 
    
    def write(self, log_msg: str, mode: str) -> None:
        with open(self.file_log.path(), mode) as log:
            log.write(log_msg)

    def debug(self,msg: str) -> None:
        log_msg = '\n'+ self.prefix['debug'] + msg
        self.write(log_msg=log_msg,mode='a')
        if self.print_cond: print(self.prefix['debug_c']+self.prefix['debug']+Log_File.RESET+msg)

    def info(self,msg: str) -> None:
        log_msg = '\n'+ self.prefix['info'] + msg
        self.write(log_msg=log_msg,mode='a')
        if self.print_cond: print(self.prefix['log_c']+self.prefix['info']+Log_File.RESET+msg)

    def set_prc(self, process: str) -> None:
        self.prefix['process'] = ' '*(len(process)-2)+'| '

    def log_prc(self,process_name: str) -> None:
        prc_prefix = self.prefix['process']
        log_msg = '\n' + prc_prefix + process_name
        self.write(log_msg=log_msg, mode='a')
        if self.print_cond: print(self.prefix['process_c']+prc_prefix+Log_File.RESET+process_name)
    
    def warn(self, warn_msg: str) -> None:
        log_msg = '\n' + self.prefix['warn'] + warn_msg
        self.write(log_msg=log_msg,mode='a')
        if self.print_cond: print(self.prefix['warn_c']+self.prefix['warn']+Log_File.RESET+warn_msg)

    def error(self, err_msg: str) -> None:
        log_msg = '\n' + self.prefix['warn'] + err_msg
        self.write(log_msg=log_msg,mode='a')
        if self.print_cond: print(self.prefix['error_c']+self.prefix['error']+Log_File.RESET+err_msg)

    def critical(self, crt_msg: str) -> None:
        log_msg = '\n' + self.prefix['warn'] + crt_msg
        self.write(log_msg=log_msg,mode='a')
        if self.print_cond: print(self.prefix['critical_c']+self.prefix['critical']+Log_File.RESET+crt_msg)

    def separe(self) -> None:
        log_msg = Log_File.SEP
        self.write(log_msg=log_msg,mode='a')
        if self.print_cond: print(self.prefix['log_c']+log_msg+Log_File.RESET)

    def copy(self) -> 'Log_File':
        new_log = Log_File(file_name=self.file,log_name=self.file_log,prefix=self.prefix)
        return new_log    

    def reset(self) -> None:
        self.prefix = Log_File.PREFIX.copy()

def log_path(file_path: FileVar) -> str:
    log_name = ''.join(file_path.FILE.split('.')[:-1]+['.log'])
    return file_path.PATH.__add__(log_name).PATH

def reorganize_index(idxes: tuple | NDArray, axis: int | None, shape: tuple) -> tuple:
    if axis is None:
        return np.unravel_index(idxes,shape)
    else:
        axes = [ np.arange(s) for s in shape]
        axes[axis] = idxes
        return tuple(axes) 

def find_argmax(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    obj = np.asarray(obj)
    maxpos = np.argmax(obj, axis=axis)
    if len(obj.shape) == 1:
        return maxpos
    else:
        return reorganize_index(maxpos,axis=axis,shape=obj.shape)
        

def find_max(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    obj = np.asarray(obj)
    return obj[find_argmax(obj,axis=axis)]

def find_argmin(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    obj = np.asarray(obj)
    minpos = np.argmin(obj,axis=axis)
    if len(obj.shape) == 1:
        return minpos
    else:
        return reorganize_index(minpos,axis=axis,shape=obj.shape)

def find_min(obj: ArrayLike, axis: ArrayLike | None = None) -> ArrayLike:
    obj = np.asarray(obj)
    return obj[find_argmin(obj,axis=axis)]

def timeit(func):
    """Source: https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk"""
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function `{func.__name__}`: {total_time:.4f} s')
        return result
    return timeit_wrapper

def distance(p1: tuple[int,int] | np.ndarray, p2: tuple[int,int] | np.ndarray) -> float | np.ndarray:
    """Compute the Euclidean distance between two projectionist

    Parameters
    ----------
    p1 : tuple[int,int] | np.ndarray
        point 1
    p2 : tuple[int,int] | np.ndarray
        point 2

    Returns
    -------
    distance : float | np.ndarray
        Euclidean distance
    """
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


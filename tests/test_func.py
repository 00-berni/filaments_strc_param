from .. import filpy
import numpy as np
import tracemalloc
import linecache
import logging
import psutil
from numpy.typing import ArrayLike

TEST_DIR = filpy.PROJECT_DIR + 'tests' 

__all__ = ['filpy',
           'tracemalloc',
           'logging',
           'ArrayLike', 
           'display_top', 
           'log_path',
           'ram_usage',
           'TEST_DIR'
           ]

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


def log_path(file_path: filpy.FileVar) -> str:
    """Compute the path of the log file

    Parameters
    ----------
    file_path : FileVar
        path of the current file

    Returns
    -------
    log_path : str
        log path
    """
    log_name = ''.join(file_path.FILE.split('.')[:-1]+['.log'])
    return file_path.DIR.__add__(log_name).PATH

def ram_usage() -> float:
    process = psutil.Process()
    usage = process.memory_info().rss
    return usage

def display_top(snapshot, key_type='lineno', limit=10, logger: logging.Logger = logging.getLogger(__name__)) -> None:
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    str_mem = "Top %s lines" % limit + '\n'
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        str_mem = str_mem + "#%s: %s:%s: %.4f MiB" % (index, frame.filename, frame.lineno, stat.size / 1024**2) + '\n'
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            str_mem = str_mem + '    %s\n' % line
    logger.debug(str_mem[:-1])

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        logger.debug("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    logger.debug("Total allocated size: %.4f MiB" % (total / 1024**2))

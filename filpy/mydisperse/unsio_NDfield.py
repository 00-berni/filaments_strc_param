#!/usr/bin/env python

# Jean-Charles Lambert (September 2019)
#
# Use UNSIO package to load data and save it in NDFIELD coordinates
# You must install python-unsio usinge command :
# pip install python-unsio


from __future__ import print_function

#import unsio
import unsio.input as uns_in
import numpy as np
from .NDfield import write_NDfield_coord
from argparse import Namespace

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# process function
def process(args: Namespace) -> None:
    """_summary_

    Parameters
    ----------
    args : Namespace
        arguments of parse in `commandLine()`. Arguments are:
            * `input`      : str  = input snapshot
            * `output`     : str  = out NDfiled file
            * `components` : list = list of components eg:gas
            * `verbose`    : bool = parameter
    """
    uns=uns_in.CUNS_IN(args.input,args.components,"all")

    # load snapshot
    if uns.nextFrame("") : 
        # return snasphot time
        ok,tsnap=uns.getData("time") 
        # read positions
        ok,pos=uns.getData(args.components,"pos") 
        if ok:
            # reshape pos to 2D array (3,nbody)
            pos=np.reshape(pos,(-1,3)).T 
            write_NDfield_coord(pos,args.output)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# commandLine, parse the command line
def commandLine() -> None:
    """Interactive command to save in .NDfield file"""
    import argparse
    # help
    parser = argparse.ArgumentParser(description="Build NDfield file for DISPERSE analysis",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # options
    parser.add_argument('input',      help='input snapshot (unsio compliant)', type=str)
    parser.add_argument('output',     help='out NDfiled file', type=str)
    parser.add_argument('components', help='list of components eg:gas')
    parser.add_argument('--verbose',  help='verbose mode',dest="verbose", action="store_true", default=False)
    # parse
    args = parser.parse_args()

    # start main funciton
    process(args)

# -----------------------------------------------------
# main program
if __name__ == '__main__':
  commandLine()

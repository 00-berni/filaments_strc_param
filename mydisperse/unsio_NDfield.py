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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# process function
def process(args):
    uns=uns_in.CUNS_IN(args.input,args.components,"all")

    if uns.nextFrame("") : # load snapshot
        ok,tsnap=uns.getData("time") # return snasphot time
        ok,pos=uns.getData(args.components,"pos") # read positions
        if ok:
            pos=np.reshape(pos,(-1,3)).T # reshape pos to 2D array (3,nbody)
            write_NDfield_coord(pos,args.output)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# commandLine, parse the command line
def commandLine():
    import argparse
     # help
    parser = argparse.ArgumentParser(description="Build NDfield file for DISPERSE analysis",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # options
    parser.add_argument('input', help='input snaphot (unsio compliant)', type=str)
    parser.add_argument('output', help='out NDfiled file', type=str)
    parser.add_argument('components', help='list of components eg:gas')
    parser.add_argument('--verbose',help='verbose mode',dest="verbose", action="store_true", default=False)
     # parse
    args = parser.parse_args()

    # start main funciton
    process(args)

# -----------------------------------------------------
# main program
if __name__ == '__main__':
  commandLine()

#!/usr/bin/env python

import os, sys, csv, argparse, string
import numpy as np

from   itertools        import ifilter, imap
from   numpy.lib.format import open_memmap

try:
    import ROOT
    # Map ROOT leaf types to numpy types
    typemap = {
        ROOT.TLeafB : np.int8,     # an 8 bit signed integer (Char_t)
        ROOT.TLeafS : np.int16,    # a 16 bit signed integer (Short_t)
        ROOT.TLeafI : np.int32,    # a 32 bit signed integer (Int_t)
        ROOT.TLeafF : np.float32,  # a 32 bit floating point (Float_t)
        ROOT.TLeafD : np.float64,  # a 64 bit floating point (Double_t)
        ROOT.TLeafL : np.int64,    # a 64 bit signed integer (Long64_t)
        ROOT.TLeafO : np.bool,     # [the letter o, not a zero] a boolean (Bool_t)
    }
    '''
        ROOT.TLeafC : np.string,   # a character string terminated by the 0 character
        ROOT.TLeafb : np.uint8,    # an 8 bit unsigned integer (UChar_t)
        ROOT.TLeafs : np.uint16,   # a 16 bit unsigned integer (UShort_t)
        ROOT.TLeafi : np.uint32,   # a 32 bit unsigned integer (UInt_t)
        ROOT.TLeafl : np.uint64,   # a 64 bit unsigned integer (ULong64_t)
    '''
    hasROOT = True
except:
    print "WARNING: Couldn't import ROOT; any input NTuples will be ignored." 
    hasROOT = False

#
def mkCache(base, name, shape, **kwargs):
    opath = os.path.join(base, "Data", name)
    try:
        os.makedirs(opath)
    except OSError:
        pass

    return { k : open_memmap(os.path.join(opath, k + ".npy"), dtype=t, mode='w+', shape=shape) for k, t in kwargs.items() }

### For CSV import.
def getReader(csvfile):
    csvFilt = ifilter(lambda x: x[0] != ';', csvfile)
    csvFilt = imap(string.strip, csvFilt)
    header  = [h.strip() for h in csvFilt.next().split(',') if h.strip() != '']
    reader  = csv.DictReader(csvFilt, fieldnames=header)

    return header, reader

def importCSV(base, name, fname, treeName):
    with open(fname, 'rb') as csvfile:
        head, read = getReader(csvfile)
        types      = { k : np.double for k in head }
        nrow       = 0
        n          = 0

        # Count lines and create mmap'ed arrays.
        for _ in read:
            if nrow % 10000 == 0:
                print "\r  Counting...           %d" % nrow,
                sys.stdout.flush()
            nrow += 1
        Sample = mkCache(base, name, (nrow,), **types)

        # Reset to the first row and parse entries.
        csvfile.seek(0); csvfile.next()
        for row in read:
            for k in head:
                Sample[k][n] = float(row[k])
            n += 1
            if n % 10000 == 0 or n == nrow:
                print "\r  Reading...  % 8d / % 8d" % (n, nrow),
                sys.stdout.flush()
        print

### For ROOT import.
def importROOT(base, name, fname, treeName):
    f      = ROOT.TFile.Open(fname)

    t      = getattr(f, treeName)
    bnames = [ b.GetName() for b in t.GetListOfBranches() ]
    types  = { b.GetName() : typemap[type(b.GetLeaf(b.GetName()))] for b in t.GetListOfBranches() }

    nEvt   = t.GetEntries()
    n      = 0

    Sample = mkCache(base, name, (nEvt,), **types)

    for n, event in enumerate(t):
        for bname in bnames:
            Sample[bname][n] = getattr(event, bname)
        n += 1
        if n % 10000 == 0 or n == nEvt:
            print "\r  Reading...  % 8d / % 8d" % (n, nEvt),
            sys.stdout.flush()
    f.Close()


###### Map file types to importers.
mapping = {
  'csv'    : importCSV,
  'root'   : importROOT if hasROOT else None,
  'root.1' : importROOT if hasROOT else None,
}



###### Okay, start it up.
ArgP      = argparse.ArgumentParser(description=' === Functional Decomposition Importer ===')
ArgP.add_argument('--base', type=str, default=".", help="FD base directory.")
ArgP.add_argument('--tree', type=str,              help="Name of tree to import (ROOT files only).")
ArgP.add_argument('files', default=[], nargs='*',  help="List of files to import.")

ArgC      = ArgP.parse_args()
ipath     = os.path.join(ArgC.base, "Input")

# Make list of input files
if len(ArgC.files) > 0:
    fpath = ArgC.files
    files = [ os.path.basename(x)    for x in fpath ]
else:
    files = os.listdir(ipath)
    fpath = [ os.path.join(ipath, x) for x in l ]

# And read them in
for fname in fpath:
    name, ext = os.path.basename(fname).split(os.extsep, 1)

    print name, ext

    try:
        func = mapping[ext]
    except KeyError:
        print "  WARNING: Skipping file with unrecognized extension."
        continue

    if func is not None:
        func(ArgC.base, name, fname, ArgC.tree)
    else:
        print "  WARNING: Skipping disabled filetype."


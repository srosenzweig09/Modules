## Open ROOT files using uproot without having to go through the entire rigamaroll each time (load file, get branches, convert to arrays)

import uproot
import numpy as np

def open_up(filename, tree_name='sixBtree', open_tree=True):
    """Opens a ROOT file using uproot and prints branch names.
    """
    
    if open_tree:
        tree = uproot.open(filename+':'+tree_name)
    else:
        file = uproot.open(filename)
        tree = file[tree_name]

    table = tree.arrays()
    nptab = tree.arrays(library='np')
    ncols = 3
    keys  = tree.keys()
    n     = len(keys)
    modu  = n%ncols
    
    print("-"*100)
    print(" "*44 + "TABLE COLUMNS" + " "*43)
    print("-"*100)
    
    for i in np.arange(0,n-modu,ncols):
        row = ""
        for j in np.arange(i,i+ncols):
            row += "{:<34}".format(keys[j])
        print(row)
    if modu != 0:
        row = ""
        for i in np.arange(n-modu, n):
            row += "{:<34}".format(keys[i])
        print(row)
    print("-"*100)

    return tree, table, nptab
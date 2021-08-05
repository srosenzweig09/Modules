## Open ROOT files using uproot without having to go through the entire rigamaroll each time (load file, get branches, convert to arrays)

import colorama
colorama.init(autoreset=True)
from colorama import Fore, Back
import uproot
import numpy as np
from logger import info

def unfiltered_tree(tree):

    ak_dict = tree.arrays()
    keys     = tree.keys()
    n        = len(keys)
    nevents  = len(table)
    np_dict = tree.arrays(library='np')

    ncols = 3
    modu  = n%ncols
    
    print("-"*100)
    print(" "*44 + "TABLE COLUMNS" + " "*43)
    print("-"*100)
    info(f"Tree contains {nevents} events.")
    
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

    return ak_dict, np_dict

def filtered_tree(tree, filter, dict_keys):

    ak_dicts = {}
    np_dicts = {}
    for f,k in zip(filter, dict_keys):
        ak_dicts[k] = tree.arrays(filter_name=f)
        np_dicts[k] = tree.arrays(filter_name=f, library='np')

    print(f"Passing ak and np dicts with keys {ak_dicts.keys()}.")

    return ak_dicts, np_dicts

def open_up(fileName, treeName='sixBtree', filter=None, dict_keys=None): #, open_tree=True):
    """
    Opens a ROOT file using uproot and prints branch names.

    :param fileName: ROOT file to open
    :param treeName: tree in ROOT file to open
    :return: uprootTTree, awkwardArrayTree, NumPyDict
    """

    info(f"Opening ROOT file {fileName} with columns")
    tree = uproot.open(fileName+':'+treeName)        

    if filter:
        assert dict_keys, print("ERROR: Please provide dict keys with filter arg.")
        ak_dict, np_dict = filtered_tree(tree, filter, dict_keys)
        return tree, ak_dict, np_dict
    else:
        ak_dict, np_dict = unfiltered_tree(tree)
        return tree, ak_dict, np_dict




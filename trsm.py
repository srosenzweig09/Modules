"""
Project: 6b Final States - TRSM
Author: Suzanne Rosenzweig

This class sorts MC-generated events into an array of input features for use in training a neural network and in evaluating the performance of the model using a test set of examples.

Notes:
Training samples are prepared such that these requirements are already imposed:
- n_jet > 6
- n_sixb == 6
"""

# from particle import Particle
import awkward as ak
import colorama
colorama.init(autoreset=True)
from colorama import Fore, Back
from icecream import ic
import itertools
from math import comb
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import numpy as np
import random
import sys  # JAKE DELETE
import uproot
import vector

from keras.models import model_from_json
from pickle import load

from logger import info
from myuproot import open_up
from tqdm import tqdm # progress bar

vector.register_awkward()

def get_6jet_p4(p4):
    combos = ak.combinations(p4, 6)
    part0, part1, part2, part3, part4, part5 = ak.unzip(combos)
    evt_p4 = part0 + part1 + part2 + part3 + part4 + part5
    boost_0 = part0.boost_p4(evt_p4)
    boost_1 = part1.boost_p4(evt_p4)
    boost_2 = part2.boost_p4(evt_p4)
    boost_3 = part3.boost_p4(evt_p4)
    boost_4 = part4.boost_p4(evt_p4)
    boost_5 = part5.boost_p4(evt_p4)
    return evt_p4, [boost_0, boost_1, boost_2, boost_3, boost_4, boost_5]

class TRSM():

    # b_mass = Particle.from_pdgid(5).mass / 1000 # Convert from MeV to GeV

    def open_file(self, fileName, treeName='sixBtree'):

        tree = uproot.open(fileName + ":" + treeName)

        self.tree = tree
        
        self.jet_idx  = tree['jet_idx'].array(library='np')

        self.n_jet    = tree['n_jet'].array(library='np')
        self.n_sixb   = tree['n_sixb'].array(library='np')
        self.jet_pt   = tree['jet_pt'].array(library='np')
        self.jet_eta  = tree['jet_eta'].array(library='np')
        self.jet_phi  = tree['jet_phi'].array(library='np')
        self.jet_m    = tree['jet_m'].array(library='np')
        self.jet_btag = tree['jet_btag'].array(library='np')
        self.jet_qgl  = tree['jet_qgl'].array(library='np')
        self.jet_partonFlav = tree['jet_partonFlav'].array(library='np')
        self.jet_hadronFlav = tree['jet_hadronFlav'].array(library='np')

        self.HX_b1_pt  = tree['gen_HX_b1_recojet_ptRegressed'].array(library='np')
        self.HX_b2_pt  = tree['gen_HX_b2_recojet_ptRegressed'].array(library='np')
        self.HX_b1_eta = tree['gen_HX_b1_recojet_eta'].array(library='np')
        self.HX_b2_eta = tree['gen_HX_b2_recojet_eta'].array(library='np')
        self.HX_b1_phi = tree['gen_HX_b1_recojet_phi'].array(library='np')
        self.HX_b2_phi = tree['gen_HX_b2_recojet_phi'].array(library='np')
        self.HX_b1_m   = tree['gen_HX_b1_recojet_m'].array(library='np')
        self.HX_b2_m   = tree['gen_HX_b2_recojet_m'].array(library='np')
        self.H1_b1_pt  = tree['gen_HY1_b1_recojet_ptRegressed'].array(library='np')
        self.H1_b2_pt  = tree['gen_HY1_b2_recojet_ptRegressed'].array(library='np')
        self.H1_b1_eta = tree['gen_HY1_b1_recojet_eta'].array(library='np')
        self.H1_b2_eta = tree['gen_HY1_b2_recojet_eta'].array(library='np')
        self.H1_b1_phi = tree['gen_HY1_b1_recojet_phi'].array(library='np')
        self.H1_b2_phi = tree['gen_HY1_b2_recojet_phi'].array(library='np')
        self.H1_b1_m   = tree['gen_HY1_b1_recojet_m'].array(library='np')
        self.H1_b2_m   = tree['gen_HY1_b2_recojet_m'].array(library='np')
        self.H2_b1_pt  = tree['gen_HY2_b1_recojet_ptRegressed'].array(library='np')
        self.H2_b2_pt  = tree['gen_HY2_b2_recojet_ptRegressed'].array(library='np')
        self.H2_b1_eta = tree['gen_HY2_b1_recojet_eta'].array(library='np')
        self.H2_b2_eta = tree['gen_HY2_b2_recojet_eta'].array(library='np')
        self.H2_b1_phi = tree['gen_HY2_b1_recojet_phi'].array(library='np')
        self.H2_b2_phi = tree['gen_HY2_b2_recojet_phi'].array(library='np')
        self.H2_b1_m   = tree['gen_HY2_b1_recojet_m'].array(library='np')
        self.H2_b2_m   = tree['gen_HY2_b2_recojet_m'].array(library='np')

        self.nevents = len(self.jet_pt)

    def open_files(self, fileList, treeName='sixBtree'):

        file1 = fileList[0]
        tree = uproot.open(file1 + ":" + treeName)

        self.jet_idx  = tree['jet_signalId'].array(library='np')

        self.n_jet    = tree['n_jet'].array(library='np')
        self.n_sixb   = tree['n_sixb'].array(library='np')
        self.jet_pt   = tree['jet_pt'].array(library='np')
        self.jet_eta  = tree['jet_eta'].array(library='np')
        self.jet_phi  = tree['jet_phi'].array(library='np')
        self.jet_m    = tree['jet_m'].array(library='np')
        self.jet_btag = tree['jet_btag'].array(library='np')
        self.jet_qgl  = tree['jet_qgl'].array(library='np')
        self.jet_partonFlav = tree['jet_partonFlav'].array(library='np')
        self.jet_hadronFlav = tree['jet_hadronFlav'].array(library='np')

        self.HX_b1_pt  = tree['gen_HX_b1_recojet_ptRegressed'].array(library='np')
        self.HX_b2_pt  = tree['gen_HX_b2_recojet_ptRegressed'].array(library='np')
        self.HX_b1_eta = tree['gen_HX_b1_recojet_eta'].array(library='np')
        self.HX_b2_eta = tree['gen_HX_b2_recojet_eta'].array(library='np')
        self.HX_b1_phi = tree['gen_HX_b1_recojet_phi'].array(library='np')
        self.HX_b2_phi = tree['gen_HX_b2_recojet_phi'].array(library='np')
        self.HX_b1_m   = tree['gen_HX_b1_recojet_m'].array(library='np')
        self.HX_b2_m   = tree['gen_HX_b2_recojet_m'].array(library='np')
        self.H1_b1_pt  = tree['gen_HY1_b1_recojet_ptRegressed'].array(library='np')
        self.H1_b2_pt  = tree['gen_HY1_b2_recojet_ptRegressed'].array(library='np')
        self.H1_b1_eta = tree['gen_HY1_b1_recojet_eta'].array(library='np')
        self.H1_b2_eta = tree['gen_HY1_b2_recojet_eta'].array(library='np')
        self.H1_b1_phi = tree['gen_HY1_b1_recojet_phi'].array(library='np')
        self.H1_b2_phi = tree['gen_HY1_b2_recojet_phi'].array(library='np')
        self.H1_b1_m   = tree['gen_HY1_b1_recojet_m'].array(library='np')
        self.H1_b2_m   = tree['gen_HY1_b2_recojet_m'].array(library='np')
        self.H2_b1_pt  = tree['gen_HY2_b1_recojet_ptRegressed'].array(library='np')
        self.H2_b2_pt  = tree['gen_HY2_b2_recojet_ptRegressed'].array(library='np')
        self.H2_b1_eta = tree['gen_HY2_b1_recojet_eta'].array(library='np')
        self.H2_b2_eta = tree['gen_HY2_b2_recojet_eta'].array(library='np')
        self.H2_b1_phi = tree['gen_HY2_b1_recojet_phi'].array(library='np')
        self.H2_b2_phi = tree['gen_HY2_b2_recojet_phi'].array(library='np')
        self.H2_b1_m   = tree['gen_HY2_b1_recojet_m'].array(library='np')
        self.H2_b2_m   = tree['gen_HY2_b2_recojet_m'].array(library='np')

        for files in fileList[1:]:
            tree = uproot.open(files + ":" + treeName)

            self.jet_idx  = np.append(self.jet_idx, tree['jet_signalId'].array(library='np'))

            # self.n_sixb   = np.append(self.n_sixb, tree['nfound_sixb'].array(library='np'))
            # self.n_jet    = np.append(self.n_sixb, tree['n_jet'].array(library='np'))
            self.jet_pt   = np.append(self.jet_pt, tree['jet_pt'].array(library='np'))
            self.jet_eta  = np.append(self.jet_eta, tree['jet_eta'].array(library='np'))
            self.jet_phi  = np.append(self.jet_phi, tree['jet_phi'].array(library='np'))
            self.jet_m    = np.append(self.jet_m, tree['jet_m'].array(library='np'))
            self.jet_btag = np.append(self.jet_btag, tree['jet_btag'].array(library='np'))
            self.jet_qgl  = np.append(self.jet_qgl, tree['jet_qgl'].array(library='np'))
            self.jet_partonFlav = np.append(self.jet_partonFlav, tree['jet_partonFlav'].array(library='np'))
            self.jet_hadronFlav = np.append(self.jet_hadronFlav, tree['jet_hadronFlav'].array(library='np'))

            self.HX_b1_pt  = np.append(self.HX_b1_pt, tree['gen_HX_b1_recojet_ptRegressed'].array(library='np'))
            self.HX_b2_pt  = np.append(self.HX_b2_pt, tree['gen_HX_b2_recojet_ptRegressed'].array(library='np'))
            self.HX_b1_eta = np.append(self.HX_b1_eta, tree['gen_HX_b1_recojet_eta'].array(library='np'))
            self.HX_b2_eta = np.append(self.HX_b2_eta, tree['gen_HX_b2_recojet_eta'].array(library='np'))
            self.HX_b1_phi = np.append(self.HX_b1_phi, tree['gen_HX_b1_recojet_phi'].array(library='np'))
            self.HX_b2_phi = np.append(self.HX_b2_phi, tree['gen_HX_b2_recojet_phi'].array(library='np'))
            self.HX_b1_m   = np.append(self.HX_b1_m, tree['gen_HX_b1_recojet_m'].array(library='np'))
            self.HX_b2_m   = np.append(self.HX_b2_m, tree['gen_HX_b2_recojet_m'].array(library='np'))
            self.H1_b1_pt  = np.append(self.H1_b1_pt, tree['gen_HY1_b1_recojet_ptRegressed'].array(library='np'))
            self.H1_b2_pt  = np.append(self.H1_b2_pt, tree['gen_HY1_b2_recojet_ptRegressed'].array(library='np'))
            self.H1_b1_eta = np.append(self.H1_b1_eta, tree['gen_HY1_b1_recojet_eta'].array(library='np'))
            self.H1_b2_eta = np.append(self.H1_b2_eta, tree['gen_HY1_b2_recojet_eta'].array(library='np'))
            self.H1_b1_phi = np.append(self.H1_b1_phi, tree['gen_HY1_b1_recojet_phi'].array(library='np'))
            self.H1_b2_phi = np.append(self.H1_b2_phi, tree['gen_HY1_b2_recojet_phi'].array(library='np'))
            self.H1_b1_m   = np.append(self.H1_b1_m, tree['gen_HY1_b1_recojet_m'].array(library='np'))
            self.H1_b2_m   = np.append(self.H1_b2_m, tree['gen_HY1_b2_recojet_m'].array(library='np'))
            self.H2_b1_pt  = np.append(self.H2_b1_pt, tree['gen_HY2_b1_recojet_ptRegressed'].array(library='np'))
            self.H2_b2_pt  = np.append(self.H2_b2_pt, tree['gen_HY2_b2_recojet_ptRegressed'].array(library='np'))
            self.H2_b1_eta = np.append(self.H2_b1_eta, tree['gen_HY2_b1_recojet_eta'].array(library='np'))
            self.H2_b2_eta = np.append(self.H2_b2_eta, tree['gen_HY2_b2_recojet_eta'].array(library='np'))
            self.H2_b1_phi = np.append(self.H2_b1_phi, tree['gen_HY2_b1_recojet_phi'].array(library='np'))
            self.H2_b2_phi = np.append(self.H2_b2_phi, tree['gen_HY2_b2_recojet_phi'].array(library='np'))
            self.H2_b1_m   = np.append(self.H2_b1_m, tree['gen_HY2_b1_recojet_m'].array(library='np'))
            self.H2_b2_m   = np.append(self.H2_b2_m, tree['gen_HY2_b2_recojet_m'].array(library='np'))

        self.nevents = len(self.H2_b2_m)

    ## Open ROOT file
    def __init__(self, filename, evan_preselections=True):

        if type(filename) == str:
            self.open_file(filename)
        elif type(filename) == list:
            self.open_files(filename)
        else:
            raise

        self.HX_b1_p4 = ak.Array({"pt":self.HX_b1_pt, "eta":self.HX_b1_eta, "phi":self.HX_b1_phi, "m":self.HX_b1_m}, with_name="Momentum4D")
        self.HX_b2_p4 = ak.Array({"pt":self.HX_b2_pt, "eta":self.HX_b2_eta, "phi":self.HX_b2_phi, "m":self.HX_b2_m}, with_name="Momentum4D")
        self.H1_b1_p4 = ak.Array({"pt":self.H1_b1_pt, "eta":self.H1_b1_eta, "phi":self.H1_b1_phi, "m":self.H1_b1_m}, with_name="Momentum4D")
        self.H1_b2_p4 = ak.Array({"pt":self.H1_b2_pt, "eta":self.H1_b2_eta, "phi":self.H1_b2_phi, "m":self.H1_b2_m}, with_name="Momentum4D")
        self.H2_b1_p4 = ak.Array({"pt":self.H2_b1_pt, "eta":self.H2_b1_eta, "phi":self.H2_b1_phi, "m":self.H2_b1_m}, with_name="Momentum4D")
        self.H2_b2_p4 = ak.Array({"pt":self.H2_b2_pt, "eta":self.H2_b2_eta, "phi":self.H2_b2_phi, "m":self.H2_b2_m}, with_name="Momentum4D")

        self.HX_p4 = self.HX_b1_p4 + self.HX_b2_p4
        self.H1_p4 = self.H1_b1_p4 + self.H1_b2_p4
        self.H2_p4 = self.H2_b1_p4 + self.H2_b2_p4
    
    def construct_6j_features(self, combo_p4, combo_btag, boosted):

        inputs = []

        combo_pt = np.asarray(combo_p4.pt)
        combo_eta = np.asarray(combo_p4.eta)
        combo_phi = np.asarray(combo_p4.phi)
        boosted_pt = []

        for boost in boosted:
            boosted_pt.append(boost.pt)
        boosted_pt = np.asarray(boosted_pt)

        inputs = np.column_stack((combo_pt, combo_eta, combo_phi, combo_btag, boosted_pt[0,:], boosted_pt[1,:], boosted_pt[2,:], boosted_pt[3,:], boosted_pt[4,:], boosted_pt[5,:]))
        # inputs = np.column_stack((combo_pt, combo_eta, combo_phi, combo_btag))

        return inputs

class training_6j(TRSM):
    ## Build p4 for all events
    @jit(forceobj=True)
    def __init__(self, trsm):

        signal_builder = ak.ArrayBuilder()
        bkgd_builder = ak.ArrayBuilder()
        
        n_jet          = trsm.n_jet
        n_sixb         = trsm.n_sixb
        jet_idx        = trsm.jet_idx
        jet_pt         = trsm.jet_pt
        jet_eta        = trsm.jet_eta
        jet_phi        = trsm.jet_phi
        jet_m          = trsm.jet_m
        jet_btag       = trsm.jet_btag
        jet_qgl        = trsm.jet_qgl
        jet_partonFlav = trsm.jet_partonFlav
        jet_hadronFlav = trsm.jet_hadronFlav
        nevents   = trsm.nevents

        signal_btag = []
        background_btag = []

        print("Looping through events. This may take a few minutes.")
        for evt in tqdm(range(nevents)):

            # signal and background masks
            signal_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj > -1]
            signal_mask = np.array((signal_mask))

            background_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj == -1]
            background_mask = np.array((background_mask))
            # Cound the number of background jets
            n_bkgd = len(jet_pt[evt][background_mask])

            ## Not sure if this is necessary but keeping just in case
            # Skip any events with duplicate matches (for now)
            if len(np.unique(jet_idx[evt][signal_mask])) < len(jet_idx[evt][signal_mask]): 
                combo_mask.append(False)
                continue
            
            # Choose random signal jets to swap out for background
            if n_bkgd < 6:
                n_sgnl = 6 - n_bkgd
                sgnl_jet_mask = np.random.choice(signal_mask, size=n_sgnl, replace=False)
                incorrect_mask = np.concatenate((sgnl_jet_mask, background_mask))
            else:
                incorrect_mask = np.random.choice(background_mask, size=6, replace=False)
            
            sixb_pt = jet_pt[evt][signal_mask]
            non_sixb_pt = jet_pt[evt][incorrect_mask]
            
            sixb_eta = jet_eta[evt][signal_mask]
            non_sixb_eta = jet_eta[evt][incorrect_mask]
            
            sixb_phi = jet_phi[evt][signal_mask]
            non_sixb_phi = jet_phi[evt][incorrect_mask]

            sixb_m = jet_m[evt][signal_mask]
            non_sixb_m = jet_m[evt][incorrect_mask]

            sixb_btag = jet_btag[evt][signal_mask]
            non_sixb_btag = jet_btag[evt][incorrect_mask]

            # sort events by pT (perhaps later change to sort by btag?)
            sixb_idx  = signal_mask[np.argsort(sixb_pt)][::-1]
            sixb_eta  = sixb_eta[np.argsort(sixb_pt)][::-1]
            sixb_phi  = sixb_phi[np.argsort(sixb_pt)][::-1]
            sixb_m    = sixb_m[np.argsort(sixb_pt)][::-1]
            sixb_btag = sixb_btag[np.argsort(sixb_pt)][::-1]
            signal_btag.append(sixb_btag)
            # pt must be sorted last because it is used to sort everything else
            sixb_pt   = np.sort(sixb_pt)[::-1]
            # signal_idx.append(sixb_idx)
            
            non_sixb_eta  = non_sixb_eta[np.argsort(non_sixb_pt)][::-1]
            non_sixb_phi  = non_sixb_phi[np.argsort(non_sixb_pt)][::-1]
            non_sixb_m    = non_sixb_m[np.argsort(non_sixb_pt)][::-1]
            non_sixb_btag = non_sixb_btag[np.argsort(non_sixb_pt)][::-1]
            background_btag.append(non_sixb_btag)
            # pt must be sorted last because it is used to sort everything else
            non_sixb_pt   = np.sort(non_sixb_pt)[::-1]

            signal_builder.begin_list()
            for pt, eta, phi, m in zip(sixb_pt, sixb_eta, sixb_phi, sixb_m):
                signal_builder.begin_record("Momentum4D")

                signal_builder.field("pt").real(pt)
                signal_builder.field("eta").real(eta)
                signal_builder.field("phi").real(phi)
                signal_builder.field("m").real(m)
                
                signal_builder.end_record()
            signal_builder.end_list()

            bkgd_builder.begin_list()
            for pt, eta, phi,m  in zip(non_sixb_pt, non_sixb_eta, non_sixb_phi, non_sixb_m):
                bkgd_builder.begin_record("Momentum4D")

                bkgd_builder.field("pt").real(pt)
                bkgd_builder.field("eta").real(eta)
                bkgd_builder.field("phi").real(phi)
                bkgd_builder.field("m").real(m)

                bkgd_builder.end_record()
            bkgd_builder.end_list()
            # combo_mask.append(True)

        self.sgnl_p4 = signal_builder.snapshot()
        self.bkgd_p4 = bkgd_builder.snapshot()

        self.sgnl_evt_p4, self.sgnl_boosted = get_6jet_p4(self.sgnl_p4)
        self.bkgd_evt_p4, self.bkgd_boosted = get_6jet_p4(self.bkgd_p4)

        # self.signal_idx = np.array((signal_idx))
        self.signal_btag = np.array((signal_btag))
        self.bkgd_btag = np.array((background_btag))

        # self.combo_mask = np.asarray(combo_mask)
        sgnl_inputs = self.construct_6j_features(self.sgnl_p4, self.signal_btag, self.sgnl_boosted)
        bkgd_inputs = self.construct_6j_features(self.bkgd_p4, self.bkgd_btag, self.bkgd_boosted)

        self.inputs = np.row_stack((sgnl_inputs, bkgd_inputs))
        print("Combined (sgnl + bkgd) input shape =",self.inputs.shape)

        sgnl_node_targets = np.concatenate((np.repeat(1, len(self.inputs)/2), np.repeat(0, len(self.inputs)/2)))
        bkgd_node_targets = np.where(sgnl_node_targets == 1, 0, 1)
        self.targets = np.column_stack((sgnl_node_targets, bkgd_node_targets))

class training_2j(TRSM):
    @jit(forceobj=True)
    def generate_incorrect_pairings(self, evt_mask, H_mask, H1_mask, H2_mask):

        incorrect_masks = []

        while len(incorrect_masks) < 3:
            mask = np.random.choice(evt_mask, size=2, replace=False)
            if np.all(np.isin(mask, H_mask)):
                continue
            if np.all(np.isin(mask, H1_mask)):
                continue
            if np.all(np.isin(mask, H2_mask)):
                continue
            incorrect_masks.append(mask)

        return incorrect_masks


    @jit(forceobj=True)
    def __init__(self, trsm):

        jet1 = ak.ArrayBuilder()
        jet2 = ak.ArrayBuilder()
        
        n_jet          = trsm.n_jet
        n_sixb         = trsm.n_sixb
        jet_idx        = trsm.jet_idx
        jet_pt         = trsm.jet_pt
        jet_eta        = trsm.jet_eta
        jet_phi        = trsm.jet_phi
        jet_m          = trsm.jet_m
        jet_btag       = trsm.jet_btag
        jet_qgl        = trsm.jet_qgl
        jet_partonFlav = trsm.jet_partonFlav
        jet_hadronFlav = trsm.jet_hadronFlav
        nevents   = trsm.nevents

        jet1_btag = []
        jet1_idx = []
        jet2_btag = []
        jet2_idx = []

        print("Looping through events. This may take a few minutes.")
        for evt in tqdm(range(trsm.nevents)):

            evt_idx = jet_idx[evt]
            evt_btag = jet_btag[evt]
            evt_pt = jet_pt[evt]
            evt_eta = jet_eta[evt]
            evt_phi = jet_phi[evt]
            evt_m = jet_m[evt]

            signal_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj > -1]
            signal_mask = np.array((signal_mask))

            H_mask = signal_mask[evt_idx[signal_mask] < 2]
            H2_mask = signal_mask[evt_idx[signal_mask] > 3]
            H1_mask = signal_mask[np.logical_or(evt_idx[signal_mask] == 2, evt_idx[signal_mask] == 3)]

            H_mask = H_mask[np.argsort(jet_pt[evt][H_mask])][::-1]
            H1_mask = H1_mask[np.argsort(jet_pt[evt][H1_mask])][::-1]
            H2_mask = H2_mask[np.argsort(jet_pt[evt][H2_mask])][::-1]

            background_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj == -1]
            background_mask = np.array((background_mask))

            ordered_evt_mask = np.concatenate((H_mask, H1_mask, H2_mask, background_mask))

            incorrect_pairings_mask = self.generate_incorrect_pairings(ordered_evt_mask, H_mask, H1_mask, H2_mask)
            pair1_mask = incorrect_pairings_mask[0][np.argsort(jet_pt[evt][incorrect_pairings_mask[0]])][::-1]
            pair2_mask = incorrect_pairings_mask[1][np.argsort(jet_pt[evt][incorrect_pairings_mask[1]])][::-1]
            pair3_mask = incorrect_pairings_mask[2][np.argsort(jet_pt[evt][incorrect_pairings_mask[2]])][::-1]

            jet1_ind = np.array((H_mask[0], H1_mask[0], H2_mask[0], pair1_mask[0], pair2_mask[0], pair3_mask[0]))
            jet2_ind = np.array((H_mask[1], H1_mask[1], H2_mask[1], pair1_mask[1], pair2_mask[1], pair3_mask[1]))

            for ind1, ind2 in zip(jet1_ind, jet2_ind):

                jet1_btag.append(evt_btag[ind1])
                jet1_idx.append(evt_idx[ind1])
                jet2_btag.append(evt_btag[ind2])
                jet2_idx.append(evt_idx[ind2])

                jet1.begin_list()
                jet1.begin_record("Momentum4D")
                jet1.field("pt"); jet1.real(evt_pt[ind1])
                jet1.field("eta"); jet1.real(evt_eta[ind1])
                jet1.field("phi"); jet1.real(evt_phi[ind1])
                jet1.field("m"); jet1.real(evt_m[ind1])
                jet1.end_record()
                jet1.end_list()
            
                jet2.begin_list()
                jet2.begin_record("Momentum4D")
                jet2.field("pt"); jet2.real(evt_pt[ind2])
                jet2.field("eta"); jet2.real(evt_eta[ind2])
                jet2.field("phi"); jet2.real(evt_phi[ind2])
                jet2.field("m"); jet2.real(evt_m[ind2])
                jet2.end_record()
                jet2.end_list()
            
        jet1 = jet1.snapshot()
        jet2 = jet2.snapshot()
        
        jet_pair = jet1 + jet2

        jet1_btag = np.asarray(jet1_btag)
        jet1_idx = np.asarray(jet1_idx)
        jet2_btag = np.asarray(jet2_btag)
        jet2_idx = np.asarray(jet2_idx)

        node1 = np.tile(np.array((1,1,1,0,0,0)), trsm.nevents)
        node2 = np.tile(np.array((0,0,0,1,1,1)), trsm.nevents)

        targets = np.column_stack((node1, node2))

        self.pair_features = np.column_stack((np.asarray(jet1.pt), np.asarray(jet2.pt), np.asarray(jet1.eta), np.asarray(jet2.eta), np.asarray(jet1.phi), np.asarray(jet2.phi), jet1_btag, jet2_btag, np.asarray(jet1.deltaR(jet2))))
        self.pair_targets = targets
    
class combos_6j(TRSM):
    @jit
    def __init__(self, trsm, n, k=6):
        combo_builder = ak.ArrayBuilder()
        combo_builder = ak.ArrayBuilder()

        self.trsm = trsm

        n_jet    = trsm.n_jet
        # n_sixb   = trsm.n_sixb
        jet_idx  = trsm.jet_idx
        jet_pt   = trsm.jet_pt
        jet_eta  = trsm.jet_eta
        jet_phi  = trsm.jet_phi
        jet_m    = trsm.jet_m
        jet_btag = trsm.jet_btag
        jet_qgl  = trsm.jet_qgl
        jet_partonFlav = trsm.jet_partonFlav
        jet_hadronFlav = trsm.jet_hadronFlav

        HX_p4 = trsm.HX_p4
        H1_p4 = trsm.H1_p4
        H2_p4 = trsm.H2_p4

        HX_b1_p4 = trsm.HX_b1_p4
        HX_b2_p4 = trsm.HX_b2_p4
        H1_b1_p4 = trsm.H1_b1_p4
        H1_b2_p4 = trsm.H1_b2_p4
        H2_b1_p4 = trsm.H2_b1_p4
        H2_b2_p4 = trsm.H2_b2_p4

        nevents = trsm.nevents

        sgnl_mask = []
        combo_mask = []

        combos_btag = []

        combo_jet_binary_mask = []
        combo_H_binary_mask = []

        evt_idx_combos = []

        ## 2 jet classifier inputs
        combo_pt = []
        combo_eta = []
        combo_phi = []
        combo_m = []

        counter = 0

        nevents = np.arange(trsm.nevents)
        nevents = nevents[n_jet == n]
        
        for evt in tqdm(nevents):

            evt_n          = np.arange(n_jet[evt])
            evt_idx        = jet_idx[evt]
            evt_pt         = jet_pt[evt]
            evt_eta        = jet_eta[evt]
            evt_phi        = jet_phi[evt]
            evt_m          = jet_m[evt]
            evt_btag       = jet_btag[evt]
            evt_qgl        = jet_qgl[evt]
            evt_partonFlav = jet_partonFlav[evt]
            evt_hadronFlav = jet_hadronFlav[evt]

            # signal mask
            signal_mask = [i for i,obj in enumerate(evt_idx) if obj > -1]
            signal_mask = np.array((signal_mask))

            # flag as full event if it contains all six signal jets
            if len(np.unique(signal_mask)) == 6:
                signal_flag = True
            else:
                signal_flag = False

            # background mask
            background_mask = [i for i,obj in enumerate(evt_idx) if obj == -1]
            background_mask = np.array((background_mask))

            # Cound the number of background jets
            n_bkgd = len(evt_pt[background_mask])
            counter += 1

            # Create combos of k jets from all jets in the event
            jet_combos  = list(itertools.combinations(evt_n, k))
            pt_combos   = list(itertools.combinations(evt_pt, k))
            eta_combos  = list(itertools.combinations(evt_eta, k))
            phi_combos  = list(itertools.combinations(evt_phi, k))
            m_combos    = list(itertools.combinations(evt_m, k))
            btag_combos = list(itertools.combinations(evt_btag, k))
            idx_combos  = list(itertools.combinations(evt_idx, k))
            
            idx_array = np.array(())

            for pt, eta, phi, m, btag, idx, jet_ind in zip(pt_combos, eta_combos, phi_combos, m_combos, btag_combos, idx_combos, jet_combos):
                
                evt_idx_combos.append(idx)

                # Check if the current combo idx is the same as the signal idx
                if -1 not in idx:
                    sgnl_mask.append(True)
                else:
                    # Save location of incorrect jets in evt arrays
                    bkgd_jets_in_combo = [ind for ind, sig_idx in zip(jet_ind,idx) if sig_idx == -1]
                    signal_jets_in_combo = [ind for ind, sig_idx in zip(jet_ind,idx) if sig_idx > -1]

                    assert(np.all(np.isin(bkgd_jets_in_combo, background_mask)))

                    # binary key indicates which signal jets are in the combo
                    combo_jet_info = 0b000000
                    for i in range(6):
                        if i in idx:
                            combo_jet_info = combo_jet_info | (1 << (5-i))
                    combo_jet_binary_mask.append(combo_jet_info)
                    
                    combo_H_info = 0b000
                    if ((0b11 << 4) & combo_jet_info) == 0b110000:
                        combo_H_info = combo_H_info | (11 << 4)
                    if ((0b11 << 2) & combo_jet_info) == 0b001100:
                        combo_H_info = combo_H_info | (11 << 2)
                    if ((0b11 << 2) & combo_jet_info) == 0b000011:
                        combo_H_info = combo_H_info | (11 << 0)
                    combo_H_binary_mask.append(combo_H_info)
                        
                    sgnl_mask.append(False)
  
                # sort by pT
                sort_mask = np.array((np.argsort(pt)[::-1]))
                # sort by btag
                # sort_mask = np.array((np.argsort(btag)[::-1]))

                arr_eta  = np.asarray(eta)[sort_mask]
                arr_phi  = np.asarray(phi)[sort_mask]
                arr_m    = np.asarray(m)[sort_mask]
                idx_array = np.append(idx_array, np.asarray(idx)[sort_mask])
                btag = np.asarray(btag)[sort_mask]
                combos_btag.append(btag)
                # pt must be sorted last because it is used to sort everything else
                arr_pt   = np.sort(np.asarray(pt))[::-1]

                combo_pt.append(pt)
                combo_eta.append(eta)
                combo_phi.append(phi)
                combo_m.append(m)

                combo_builder.begin_list()
                for pt, eta, phi, m in zip(arr_pt, arr_eta, arr_phi, arr_m):
                    combo_builder.begin_record("Momentum4D")
                    combo_builder.field("pt").real(pt)
                    combo_builder.field("eta").real(eta)
                    combo_builder.field("phi").real(phi)
                    combo_builder.field("m").real(m)
                    combo_builder.end_record()
                combo_builder.end_list()
                

                # with combo_builder.list():

                #     for pt, eta, phi, m in zip(arr_pt, arr_eta, arr_phi, arr_m):

                #         with combo_builder.record("Momentum4D"):   # not MomentumObject4D

                #             combo_builder.field("pt"); combo_builder.real(pt)
                #             combo_builder.field("eta"); combo_builder.real(eta)
                #             combo_builder.field("phi"); combo_builder.real(phi)
                #             combo_builder.field("m"); combo_builder.real(m)


        combos_builder = combo_builder.snapshot()

        self.combo_p4 = vector.obj(pt=ak.Array(combo_pt), eta=ak.Array(combo_eta), phi=ak.Array(combo_phi), m=ak.Array(combo_m))

        self.sgnl_mask = np.asarray(sgnl_mask)

        self.combo_jet_binary_mask = np.asarray(combo_jet_binary_mask)
        self.combo_H_binary_mask = np.asarray(combo_H_binary_mask)

        self.idx_combos = np.asarray(evt_idx_combos)
        self.btag_combos = np.asarray(combos_btag)

        self.sixjet_p4, boosted = get_6jet_p4(combos_builder)
        self.combo_features_6j = self.construct_6j_features(combos_builder, combos_btag, boosted)

        print("Total events chosen:",counter)

    def load_model(self, tag):
        json_file = open(f'../models/{tag}/model/model_1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(f'../models/{tag}/model/model_1.h5')
        scaler = load(open(f'../models/{tag}/model/scaler_1.pkl', 'rb'))

        return scaler, loaded_model

    def apply_6j_model(self, tag):

        scaler, model = self.load_model(tag)
        test_features = scaler.transform(self.combo_features_6j)
        self.scores_combo = model.predict(test_features)[:,0]

    @jit()
    def create_pairs(self, tag, score_cut=0):
        jet1 = ak.ArrayBuilder()
        jet2 = ak.ArrayBuilder()

        score_mask = self.scores_combo > score_cut

        pt_arr = self.combo_p4.pt[score_mask]
        eta_arr = self.combo_p4.eta[score_mask]
        phi_arr = self.combo_p4.phi[score_mask]
        m_arr = self.combo_p4.m[score_mask]
        btag_arr = self.btag_combos[score_mask]
        idx_arr = self.idx_combos[score_mask]

        jet1_btag = []
        jet2_btag = []
        jet1_idx = []
        jet2_idx = []

        Higgs_mask = []

        for pt, eta, phi, m, btag, idx in zip(pt_arr, eta_arr, phi_arr, m_arr, btag_arr, idx_arr):

            pt_combos   = list(itertools.combinations(pt, 2))
            eta_combos  = list(itertools.combinations(eta, 2))
            phi_combos  = list(itertools.combinations(phi, 2))
            m_combos    = list(itertools.combinations(m, 2))
            btag_combos = list(itertools.combinations(btag, 2))
            idx_combos = list(itertools.combinations(idx, 2))
            
            for pair_pt, pair_eta, pair_phi, pair_m, pair_btag, pair_idx in zip(pt_combos, eta_combos, phi_combos, m_combos, btag_combos, idx_combos):

                sort_mask = np.array((np.argsort(np.asarray(pair_pt))[::-1]))

                pair_eta  = np.asarray(pair_eta)[sort_mask]
                pair_phi  = np.asarray(pair_phi)[sort_mask]
                pair_m    = np.asarray(pair_m)[sort_mask]
                pair_idx = np.asarray(pair_idx)[sort_mask]
                pair_btag = np.asarray(btag)[sort_mask]

                jet1_idx.append(pair_idx[0])
                jet2_idx.append(pair_idx[1])
                jet1_btag.append(pair_btag[0])
                jet2_btag.append(pair_btag[1])

                if 0 in pair_idx and 1 in pair_idx:
                    Higgs_mask.append(0)
                elif 2 in pair_idx and 3 in pair_idx:
                    Higgs_mask.append(1)
                elif 4 in pair_idx and 5 in pair_idx:
                    Higgs_mask.append(2)
                else:
                    Higgs_mask.append(-1)
                

                # pt must be sorted last because it is used to sort everything else
                pair_pt   = np.sort(np.asarray(pair_pt))[::-1]

                jet1.begin_list()
                jet1.begin_record("Momentum4D")
                jet1.field("pt").real(pair_pt[0])
                jet1.field("eta").real(pair_eta[0])
                jet1.field("phi").real(pair_phi[0])
                jet1.field("m").real(pair_m[0])
                jet1.end_record("Momentum4D")
                jet1.end_list()

                jet2.begin_list()
                jet2.begin_record("Momentum4D")
                jet2.field("pt").real(pair_pt[1])
                jet2.field("eta").real(pair_eta[1])
                jet2.field("phi").real(pair_phi[1])
                jet2.field("m").real(pair_m[1])
                jet2.end_record("Momentum4D")
                jet2.end_list()
            
        jet1 = jet1.snapshot()
        jet2 = jet2.snapshot()
        
        jet_pair = jet1 + jet2

        jet1_idx = np.asarray(jet1_idx)
        jet2_idx = np.asarray(jet2_idx)
        jet1_btag = np.asarray(jet1_btag)
        jet2_btag = np.asarray(jet2_btag)

        self.pair_features = np.column_stack((np.asarray(jet1.pt), np.asarray(jet2.pt), np.asarray(jet1.eta), np.asarray(jet2.eta), np.asarray(jet1.phi), np.asarray(jet2.phi), jet1_btag, jet2_btag, np.asarray(jet1.deltaR(jet2))))
        self.pair_target = np.asarray(Higgs_mask)


    def apply_2j_model(self, tag):
        
        scaler, model = self.load_model(tag)
        test_features = scaler.transform(self.combo_features_2j)
        self.scores_pairs = model.predict(test_features)


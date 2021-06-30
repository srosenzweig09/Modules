"""
Project: 6b Final States - TRSM
Author: Suzanne Rosenzweig

This class sorts MC-generated events into an array of input features for use in training a neural network and in evaluating the performance of the model using a test set of examples.
"""

# from particle import Particle
import awkward as ak
from icecream import ic
import itertools
from math import comb
import numpy as np
import random
import vector
import sys  # JAKE DELETE

from logger import info
from myuproot import open_up
from tqdm import tqdm # progress bar

vector.register_awkward()

def get_evt_p4(p4, num=6):
    combos = ak.combinations(p4, num)
    part0, part1, part2, part3, part4, part5 = ak.unzip(combos)
    evt_p4 = part0 + part1 + part2 + part3 + part4 + part5
    boost_0 = part0.boost_p4(evt_p4)
    boost_1 = part1.boost_p4(evt_p4)
    boost_2 = part2.boost_p4(evt_p4)
    boost_3 = part3.boost_p4(evt_p4)
    boost_4 = part4.boost_p4(evt_p4)
    boost_5 = part5.boost_p4(evt_p4)
    return evt_p4, boost_0, boost_1, boost_2, boost_3, boost_4, boost_5

class TRSM():

    # b_mass = Particle.from_pdgid(5).mass / 1000 # Convert from MeV to GeV

    ## Open ROOT file
    def __init__(self, filename):

        info(f"Opening ROOT file {filename} with columns")

        tree, ak_table, np_table = open_up(filename)

        self.tree     = tree
        self.ak_table = ak_table
        self.np_table = np_table
        self.nevents  = len(ak_table)

        self.jet_idx  = np_table['jet_idx']
        self.jet_pt   = np_table['jet_pt']
        self.jet_eta  = np_table['jet_eta']
        self.jet_phi  = np_table['jet_phi']
        self.jet_m    = np_table['jet_m']
        self.jet_btag = np_table['jet_btag']
        self.jet_qgl  = np_table['jet_qgl']
        self.jet_partonFlav = np_table['jet_partonFlav']
        self.jet_hadronFlav = np_table['jet_hadronFlav']

        self.HX_b1_pt  = np_table['HX_b1_recojet_ptRegressed']
        self.HX_b2_pt  = np_table['HX_b2_recojet_ptRegressed']
        self.HX_b1_eta = np_table['HX_b1_recojet_eta']
        self.HX_b2_eta = np_table['HX_b2_recojet_eta']
        self.HX_b1_phi = np_table['HX_b1_recojet_phi']
        self.HX_b2_phi = np_table['HX_b2_recojet_phi']
        self.HX_b1_m   = np_table['HX_b1_recojet_m']
        self.HX_b2_m   = np_table['HX_b2_recojet_m']
        self.H1_b1_pt  = np_table['HY1_b1_recojet_ptRegressed']
        self.H1_b2_pt  = np_table['HY1_b2_recojet_ptRegressed']
        self.H1_b1_eta = np_table['HY1_b1_recojet_eta']
        self.H1_b2_eta = np_table['HY1_b2_recojet_eta']
        self.H1_b1_phi = np_table['HY1_b1_recojet_phi']
        self.H1_b2_phi = np_table['HY1_b2_recojet_phi']
        self.H1_b1_m   = np_table['HY1_b1_recojet_m']
        self.H1_b2_m   = np_table['HY1_b2_recojet_m']
        self.H2_b1_pt  = np_table['HY2_b1_recojet_ptRegressed']
        self.H2_b2_pt  = np_table['HY2_b2_recojet_ptRegressed']
        self.H2_b1_eta = np_table['HY2_b1_recojet_eta']
        self.H2_b2_eta = np_table['HY2_b2_recojet_eta']
        self.H2_b1_phi = np_table['HY2_b1_recojet_phi']
        self.H2_b2_phi = np_table['HY2_b2_recojet_phi']
        self.H2_b1_m   = np_table['HY2_b1_recojet_m']
        self.H2_b2_m   = np_table['HY2_b2_recojet_m']

        self.HX_b1_p4 = vector.obj(pt=self.HX_b1_pt, eta=self.HX_b1_eta, phi=self.HX_b1_phi, m=self.HX_b1_m)
        self.HX_b2_p4 = vector.obj(pt=self.HX_b2_pt, eta=self.HX_b2_eta, phi=self.HX_b2_phi, m=self.HX_b2_m)
        self.H1_b1_p4 = vector.obj(pt=self.H1_b1_pt, eta=self.H1_b1_eta, phi=self.H1_b1_phi, m=self.H1_b1_m)
        self.H1_b2_p4 = vector.obj(pt=self.H1_b2_pt, eta=self.H1_b2_eta, phi=self.H1_b2_phi, m=self.H1_b2_m)
        self.H2_b1_p4 = vector.obj(pt=self.H2_b1_pt, eta=self.H2_b1_eta, phi=self.H2_b1_phi, m=self.H2_b1_m)
        self.H2_b2_p4 = vector.obj(pt=self.H2_b2_pt, eta=self.H2_b2_eta, phi=self.H2_b2_phi, m=self.H2_b2_m)

        self.HX_p4 = self.HX_b1_p4 + self.HX_b2_p4
        self.H1_p4 = self.H1_b1_p4 + self.H1_b2_p4
        self.H2_p4 = self.H2_b1_p4 + self.H2_b2_p4

    def construct_features(self, combo_p4, combo_btag, boosted):

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
    
class training(TRSM):
    ## Build p4 for all events
    def __init__(self, trsm):

        signal_builder = ak.ArrayBuilder()
        bkgd_builder = ak.ArrayBuilder()
        
        jet_idx        = trsm.jet_idx
        jet_pt         = trsm.jet_pt
        jet_eta        = trsm.jet_eta
        jet_phi        = trsm.jet_phi
        jet_m          = trsm.jet_m
        jet_btag       = trsm.jet_btag
        jet_qgl        = trsm.jet_qgl
        jet_partonFlav = trsm.jet_partonFlav
        jet_hadronFlav = trsm.jet_hadronFlav

        signal_idx  = []
        signal_btag = []

        background_btag = []
        available_bkgd  = []
        n_background = []

        pass_count = 0

        info("Looping through events. This may take a few minutes.")
        for evt in tqdm(range(self.nevents)):

            # signal and background masks
            signal_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj > -1]
            signal_mask = np.array((signal_mask))
            background_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj == -1]
            # background_mask = np.array((background_mask))

            # Cound the number of background jets
            n_bkgd = len(jet_pt[evt][background_mask])

            # Skip and events with less than 6 matching signal bs (for now)
            if len(jet_pt[evt][signal_mask]) < 6: continue

            # Skip any events with duplicate matches (for now)
            if len(np.unique(jet_idx[evt][signal_mask])) < len(jet_idx[evt][signal_mask]): continue
            
            if (n_bkgd < 1): continue
            pass_count += 1
            available_bkgd.append(n_bkgd)
            
            bkgd_ind = np.arange(1,n_bkgd)
            sgnl_ind = np.arange(0,6)

            if n_bkgd > 1:
                # Choose random background and signal jets to swap
                random_bkgd = random.choices(bkgd_ind, k=n_bkgd)
                random_sgnl = random.choices(sgnl_ind, k=n_bkgd)
            else:
                random_bkgd = [0] # Not so random lol
                random_sgnl = random.choices(sgnl_ind, k=1) # This is randomized
            
            n_background.append(len(random_bkgd))
            
            sixb_pt = jet_pt[evt][signal_mask]
            non_sixb_pt = sixb_pt.copy()
            for nH, sb in zip(random_bkgd, random_sgnl):
                bkgd_pt = jet_pt[evt][background_mask][nH]
                non_sixb_pt[sb] = bkgd_pt
            
            sixb_eta = jet_eta[evt][signal_mask]
            non_sixb_eta = sixb_eta.copy()
            for nH, sb in zip(random_bkgd, random_sgnl):
                bkgd_eta = jet_eta[evt][background_mask][nH]
                non_sixb_eta[sb] = bkgd_eta
            
            sixb_phi = jet_phi[evt][signal_mask]
            non_sixb_phi = sixb_phi.copy()
            for nH, sb in zip(random_bkgd, random_sgnl):
                bkgd_phi = jet_phi[evt][background_mask][nH]
                non_sixb_phi[sb] = bkgd_phi

            sixb_m = jet_m[evt][signal_mask]
            non_sixb_m = sixb_m.copy()
            for nH, sb in zip(random_bkgd, random_sgnl):
                bkgd_m = jet_m[evt][background_mask][nH]
                non_sixb_m[sb] = bkgd_m

            sixb_btag = jet_btag[evt][signal_mask]
            non_sixb_btag = sixb_btag.copy()
            for nH, sb in zip(random_bkgd, random_sgnl):
                bkgd_btag = jet_btag[evt][background_mask][nH]
                non_sixb_btag[sb] = bkgd_btag

            sixb_idx = signal_mask[np.argsort(sixb_pt)][::-1]
            signal_idx.append(sixb_idx)
            
            sixb_eta  = sixb_eta[np.argsort(sixb_pt)][::-1]
            sixb_phi  = sixb_phi[np.argsort(sixb_pt)][::-1]
            sixb_m    = sixb_m[np.argsort(sixb_pt)][::-1]
            sixb_btag = sixb_btag[np.argsort(sixb_pt)][::-1]
            signal_btag.append(sixb_btag)
            # pt must be sorted last because it is used to sort everything else
            sixb_pt   = np.sort(sixb_pt)[::-1]
            
            non_sixb_eta  = non_sixb_eta[np.argsort(non_sixb_pt)][::-1]
            non_sixb_phi  = non_sixb_phi[np.argsort(non_sixb_pt)][::-1]
            non_sixb_m    = non_sixb_m[np.argsort(non_sixb_pt)][::-1]
            non_sixb_btag = non_sixb_btag[np.argsort(non_sixb_pt)][::-1]
            background_btag.append(non_sixb_btag)
            # pt must be sorted last because it is used to sort everything else
            non_sixb_pt   = np.sort(non_sixb_pt)[::-1]

            with signal_builder.list():

                for pt, eta, phi, m in zip(sixb_pt, sixb_eta, sixb_phi, sixb_m):

                    with signal_builder.record("Momentum4D"):   # not MomentumObject4D

                        signal_builder.field("pt"); signal_builder.real(pt)
                        signal_builder.field("eta"); signal_builder.real(eta)
                        signal_builder.field("phi"); signal_builder.real(phi)
                        signal_builder.field("m"); signal_builder.real(m)

            with bkgd_builder.list():

                    for pt, eta, phi,m  in zip(non_sixb_pt, non_sixb_eta, non_sixb_phi, non_sixb_m):

                        with bkgd_builder.record("Momentum4D"):   # not MomentumObject4D

                            bkgd_builder.field("pt"); bkgd_builder.real(pt)
                            bkgd_builder.field("eta"); bkgd_builder.real(eta)
                            bkgd_builder.field("phi"); bkgd_builder.real(phi)
                            bkgd_builder.field("m"); bkgd_builder.real(m)

        info(f"Number of events saved: {pass_count}")

        self.sgnl_p4 = signal_builder.snapshot()
        self.bkgd_p4 = bkgd_builder.snapshot()

        self.sgnl_evt_p4, *self.sgnl_boosted = get_evt_p4(self.sgnl_p4)
        self.bkgd_evt_p4, *self.bkgd_boosted = get_evt_p4(self.bkgd_p4)

        self.signal_idx = np.array((signal_idx))
        self.signal_btag = np.array((signal_btag))
        self.bkgd_btag = np.array((background_btag))
        self.bkgd = np.array((available_bkgd))

        self.n_bkgd = np.array((n_background))

    def construct_training_features(self):

        sgnl_inputs = self.construct_features(self.sgnl_p4, self.signal_btag, self.sgnl_boosted)

        bkgd_inputs = self.construct_features(self.bkgd_p4, self.bkgd_btag, self.bkgd_boosted)

        inputs = np.row_stack((sgnl_inputs, bkgd_inputs))
        print(f"Combined (sgnl + bkgd) input shape = {inputs.shape}")

        return inputs
    

class combos(TRSM):

    def __init__(self, trsm, n):
        combo_builder = ak.ArrayBuilder()

        self.jet_idx  = trsm.jet_idx
        self.jet_pt   = trsm.jet_pt
        self.jet_eta  = trsm.jet_eta
        self.jet_phi  = trsm.jet_phi
        self.jet_m    = trsm.jet_m
        self.jet_btag = trsm.jet_btag
        self.jet_qgl  = trsm.jet_qgl
        self.jet_partonFlav = trsm.jet_partonFlav
        self.jet_hadronFlav = trsm.jet_hadronFlav

        self.HX_p4 = trsm.HX_p4
        self.H1_p4 = trsm.H1_p4
        self.H2_p4 = trsm.H2_p4

        k = 6 # 6 jets originating from X in our signal event

        sgnl_mask = []
        combos_btag = []

        swapped_bkgd_ind = []
        swapped_bkgd_partonFlav = []
        swapped_bkgd_hadronFlav = []
        swapped_bkgd_pt = []

        swapped_sgnl_idx = []
        swapped_sgnl_ind = []
        swapped_sgnl_pt = []

        incorrect_H_bkgd_pt = []
        incorrect_H_bkgd_eta = []
        incorrect_H_bkgd_phi = []
        incorrect_H_bkgd_m = []

        incorrect_H_sgnl_pt = []
        incorrect_H_sgnl_eta = []
        incorrect_H_sgnl_phi = []
        incorrect_H_sgnl_m = []

        combo_mask = []

        counter = 0
        
        for evt in tqdm(range(trsm.nevents)):

            evt_idx        = self.jet_idx[evt]
            evt_pt         = self.jet_pt[evt]
            evt_eta        = self.jet_eta[evt]
            evt_phi        = self.jet_phi[evt]
            evt_m          = self.jet_m[evt]
            evt_btag       = self.jet_btag[evt]
            evt_qgl        = self.jet_qgl[evt]
            evt_partonFlav = self.jet_partonFlav[evt]
            evt_hadronFlav = self.jet_hadronFlav[evt]

            # signal and background masks
            signal_mask = [i for i,obj in enumerate(evt_idx) if obj > -1]
            signal_mask = np.array((signal_mask))

            background_mask = [i for i,obj in enumerate(evt_idx) if obj == -1]
            # background_mask = np.array((background_mask))

            # Cound the number of background jets
            n_bkgd = len(evt_pt[background_mask])

            # Skip and events with less than 6 matching signal bs (for now)
            if len(signal_mask) < k: 
                combo_mask.append(False)
                continue

            # Skip any events with duplicate matches (for now)
            if len(np.unique(evt_idx[signal_mask])) < len(evt_idx[signal_mask]): 
                combo_mask.append(False)
                continue
            
            if (n_bkgd < n - k) or (n_bkgd == 0): 
                combo_mask.append(False)
                continue
            combo_mask.append(True)
            counter += 1
            # if counter > 2: break

            if n_bkgd > n - k:
                if n - k == 1:
                    rand_bkgd_ind =random.choice(background_mask)
                    swap_mask = np.append(signal_mask, rand_bkgd_ind)
                else:
                    r = n - k
                    rand_bkgd_ind = random.choices(background_mask, k=r)
                    swap_mask = np.append(signal_mask, rand_bkgd_ind)
            else:
                swap_mask = np.arange(n)

            N_jets = np.arange(len(evt_pt))

            jet_combos  = list(itertools.combinations(N_jets[swap_mask], k))
            pt_combos   = list(itertools.combinations(evt_pt[swap_mask], k))
            eta_combos  = list(itertools.combinations(evt_eta[swap_mask], k))
            phi_combos  = list(itertools.combinations(evt_phi[swap_mask], k))
            m_combos    = list(itertools.combinations(evt_m[swap_mask], k))
            btag_combos = list(itertools.combinations(evt_btag[swap_mask], k))
            idx_combos  = list(itertools.combinations(evt_idx[swap_mask], k))
            
            idx_array = np.array(())

            signal_flag = False
            for pt, eta, phi, m, btag, idx, jet_ind in zip(pt_combos, eta_combos, phi_combos, m_combos, btag_combos, idx_combos, jet_combos):

                # Check if the current combo idx is the same as the signal idx
                # ic(idx, evt_idx[signal_mask])
                if np.array_equal(np.sort(np.asarray(idx)), np.sort(evt_idx[signal_mask])):
                    sgnl_mask.append(True)
                    signal_flag = True
                    # ic()
                else:
                    # ic()
                    # Save incorrect combo info
                    bkgd_jets_in_combo = [ind for ind, sig_idx in zip(jet_ind,idx) if sig_idx == -1]
                    signal_jets_in_combo = [ind for ind, sig_idx in zip(jet_ind,idx) if sig_idx > -1]
                    swapped_sgnl_jets = np.logical_not(np.isin(signal_mask, signal_jets_in_combo))

                    assert np.all(np.isin(bkgd_jets_in_combo, background_mask)), print(f"bkgd_jets_in_combo = {bkgd_jets_in_combo}\nbackground_mask = {background_mask}")

                    swapped_bkgd_pt.append(evt_pt[bkgd_jets_in_combo])
                    swapped_bkgd_ind.append(N_jets[bkgd_jets_in_combo])
                    swapped_bkgd_partonFlav.append(evt_partonFlav[bkgd_jets_in_combo])
                    swapped_bkgd_hadronFlav.append(evt_hadronFlav[bkgd_jets_in_combo])

                    swapped_sgnl_pt.append(evt_pt[signal_mask][swapped_sgnl_jets])
                    swapped_sgnl_idx.append(evt_idx[signal_mask][swapped_sgnl_jets])

                    HX_b_mask = np.isin([0,1], idx)
                    H1_b_mask = np.isin([2,3], idx)
                    H2_b_mask = np.isin([4,5], idx)

                    # ic(HX_b_mask, H1_b_mask, H2_b_mask, idx)

                    HX_mask = np.all(HX_b_mask)
                    H1_mask = np.all(H1_b_mask)
                    H2_mask = np.all(H2_b_mask)

                    incorrect_H_bkgd_pt.append(evt_pt[bkgd_jets_in_combo])
                    incorrect_H_bkgd_eta.append(evt_eta[bkgd_jets_in_combo])
                    incorrect_H_bkgd_phi.append(evt_phi[bkgd_jets_in_combo])
                    incorrect_H_bkgd_m.append(evt_m[bkgd_jets_in_combo])

                    if not HX_mask:
                        HX_b_idx = np.array((0,1))[HX_b_mask] # which signal b
                        HX_b_ind = np.argwhere(idx == HX_b_idx) # where is it in this combo
                        # ic(HX_b_ind)
                        incorrect_H_sgnl_pt.append(evt_pt[HX_b_ind])
                        incorrect_H_sgnl_eta.append(evt_eta[HX_b_ind])
                        incorrect_H_sgnl_phi.append(evt_phi[HX_b_ind])
                        incorrect_H_sgnl_m.append(evt_m[HX_b_ind])
                    if not H1_mask:
                        H1_b_idx = np.array((2,3))[H1_b_mask] # which signal b
                        H1_b_ind = np.argwhere(idx == H1_b_idx) # where is it in this combo
                        incorrect_H_sgnl_pt.append(evt_pt[H1_b_ind])
                        incorrect_H_sgnl_eta.append(evt_eta[H1_b_ind])
                        incorrect_H_sgnl_phi.append(evt_phi[H1_b_ind])
                        incorrect_H_sgnl_m.append(evt_m[H1_b_ind])
                        # ic(H1_b_ind)
                    if not H2_mask:
                        H2_b_idx = np.array((4,5))[H2_b_mask] # which signal b
                        H2_b_ind = np.argwhere(idx == H2_b_idx) # where is it in this combo           
                        incorrect_H_sgnl_pt.append(evt_pt[H2_b_ind])
                        incorrect_H_sgnl_eta.append(evt_eta[H2_b_ind])
                        incorrect_H_sgnl_phi.append(evt_phi[H2_b_ind])
                        incorrect_H_sgnl_m.append(evt_m[H2_b_ind])
                        # ic(H2_b_ind)
                        
                        
                    sgnl_mask.append(False)
  
                sort_mask = np.array((np.argsort(pt)[::-1]))

                eta  = np.asarray(eta)[sort_mask]
                phi  = np.asarray(phi)[sort_mask]
                m    = np.asarray(m)[sort_mask]
                btag = np.asarray(btag)[sort_mask]
                combos_btag.append(btag)
                idx_array = np.append(idx_array, idx)
                # pt must be sorted last because it is used to sort everything else
                pt   = np.sort(np.asarray(pt))[::-1]


                with combo_builder.list():

                    for pt, eta, phi, m in zip(pt, eta, phi, m):

                        with combo_builder.record("Momentum4D"):   # not MomentumObject4D

                            combo_builder.field("pt"); combo_builder.real(pt)
                            combo_builder.field("eta"); combo_builder.real(eta)
                            combo_builder.field("phi"); combo_builder.real(phi)
                            combo_builder.field("m"); combo_builder.real(m)
            
            assert signal_flag, print(f"evt = {evt}\nn mask = {swap_mask}\nevt_idx[signal_mask] = {evt_idx[signal_mask]}\nidx_combos = {idx_combos}\nn_bkgd = {n_bkgd}")

        self.combo_mask = ak.Array(combo_mask)

        incorrect_H_bkgd_pt = ak.Array(incorrect_H_bkgd_pt)
        incorrect_H_bkgd_eta = ak.Array(incorrect_H_bkgd_eta)
        incorrect_H_bkgd_phi = ak.Array(incorrect_H_bkgd_phi)
        incorrect_H_bkgd_m = ak.Array(incorrect_H_bkgd_m)
        incorrect_H_sgnl_pt = ak.Array(incorrect_H_sgnl_pt)
        incorrect_H_sgnl_eta = ak.Array(incorrect_H_sgnl_eta)
        incorrect_H_sgnl_phi = ak.Array(incorrect_H_sgnl_phi)
        incorrect_H_sgnl_m = ak.Array(incorrect_H_sgnl_m)

        incorrect_sgnl_p4 = vector.obj(pt=incorrect_H_sgnl_pt,
                                       eta=incorrect_H_sgnl_eta,
                                       phi=incorrect_H_sgnl_phi,
                                       m=incorrect_H_sgnl_m)
        # ic(incorrect_sgnl_p4)
        incorrect_bkgd_p4 = vector.obj(pt=incorrect_H_bkgd_pt,
                                       eta=incorrect_H_bkgd_eta,
                                       phi=incorrect_H_bkgd_phi,
                                       m=incorrect_H_bkgd_m)
        # ic(incorrect_bkgd_p4)
        
        self.incorrect_H_p4 = incorrect_sgnl_p4 + incorrect_bkgd_p4

        swapped_bkgd_ind = ak.Array(swapped_bkgd_ind)
        self.swapped_bkgd_pt = ak.Array(swapped_bkgd_pt)
        swapped_bkgd_partonFlav = ak.Array(swapped_bkgd_partonFlav)
        swapped_bkgd_hadronFlav = ak.Array(swapped_bkgd_hadronFlav)

        swapped_sgnl_idx = ak.Array(swapped_sgnl_idx)
        self.swapped_sgnl_pt = ak.Array(swapped_sgnl_pt)

        self.bkgd_dict = {'idx':swapped_sgnl_idx, 'ind':swapped_bkgd_ind, 'partonFlav':swapped_bkgd_partonFlav, 'hadronFlav':swapped_bkgd_hadronFlav}

        combos_builder = combo_builder.snapshot()
        combo_evt_p4, boost_0, boost_1, boost_2, boost_3, boost_4, boost_5 = get_evt_p4(combos_builder)
        self.sgnl_mask = np.array((sgnl_mask))

        boosted = [boost_0, boost_1, boost_2, boost_3, boost_4, boost_5]

        self.combo_features = self.construct_features(combos_builder, combos_btag, boosted)

        assert len(sgnl_mask) == counter*comb(n,k), print(len(sgnl_mask), counter*comb(n,k))
        assert np.sum(sgnl_mask) == counter, print(np.sum(sgnl_mask*1),counter, sgnl_mask)

        print(f"Total events chosen: {counter}")
        


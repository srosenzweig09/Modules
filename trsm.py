"""
trsm_combos is a class that ...





"""

# from particle import Particle
import awkward as ak
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

class trsm_combos():

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
    
    ## Build p4 for all events
    def build_p4(self, filename=None):

        signal_builder = ak.ArrayBuilder()
        bkgd_builder = ak.ArrayBuilder()
        
        jet_idx  = self.jet_idx
        jet_pt   = self.jet_pt
        jet_eta  = self.jet_eta
        jet_phi  = self.jet_phi
        jet_m    = self.jet_m
        jet_btag = self.jet_btag
        jet_qgl  = self.jet_qgl
        jet_partonFlav = self.jet_partonFlav
        jet_hadronFlav = self.jet_hadronFlav

        signal_idx  = []
        signal_btag = []

        background_btag = []
        available_bkgd  = []

        n_background = []

        mask_7 = []
        mask_8 = []

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

        self.sgnl_p4 = signal_builder.snapshot()
        self.bkgd_p4 = bkgd_builder.snapshot()

        self.sgnl_evt_p4, *self.sgnl_boosted = get_evt_p4(self.sgnl_p4)
        self.bkgd_evt_p4, *self.bkgd_boosted = get_evt_p4(self.bkgd_p4)

        self.signal_idx = np.array((signal_idx))
        self.signal_btag = np.array((signal_btag))
        self.bkgd_btag = np.array((background_btag))
        self.bkgd = np.array((available_bkgd))

        self.n_bkgd = np.array((n_background))

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
        print(f"Input shape = {inputs.shape}")

        return inputs

    def construct_training_features(self):

        sgnl_inputs = self.construct_features(self.sgnl_p4, self.signal_btag, self.sgnl_boosted)

        bkgd_inputs = self.construct_features(self.bkgd_p4, self.bkgd_btag, self.bkgd_boosted)

        inputs = np.concatenate((sgnl_inputs, bkgd_inputs))
        print(f"Input shape = {inputs.shape}")

        return inputs
        
    def nCk(self, n=7, k=6):

        combo_builder = ak.ArrayBuilder()

        jet_idx  = self.jet_idx
        jet_pt   = self.jet_pt
        jet_eta  = self.jet_eta
        jet_phi  = self.jet_phi
        jet_m    = self.jet_m
        jet_btag = self.jet_btag
        jet_qgl  = self.jet_qgl
        jet_partonFlav = self.jet_partonFlav
        jet_hadronFlav = self.jet_hadronFlav

        combo_btag = []
        evt_tag = []

        swapped_bkgd_ind = []
        swapped_bkgd_partonFlav = []
        swapped_bkgd_hadronFlav = []

        swapped_sgnl_ind = []
        swapped_sgnl_partonFlav = []
        swapped_sgnl_hadronFlav = []

        counter = 0
        
        for evt in tqdm(range(self.nevents)):

            # signal and background masks
            signal_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj > -1]
            signal_mask = np.array((signal_mask))
            background_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj == -1]
            # background_mask = np.array((background_mask))

            # Cound the number of background jets
            n_bkgd = len(jet_pt[evt][background_mask])

            # Skip and events with less than 6 matching signal bs (for now)
            if len(jet_pt[evt][signal_mask]) < k: continue

            # Skip any events with duplicate matches (for now)
            if len(np.unique(jet_idx[evt][signal_mask])) < len(jet_idx[evt][signal_mask]): continue
            
            if (n_bkgd < n - k) or (n_bkgd == 0): continue
            counter += 1

            if n_bkgd > n - k:
                if n-k == 1:
                    rand_bkgd_ind =random.choice(background_mask)
                    n_mask = np.append(signal_mask, rand_bkgd_ind)
                else:
                    try:
                        r = n - k
                        rand_bkgd_ind = random.choices(background_mask
                        n_mask = np.append(signal_mask, rand_bkgd_ind, k=r))
                    except:
                        print(signal_mask)
                        print(r)
                        print(background_mask)
                        print(random.choices(background_mask, k=r))
            else:
                n_mask = np.arange(n)

            N_jets = np.arange(len(jet_pt[evt]))



            jet_combos  = list(itertools.combinations(N_jets[n_mask], k))
            pt_combos   = list(itertools.combinations(jet_pt[evt][n_mask], k))
            eta_combos  = list(itertools.combinations(jet_eta[evt][n_mask], k))
            phi_combos  = list(itertools.combinations(jet_phi[evt][n_mask], k))
            m_combos    = list(itertools.combinations(jet_m[evt][n_mask], k))
            btag_combos = list(itertools.combinations(jet_btag[evt][n_mask], k))
            idx_combos  = list(itertools.combinations(jet_idx[evt][n_mask], k))
            
            signal_flag = False
            for pt, eta, phi, m, btag, idx, jet_ind in zip(pt_combos, eta_combos, phi_combos, m_combos, btag_combos, idx_combos, jet_combos):


                if np.array_equal(np.sort(np.asarray(idx)), np.sort(jet_idx[evt][signal_mask])):
                    evt_tag.append(True)
                    signal_flag = True
                else:
                    unmatched = np.array(([i for i,(ind, sig_idx) in enumerate(zip(jet_ind,idx)) where sig_idx == -1]))

                    swapped_bkgd_idx.append(unmatched)
                    swapped_bkgd_partonFlav.append(jet_partonFlav[unmatched])
                    swapped_bkgd_hadronFlav.append(jet_hadronFlav[unmatched])
                    evt_tag.append(False)
  
                sort_mask = np.array((np.argsort(pt)[::-1]))

                eta  = np.asarray(eta)[sort_mask]
                phi  = np.asarray(phi)[sort_mask]
                m    = np.asarray(m)[sort_mask]
                btag = np.asarray(btag)[sort_mask]
                combo_btag.append(btag)
                # pt must be sorted last because it is used to sort everything else
                pt   = np.sort(np.asarray(pt))[::-1]

                with combo_builder.list():

                    for pt, eta, phi, m in zip(pt, eta, phi, m):

                        with combo_builder.record("Momentum4D"):   # not MomentumObject4D

                            combo_builder.field("pt"); combo_builder.real(pt)
                            combo_builder.field("eta"); combo_builder.real(eta)
                            combo_builder.field("phi"); combo_builder.real(phi)
                            combo_builder.field("m"); combo_builder.real(m)
            assert signal_flag, print(f"evt = {evt}\nn mask = {n_mask}\njet_idx[evt][signal_mask] = {jet_idx[evt][signal_mask]}\nidx_combos = {idx_combos}")


        combos_builder = combo_builder.snapshot()
        combo_evt_p4, boost_0, boost_1, boost_2, boost_3, boost_4, boost_5 = get_evt_p4(combos_builder)
        evt_tag = np.array((evt_tag))

        combo_btag = np.array((combo_btag))
        boosted = [boost_0, boost_1, boost_2, boost_3, boost_4, boost_5]

        combo_features = self.construct_features(combos_builder, combo_btag, boosted)

        assert len(evt_tag) == counter*comb(n,k), print(len(evt_tag), counter*comb(n,k))
        assert np.sum(evt_tag) == counter, print(np.sum(evt_tag*1),counter, evt_tag)

        print(f"Total events chosen: {counter}")

        return combo_features, evt_tag

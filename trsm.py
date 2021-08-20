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

from consistent_plots import hist, hist2d

import matplotlib.pyplot as plt

from keras.models import model_from_json
from pickle import load

from consistent_plots import hist
from logger import info
from myuproot import open_up
from tqdm import tqdm # progress bar

vector.register_awkward()

def norm_hist(arr, bins=100):
    n, b = np.histogram(arr, bins=bins)
    x = (b[:-1] + b[1:]) / 2
    
    return n/n.max(), b, x

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
    return evt_p4, np.asarray([boost_0, boost_1, boost_2, boost_3, boost_4, boost_5])

def load_model(location, tag):
    json_file = open(location + f'models/{tag}/model/model_1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(location + f'models/{tag}/model/model_1.h5')
    scaler = load(open(location + f'models/{tag}/model/scaler_1.pkl', 'rb'))
    return scaler, loaded_model


def plot_highest_score(combos):
    score_mask = combos.high_score_combo_mask
    high_scores = combos.evt_highest_score
    signal_mask = combos.signal_evt_mask
    signal_highs = combos.signal_high_score
    high_nsignal = combos.highest_score_nsignal
    n_signal = combos.n_signal
    
    fig, ax = plt.subplots()
    score_bins = np.arange(0, 1.01, 0.01)

    n_signal, edges = np.histogram(high_scores[signal_mask], bins=score_bins)
    n_bkgd, edges   = np.histogram(high_scores[~signal_mask], bins=score_bins)

    x = (edges[1:] + edges[:-1])/2

    n_signal = n_signal / np.sum(n_signal)
    n_bkgd = n_bkgd / np.sum(n_bkgd)

    n_signal, edges, _ = hist(ax, x, weights=n_signal, bins=score_bins, label='Events with correct combos')
    n_bkgd, edges, _ = hist(ax, x, weights=n_bkgd, bins=score_bins, label='Events with no correct combos')

    ax.legend(loc=2)
    ax.set_xlabel('Highest Assigned Score in Event')
    ax.set_ylabel('AU')
    ax.set_title('Distribution of Highest Scoring Combination')

    # hi_score = np.sum(n_signal[x > 0.8]) / (np.sum(n_signal[x > 0.8]) + np.sum(n_bkgd[x > 0.8]))
    # ax.text(0.2, 0.5, f"Ratio of signal to sgnl+bkgd above 0.8 = {hi_score*100:.0f}%", transform=ax.transAxes)

    return fig, ax

def plot_combo_scores(combos, normalize=True):

    fig, ax = plt.subplots()

    if normalize:
        c, b, x = norm_hist(combos.scores_combo[combos.signal_mask])
        w, b, x = norm_hist(combos.scores_combo[~combos.signal_mask])
        ax.set_ylabel('AU')
    else:
        c, b = np.histogram(combos.scores_combo[combos.signal_mask], bins=100)
        w, b = np.histogram(combos.scores_combo[~combos.signal_mask], bins=100)
        x = (b[1:] + b[:-1]) / 2
        ax.set_ylabel('Entries Per Bin')

    hist(ax, x, weights=c, bins=b, label='Correct 6-jet combo')
    hist(ax, x, weights=w, bins=b, label='Incorrect 6-jet combo')
    ax.legend(fontsize='small', loc=9)

    ax.set_xlabel('Assigned Score')
    

    textstr = f'Entries = {len(combos.scores_combo)}'
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax.text(0.8, 1.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    return fig, ax

def plot_combo_score_v_mass(combos):
    combo_m = ak.to_numpy(combos.sixjet_p4.mass)
    combo_m = combo_m.reshape(combo_m.shape[0])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))

    fig.suptitle("Combination Analysis")

    ax[0].set_title("Correct Combos")
    ax[1].set_title("Incorrect Combos")

    n, xedges, yedges, ims = hist2d(ax[0], combo_m[combos.signal_mask], combos.scores_combo[combos.signal_mask], xbins=np.linspace(400,900,100))
    n, xedges, yedges, imb = hist2d(ax[1], combo_m[~combos.signal_mask], combos.scores_combo[~combos.signal_mask], xbins=np.linspace(0,2000,100))

    plt.colorbar(ims, ax=ax[0])
    plt.colorbar(imb, ax=ax[1])

    ax[0].set_xlabel('Invariant Mass of 6-jet System [GeV]')
    ax[1].set_xlabel('Invariant Mass of 6-jet System [GeV]')
    ax[0].set_ylabel('Assigned Score')
    ax[1].set_ylabel('Assigned Score')

    plt.tight_layout()
    return fig, ax

def plot_highest_score_v_mass(combos):
    combo_m = ak.to_numpy(combos.sixjet_p4.mass)
    combo_m = combo_m.reshape(combo_m.shape[0])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))

    fig.suptitle("Combination Analysis")

    ax[0].set_title("Correct Combos")
    ax[1].set_title("Incorrect Combos")
    
    signal_mask = combos.signal_evt_mask
    high_score_mask = combos.high_score_combo_mask

    n, xedges, yedges, ims = hist2d(ax[0], combo_m[high_score_mask][signal_mask], combos.scores_combo[high_score_mask][signal_mask], xbins=np.linspace(400,900,100))
    n, xedges, yedges, imb = hist2d(ax[1], combo_m[high_score_mask][~signal_mask], combos.scores_combo[high_score_mask][~signal_mask], xbins=np.linspace(0,2000,100))

    plt.colorbar(ims, ax=ax[0])
    plt.colorbar(imb, ax=ax[1])

    ax[0].set_xlabel('Invariant Mass of 6-jet System [GeV]')
    ax[1].set_xlabel('Invariant Mass of 6-jet System [GeV]')
    ax[0].set_ylabel('Assigned Score')
    ax[1].set_ylabel('Assigned Score')

    plt.tight_layout()
    return fig, ax

class TRSM():

    # b_mass = Particle.from_pdgid(5).mass / 1000 # Convert from MeV to GeV

    def open_file(self, fileName, treeName='sixBtree'):

        tree = uproot.open(fileName + ":" + treeName)

        self.tree = tree
        
        try:
            self.jet_idx  = tree['jet_idx'].array(library='np')
            self.n_sixb   = tree['n_sixb'].array(library='np')
        except:
            self.jet_idx  = tree['jet_signalId'].array(library='np')
            self.n_sixb   = tree['nfound_sixb'].array(library='np')
            

        self.n_jet    = tree['n_jet'].array(library='np')
        self.jet_pt   = tree['jet_pt'].array(library='np')
        self.jet_eta  = tree['jet_eta'].array(library='np')
        self.jet_phi  = tree['jet_phi'].array(library='np')
        self.jet_m    = tree['jet_m'].array(library='np')
        self.jet_btag = tree['jet_btag'].array(library='np')
        self.jet_qgl  = tree['jet_qgl'].array(library='np')
        # self.jet_partonFlav = tree['jet_partonFlav'].array(library='np')
        # self.jet_hadronFlav = tree['jet_hadronFlav'].array(library='np')

        try:
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
        except:
            self.HX_b1_pt  = tree['HX_b1_ptRegressed'].array(library='np')
            self.HX_b2_pt  = tree['HX_b2_ptRegressed'].array(library='np')
            self.HX_b1_eta = tree['HX_b1_eta'].array(library='np')
            self.HX_b2_eta = tree['HX_b2_eta'].array(library='np')
            self.HX_b1_phi = tree['HX_b1_phi'].array(library='np')
            self.HX_b2_phi = tree['HX_b2_phi'].array(library='np')
            self.HX_b1_m   = tree['HX_b1_m'].array(library='np')
            self.HX_b2_m   = tree['HX_b2_m'].array(library='np')
            self.H1_b1_pt  = tree['HY1_b1_ptRegressed'].array(library='np')
            self.H1_b2_pt  = tree['HY1_b2_ptRegressed'].array(library='np')
            self.H1_b1_eta = tree['HY1_b1_eta'].array(library='np')
            self.H1_b2_eta = tree['HY1_b2_eta'].array(library='np')
            self.H1_b1_phi = tree['HY1_b1_phi'].array(library='np')
            self.H1_b2_phi = tree['HY1_b2_phi'].array(library='np')
            self.H1_b1_m   = tree['HY1_b1_m'].array(library='np')
            self.H1_b2_m   = tree['HY1_b2_m'].array(library='np')
            self.H2_b1_pt  = tree['HY2_b1_ptRegressed'].array(library='np')
            self.H2_b2_pt  = tree['HY2_b2_ptRegressed'].array(library='np')
            self.H2_b1_eta = tree['HY2_b1_eta'].array(library='np')
            self.H2_b2_eta = tree['HY2_b2_eta'].array(library='np')
            self.H2_b1_phi = tree['HY2_b1_phi'].array(library='np')
            self.H2_b2_phi = tree['HY2_b2_phi'].array(library='np')
            self.H2_b1_m   = tree['HY2_b1_m'].array(library='np')
            self.H2_b2_m   = tree['HY2_b2_m'].array(library='np')

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
    def __init__(self, filename, treename='sixBtree'):

        self.tree = uproot.open(f"{filename}:{treename}")

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

        combo_pt  = ak.to_numpy(combo_p4.pt)
        combo_eta = ak.to_numpy(combo_p4.eta)
        combo_phi = ak.to_numpy(combo_p4.phi)
        boosted_pt = []

        for boost in boosted:
            boosted_pt.append(boost.pt)
        boosted_pt = np.asarray(boosted_pt)

        inputs = np.column_stack((combo_pt, combo_eta, combo_phi, combo_btag, boosted_pt[0,:], boosted_pt[1,:], boosted_pt[2,:], boosted_pt[3,:], boosted_pt[4,:], boosted_pt[5,:]))

        return inputs

class training_6j(TRSM):
    ## Build p4 for all events
    @jit(forceobj=True)
    def __init__(self, trsm):

        self.trsm = trsm

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
        # jet_partonFlav = trsm.jet_partonFlav
        # jet_hadronFlav = trsm.jet_hadronFlav
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

            assert (len(sixb_pt) == 6)
            assert (len(non_sixb_pt) == 6)

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

        self.signal_evt_mask_p4, self.sgnl_boosted = get_6jet_p4(self.sgnl_p4)
        self.bkgd_evt_p4, self.bkgd_boosted = get_6jet_p4(self.bkgd_p4)

        # self.signal_idx = np.array((signal_idx))
        self.signal_btag = np.array((signal_btag))
        self.bkgd_btag = np.array((background_btag))

        # self.combo_mask = np.asarray(combo_mask)
        sgnl_inputs = self.trsm.construct_6j_features(self.sgnl_p4, self.signal_btag, self.sgnl_boosted)
        bkgd_inputs = self.trsm.construct_6j_features(self.bkgd_p4, self.bkgd_btag, self.bkgd_boosted)

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
        # jet_partonFlav = trsm.jet_partonFlav
        # jet_hadronFlav = trsm.jet_hadronFlav
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
    
    def __init__(self, trsm, n=6, k=6, fast_build=False, tag=False):
        """
        Builds combinations of six jets from events in the TRSM object.

        If n is an integer, only events with n jets will be considered. Otherwise, all events will be considered.
        """

        if fast_build: 
            assert tag, print("Include tag for quick building!")

        if fast_build: self.quick_build(trsm, tag)
        else: self.build_combos(trsm, n, k)

    @jit(forceobj=True)
    def build_combos(self, trsm, n=6, k=6):

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
        # jet_partonFlav = trsm.jet_partonFlav
        # jet_hadronFlav = trsm.jet_hadronFlav

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

        n_evt_mask = [] # each event makes several combos - keep track of evt # for each combo

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

        self.nevents = np.arange(trsm.nevents)
        if not n:
            # if n = False, find combos for all events
            pass
        else:
            # if n is specified as an int, find combos for events with n jets
            self.nevents = self.nevents[n_jet == n]
        
        for evt in tqdm(self.nevents):

            evt_n          = np.arange(n_jet[evt])
            evt_idx        = jet_idx[evt]
            evt_pt         = jet_pt[evt]
            evt_eta        = jet_eta[evt]
            evt_phi        = jet_phi[evt]
            evt_m          = jet_m[evt]
            evt_btag       = jet_btag[evt]
            evt_qgl        = jet_qgl[evt]
            # evt_partonFlav = jet_partonFlav[evt]
            # evt_hadronFlav = jet_hadronFlav[evt]

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
            n_bkgd = np.sum(background_mask)
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
                
                n_evt_mask.append(evt)
                evt_idx_combos.append(idx)

                # Check if the current combo idx is the same as the signal idx
                if -1 not in idx:
                    sgnl_mask.append(True)
                else:
                    # Save location of incorrect jets in evt arrays
                    bkgd_jets_in_combo = [ind for ind, sig_idx in zip(jet_ind,idx) if sig_idx == -1]
                    signal_jets_in_combo = [ind for ind, sig_idx in zip(jet_ind,idx) if sig_idx > -1]

                    assert(np.all(np.isin(bkgd_jets_in_combo, background_mask)))
                        
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

        combo_p4 = vector.obj(pt=ak.Array(combo_pt), eta=ak.Array(combo_eta), phi=ak.Array(combo_phi), m=ak.Array(combo_m))

        self.signal_mask = np.asarray(sgnl_mask)
        self.n_evt_mask = np.asarray(n_evt_mask)

        self.combo_jet_binary_mask = np.asarray(combo_jet_binary_mask)
        self.combo_H_binary_mask = np.asarray(combo_H_binary_mask)

        self.idx_combos = np.asarray(evt_idx_combos)
        self.btag_combos = np.asarray(combos_btag)

        self.sixjet_p4, boosted = get_6jet_p4(combos_builder)
        self.combo_features = self.construct_6j_features(combos_builder, combos_btag, boosted)

        print("Total events chosen:",counter)

    @jit(forceobj=True)
    def quick_build(self, trsm, tag):

        jet_pt = trsm.jet_pt
        jet_eta = trsm.jet_eta
        jet_phi = trsm.jet_phi
        jet_m = trsm.jet_m
        jet_btag = trsm.jet_btag
        jet_idx = trsm.jet_idx

        signal_counter = 0
        signal_highest = 0

        for evt in tqdm(range(trsm.nevents)):
            signal_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj > -1]
            signal_mask = np.array((signal_mask))

            background_mask = [i for i,obj in enumerate(jet_idx[evt]) if obj == -1]
            background_mask = np.array((background_mask))

            pt_combos   = list(itertools.combinations(jet_pt[evt], 6))
            eta_combos  = list(itertools.combinations(jet_eta[evt], 6))
            phi_combos  = list(itertools.combinations(jet_phi[evt], 6))
            m_combos    = list(itertools.combinations(jet_m[evt], 6))
            btag_combos = list(itertools.combinations(jet_btag[evt], 6))
            idx_combos  = list(itertools.combinations(jet_idx[evt], 6))

            jet0_p4 = ak.Array({"pt":pt_combos.copy()[0], "eta":eta_combos.copy()[0], "phi":phi_combos.copy()[0], "m":m_combos.copy()[0]})
            jet1_p4 = ak.Array({"pt":pt_combos.copy()[1], "eta":eta_combos.copy()[1], "phi":phi_combos.copy()[1], "m":m_combos.copy()[1]})
            jet2_p4 = ak.Array({"pt":pt_combos.copy()[2], "eta":eta_combos.copy()[2], "phi":phi_combos.copy()[2], "m":m_combos.copy()[2]})
            jet3_p4 = ak.Array({"pt":pt_combos.copy()[3], "eta":eta_combos.copy()[3], "phi":phi_combos.copy()[3], "m":m_combos.copy()[3]})
            jet4_p4 = ak.Array({"pt":pt_combos.copy()[4], "eta":eta_combos.copy()[4], "phi":phi_combos.copy()[4], "m":m_combos.copy()[4]})
            jet5_p4 = ak.Array({"pt":pt_combos.copy()[5], "eta":eta_combos.copy()[5], "phi":phi_combos.copy()[5], "m":m_combos.copy()[5]})

            evt_p4 = jet0_p4 + jet1_p4 + jet2_p4 + jet3_p4 + jet4_p4 + jet5_p4

            highest = 0
            false_pos = 0

            signal_score = -1
            wrong_score = -1

            for ii in range(len(pt_combos)):
                pt = pt_combos[ii]
                eta = eta_combos[ii]
                phi = phi_combos[ii]
                m = m_combos[ii]
                btag = btag_combos[ii]
                idx = idx_combos[ii]

                # sort by pT
                sort_mask = np.array((np.argsort(pt)[::-1]))
                # sort by btag
                # sort_mask = np.array((np.argsort(btag)[::-1]))

                arr_eta  = np.asarray(eta)[sort_mask]
                arr_phi  = np.asarray(phi)[sort_mask]
                arr_m    = np.asarray(m)[sort_mask]
                btag = np.asarray(btag)[sort_mask]
                # pt must be sorted last because it is used to sort everything else
                arr_pt   = np.sort(np.asarray(pt))[::-1]

                print(jet0_p4[ii])

                boosted_pt = np.array([
                    jet0_p4[ii].boost_p4(evt_p4[ii]),
                    jet1_p4[ii].boost_p4(evt_p4[ii]),
                    jet2_p4[ii].boost_p4(evt_p4[ii]),
                    jet3_p4[ii].boost_p4(evt_p4[ii]),
                    jet4_p4[ii].boost_p4(evt_p4[ii]),
                    jet5_p4[ii].boost_p4(evt_p4[ii]),
                    ])

                inputs = np.concatenate((arr_pt, arr_eta, arr_phi, btag, boosted_pt))
                inputs = inputs.reshape(1, 30)
                scaler, model = load_model('../', tag)

                score = model.predict(inputs)[0][0]

                if score > highest:
                    highest = score

                if -1 not in idx: 
                    signal_counter += 1
                    signal_score = score
                else:
                    wrong_score = score

            if signal_score == highest:
                signal_highest += 1
            if wrong_score == highest:
                if wrong_score > 0.8: false_pos += 1

        ic(signal_counter)
        ic(signal_highest)
        ic(false_pos)

    @jit(forceobj=True)
    def select_highest_scoring_combos(self):

        highest_scoring_combo_mask = []
        highest_score = []
        highest_score_nsignal = []

        signal_evt_mask = []
        signal_high_score = []

        n_signal = []

        count = 0
        for evt in tqdm(self.nevents):
            temp_scores = self.scores_combo[self.n_evt_mask == evt]
            ind_below = np.sum(self.n_evt_mask < evt)
            temp_loc = np.argmax(temp_scores)

            max_ind = ind_below + temp_loc

            n_sig = np.sum(self.idx_combos[self.n_evt_mask == evt] > -1, axis=1)
            n_signal.extend(n_sig)

            highest_score_nsignal.append(n_sig[temp_loc])

            if 6 in n_sig: signal_evt_mask.append(True)
            else: signal_evt_mask.append(False)

            if self.signal_mask[max_ind] == True: signal_high_score.append(True)
            else: signal_high_score.append(False)

            highest_scoring_combo_mask.append(max_ind)
            highest_score.append(np.max(temp_scores))

        self.high_score_combo_mask = np.asarray(highest_scoring_combo_mask)
        self.evt_highest_score = np.asarray(highest_score)

        self.signal_evt_mask = np.asarray(signal_evt_mask)
        self.signal_high_score = np.asarray(signal_high_score) 

        self.highest_score_nsignal = np.asarray(highest_score_nsignal)
        self.n_signal = np.asarray(highest_score_nsignal)
            
    def apply_6j_model(self, tag, location='../'):

        print("Applying 6b Model to combinations. Please wait.")
        scaler, model = load_model(location, tag)
        test_features = scaler.transform(self.combo_features)
        self.scores_combo = model.predict(test_features)[:,0]

        print("Selecting highest scoring combination from each event.")
        self.select_highest_scoring_combos()

    @jit(forceobj=True)
    def create_pairs(self, tag, score_cut=0):
        """
        Pairs of jets are generated from the combinations of six jets that were assigned the highest 6b-model score in the event.
        """

        jet1 = ak.ArrayBuilder()
        jet2 = ak.ArrayBuilder()

        combo_mask = self.high_score_combo_mask
        score_mask = self.evt_highest_score > score_cut

        pt_arr   = self.combo_p4.pt[combo_mask]
        eta_arr  = self.combo_p4.eta[combo_mask]
        phi_arr  = self.combo_p4.phi[combo_mask]
        m_arr    = self.combo_p4.m[combo_mask]
        btag_arr = self.btag_combos[combo_mask]
        idx_arr  = self.idx_combos[combo_mask]

        jet1_btag = []
        jet2_btag = []
        jet1_idx = []
        jet2_idx = []

        Higgs_mask = []
        pair_evt_mask = []

        for i, (pt, eta, phi, m, btag, idx) in tqdm(enumerate(zip(pt_arr, eta_arr, phi_arr, m_arr, btag_arr, idx_arr)), total=len(pt_arr)):

            pt   = ak.to_numpy(pt)
            eta  = ak.to_numpy(eta)
            phi  = ak.to_numpy(phi)
            m    = ak.to_numpy(m)
            btag = np.asarray(btag)
            idx  = np.asarray(idx)

            pt_combos   = list(itertools.combinations(pt, 2))
            eta_combos  = list(itertools.combinations(eta, 2))
            phi_combos  = list(itertools.combinations(phi, 2))
            m_combos    = list(itertools.combinations(m, 2))
            btag_combos = list(itertools.combinations(btag, 2))
            idx_combos = list(itertools.combinations(idx, 2))
            
            for pair_pt, pair_eta, pair_phi, pair_m, pair_btag, pair_idx in zip(pt_combos, eta_combos, phi_combos, m_combos, btag_combos, idx_combos):

                pair_evt_mask.append(combo_mask[i])

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
                jet1.field("pt"); jet1.real(pair_pt[0])
                jet1.field("eta"); jet1.real(pair_eta[0])
                jet1.field("phi"); jet1.real(pair_phi[0])
                jet1.field("m"); jet1.real(pair_m[0])
                # jet1.field("pt").real(pair_pt[0])
                # jet1.field("eta").real(pair_eta[0])
                # jet1.field("phi").real(pair_phi[0])
                # jet1.field("m").real(pair_m[0])
                jet1.end_record()
                jet1.end_list()

                jet2.begin_list()
                jet2.begin_record("Momentum4D")
                jet2.field("pt"); jet2.real(pair_pt[1])
                jet2.field("eta"); jet2.real(pair_eta[1])
                jet2.field("phi"); jet2.real(pair_phi[1])
                jet2.field("m"); jet2.real(pair_m[1])
                jet2.end_record()
                jet2.end_list()
            
        jet1 = jet1.snapshot()
        jet2 = jet2.snapshot()
        
        jet_pair = jet1 + jet2

        jet1_idx = np.asarray(jet1_idx)
        jet2_idx = np.asarray(jet2_idx)
        jet1_btag = np.asarray(jet1_btag)
        jet2_btag = np.asarray(jet2_btag)

        self.Higgs_mask = np.asarray(Higgs_mask)
        self.pair_evt_mask = np.asarray(pair_evt_mask)

        self.pair_features = np.column_stack((np.asarray(jet1.pt), np.asarray(jet2.pt), np.asarray(jet1.eta), np.asarray(jet2.eta), np.asarray(jet1.phi), np.asarray(jet2.phi), jet1_btag, jet2_btag, np.asarray(jet1.deltaR(jet2))))
        self.pair_target = np.asarray(Higgs_mask)

    def apply_2j_model(self, tag, location='../../2jet_classifier/'):

        print("Applying 6b Model to combinations. Please wait.")
        scaler, model = load_model(location, tag)
        test_features = scaler.transform(self.pair_features)
        self.scores_pairs = model.predict(test_features)

        self.select_highest_scoring_pairs()

    @jit(forceobj=True)
    def select_highest_scoring_pairs(self):

        high1_score_mask = []
        high2_score_mask = []
        high3_score_mask = []
        high1_score = []
        high2_score = []
        high3_score = []

        all3Higgs = []

        evt_3Higgs_mask = []

        count = 0
        for evt in tqdm(self.high_score_combo_mask):
            temp_scores = self.scores_pairs[self.pair_evt_mask == evt][:,0]
            ind_below = np.sum(self.pair_evt_mask < evt)

            max1_loc = np.argmax(temp_scores)
            max1 = temp_scores[max1_loc]
            temp_scores[max1_loc] = -1
            max2_loc = np.argmax(temp_scores)
            max2 = temp_scores[max2_loc]
            temp_scores[max2_loc] = -1
            max3_loc = np.argmax(temp_scores)
            max3 = temp_scores[max3_loc]

            max1_ind = ind_below + max1_loc
            max2_ind = ind_below + max2_loc
            max3_ind = ind_below + max3_loc

            high1_score_mask.append(max1_ind)
            high2_score_mask.append(max2_ind)
            high3_score_mask.append(max3_ind)

            high1_score.append(max1)
            high2_score.append(max2)
            high3_score.append(max3)

            Higgs_match_1 = self.Higgs_mask[max1_ind]
            Higgs_match_2 = self.Higgs_mask[max2_ind]
            Higgs_match_3 = self.Higgs_mask[max3_ind]

            Higgs_match = Higgs_match_1 & Higgs_match_2 & Higgs_match_3
            all3Higgs.append(Higgs_match)

            if Higgs_match: evt_3Higgs_mask.append(True)
            else: evt_3Higgs_mask.append(False)

        self.high1_score_mask = np.asarray(high1_score_mask)
        self.high2_score_mask = np.asarray(high2_score_mask)
        self.high3_score_mask = np.asarray(high3_score_mask)
        self.high1_score = np.asarray(high1_score)
        self.high2_score = np.asarray(high2_score)
        self.high3_score = np.asarray(high3_score)

        self.all3Higgs_mask = np.asarray(all3Higgs)

    @jit(forceobj=True)
    def get_stats(self, score_cut = 0):
        # what percentage of events contain signal jets
        percent_signal = np.sum(self.signal_evt_mask)/len(self.signal_evt_mask) * 100
        
        # what percentage of signal events are the highest scoring
        percent_signal_highest_score = np.sum(self.signal_high_score)/np.sum(self.signal_evt_mask) * 100
        
        # score cut removes this percentage
        percent_removal = (1 - np.sum(self.evt_highest_score > score_cut)/len(self.evt_highest_score)) * 100
        
        # of events with highest score above threshold, what percentage of those contain signal combo?
        percent_high_score_are_signal = np.sum(np.logical_and(self.signal_evt_mask, self.evt_highest_score > score_cut))/np.sum(self.evt_highest_score > score_cut) * 100
        
        # of events with highest score above threshold, what percentage of those are signal combos with the highest score in the event?
        percent_high_score_signal_with_highest_score = np.sum(np.logical_and(self.signal_high_score, self.evt_highest_score > score_cut))/np.sum(self.evt_highest_score > score_cut) * 100
        
        print(round(percent_signal), "% of all events contain signal combos")
        print(round(percent_signal_highest_score), "% of events containing signal combo assigned highest score to signal combo")
        print(round(percent_removal), "% of events removed by applying score cut", score_cut)
        print(round(percent_high_score_are_signal), "% of events above score cut", score_cut, "contain signal combo")
        print(round(percent_high_score_signal_with_highest_score), "% of events above score cut", score_cut, "contain signal combo assigned highest score")
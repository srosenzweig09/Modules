# Common kinematic calculations performed in the analysis.
# eta*, phi* = calcStar()

import numpy as np

z_ME2 = 808 # cm

def calcStar(eta, phi, vx, vy, vz):
    """Calculate etastar and phistar for a gen particle."""
    
    r = np.where(eta > 0, (z_ME2 - abs(vz))/abs(np.sinh(eta)), abs(-z_ME2 - vz)/abs(np.sinh(eta)))
    xStar = vx + r * np.cos(phi)
    yStar = vy + r * np.sin(phi)
    rStar = np.sqrt(xStar*xStar + yStar*yStar)
    
    etaStar = np.arcsinh(z_ME2/rStar) * (eta/abs(eta))
    phiStar = []

    for x,y in zip(xStar, yStar):
        if x >= 0:
            phiStar.append(np.arctan(y/x))
        elif y >= 0 and x < 0:
            phiStar.append(np.pi + np.arctan(y/x))
        elif y <= 0 and x < 0:
            phiStar.append(np.arctan(y/x) - np.pi)

    return etaStar, phiStar

def calc_d0(pt, phi, vx, vy, q, B=3.811):
    R = -pt/(q*0.003*B) # [cm]
    xc = vx - R*np.sin(phi)
    yc = vy + R*np.cos(phi)
    d0 = R - np.sign(R)*np.sqrt(xc**2 + yc**2)
    
    return d0

def calcDeltaR(eta1, eta2, phi1, phi2):
    deltaEta = eta1 - eta2
    deltaPhi = phi1 - phi2
    # Add and subtract 2np.pi to values below and above -np.pi and np.pi, respectively.
    # This limits the range of deltaPhi to (-np.pi, np.pi).
    deltaPhi = np.where(deltaPhi < -np.pi, deltaPhi + 2*np.pi, deltaPhi)
    deltaPhi = np.where(deltaPhi > +np.pi, deltaPhi - 2*np.pi, deltaPhi)
    deltaR = np.sqrt(deltaEta**2 + deltaPhi**2)
    
    return deltaR

def convert_emtf(local_pt,  local_eta, local_phi, flatten=False):
    """Convert BDT-assigned EMTF variables."""

    global_pt  = local_pt*0.5
    global_eta = local_eta*0.010875
    global_phi = local_phi / 576 *2*np.pi - np.pi*15/180
    global_phi = np.where(global_phi > np.pi, global_phi-2*np.pi, global_phi)

    if flatten:
        global_pt  = global_pt.flatten()
        global_eta = global_eta.flatten()
        global_phi = global_phi.flatten()

    return global_pt, global_eta, global_phi
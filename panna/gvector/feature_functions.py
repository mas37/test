import numpy as np


def _cutoff(r_ij, r_c):
    return 0.5 * (1.0 + np.cos(np.pi * r_ij / r_c))


def G_radial(r_ij, R_s, eta_rad, Rc_rad):
    G = np.exp(-eta_rad * (r_ij - R_s)**2) * _cutoff(r_ij, Rc_rad)
    return G


def G_angular_mBP(r_ij, r_ik, theta_ijk, R_p, theta_s, eta_ang, zeta, Rc_ang):
    """
    Just the function to calculate G_radial element
    """
    G = np.exp(-eta_ang * (0.5 * (r_ij + r_ik) - R_p)**2)
    G = G * 2.0 * (0.5 + np.cos(theta_ijk - theta_s) * 0.5)**zeta
    G = G * _cutoff(r_ij, Rc_ang)
    G = G * _cutoff(r_ik, Rc_ang)

    return G


def G_angular_BP(r_ij, r_ik, r_jk, theta_ijk, eta_ang, zeta, lamb, Rc_ang):
    G = np.exp(-eta_ang * (r_ij**2 + r_ik**2 + r_jk**2))
    G = G * _cutoff(r_ij, Rc_ang)
    G = G * _cutoff(r_ik, Rc_ang)
    G = G * _cutoff(r_jk, Rc_ang)
    G = G * 2.0 * (0.5 + lamb * np.cos(theta_ijk) * 0.5)**zeta
    return G

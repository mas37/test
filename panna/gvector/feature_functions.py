###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import numpy as np


def _cutoff(r_ij, r_c):
    return 0.5 * (1.0 + np.cos(np.pi * r_ij / r_c))


def G_radial(r_ij, R_s, eta_rad, Rc_rad):
    g_rad = np.exp(-eta_rad * (r_ij - R_s)**2) * _cutoff(r_ij, Rc_rad)
    return g_rad


def G_angular_mBP(r_ij, r_ik, theta_ijk, R_p, theta_s, eta_ang, zeta, Rc_ang):
    """
    Just the function to calculate G_radial element
    """
    eps = 1e-2
    G = np.exp(-eta_ang * (0.5 * (r_ij + r_ik) - R_p)**2)
    corr = np.cos(theta_ijk) * np.cos(theta_s) + \
        np.sqrt(1 - np.cos(theta_ijk)**2 + eps * np.sin(theta_s)**2) *\
        np.sin(theta_s)
    norm = (2 / (1+np.sqrt(1+eps *np.sin(theta_s)**2)))**zeta
    G = G * 2.0 * norm * (0.5 + corr * 0.5)**zeta
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

###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################

"""
This module contains the routines to calculate derivatives of the g's
"""
import numpy as np
import itertools
from .pbc import replicas_max_idx


def _G_radial_mBP_dGvect(rdiff, index, eta_rad, Rs0_rad, Rsst_rad, Rc_rad):
    """ The function to calculate G_radial element with
    certain R_s index, plus the factor for the derivatives
    """
    Gauss = np.exp(-eta_rad * (rdiff - Rs0_rad - index * Rsst_rad)**2)
    f_c = 0.5 * (1.0 + np.cos(np.pi * rdiff / Rc_rad))
    G = Gauss * f_c
    dG = (( 0.5 * np.pi * np.sin(np.pi * rdiff / Rc_rad) / Rc_rad ) +
        ( 2.0 * eta_rad * f_c * (rdiff - Rs0_rad - index * Rsst_rad) ) ) * \
        Gauss / rdiff
    return (G, dG)


def _G_angular_mBP_dGvect(rdiff1, rdiff2, costheta, indexR, cosThi, sinThi,
                          eta_ang, Rs0_ang, Rsst_ang, Thetasst, zeta, Rc_ang):
    """
    The function to calculate G_radial element with certain R_s index,
    plus the factors to compute the derivatives.
    Cosine centers are shifted by half, differently than mBP reference.
    The formula is also corrected with epsilon to fix for the discontinuous derivative
    See PANNA paper.
    """
    # introduce this to fix for discontinuous derivatives
    epscorr = 1.0e-2
    norm = ((1. + np.sqrt(1. + epscorr * sinThi**2))/2.)**zeta
    sintheta = np.sqrt(1.0 - costheta**2 + epscorr * sinThi**2)

    R_cent = (0.5 * (rdiff1 + rdiff2) - Rs0_ang - indexR * Rsst_ang)
    Gauss = 2.0 * np.exp(-eta_ang * R_cent**2)

    # normalized angular part
    f_c_th = 0.5 * (1.0 + cosThi * costheta + sinThi * sintheta)
    f_c_1 = 0.5 * (1.0 + np.cos(np.pi * rdiff1 / Rc_ang))
    f_c_2 = 0.5 * (1.0 + np.cos(np.pi * rdiff2 / Rc_ang))

    tmp0 = Gauss * f_c_th**(zeta - 1)/norm
    tmp1 = tmp0 * f_c_1 * f_c_2

    tmp2 = 0.5 * zeta * tmp1 * (cosThi - sinThi * costheta / sintheta)

    #derivatives of cutoff
    tmp3 = 0.5 * np.pi * tmp0 * f_c_th / Rc_ang
    G = tmp1 * f_c_th
    dG1 = -(eta_ang * R_cent * G + tmp2 * costheta / rdiff1 +
            tmp3 * f_c_2 * np.sin(np.pi * rdiff1 / Rc_ang)) / rdiff1
    dG2 = tmp2 / (rdiff1 * rdiff2)
    dG3 = -(eta_ang * R_cent * G + tmp2 * costheta / rdiff2 +
            tmp3 * f_c_1 * np.sin(np.pi * rdiff2 / Rc_ang)) / rdiff2

    return (G, dG1, dG2, dG3)


def calculate_Gvector_dGvect(key, positions, species, lattice_vectors,
                             Nspecies, eta_rad, Rc_rad, Rs0_rad, RsN_rad,
                             Rsst_rad, eta_ang, Rc_ang, Rs0_ang, Rsst_ang,
                             RsN_ang, zeta, ThetasN, **kwargs):
    """Calculate the gvector and its derivative based on given parameters

    Args:
        key: key of the simulation
        positions: list of atomic positions

        species: list of species as name

        lattice_vectors: list [a1, a2, a3]

        For all the other parameters refer to the article
    kwargs:
        pbc: Normally pbc are recovered from file,
             if in the lattice_vectors a direction is set to zero
             then no pbc is applied in that direction.
             This argument allow you to turn off specific directions
             by passing an array of 3 logical value (one for each
             direction). False value turn off that specific direction
             eg. pbc = [True, False, True] => pbc along a1 and a3

    Return:
        (key, gvector as list)
        gvector is a matrix (natoms * gsize)

    note:
        All kind of PBC : 1D, 2D and 3D are implemented..
        Check the function "replicas_max_idx" below

    """

    # extrapolation of useful quantities
    Thetasst = np.pi / ThetasN
    G_size = int(Nspecies * RsN_rad +
                 0.5 * Nspecies * (Nspecies + 1) * RsN_ang * ThetasN)
    number_of_atoms = len(positions)
    cThi = []
    sThi = []
    for Thi in range(ThetasN):
        cThi.append(np.cos((Thi + 0.5) * Thetasst))
        sThi.append(np.sin((Thi + 0.5) * Thetasst))

    # start of the algorithm
    Gvector = np.zeros((number_of_atoms, G_size), dtype=float)
    dGvector = np.zeros((number_of_atoms, G_size, number_of_atoms, 3),
                        dtype=float)
    positions = np.asarray(positions)
    lattice_vectors = np.asarray(lattice_vectors)

    if 'pbc' in kwargs:
        max_indices = replicas_max_idx(
            lattice_vectors, max(Rc_rad, Rc_ang), pbc=kwargs['pbc'])
    else:
        max_indices = replicas_max_idx(lattice_vectors, max(Rc_rad, Rc_ang))

    l_max, m_max, n_max = max_indices

    l_list = range(-l_max, l_max + 1)
    m_list = range(-m_max, m_max + 1)
    n_list = range(-n_max, n_max + 1)

    # radial: i==central atom + j==second atom
    # angular: i==central atom + j==second atom +k==third atom
    for i, kind_i in enumerate(species):
        pos_i = positions[i]
        # list of neighbors per kind,
        # [
        #   [[rij, (xi, yi, zi)],[].....],
        #   [[rij, (xi, yi, zi)],[].....],
        #   .....,
        # ]
        neighbors = []
        # radial part starts here
        for kind_j_selector in range(Nspecies):
            R_ij_neighbors = []
            # loop over atoms of j in the unit cell
            for j, kind_j in enumerate(species):
                if (kind_j == kind_j_selector):
                    # compute the contributions of type_j
                    # to radial part of type_i;
                    # loop over all cells around
                    for l, m, n in itertools.product(l_list, m_list, n_list):
                        pos_j = positions[j] +\
                            l * lattice_vectors[0] +\
                            m * lattice_vectors[1] +\
                            n * lattice_vectors[2]
                        Rij_v = pos_j - pos_i
                        Rij = np.linalg.norm(Rij_v)
                        # This part create neighbor lists for each atom type_i
                        # Which are needed for the angular part
                        if (Rc_ang > Rc_rad):
                            if (Rij < Rc_ang and Rij > 1e-8):
                                R_ij_neighbors.append([j, Rij_v, Rij])
                                if (Rij < Rc_rad):
                                    # loop over radial centers
                                    for idx_rs_rad in range(RsN_rad):
                                        Gv, Gvd = _G_radial_mBP_dGvect(
                                            Rij, idx_rs_rad, eta_rad, Rs0_rad,
                                            Rsst_rad, Rc_rad)
                                        Gvector[i][kind_j * RsN_rad +
                                                   idx_rs_rad] += Gv
                                        dGvector[i][kind_j * RsN_rad +
                                                   idx_rs_rad][i] +=\
                                                   Gvd*Rij_v
                                        if (i != j):
                                            dGvector[i][kind_j * RsN_rad +
                                                       idx_rs_rad][j] -=\
                                                       Gvd*Rij_v

                        if (Rc_ang <= Rc_rad):
                            if (Rij < Rc_rad and Rij > 1e-8):
                                # loop over radial centers
                                for idx_rs_rad in range(RsN_rad):
                                    Gv, Gvd = _G_radial_mBP_dGvect(
                                        Rij, idx_rs_rad, eta_rad, Rs0_rad,
                                        Rsst_rad, Rc_rad)
                                    Gvector[i][kind_j * RsN_rad +
                                               idx_rs_rad] += Gv
                                    dGvector[i][kind_j * RsN_rad +
                                               idx_rs_rad][i] +=\
                                               Gvd*Rij_v
                                    if (i != j):
                                        dGvector[i][kind_j * RsN_rad +
                                                   idx_rs_rad][j] -=\
                                                   Gvd*Rij_v
                                if (Rij < Rc_ang):
                                    R_ij_neighbors.append([j, Rij_v, Rij])

            neighbors.append(R_ij_neighbors)

        for kind_j, kind_j_list in enumerate(neighbors):
            # loop over all positions associated to type_j atoms
            for j, Rij_v, Rij in kind_j_list:
                # third species
                for kind_k in range(kind_j, Nspecies):
                    if (kind_k != kind_j):
                        prefactor = 1.0
                    else:
                        prefactor = 0.5
                    kind_k_list = neighbors[kind_k]
                    temp_ind = int(RsN_rad * Nspecies +
                                   (kind_j * (Nspecies - (kind_j + 1) / 2) +
                                    kind_k) * RsN_ang * ThetasN)

                    for k, Rik_v, Rik in kind_k_list:
                        cos_theta_ijk = np.dot(Rij_v, Rik_v) / (Rij * Rik)
                        # for some numerical reasons,
                        # cos_theta_ijk may be slightly
                        # greater/less than 1.0/-1.0
                        if (cos_theta_ijk > 1.0):
                            cos_theta_ijk = np.float(1.0)
                        if (cos_theta_ijk < -1.0):
                            cos_theta_ijk = np.float(-1.0)

                        if (np.abs(Rij - Rik) > 1e-5
                                or cos_theta_ijk < 0.99999):

                            # loop over angular centers
                            for Rsi in range(RsN_ang):
                                # loop over the angles
                                for Thi in range(ThetasN):
                                    Gv, Gvd1, Gvd2, Gvd3 = \
                                    _G_angular_mBP_dGvect(Rij,
                                                          Rik,
                                                          cos_theta_ijk,
                                                          Rsi,
                                                          cThi[Thi],
                                                          sThi[Thi],
                                                          eta_ang,
                                                          Rs0_ang,
                                                          Rsst_ang,
                                                          Thetasst,
                                                          zeta,
                                                          Rc_ang)
                                    Gvector[i][temp_ind + Rsi * ThetasN +
                                               Thi] += Gv * prefactor
                                    dGdxj = Gvd1 * Rij_v + Gvd2 * Rik_v
                                    dGdxk = Gvd2 * Rij_v + Gvd3 * Rik_v
                                    dGvector[i][temp_ind + Rsi * ThetasN +
                                                Thi][j] += dGdxj * prefactor
                                    dGvector[i][temp_ind + Rsi * ThetasN +
                                                Thi][k] += dGdxk * prefactor
                                    dGvector[i][
                                        temp_ind + Rsi * ThetasN +
                                        Thi][i] -= (dGdxj + dGdxk) * prefactor

    return (key, Gvector, dGvector)

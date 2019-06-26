###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
"""
This module contains the routines to calculate the gvector as
defined in 10.1103/PhysRevLetter.98.146401
"""
import numpy as np
import itertools
from .pbc import replicas_max_idx


def _G_radial_BP(rdiff, index, eta_rad, Rs0_rad, Rsst_rad, Rc_rad):
    """ The function to calculate G_radial element with
    certain R_s index
    PROPhet G2 function
    send here the element of eta_rad list
    """
    G = np.exp(- eta_rad * (rdiff - Rs0_rad - index * Rsst_rad)**2) * \
        0.5 * (1.0 + np.cos(np.pi * rdiff / Rc_rad))
    return G


def _G_angular_BP(rdiff1, rdiff2, theta, eta_ang, zeta, Rc_ang):
    """
    PROPhet G4 function
    send here the elements of eta_ang and zeta list
    """
    G = 2.0 * \
        np.exp(-eta_ang *(rdiff1**2 + rdiff2**2)) * \
        (0.5 + np.cos(theta) * 0.5)**zeta

    G *= 0.25 * (1.0 + np.cos(np.pi * rdiff1 / Rc_ang)) * \
        (1.0 + np.cos(np.pi * rdiff2 / Rc_ang))
    return G


def calculate_Gvector_BP(key, positions, species, lattice_vectors, Nspecies,
                         eta_rad, Rc_rad, Rs0_rad, RsN_rad, Rsst_rad, eta_ang,
                         Rc_ang, zeta, ThetasN, **kwargs):
    """Calculate the gvector based on given parameters, using list

    Args:
        key: key of the simulation
        positions: list of atomic positions

        # species: List of distinct species considered in the system no way man
        # list_at_species_per_cell: List of atomic species chemical
        #                          symbols in unit cell
        # this two must be exchanged

        species: list of species as name

        # species must be moved to another name...

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

    #emine - new Gsize is smaller
    G_size = int(Nspecies * RsN_rad * len(eta_rad) +\
                  0.5 * Nspecies * (Nspecies + 1) * len(eta_ang) * len(zeta))
    number_of_atoms = len(positions)

    # start of the algorithm
    Gvector = np.zeros((number_of_atoms, G_size), dtype=float)
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
                        Rij = np.linalg.norm(pos_j - pos_i)
                        # This part create neighbor lists for each atom type_i
                        # Which are needed for the angular part
                        if (Rc_ang > Rc_rad):
                            if (Rij < Rc_ang and Rij > 1e-8):
                                R_ij_neighbors.append([Rij, pos_j])
                                if (Rij < Rc_rad):
                                    # loop over radial centers
                                    for idx_rs_rad in range(RsN_rad):
                                        for idx_eta_rad, eta in enumerate(
                                                eta_rad):
                                            Gvector[i][kind_j * (RsN_rad + len(eta_rad)) +
                                                       idx_rs_rad + idx_eta_rad] +=\
                                                _G_radial_BP(
                                                Rij, idx_rs_rad, eta,
                                                Rs0_rad, Rsst_rad, Rc_rad)
                        if (Rc_ang <= Rc_rad):
                            if (Rij < Rc_rad and Rij > 1e-8):
                                # loop over radial centers
                                for idx_rs_rad in range(RsN_rad):
                                    #loop over eta_rad
                                    for idx_eta_rad, eta in enumerate(eta_rad):
                                        Gvector[i][kind_j * (RsN_rad + len(eta_rad)) +
                                                   idx_rs_rad + idx_eta_rad] +=\
                                            _G_radial_BP(Rij, idx_rs_rad, eta,
                                                          Rs0_rad, Rsst_rad,
                                                          Rc_rad)
                                if (Rij < Rc_ang):
                                    R_ij_neighbors.append([Rij, pos_j])

            neighbors.append(R_ij_neighbors)

        for kind_j, kind_j_list in enumerate(neighbors):
            # loop over all positions associated to type_j atoms
            for Rij, pos_j in kind_j_list:
                # third species
                for kind_k in range(kind_j, Nspecies):
                    # emine: reproducing prophet requires double counting angles when
                    # type2 != type3, hence multiply the angulars by two..
                    # it is not a bad idea coz
                    # we already double count the typeA-central atom-typeA kind of angles
                    if kind_j == kind_k:
                        angular_factor = 1
                    else:
                        angular_factor = 2
                    kind_k_list = neighbors[kind_k]
                    temp_ind = int(
                        len(eta_rad) * RsN_rad * Nspecies +
                        (kind_j * (Nspecies -
                                   (kind_j + 1) / 2) + kind_k) * len(zeta))

                    for Rik, pos_k in kind_k_list:
                        cos_theta_ijk = np.dot((pos_j - pos_i),
                                               (pos_k - pos_i)) / (Rij * Rik)
                        # for some numerical reasons,
                        # cos_theta_ijk may be slightly
                        # greater/less than 1.0/-1.0
                        if (cos_theta_ijk > 1.0):
                            cos_theta_ijk = np.float(1.0)
                        if (cos_theta_ijk < -1.0):
                            cos_theta_ijk = np.float(-1.0)

                        if (np.abs(Rij - Rik) > 1e-8
                                or cos_theta_ijk < 0.99999999):

                            theta_ijk = np.arccos(cos_theta_ijk)
                            # loop over zeta
                            for idx_zeta, zeta_i in enumerate(zeta):
                                #loop over eta
                                for idx_eta, eta in enumerate(eta_ang):
                                    Gvector[i][temp_ind + idx_zeta * len(eta_ang)+idx_eta]\
                                         += _G_angular_BP(Rij,
                                                          Rik,
                                                          theta_ijk,
                                                          eta,
                                                          zeta_i,
                                                          Rc_ang)*angular_factor

    return (key, Gvector.tolist())

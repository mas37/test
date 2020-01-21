###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################

import itertools
import numpy as np

from .gvect_routine import calculate_Gvector_dGvect as calculate_Gvector_dGvect_mBP

from .pbc import replicas_max_idx

from .feature_functions import G_radial
from .feature_functions import G_angular_mBP
from .feature_functions import G_angular_BP

from lib.units_converter import distances_to_angstrom


class GvectBase():
    """Base class for Gvectors, BP, mBP
    """

    def __init__(self,
                 compute_dgvect,
                 sparse_dgvect,
                 species,
                 param_unit,
                 pbc_directions,
                 Rc_rad,
                 Rs0_rad,
                 RsN_rad,
                 Rc_ang,
                 Rsst_rad=None):

        # ==== generic parameters ====
        self.compute_dgvect = compute_dgvect
        self.sparse_dgvect = sparse_dgvect
        self.species = species
        self.param_unit = param_unit
        self.pbc_directions = pbc_directions

        # ==== gvect ====
        self._unit_2_angstrom = distances_to_angstrom(self.param_unit)
        # RADIAL_COMPONENTS
        self.Rc_rad = Rc_rad * self._unit_2_angstrom
        self.Rs0_rad = Rs0_rad * self._unit_2_angstrom
        self.RsN_rad = RsN_rad

        # infer if None:
        self._Rsst_rad = Rsst_rad

        # ANGULAR_COMPONENTS
        self.Rc_ang = Rc_ang * self._unit_2_angstrom

    @property
    def species_idx_2str(self):
        """ converter index atomic type to string
        """
        return [x.strip() for x in self.species.split(',')]

    @property
    def number_of_species(self):
        """number of species in the system
        """
        return len(self.species_idx_2str)

    @property
    def species_str_2idx(self):
        """ converter string atomic type to idx
        """
        return {
            x: y
            for x, y in zip(self.species_idx_2str, range(
                self.number_of_species))
        }

    @property
    def Rsst_rad(self):
        """ Radial step size
        """
        if self._Rsst_rad is not None:
            return self._Rsst_rad
        return (self.Rc_rad - self.Rs0_rad) / self.RsN_rad


class GvectmBP(GvectBase):
    def __init__(self,
                 compute_dgvect,
                 sparse_dgvect,
                 species,
                 param_unit,
                 pbc_directions,
                 Rc_rad,
                 Rs0_rad,
                 RsN_rad,
                 eta_rad,
                 Rc_ang,
                 ThetasN,
                 Rs0_ang,
                 RsN_ang,
                 eta_ang,
                 zeta,
                 Rsst_rad=None,
                 Rsst_ang=None):

        super().__init__(
            compute_dgvect=compute_dgvect,
            sparse_dgvect=sparse_dgvect,
            species=species,
            param_unit=param_unit,
            pbc_directions=pbc_directions,
            Rc_rad=Rc_rad,
            Rs0_rad=Rs0_rad,
            RsN_rad=RsN_rad,
            Rc_ang=Rc_ang,
            Rsst_rad=Rsst_rad)

        # ==== gvect ====
        unit2A = self._unit_2_angstrom
        # RADIAL_COMPONENTS
        self.eta_rad = eta_rad / (unit2A * unit2A)

        # ANGULAR_COMPONENTS
        self.eta_ang = eta_ang / (unit2A * unit2A)
        self.Rs0_ang = Rs0_ang * unit2A
        self.RsN_ang = RsN_ang
        self.ThetasN = ThetasN

        # infer if not present
        self._Rsst_ang = Rsst_ang

        self.zeta = zeta

    @property
    def Rsst_ang(self):
        if self._Rsst_ang is not None:
            return self._Rsst_ang
        return (self.Rc_ang - self.Rs0_ang) / self.RsN_ang

    @property
    def gsize(self):
        ns = self.number_of_species
        return int(ns * self.RsN_rad +
                   0.5 * ns * (ns + 1) * self.RsN_ang * self.ThetasN)

    @property
    def gvect(self):
        gvect = {
            # RADIAL_COMPONENTS
            'eta_rad': self.eta_rad,
            'Rc_rad': self.Rc_rad,
            'Rs0_rad': self.Rs0_rad,
            'RsN_rad': int(self.RsN_rad),
            'Rsst_rad': self.Rsst_rad,

            # ANGULAR_COMPONENTS
            'eta_ang': self.eta_ang,
            'Rc_ang': self.Rc_ang,
            'Rs0_ang': self.Rs0_ang,
            'RsN_ang': self.RsN_ang,
            'Rsst_ang': self.Rsst_ang,
            'zeta': self.zeta,
            'ThetasN': self.ThetasN
        }
        return gvect

    def __call__(self, key, positions, species, lattice_vectors):
        """Calculate the gvector based on given parameters, using list

        Parameters
        ----------
        key: key of the simulation
        positions: list of atomic positions
        species: list of species as name
        lattice_vectors: list [a1, a2, a3]

        For all the other parameters refer to the article

        Returns
        -------
        tuple: (key, gvector as list, dgvecotor as a list if required)
        key: key of the simulation
        gvector: list of numpy arrays n_atoms, gsize
        dgvecotor: list of numpy arrays n_atoms, gsize * 3 * n_atoms

        Note
        ----
        All kind of PBC : 1D, 2D and 3D are implemented..
        Check the function "replicas_max_idx" below

        if derivatives are required code is legacy and not very optimized
        """
        # better to have name closer to equations
        # pylint: disable=invalid-name

        # define shorter names:
        n_species = self.number_of_species
        eta_rad = self.eta_rad
        Rc_rad = self.Rc_rad
        Rs0_rad = self.Rs0_rad
        Rsst_rad = self.Rsst_rad
        eta_ang = self.eta_ang
        Rc_ang = self.Rc_ang
        Rs0_ang = self.Rs0_ang
        Rsst_ang = self.Rsst_ang
        zeta = self.zeta
        ThetasN = self.ThetasN

        # ensure that everything is a numpy array
        positions = np.asarray(positions)
        species = np.asarray(species)
        lattice_vectors = np.asarray(lattice_vectors)

        if self.compute_dgvect:
            # legacy code for derivative.
            key, Gvector, dGvector = calculate_Gvector_dGvect_mBP(
                key,
                positions,
                species,
                lattice_vectors,
                self.number_of_species,
                self.eta_rad,
                self.Rc_rad,
                self.Rs0_rad,
                self.RsN_rad,
                self.Rsst_rad,
                self.eta_ang,
                self.Rc_ang,
                self.Rs0_ang,
                self.Rsst_ang,
                self.RsN_ang,
                self.zeta,
                self.ThetasN,
                pcb=self.pbc_directions)
            if self.sparse_dgvect:
                # Done by hand from the dense one for now...
                n_atoms = len(positions)
                G_size = self.gsize
                dGdx_values = []
                dGdx_indices = []
                for i in range(n_atoms):
                    for j in range(G_size):
                        for k in range(n_atoms):
                            if dGvector[i,j,k,0] != 0.0:
                                dGdx_values.append(dGvector[i,j,k,0])
                                dGdx_indices.append([i*G_size+j,3*k  ])
                            if dGvector[i,j,k,1] != 0.0:
                                dGdx_values.append(dGvector[i,j,k,1])
                                dGdx_indices.append([i*G_size+j,3*k+1])
                            if dGvector[i,j,k,2] != 0.0:
                                dGdx_values.append(dGvector[i,j,k,2])
                                dGdx_indices.append([i*G_size+j,3*k+2])
                return (key, Gvector.tolist(), dGdx_values, dGdx_indices)
            else:
                return (key, Gvector.tolist(), dGvector.tolist())

        n_atoms = len(positions)
        Gs = [np.empty(0) for x in range(n_atoms)]

        # compute how many cell replica we need in each direction

        if self.pbc_directions is not None:
            max_indices = replicas_max_idx(
                lattice_vectors, max(Rc_rad, Rc_ang), pbc=self.pbc_directions)
        else:
            max_indices = replicas_max_idx(lattice_vectors, max(
                Rc_rad, Rc_ang))
        l_max, m_max, n_max = max_indices
        l_list = range(-l_max, l_max + 1)
        m_list = range(-m_max, m_max + 1)
        n_list = range(-n_max, n_max + 1)
        # create a matrix with all the idx of the extra cell we need
        replicas = np.asarray(list(itertools.product(l_list, m_list, n_list)))
        # map the index of the cell on to the lattice vectors
        # all needed translation for each atom in the unit cell
        replicas = replicas @ lattice_vectors
        n_replicas = len(replicas)
        # creating a tensor with the coordinate of all the needed atoms
        # and reshape it as positions tensor
        # shape: n_atoms, n_replicas, 3
        positions_extended = positions[:, np.newaxis, :] + replicas
        positions_extended = positions_extended.reshape(
            n_atoms * n_replicas, 3)
        # creating the equivalent species tensor
        # the order is [...all atom 1 replicas.....,... all atom2 replicas....,]
        species = np.tile(species[:, np.newaxis],
                          (1, n_replicas)).reshape(n_atoms * n_replicas)
        # computing x_i - x_j between all the atom in the unit cell and all the atom
        # in the cell + replica tensor
        # shape: n_atoms, n_replicas, 3
        deltas = positions[:, np.newaxis, :] - positions_extended
        # using the deltas to compute rij for each atom in the unit cell and all the atom
        # in the cell + replica tensor
        # shape: n_atoms, n_replicas
        rij = np.linalg.norm(deltas, axis=-1)

        # === RADIAL PART 1 ===
        # create a boolean mask to extrapolate only atoms inside the cutoff
        # shape: n_atoms, n_replicas
        # each row contains True value if the given j atom is inside the
        # cutoff wrt the i atom, logical "and" to exclude counting the
        # central atom
        radial_mask = np.logical_and(rij < Rc_rad, rij > 1e-8)

        # create tensors of parameters for the sampling to allow proper broadcasting
        sampling_rad = np.arange(Rs0_rad, Rc_rad, Rsst_rad)
        # ===  END RADIAL PART 1 ===

        # === ANGULAR PART 1 ===
        # same mask as the radial part
        # shape: n_atoms, n_replicas
        angular_mask = np.logical_and(rij < Rc_ang, rij > 1e-8)

        # create tensors of parameters for the sampling to allow proper broadcasting
        # angles centers are shift by half differently than mBP reference.
        step_theta = np.pi / ThetasN
        sampling_theta = np.arange(0, np.pi, step_theta) + .5 * step_theta

        sampling_rad_ang = np.arange(Rs0_ang, Rc_ang, Rsst_ang)
        # === END ANGULAR PART 1 ===

        for idx in range(n_atoms):
            # === RADIAL PART 2 ===
            # for each relevant quantity for the radial part we extract only the
            # elemnts inside the cutoff
            # shape: atoms_inside_cutoff
            cutoff_species = species[radial_mask[idx]]
            cutoff_rij = rij[idx, radial_mask[idx]]

            # for each atom inside the cutoff we compute the corresponding
            # sampling
            per_atom_contrib = G_radial(cutoff_rij[:, np.newaxis],
                                        sampling_rad[np.newaxis, :], eta_rad,
                                        Rc_rad)
            for atom_kind in range(n_species):
                # for each atom_kind we take the idxs of all atoms of that
                # kind inside the cutoff
                species_idxs = np.where(cutoff_species == atom_kind)
                # we use the indexs to extract the g's contribution
                # of that species
                species_per_atom_contrib = per_atom_contrib[species_idxs]
                # contract over all the contibutions
                kind_radial_g = species_per_atom_contrib.sum(0)
                # creation of the gs
                Gs[idx] = np.append(Gs[idx], kind_radial_g)
            # ===  END RADIAL PART 2 ===

            # === ANGULAR PART 2 ===
            # same extraction as the radial part
            # shape: atoms_inside_cutoff, 3
            # cutoff_positions = positions_extended[angular_mask[idx]]
            cutoff_deltas = deltas[idx, angular_mask[idx]]
            # shape: atoms_inside_cutoff
            cutoff_species = species[angular_mask[idx]]
            cutoff_rij = rij[idx, angular_mask[idx]]

            for atom_kind_1 in range(n_species):
                for atom_kind_2 in range(atom_kind_1, n_species):
                    # prevent double counting if atom i is equal tom atom j
                    prefactor = 1.0 if atom_kind_1 != atom_kind_2 else 0.5
                    # for each atom_kind we take the idxs of all atoms of that
                    # kind inside the cutoff, we do this for both the species
                    species_idxs_1 = np.where(cutoff_species == atom_kind_1)
                    species_idxs_2 = np.where(cutoff_species == atom_kind_2)
                    # in the same way we extract all the quanities needed to compute
                    # the angular contribution
                    cutoff_deltas_ij = cutoff_deltas[species_idxs_1]
                    cutoff_deltas_ik = cutoff_deltas[species_idxs_2]
                    cutoff_r_ij = cutoff_rij[species_idxs_1]
                    cutoff_r_ik = cutoff_rij[species_idxs_2]
                    # = computation of the angle between ikj triplet =
                    # numerator: ij dot ik
                    # shape: n_atom_species1_inside_cutoff, n_atom_species2_inside_cutoff
                    a = np.sum(
                        cutoff_deltas_ij[:, np.newaxis, :] * cutoff_deltas_ik,
                        2)
                    # denominator: ij * ik
                    # shape: n_atom_species1_inside_cutoff, n_atom_species2_inside_cutoff
                    b = cutoff_r_ij[:, np.newaxis] * cutoff_r_ik
                    # element by element ratio
                    cos_theta_ijk = a / b
                    # correct numerical error
                    cos_theta_ijk[cos_theta_ijk >= 1.0] = 1.0
                    cos_theta_ijk[cos_theta_ijk <= -1.0] = -1.0
                    # compute the angle
                    # shape: n_atom_species1_inside_cutoff, n_atom_species2_inside_cutoff
                    theta_ijk = np.arccos(cos_theta_ijk)
                    # computation of all the elements
                    # shape: n_atom_species1_inside_cutoff, n_atom_species2_inside_cutoff,
                    #        radial_ang_sampling, ang_sampling
                    kind1_kind2_angular_g = G_angular_mBP(
                        cutoff_r_ij[:, np.newaxis, np.newaxis, np.newaxis],
                        cutoff_r_ik[np.newaxis, :, np.newaxis, np.newaxis],
                        theta_ijk[:, :, np.newaxis, np.newaxis],
                        sampling_rad_ang[np.newaxis, np.newaxis, :, np.
                                         newaxis],
                        sampling_theta[np.newaxis, np.newaxis, np.newaxis, :],
                        eta_ang, zeta, Rc_ang)
                    # creation of a mask to esclude counting of j==k
                    # shape: n_atom_species1_inside_cutoff, n_atom_species2_inside_cutoff
                    f = np.logical_or(
                        np.abs(cutoff_r_ij[:, np.newaxis] - cutoff_r_ik) >
                        1e-5, cos_theta_ijk < .99999)
                    # collection of the contributions:
                    # shape: n_of_triplet 1x2,
                    #        radial_ang_sampling, ang_sampling
                    kind1_kind2_angular_g = kind1_kind2_angular_g[f, :, :]
                    # contraction over the number of triplet, flattenig and adding it
                    # to the total G
                    kind1_kind2_angular_g = prefactor * np.sum(
                        kind1_kind2_angular_g, 0).reshape(-1)
                    Gs[idx] = np.append(Gs[idx], kind1_kind2_angular_g)
            # === END ANGULAR PART 2 ===

        return key, np.asarray(Gs).tolist()


class GvectBP(GvectBase):
    def __init__(self,
                 compute_dgvect,
                 sparse_dgvect,
                 species,
                 param_unit,
                 pbc_directions,
                 Rc,
                 RsN,
                 eta,
                 zeta,
                 Rs0=0.0,
                 Rc_ang=None,
                 eta_ang=np.empty(0),
                 Rsst_rad=None):

        eta_rad = np.asarray(eta)
        try:
            if eta_ang.size == 0:
                eta_ang = eta_rad
        except AttributeError:
            eta_ang = np.asarray(eta_ang)

        if not Rc_ang:
            Rc_ang = Rc

        super().__init__(
            compute_dgvect=compute_dgvect,
            sparse_dgvect=sparse_dgvect,
            species=species,
            param_unit=param_unit,
            pbc_directions=pbc_directions,
            Rc_rad=Rc,
            Rs0_rad=Rs0,
            RsN_rad=RsN,
            Rc_ang=Rc_ang,
            Rsst_rad=Rsst_rad)

        # ==== gvect ====
        # RADIAL_COMPONENTS
        unit2A = self._unit_2_angstrom
        self.eta_rad = np.asarray(eta_rad) / (unit2A * unit2A)

        # ANGULAR_COMPONENTS
        self.eta_ang = np.asarray(eta_ang) / (unit2A * unit2A)
        self.zeta = np.asarray(zeta)

    @property
    def gsize(self):
        return int(
            self.RsN_rad * len(self.eta_rad) * self.number_of_species +
            len(self.eta_ang) * len(self.zeta) * 2 * self.number_of_species)

    @property
    def gvect(self):
        gvect = {
            # RADIAL_COMPONENTS
            'eta_rad': self.eta_rad,
            'Rc_rad': self.Rc_rad,
            'Rs0_rad': self.Rs0_rad,
            'RsN_rad': int(self.RsN_rad),
            'Rsst_rad': self.Rsst_rad,

            # ANGULAR_COMPONENTS
            'eta_ang': self.eta_ang,
            'Rc_ang': self.Rc_ang,
            'zeta': self.zeta,
        }
        return gvect

    def __call__(self, key, positions, species, lattice_vectors, **kwargs):
        """ Calculate the gvector

        as defined in 10.1103/PhysRevLetter.98.146401
        """
        # better to have name closer to equations
        # pylint: disable=invalid-name

        # define shorter names:
        n_species = self.number_of_species
        eta_rad = np.asarray(self.eta_rad)
        Rc_rad = self.Rc_rad
        Rs0_rad = self.Rs0_rad
        Rsst_rad = self.Rsst_rad

        Rc_ang = self.Rc_ang
        eta_ang = np.asarray(self.eta_ang)
        zeta = np.asarray(self.zeta)
        lamb = np.asanyarray([1.0, -1.0])

        # ensure that everything is a numpy array
        positions = np.asarray(positions)
        species = np.asarray(species)
        lattice_vectors = np.asarray(lattice_vectors)

        if self.compute_dgvect:
            raise ValueError('derivative not implemented')

        n_atoms = len(positions)
        Gs = [np.empty(0) for x in range(n_atoms)]

        # compute how many cell replica we need in each direction
        if 'pbc' in kwargs:
            max_indices = replicas_max_idx(
                lattice_vectors, max(Rc_rad, Rc_ang), pbc=kwargs['pbc'])
        else:
            max_indices = replicas_max_idx(lattice_vectors, max(
                Rc_rad, Rc_ang))

        l_max, m_max, n_max = max_indices
        l_list = range(-l_max, l_max + 1)
        m_list = range(-m_max, m_max + 1)
        n_list = range(-n_max, n_max + 1)
        # create a matrix with all the idx of the extra cell we need
        replicas = np.asarray(list(itertools.product(l_list, m_list, n_list)))
        # map the index of the cell on to the lattice vectors
        # all needed translation for each atom in the unit cell
        replicas = replicas @ lattice_vectors
        n_replicas = len(replicas)
        # creating a tensor with the coordinate of all the needed atoms
        # and reshape it as positions tensor
        # shape: n_atoms, n_replicas, 3
        positions_extended = positions[:, np.newaxis, :] + replicas
        positions_extended = positions_extended.reshape(
            n_atoms * n_replicas, 3)
        # creating the equivalent species tensor
        # the order is [...all atom 1 replicas.....,... all atom2 replicas....,]
        species = np.tile(species[:, np.newaxis],
                          (1, n_replicas)).reshape(n_atoms * n_replicas)
        # computing x_i - x_j between all the atom in the unit cell and all the atom
        # in the cell + replica tensor
        # shape: n_atoms, n_replicas, 3
        deltas = positions[:, np.newaxis, :] - positions_extended
        # using the deltas to compute rij for each atom in the unit cell and all the atom
        # in the cell + replica tensor
        # shape: n_atoms, n_replicas
        rij = np.linalg.norm(deltas, axis=-1)

        # === RADIAL PART 1 ===
        # create a boolean mask to extrapolate only atoms inside the cutoff
        # shape: n_atoms, n_replicas
        # each row contains True value if the given j atom is inside the
        # cutoff wrt the i atom, logical "and" to exclude counting the
        # central atom
        radial_mask = np.logical_and(rij < Rc_rad, rij > 1e-8)

        # create tensors of parameters for the sampling to allow proper broadcasting
        sampling_rad = np.arange(Rs0_rad, Rc_rad, Rsst_rad)
        # ===  END RADIAL PART 1 ===

        # === ANGULAR PART 1 ===
        # same mask as the radial part
        # shape: n_atoms, n_replicas
        angular_mask = np.logical_and(rij < Rc_ang, rij > 1e-8)
        # === END ANGULAR PART 1 ===

        for idx in range(n_atoms):
            # === RADIAL PART 2 ===
            # for each relevant quantity for the radial part we extract only the
            # elemnts inside the cutoff
            # shape: atoms_inside_cutoff
            cutoff_species = species[radial_mask[idx]]
            cutoff_rij = rij[idx, radial_mask[idx]]

            # for each atom inside the cutoff we compute the corresponding
            # sampling
            # shape:n_atom in the cutoff, n_ etas, radial samples
            per_atom_contrib = G_radial(
                cutoff_rij[:, np.newaxis, np.newaxis],
                sampling_rad[np.newaxis, np.newaxis, :],
                eta_rad[np.newaxis, :, np.newaxis], Rc_rad)
            for atom_kind in range(n_species):
                # for each atom_kind we take the idxs of all atoms of that
                # kind inside the cutoff
                species_idxs = np.where(cutoff_species == atom_kind)
                # we use the indexs to extract the g's contribution
                # of that species
                species_per_atom_contrib = per_atom_contrib[species_idxs]
                # contract over all the contibutions
                kind_radial_g = species_per_atom_contrib.sum(0).reshape(-1)
                # creation of the gs
                Gs[idx] = np.append(Gs[idx], kind_radial_g)
            # ===  END RADIAL PART 2 ===

            # === ANGULAR PART 2 ===
            # same extraction as the radial part
            # shape: atoms_inside_cutoff, 3
            # cutoff_positions = positions_extended[angular_mask[idx]]
            cutoff_deltas = deltas[idx, angular_mask[idx]]
            # shape: atoms_inside_cutoff
            cutoff_species = species[angular_mask[idx]]
            cutoff_rij = rij[idx, angular_mask[idx]]

            for atom_kind_1 in range(n_species):
                for atom_kind_2 in range(atom_kind_1, n_species):
                    # prevent double counting if atom i is equal tom atom j
                    # TODO in emine version angular prefactor = 1 if i = j else 2
                    # decide which we wont to keep
                    prefactor = 1.0 if atom_kind_1 != atom_kind_2 else 0.5
                    # for each atom_kind we take the idxs of all atoms of that
                    # kind inside the cutoff, we do this for both the species
                    species_idxs_1 = np.where(cutoff_species == atom_kind_1)
                    species_idxs_2 = np.where(cutoff_species == atom_kind_2)
                    # in the same way we extract all the quanities needed to compute
                    # the angular contribution
                    cutoff_deltas_ij = cutoff_deltas[species_idxs_1]
                    cutoff_deltas_ik = cutoff_deltas[species_idxs_2]
                    cutoff_r_ij = cutoff_rij[species_idxs_1]
                    cutoff_r_ik = cutoff_rij[species_idxs_2]
                    # compute the distance between jk
                    # shape: n_atom_species1_inside_cutoff, n_atom_species2_inside_cutoff
                    cutoff_r_jk = np.linalg.norm(
                        cutoff_deltas_ij[:, np.newaxis, :] - cutoff_deltas_ik,
                        axis=-1)
                    # = computation of the angle between ikj triplet =
                    # numerator: ij dot ik
                    # shape: n_atom_species1_inside_cutoff, n_atom_species2_inside_cutoff
                    a = np.sum(
                        cutoff_deltas_ij[:, np.newaxis, :] * cutoff_deltas_ik,
                        2)
                    # denominator: ij * ik
                    # shape: n_atom_species1_inside_cutoff, n_atom_species2_inside_cutoff
                    b = cutoff_r_ij[:, np.newaxis] * cutoff_r_ik
                    # element by element ratio
                    cos_theta_ijk = a / b
                    # correct numerical error
                    cos_theta_ijk[cos_theta_ijk >= 1.0] = 1.0
                    cos_theta_ijk[cos_theta_ijk <= -1.0] = -1.0
                    # compute the angle
                    # shape: n_atom_species1_inside_cutoff, n_atom_species2_inside_cutoff
                    theta_ijk = np.arccos(cos_theta_ijk)
                    # computation of all the elements
                    # shape: n_atom_species1_inside_cutoff, n_atom_species2_inside_cutoff,
                    #        eta_ang_elements, zeta_elements, 2
                    kind1_kind2_angular_g = G_angular_BP(
                        cutoff_r_ij[:, np.newaxis, np.newaxis, np.newaxis, np.
                                    newaxis],
                        cutoff_r_ik[np.newaxis, :, np.newaxis, np.newaxis, np.
                                    newaxis],
                        cutoff_r_jk[:, :, np.newaxis, np.newaxis, np.newaxis],
                        theta_ijk[:, :, np.newaxis, np.newaxis, np.newaxis],
                        eta_ang[np.newaxis, np.newaxis, :, np.newaxis, np.
                                newaxis], zeta[np.newaxis, np.newaxis, np.
                                               newaxis, :, np.newaxis],
                        lamb[np.newaxis, np.newaxis, np.newaxis, np.
                             newaxis, :], Rc_ang)
                    # creation of a mask to esclude counting of j==k
                    # shape: n_atom_species1_inside_cutoff, n_atom_species2_inside_cutoff
                    f = np.logical_or(
                        np.abs(cutoff_r_ij[:, np.newaxis] - cutoff_r_ik) >
                        1e-5, cos_theta_ijk < .99999)
                    # collection of the contributions:
                    # shape: n_of_triplet 1x2,
                    #        eta_ang_elements, zeta_elements, 2
                    kind1_kind2_angular_g = kind1_kind2_angular_g[f]
                    # contraction over the number of triplet, flattenig and adding it
                    # to the total G
                    # when reshaping the unroll is done right to left
                    # so the sequence is lambda, for each zeta, for each eta_ang
                    kind1_kind2_angular_g = prefactor * np.sum(
                        kind1_kind2_angular_g, 0).reshape(-1)
                    Gs[idx] = np.append(Gs[idx], kind1_kind2_angular_g)
            # === END ANGULAR PART 2 ===
        return key, np.asarray(Gs).tolist()

import os
import json
import logging

import numpy as np

# logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(levelname)s - %(message)s')
logger.setLevel(logging.INFO)

# console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# Conversion constants
BOHR2A = np.float32(0.52917721067)
RY2EV = 13.6056980659


class ExampleJsonWrapper(object):
    def __init__(self, file_name, species_str_2idx):
        with open(file_name) as f:
            example = json.load(f)

        logger.info('parsing {}'.format(file_name))
        # === THE CODE WORKS IN ANGSTROM AND EV ===
        # so to grant consistency we convert everything to these units

        # energy info
        energy_list = example.get('energy', None)
        if energy_list[1] in ["Ry", "Ryd", "ryd", "ry", "RY"]:
            unit2eV = np.float32(1) * RY2EV
        elif energy_list[1] == "Ha":
            unit2eV = np.float32(2) * RY2EV
        elif energy_list[1] in ["eV", "ev", "EV"]:
            unit2eV = np.float32(1)
        else:
            unit2eV = np.float32(1)
            logger.warning(
                'WARNING: unit of energy unknown, assumed eV {}'.format(key))

        # lattice info
        unit_of_length = example.get('unit_of_length', None)
        lattice_vectors = example.get('lattice_vectors', None)

        if unit_of_length == 'bohr':
            alat = np.float32(1) * BOHR2A
        elif unit_of_length == 'angstrom':
            alat = np.float32(1)
        else:
            logger.warning(
                'unit_of_length unknown, assumed angstrom {}'.format(key))
            unit_of_length = "angstrom"
            alat = np.float32(1)

        if not lattice_vectors:
            logger.warning(
                'key {} didn\'t provide information about lattice vectors, '
                'lattice vectors are set to zero, no PBC will be applied'.
                format(key))
            lattice_vectors = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]
            example['lattice_vectors'] = lattice_vectors

        # read and scale lattice_vectors to angstrom
        # (lattice vectors units are specified in unit of length keyword
        lattice_vectors = np.multiply(
            np.asarray(example['lattice_vectors']).astype(float), alat)

        species_idxs = []
        pos = []
        forces = []

        atomic_position_unit = example.get('atomic_position_unit', None)
        if not atomic_position_unit:
            atomic_position_unit = example.get('atomic_coordinates', None)

        for atom in example['atoms']:
            species_idxs.append(species_str_2idx[atom[1]])
            if atomic_position_unit == 'cartesian':
                if unit_of_length == 'angstrom':
                    pos.append(np.asarray(atom[2]).astype(float))
                elif unit_of_length == 'bohr':
                    # the coordinates are just converted to angstrom
                    pos.append(np.asarray(atom[2]).astype(float) * BOHR2A)
                else:
                    raise ValueError('unit_of_length unknown')
            elif atomic_position_unit == 'crystal':
                # Here the lattice vectors have been already converted in
                # angstrom
                pos.append(
                    np.dot(
                        np.transpose(lattice_vectors),
                        np.asarray(atom[2]).astype(float)))
            else:
                raise ValueError('atomic_position_unit unknown')

            if len(atom) > 3:
                if unit_of_length == 'angstrom':
                    forces.append(np.asarray(atom[3]).astype(float) * unit2eV)
                elif unit_of_length == 'bohr':
                    forces.append(
                        np.asarray(atom[3]).astype(float) / BOHR2A * unit2eV)

        # species_symbols = [species_idx_2str[x] for x in species_idxs]

        # exposed quanities

        self.key = file_name
        self.positions = np.asarray(pos)
        self.species = np.asarray(species_idxs)
        self.lattice_vectors = lattice_vectors
        #self.symbols = species_symbols
        self.E = example['energy'][0] * unit2eV
        self.n_atoms = len(example['atoms'])
        self.species_str_2idx = species_str_2idx
        self.forces = forces

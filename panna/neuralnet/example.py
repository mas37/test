###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import json
import numpy as np
import tensorflow as tf
from os.path import join


class Example(object):
    """
    Example class
    """

    def __init__(self,
                 g_vectors,
                 species_vector,
                 true_energy,
                 d_g_vectors=np.empty(0),
                 forces=np.empty(0),
                 atomic_species=None,
                 name=None):
        """ init of Example

        Args:
            g_vectors: list of g's,
                       one row one atom in the example
            species_vector: list of idx, one per atom in
                            the example
            true_energy: value of the true energy
            zeros: list of zero value, one per atomic species
            atomic_species: list of strings,
                            with atomic species name
            name: arbitrary name
        """

        self._g_vectors = np.asarray(g_vectors)
        self._species_vector = np.asarray(species_vector)
        self._true_energy = true_energy
        self._n_species = max(set(species_vector)) + 1

        self._d_g_vectors = np.asarray(d_g_vectors)
        self._forces = np.asarray(forces)

        self._atomic_species = atomic_species
        self._name = name or ''

    @property
    def true_energy(self):
        return self._true_energy

    @property
    def n_atoms(self):
        return len(self._species_vector)

    @property
    def atoms_per_species(self):
        unique, counts = np.unique(self._species_vector, return_counts=True)
        return counts

    @property
    def species_vector(self):
        return self._species_vector

    @property
    def gvects(self):
        return self._g_vectors

    @property
    def dgvects(self):
        if self._d_g_vectors.size > 0:
            return self._d_g_vectors
        else:
            raise ValueError("g's derivative Not available")

    @property
    def forces(self):
        return self._forces

    @property
    def name(self):
        return self._name


def load_example(filename, n_species, derivatives=False):
    """ Load binary example
    """
    key = filename.split('/')[-1]
    data = np.fromfile(filename, np.float32)

    n_atoms = int(data[0])
    g_size = int(data[1])
    en = data[2]
    spec_tensor_bytes = n_atoms
    gvect_tensor_bytes = n_atoms * g_size
    prev_bytes = 3

    spec_tensor = data[prev_bytes:prev_bytes + spec_tensor_bytes]
    prev_bytes += spec_tensor_bytes
    gvect_tensor = data[prev_bytes:prev_bytes + gvect_tensor_bytes]
    prev_bytes += gvect_tensor_bytes

    if derivatives:
        if data.size == prev_bytes:
            raise ValueError('Derivatives requested but not '
                             'present in the file')
        dgvec_tensor_bytes = n_atoms**2 * g_size * 3
        dgvect_tensor = data[prev_bytes:prev_bytes + dgvec_tensor_bytes]
        prev_bytes += dgvec_tensor_bytes

        if data.size > prev_bytes:
            # If there is more, then forces are stored
            forces_bytes = n_atoms * 3
            forces = data[prev_bytes:prev_bytes + forces_bytes]
        else:
            forces = []
    else:
        dgvect_tensor = []
        forces = []

    # building the data
    en = np.reshape(en, [1])[0]
    spec_tensor = np.int64(np.reshape(spec_tensor, [n_atoms]))
    gvect_tensor = np.reshape(gvect_tensor, [n_atoms, g_size])
    if (derivatives):
        dgvect_tensor = np.reshape(dgvect_tensor, [n_atoms, g_size, \
                                                   n_atoms*3])

    return Example(
        g_vectors=gvect_tensor,
        species_vector=spec_tensor,
        true_energy=en,
        d_g_vectors=dgvect_tensor,
        forces=forces,
        name=key)


def iterator_over_tfdata(g_size, *args, **kwargs):
    """ TFdata unpacker

    Args:
        g_size: size of the G's
        args: all the tfdata files that one wants to parse
    kwargs:
        derivatives: bool, whether forces should be calculated
    Retrun:
        iterator over the record in the files
    """
    for file in args:
        record_iterator = tf.python_io.tf_record_iterator(path=file)
        for record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(record)
            exact_energy = example.features.feature['energy'].float_list.value[
                0]
            species_vector = example.features.feature[
                'species'].int64_list.value
            g_vectors = np.reshape(
                example.features.feature['gvects'].float_list.value,
                (len(species_vector), g_size))
            if 'name' in example.features.feature:
                name = example.features.feature['name'].bytes_list.value[
                    0].decode()
            else:
                name = ''
            if ('derivatives' in kwargs) and kwargs['derivatives']:
                d_g_vectors_falt = example.features.feature[
                    'dgvects'].float_list.value
                if not len(d_g_vectors_falt):
                    raise ValueError('Derivatives requested but not '
                                     'present in the tfr file')
                d_g_vectors = np.reshape(
                    d_g_vectors_falt,
                    (len(species_vector), g_size, len(species_vector) * 3))
                if len(example.features.feature['forces'].float_list.value):
                    #there are reference forces
                    ref_forces = example.features.feature[
                        'forces'].float_list.value
                else:
                    ref_forces = []
            else:
                d_g_vectors = []
                ref_forces = []
                # we dont care if you have reference forces
                # if you are not predicting forces

            yield Example(
                g_vectors=g_vectors,
                species_vector=species_vector,
                true_energy=exact_energy,
                d_g_vectors=d_g_vectors,
                name=name,
                forces=ref_forces)

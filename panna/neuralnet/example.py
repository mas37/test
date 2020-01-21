###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import numpy as np
import tensorflow as tf
import os


class Example(object):
    """
    Example class, a class to handle a binary PANNA example

    Parameters
    ----------
        g_vectors: (n_atoms, g_size)
            list of g's, one row one atom in the example
        species_vector: (n_atoms)
            list of idx, one per atom in the example
        true_energy: (1)
            value of the true energy
        d_g_vectors: (n_atoms, g_size, n_atoms * 3), opt
            derivative of the gvect
        forces: (n_atoms * 3), opt
            forces acting on the atoms
        per_atom_quantity: (n_atoms), opt
            a per atom quantity that can be learn (eg charge)
        atomic_species: list of strings,
                        with atomic species name
        name: arbitrary name
    """
    def __init__(self,
                 g_vectors,
                 species_vector,
                 true_energy,
                 d_g_vectors=np.empty(0),
                 d_g_vector_values=np.empty(0),
                 d_g_vector_indices1=np.empty(0),
                 d_g_vector_indices2=np.empty(0),
                 forces=np.empty(0),
                 per_atom_quantity=np.empty(0),
                 atomic_species=None,
                 name=None):

        self._g_vectors = np.asarray(g_vectors)
        self._species_vector = np.asarray(species_vector)
        self._true_energy = true_energy
        self._n_species = max(set(species_vector)) + 1

        self._d_g_vectors = np.asarray(d_g_vectors)
        self._d_g_vector_values = np.asarray(d_g_vector_values)
        self._d_g_vector_indices1 = np.asarray(d_g_vector_indices1)
        self._d_g_vector_indices2 = np.asarray(d_g_vector_indices2)
        self._forces = np.asarray(forces)

        self._per_atom_quantity = per_atom_quantity

        self._atomic_species = atomic_species
        self._name = name or 'NA'

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
    def g_size(self):
        return np.shape(self._g_vectors)[1]

    @property
    def dgvects(self):
        if self._d_g_vectors.size > 0:
            return self._d_g_vectors
        elif self._d_g_vector_values.size > 0:
            g_size = self.g_size
            self._d_g_vectors = np.zeros(
                (self.n_atoms, g_size, self.n_atoms * 3))
            dgv = self.dgvect_values
            dgi1 = self.dgvect_indices1.astype(np.int32)
            dgi2 = self.dgvect_indices2.astype(np.int32)
            for i in np.arange(dgv.size):
                i1, i2 = np.divmod(dgi1[i], g_size)
                self._d_g_vectors[i1, i2, dgi2[i]] = dgv[i]
            return self._d_g_vectors
        raise ValueError("g derivative Not available")

    @property
    def dgvect_values(self):
        if self._d_g_vector_values.size > 0:
            return self._d_g_vector_values
        raise ValueError("sparse g derivative values Not available")

    @property
    def dgvect_indices1(self):
        if self._d_g_vector_indices1.size > 0:
            return self._d_g_vector_indices1
        raise ValueError("sparse g derivative index1 Not available")

    @property
    def dgvect_indices2(self):
        if self._d_g_vector_indices2.size > 0:
            return self._d_g_vector_indices2
        raise ValueError("sparse g derivative index2 Not available")

    @property
    def forces(self):
        if self._forces.size > 0:
            return self._forces
        raise ValueError("forces Not available")

    @property
    def per_atom_quantity(self):
        if self._per_atom_quantity.size > 0:
            return self._per_atom_quantity
        raise ValueError("per atom quantity Not available")

    @property
    def name(self):
        filename, file_extension = os.path.splitext(self._name)
        return filename



def load_example(file_name):
    """ Load binary example

    Parameters
    ----------
      file name: a file name to load

    Return
    ------
      Example object for internal computation
    """

    key = file_name.split('/')[-1]
    with open(file_name, "rb") as binary_file:
        bin_version = int.from_bytes(binary_file.read(4),
                                     byteorder='little',
                                     signed=False)
        if bin_version != 0:
            raise NotImplementedError()
        # converting to int to avoid handling little/big endian
        flags = int.from_bytes(binary_file.read(2),
                               byteorder='little',
                               signed=False)
        derivative_flag = flags & 0b00000001
        force_flag = flags & 0b00000010
        per_atom_quantities_flag = flags & 0b00000100
        sparse_derivative_flag = flags & 0b00001000
        n_atoms = int.from_bytes(binary_file.read(4),
                                 byteorder='little',
                                 signed=False)
        g_size = int.from_bytes(binary_file.read(4),
                                byteorder='little',
                                signed=False)
        payload = binary_file.read()

    # assuming machine that created the binary file is little endian
    data = np.frombuffer(payload, dtype='<f4')

    energy = np.reshape(data[0], [1])[0]

    spec_tensor_bytes = n_atoms
    gvect_tensor_bytes = n_atoms * g_size
    prev_bytes = 1

    spec_tensor = data[prev_bytes:prev_bytes + spec_tensor_bytes]
    spec_tensor = np.int64(np.reshape(spec_tensor, [n_atoms]))

    prev_bytes += spec_tensor_bytes
    gvect_tensor = data[prev_bytes:prev_bytes + gvect_tensor_bytes]
    gvect_tensor = np.reshape(gvect_tensor, [n_atoms, g_size])
    prev_bytes += gvect_tensor_bytes

    if derivative_flag:
        if data.size == prev_bytes:
            raise ValueError('Derivatives requested but not '
                             'present in the file')
        if not sparse_derivative_flag:
            dgvec_tensor_bytes = n_atoms**2 * g_size * 3
            dgvect_tensor = data[prev_bytes:prev_bytes + dgvec_tensor_bytes]
            dgvect_tensor = np.reshape(dgvect_tensor, [n_atoms, g_size, \
                                                       n_atoms*3])
            prev_bytes += dgvec_tensor_bytes
            dgvect_tensor_values = np.empty(0)
            dgvect_tensor_indices1 = np.empty(0)
            dgvect_tensor_indices2 = np.empty(0)
        else:
            dgvect_elements = int(data[prev_bytes])
            prev_bytes += 1
            dgvect_tensor_values = data[prev_bytes:prev_bytes +
                                        dgvect_elements]
            prev_bytes += dgvect_elements
            dgvect_tensor_indices = data[prev_bytes:prev_bytes +
                                         2 * dgvect_elements]
            dgvect_tensor_indices = np.reshape(dgvect_tensor_indices,
                                               [dgvect_elements, 2])
            dgvect_tensor_indices1 = dgvect_tensor_indices[:, 0]
            dgvect_tensor_indices2 = dgvect_tensor_indices[:, 1]
            prev_bytes += 2 * dgvect_elements
            dgvect_tensor = np.empty(0)
    else:
        dgvect_tensor = np.empty(0)
        dgvect_tensor_values = np.empty(0)
        dgvect_tensor_indices1 = np.empty(0)
        dgvect_tensor_indices2 = np.empty(0)

    if force_flag:
        if data.size == prev_bytes:
            raise ValueError('Forces requested but not ' 'present in the file')
        forces_bytes = n_atoms * 3
        forces = data[prev_bytes:prev_bytes + forces_bytes]
        prev_bytes += forces_bytes
    else:
        forces = np.empty(0)

    if per_atom_quantities_flag:
        per_atom_quantity = data[prev_bytes:prev_bytes + n_atoms]
    else:
        per_atom_quantity = np.empty(0)

    return Example(name=key,
                   true_energy=energy,
                   species_vector=spec_tensor,
                   g_vectors=gvect_tensor,
                   d_g_vectors=dgvect_tensor,
                   d_g_vector_values=dgvect_tensor_values,
                   d_g_vector_indices1=dgvect_tensor_indices1,
                   d_g_vector_indices2=dgvect_tensor_indices2,
                   forces=forces,
                   per_atom_quantity=per_atom_quantity)


def iterator_over_tfdata(g_size, *args, **kwargs):
    """ TFdata unpacker

    Args:
        g_size: size of the G's
        args: all the tfdata files that one wants to parse
    kwargs:
        derivatives: bool, whether forces should be calculated
        sparse_derivatives: bool, whether derivatives are stored in sparse form
    Return:
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
                if ('sparse_derivatives' in kwargs
                    ) and kwargs['sparse_derivatives']:
                    d_g_vector_values = example.features.feature[
                        'dgvect_values'].float_list.value
                    d_g_vector_indices1 = example.features.feature[
                        'dgvect_indices1'].float_list.value
                    d_g_vector_indices2 = example.features.feature[
                        'dgvect_indices2'].float_list.value
                    if not len(d_g_vector_values) or \
                       not len(d_g_vector_indices1) or \
                       not len(d_g_vector_indices2):
                        raise ValueError(
                            'Sparse derivatives requested but not '
                            'present in the tfr file')
                    d_g_vectors = []

                else:
                    d_g_vectors_flat = example.features.feature[
                        'dgvects'].float_list.value
                    if not len(d_g_vectors_flat):
                        raise ValueError('Derivatives requested but not '
                                         'present in the tfr file')
                    d_g_vectors = np.reshape(
                        d_g_vectors_flat,
                        (len(species_vector), g_size, len(species_vector) * 3))
                    d_g_vector_values = []
                    d_g_vector_indices1 = []
                    d_g_vector_indices2 = []

                if len(example.features.feature['forces'].float_list.value):
                    #there are reference forces
                    ref_forces = example.features.feature[
                        'forces'].float_list.value
                else:
                    ref_forces = []
            else:
                d_g_vectors = []
                d_g_vector_values = []
                d_g_vector_indices1 = []
                d_g_vector_indices2 = []
                ref_forces = []
                # we dont care if you have reference forces
                # if you are not predicting forces

            yield Example(g_vectors=g_vectors,
                          species_vector=species_vector,
                          true_energy=exact_energy,
                          d_g_vectors=d_g_vectors,
                          d_g_vector_values=d_g_vector_values,
                          d_g_vector_indices1=d_g_vector_indices1,
                          d_g_vector_indices2=d_g_vector_indices2,
                          name=name,
                          forces=ref_forces)

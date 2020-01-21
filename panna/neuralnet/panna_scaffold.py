###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import json
import os
from copy import deepcopy
from functools import partial
from itertools import zip_longest

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logger

from lib.errors import NetworkNotAvailableError

from .a2affnetwork import A2affNetwork
from .tfr_parsers import parse_fn


def _normalize_network_type(nn_type):
    if nn_type in ['a2aff', 'A2AFF', 'ff', 'a2a', 'FF', 'A2A']:
        nn_type = 'a2ff'
    else:
        raise ValueError(
            '{} != a2aff : '
            'Only all-to-all feed forward networks supported'.format(nn_type))
    return nn_type


def _parse_network_base(base_net_params):
    nn_type = base_net_params.get('nn_type', 'a2aff')
    nn_type = _normalize_network_type(nn_type)

    # set architecture
    base_g_size = base_net_params.getint('g_size')
    base_layer_sizes = base_net_params.get_network_architecture('architecture')

    # set trainability
    base_layer_trainable = base_net_params.get_network_trainable(
        'trainable', None)

    # set activations
    base_layer_act = base_net_params.get_network_act('activations', None)

    return nn_type, base_g_size, base_layer_sizes, base_layer_trainable,\
        base_layer_act


def _parse_default_network(default_net_params):
    """ parse a default network for PANNA

    Parameters
    ----------
    default_net_params: default network section from config

    Return
    ------
    A default network and the path to a saved scaffold that can be
    later loaded
    """
    nn_type, default_g_size, default_layer_sizes, default_layer_trainable,\
        default_layer_act = _parse_network_base(default_net_params)

    # search for a default network stored in file
    default_networks_metadata_folder = default_net_params.get(
        'networks_metadata', None)

    if default_networks_metadata_folder:
        logger.warning('Networks in %s will be loaded',
                       default_networks_metadata_folder)

    # set architecture
    default_g_size = default_net_params.getint('g_size')
    default_layer_sizes = default_net_params.get_network_architecture(
        'architecture')
    if default_g_size and default_layer_sizes:
        logger.info('Found a default network: {}'.format([default_g_size] +
                                                         default_layer_sizes))
        logger.info('This network size will be used as '
                    'default for all species unless specified otherwise')
    else:
        return None, default_networks_metadata_folder

    if default_layer_trainable:
        logger.info(
            'Found a default trainability: {}'.format(default_layer_trainable))
    else:
        logger.warning('Set to default trainability: all trainable')

    if default_layer_act:
        logger.warning('Found a default activation list: '
                       '{}'.format(default_layer_act))
    else:
        # default is gaussian:gaussian: ... : linear
        logger.info('Set to default activation: ' 'gaussians + last linear')

    # search for a default network stored in file
    default_networks_metadata_folder = default_net_params.get(
        'networks_metadata', None)

    if default_networks_metadata_folder:
        logger.warning('Networks in %s will be loaded',
                       default_networks_metadata_folder)
    if default_g_size and default_layer_sizes:
        network = A2affNetwork(default_g_size, default_layer_sizes, 'default',
                               None, default_layer_trainable,
                               default_layer_act)
    else:
        network = None

    return network, default_networks_metadata_folder


def _parse_specie_network(species_name, species_config, system_scaffold):
    """ parse a specie config

    Parameters
    ---------
    species_name: a species name
    species_config: a species config
    system_scaffold: a scaffold where to perform the operation of storing
                    the species network
    """

    nn_type, g_size, layers_size, layers_trainable,\
        layers_act = _parse_network_base(species_config)

    # layers behavior flag:
    layers_behavior = species_config.get_network_behavior('behavior', None)
    # output_offset
    output_offset = species_config.getfloat('output_offset', None)

    # network_wb
    override_wb = {}
    if layers_behavior:
        for layer_number, behavior in enumerate(layers_behavior):
            if behavior == 'load':
                w_file_name = species_config.get(
                    'layer{}_w_file'.format(layer_number))
                b_file_name = species_config.get(
                    'layer{}_b_file'.format(layer_number))
                if not (w_file_name and b_file_name):
                    raise ValueError('not passed file names')
                override_wb[layer_number] = (np.load(w_file_name),
                                             np.load(b_file_name))

    def _local_log_helper(network):
        spe = '=={}== '.format(species_name)
        if layers_size:
            logger.info(
                spe +
                'New network architecture: {}'.format(network.layers_size))
        else:
            logger.warning(spe + 'Architecture flag was not specified '
                           '- set to {}'.format([network.feature_size] +
                                                network.layers_size))

        if layers_trainable:
            logger.info(
                spe +
                'New trainable flags: {}'.format(network.layers_trainable))
        else:
            logger.warning(spe + 'Trainable flag was not specified '
                           '- set to {}'.format(network.layers_trainable))

        if layers_behavior:
            logger.info(spe +
                        'New layer behavior flags: {}'.format(layers_behavior))

        if layers_act:
            logger.info(
                spe + 'New activations : {}'.format(network.layers_activation))
        else:
            logger.warning(spe + 'Activations flag is not specified '
                           '- set to {}'.format(network.layers_activation))
        if override_wb:
            for k in override_wb:
                logger.warning(spe + 'override layer {}'.format(k))
        else:
            logger.info(spe + 'No override asked')

        if output_offset:
            logger.info(spe + 'New output offset: {}'.format(network.offset))
        else:
            logger.warning(spe + 'Output offset flag was not specified '
                           '- set to {}'.format(network.offset))

    _load_network_on_scaffold(system_scaffold,
                              species_name,
                              network_type=nn_type,
                              g_size=g_size,
                              layers_size=layers_size,
                              layers_trainable=layers_trainable,
                              layers_behavior=layers_behavior,
                              layers_act=layers_act,
                              output_offset=output_offset,
                              override_wb=override_wb)
    _local_log_helper(system_scaffold[species_name])


def _read_network_tensors(network_layers_size, network_files, folder):
    init_wb = {}
    layers_behaviour = ['new'] * len(network_layers_size)
    if network_files:
        for layer, (w_file, b_file) in enumerate(network_files):
            weight = np.load(os.path.join(folder, w_file))
            bias = np.load(os.path.join(folder, b_file))
            init_wb[layer] = weight, bias
            layers_behaviour[layer] = 'load'
    return layers_behaviour, init_wb


def _read_networks_tensors(networks_layers_size, networks_files, folder):
    networks_init_wb = []
    networks_layers_behaviour = []
    for network_layers_size, network_files in zip_longest(
            networks_layers_size, networks_files):
        layers_behaviour, init_wb = _read_network_tensors(
            network_layers_size, network_files, folder)
        networks_init_wb.append(init_wb)
        networks_layers_behaviour.append(layers_behaviour)
    return networks_layers_behaviour, networks_init_wb


def _load_network_on_scaffold(system_scaffold,
                              species_name,
                              *,
                              network_type=None,
                              g_size=None,
                              layers_size=None,
                              layers_trainable=None,
                              layers_behavior=None,
                              layers_act=None,
                              output_offset=None,
                              override_wb=None):
    try:
        species_network = system_scaffold[species_name]
        if species_network.network_type != network_type:
            raise ValueError('Undefined behavior, '
                             'customization is not possible')

        species_network.customize_network(g_size, layers_size,
                                          layers_trainable, layers_behavior,
                                          layers_act, output_offset,
                                          override_wb)
    except NetworkNotAvailableError:
        if not layers_size:
            raise ValueError('not available network structure')
        # triggered only if a default fall back is not present
        network_wb = [(np.empty(0), np.empty(0))
                      for i in range(len(layers_size))]
        for key, value in override_wb.items():
            network_wb[key] = value

        if network_type == 'a2ff':
            species_network = A2affNetwork(g_size, layers_size, species_name,
                                           network_wb, layers_trainable,
                                           layers_act, output_offset)
        else:
            raise ValueError('type not supported {}'.format(network_type))

        system_scaffold[species_name] = species_network


class PannaScaffold():
    """ A recipe for the PANNA basic network

    Parameters
    ----------
    config: optional, a config to extract the quantities needed for the
            networks setup
    name: optional, a name for the network, if more are present

    If no config is provide scaffold is empty
    """
    def __init__(self, config=None, name='PANNA_scaffold'):
        super().__init__()

        self._name = name
        self._scaffolf_type = 'PANNA'

        self._default_network = None
        self._atomic_sequence = []
        self._networks = {}
        self._zeros = {}
        self._usable_for_training = False
        self._usable_for_validation = False

        if not config:
            return

        #=== Data option part ===
        self._en_rescale = 1.0
        self._sparse_derivatives = False
        if 'DATA_INFORMATION' in config:
            data_params = config['DATA_INFORMATION']
            self._atomic_sequence = data_params.get_comma_list(
                'atomic_sequence', self._atomic_sequence)
            offsets = data_params.get_comma_list_floats('output_offset', [])
            self._en_rescale = data_params.getfloat('energy_rescale',
                                                    self._en_rescale)
            self._sparse_derivatives = data_params.getboolean(
                'sparse_derivatives', self._sparse_derivatives)
        else:
            offsets = []
        if offsets and self._atomic_sequence:
            if len(offsets) != len(self._atomic_sequence):
                raise ValueError('output offset must be of '
                                 'the same size as atomic sequence')
            self._zeros = dict(zip(self._atomic_sequence, offsets))
        elif offsets and not self._atomic_sequence:
            raise ValueError('output offset can not be specified '
                             'without atomic_sequence')

        # === NETWORK option part ===
        # default network to be loaded
        if 'DEFAULT_NETWORK' in config:
            default_net_params = config['DEFAULT_NETWORK']
            self._default_network, default_networks_metadata_folder = \
                _parse_default_network(default_net_params)
        else:
            default_networks_metadata_folder = None
            logger.info('No default network is found, '
                        'all species must be fully specified.')

        #load species networks form metadata
        if default_networks_metadata_folder:
            self.load_panna_checkpoint_folder(default_networks_metadata_folder)

        # parsing single atomic species
        # if a default is specified every species behave as the default
        # if no default is specified than every species must be fully described
        for species in self._atomic_sequence:
            if species in config:
                logger.info(
                    '=={}== Found network specifications'.format(species))
                species_config = config[species]
                _parse_specie_network(species, species_config, self)

        #=== Training option specific part ===
        if 'TRAINING_PARAMETERS' in config:
            train_params = config['TRAINING_PARAMETERS']
            self._loss_func = train_params.get('loss_function', 'quad')
            self._floss_func = train_params.get('floss_function', 'quad')
            self._forces_cost = train_params.getfloat('forces_cost', 0.0)
        else:
            self._loss_func = 'quad'
            self._floss_func = 'quad'
            self._forces_cost = 0.0

        if self._forces_cost > 0:
            self._tf_require_gradients = True
        else:
            self._tf_require_gradients = False

        self._usable_for_training = True

    def __getitem__(self, value):
        """ recover a network
        """
        tmp_network = self._networks.get(value, None)

        if tmp_network:
            return tmp_network

        if self._default_network:
            tmp_network = deepcopy(self._default_network)
            tmp_network.name = value
            tmp_network.offset = self._zeros.get(value, 0.0)
            self._networks[value] = tmp_network
            if value not in self._atomic_sequence:
                self._atomic_sequence.append(value)
            return tmp_network

        raise NetworkNotAvailableError('default network not available')

    def __setitem__(self, index, value):
        """ set a network
        - if the network exists it can not be overwritten but must be
          recover and changed in place
        - if the network is not stored in the atomic sequence then it will be added
          as the last one
        - zero value will be stored when setting happen
        """
        if index in self._networks:
            raise ValueError('network already present, '
                             'recover with getter and change in place')
        if index != value.name:
            raise ValueError('assign inconsistent '
                             '{} != {}'.format(index, value.name))
        self._zeros[index] = value.offset
        if index not in self._atomic_sequence:
            self._atomic_sequence.append(index)
        self._networks[index] = value

    def _load(self, data, folder='.'):
        networks_files = data.get('networks_files', [])
        networks_layers_behaviour, networks_init_wb = _read_networks_tensors(
            data['networks_layers_size'], networks_files, folder)

        for i, network_type in enumerate(data['networks_type']):
            init_wb = networks_init_wb[i]
            layers_behaviour = networks_layers_behaviour[i]
            name = data['networks_species'][i]

            if name in self._zeros:
                offset = self._zeros[name]
            else:
                offset = data['networks_offset'][i]
                logger.warning(
                    'offset not specified - set to %f from checkpoint', offset)

            _load_network_on_scaffold(
                self,
                name,
                network_type=network_type,
                g_size=data['networks_feature_size'][i],
                layers_size=data['networks_layers_size'][i],
                layers_act=data['networks_layers_activation'][i],
                layers_trainable=data['networks_layers_trainable'][i],
                layers_behavior=layers_behaviour,
                override_wb=init_wb,
                output_offset=offset)
            logger.info('loading %s species from checkpoint', name)

        self._usable_for_validation = True

    def load_panna_checkpoint_folder(
            self,
            folder,
            *,
            file_name='networks_metadata.json',
    ):
        """ set up internal value form a checkpoint
        """

        info_file = os.path.join(folder, file_name)
        if not os.path.isfile(info_file):
            raise ValueError('Please specify a valid panna ckpt.')

        with open(info_file) as file_stream:
            data = json.load(file_stream)
        self._load(data, folder)

    def load_panna_metadata(self, data):
        self._load(data)

    def evaluate(self, example, forces_flag=False, add_offset=True):
        """ evaluate the network with numpy

        Parameters
        ----------
            example:
            force_flag:
            add_offset:

        Return
        ------
            numpy array
        """
        n_atoms = example.n_atoms
        species_vect, g_vects = example.species_vector, example.gvects
        energy = 0

        if forces_flag:
            forces = np.zeros(n_atoms * 3)
            dgs = example.dgvects

        for species_idx, species_symbol in enumerate(self.atomic_sequence):
            network = self[species_symbol]

            # select per species gvects
            species_indices = np.where(species_vect == species_idx)
            if len(species_indices[0]) > 0:
                in_gs = g_vects[species_indices]
                if forces_flag:
                    in_dgs = dgs[species_indices]
                    energies, minus_partial_forces = network.evaluate(
                        in_gs, in_dgs, add_offset)
                    forces -= np.sum(minus_partial_forces, axis=(0))
                else:
                    energies = network.evaluate(in_gs, add_offset=add_offset)

                energy += energies.sum()

        if forces_flag:
            return energy, forces
        return energy

    def tf_evaluate(self, batch_of_species, batch_of_gvects, batch_size):
        """ perform a tensorflow evaluation
        """
        #Species sorted list of lists [specie 1 list ,specie 2 list,..]:
        #atomic energies for all the atoms in the batch, species sorted

        # hack, for now we support only one feature size
        g_size = self[self.atomic_sequence[0]].feature_size

        atomic_energies = []
        # for each atom, the example number in the batch,
        # to keep track of where they came from, species sorted
        max_number_atoms = tf.shape(batch_of_species)[1]
        partitions = tf.cast(tf.reshape(batch_of_species, [-1]), tf.int32)
        array_gvects = tf.reshape(batch_of_gvects, [-1, g_size])
        atomic_index_list = tf.dynamic_partition(
            tf.range(batch_size * max_number_atoms), partitions,
            self.n_species + 1)

        # List of derivatives wrt Gs, needed for differentiation
        des_dgs = []

        for species_idx, species_symbol in enumerate(self.atomic_sequence):
            logger.debug('crating network: %d, %s', species_idx,
                         species_symbol)
            # recover the network
            network = self[species_symbol]

            s_n_atoms = tf.shape(atomic_index_list[species_idx])[0]
            s_input_features = tf.gather(array_gvects,
                                         atomic_index_list[species_idx])

            if self._tf_require_gradients:
                s_atomic_energies, s_des_dgs = network.tf_evaluate(
                    s_input_features, True)
                # dEs_dGs has only one element by construction
                s_des_dgs = s_des_dgs[0]
                des_dgs.append(s_des_dgs)
            else:
                s_atomic_energies = network.tf_evaluate(s_input_features)

            s_atomic_energies = tf.reshape(s_atomic_energies, [-1])

            mean, var = tf.nn.moments(s_atomic_energies, axes=[0])
            tf.summary.scalar(
                "2.Mean_energy/S{}.{}".format(species_idx, species_symbol),
                mean)
            tf.summary.scalar(
                "3.Std_energy/S{}.{}".format(species_idx, species_symbol),
                tf.sqrt(var))
            tf.summary.scalar(
                "4.N_atoms/S{}.{}".format(species_idx, species_symbol),
                s_n_atoms)

            atomic_energies.append(s_atomic_energies)

        # recover how many empty slot we have in the species matrix
        # and fill the predicted energy for those with zeros
        n_placeholder_in_species_matrix = tf.shape(
            atomic_index_list[self.n_species])[0]
        fake_species_e_contrib = tf.zeros(n_placeholder_in_species_matrix)

        atomic_energies.append(fake_species_e_contrib)

        # reconstruct the matrix
        reconstructed_energies_matrix = tf.dynamic_stitch(
            atomic_index_list, atomic_energies)
        reconstructed_energies_matrix = tf.reshape(
            reconstructed_energies_matrix, [batch_size, max_number_atoms])

        if self._tf_require_gradients:
            des_dgs.append(tf.zeros([n_placeholder_in_species_matrix, g_size]))
            reconstructed_des_dgs_matrix = tf.dynamic_stitch(
                atomic_index_list, des_dgs)
            reconstructed_des_dgs_matrix = tf.reshape(
                reconstructed_des_dgs_matrix,
                [batch_size, max_number_atoms, g_size])

        # create a hot matrix to count atom per example
        atoms_presence = tf.where(tf.less(batch_of_species, self.n_species),
                                  tf.ones(tf.shape(batch_of_species)),
                                  tf.zeros(tf.shape(batch_of_species)))
        batch_n_atoms = tf.reduce_sum(atoms_presence, axis=1)

        # helper
        _find_idx_from_str = {
            symbol: idx
            for idx, symbol in enumerate(self.atomic_sequence)
        }

        for x in tf.get_collection(tf.GraphKeys.WEIGHTS):
            name = x.op.name.split("/")[0]
            nameparts = name.split("_")
            tf.summary.histogram(
                "S{}.{}/W{}".format(nameparts[1],
                                    _find_idx_from_str[nameparts[1]],
                                    nameparts[3]), x)
        for x in tf.get_collection(tf.GraphKeys.BIASES):
            name = x.op.name.split("/")[0]
            nameparts = name.split("_")
            tf.summary.histogram(
                "S{}.{}/b{}".format(nameparts[1],
                                    _find_idx_from_str[nameparts[1]],
                                    nameparts[3]), x)

        if self._tf_require_gradients:
            return reconstructed_energies_matrix, batch_n_atoms, \
                   reconstructed_des_dgs_matrix

        return reconstructed_energies_matrix, batch_n_atoms

    def tf_energy_loss(self, batch_delta_e, batch_natoms):
        """ Cost function for this scaffold

        Parameters
        ----------
            batch_delta_e: energy differences
            batch_natoms: number of atoms

        Returns
        -------
            the loss value
            tensor with delta_e for each element of the batch
        """

        with tf.name_scope('energy_loss_calculation'):

            # loss function
            batch_delta_e2 = tf.square(batch_delta_e)

            if self._loss_func == "quad":
                tot_loss = tf.reduce_sum(batch_delta_e2,
                                         name='1.LOSS-Delta_E2')
            elif self._loss_func == "exp_quad":
                tot_loss = 0.5 * tf.exp(2.0 * tf.reduce_sum(batch_delta_e2),
                                        name='1.LOSS-Exp_Delta_E2')
            elif self._loss_func == "quad_atom":
                tot_loss = tf.reduce_sum(tf.div(batch_delta_e2,
                                                tf.square(batch_natoms)),
                                         name='1.LOSS-Delta_E2_div_Natom2')
            elif self._loss_func == "quad_std_atom":
                tot_loss = tf.reduce_sum(tf.div(batch_delta_e2, batch_natoms),
                                         name='1.LOSS-Delta_E2_div_Natom')
            elif self._loss_func == "exp_quad_atom":
                tot_loss = 0.5 * tf.exp(2.0 * tf.reduce_sum(
                    tf.div(batch_delta_e2, tf.square(batch_natoms))),
                                        name='1.LOSS-Exp_Delta_E2_div_Natom2')
            elif self._loss_func == "exp_quad_std_atom":
                tot_loss = 0.5 * tf.exp(
                    2.0 * tf.reduce_sum(tf.div(batch_delta_e2, batch_natoms)),
                    name='1.LOSS-Exp_Delta_E2_div_Natom')
            elif self._loss_func == "quad_exp_tanh_atom":
                const = tf.constant(5.0)
                red_sum = tf.div(
                    tf.reduce_sum(
                        tf.div(batch_delta_e2, tf.square(batch_natoms))),
                    const)
                tot_loss = tf.add(
                    tf.reduce_sum(batch_delta_e2),
                    tf.exp(tf.multiply(const, tf.tanh(red_sum))),
                    name='1.LOSS-quad_exp_tanh_Delta_E2_div_Natom2')
            elif self._loss_func == "quad_exp_tanh_std_atom":
                const = tf.constant(5.0)
                red_sum = tf.div(
                    tf.reduce_sum(tf.div(batch_delta_e2, batch_natoms)), const)
                tot_loss = tf.add(
                    tf.reduce_sum(batch_delta_e2),
                    tf.exp(tf.multiply(const, tf.tanh(red_sum))),
                    name='1.LOSS-quad_exp_tanh_Delta_E2_div_Natom')
            elif self._loss_func == "quad_exp_tanh":
                const = tf.constant(5.0)
                red_sum = tf.div(tf.reduce_sum(batch_delta_e2), const)
                tot_loss = tf.add(tf.reduce_sum(batch_delta_e2),
                                  tf.exp(tf.multiply(const, tf.tanh(red_sum))),
                                  name='1.LOSS-quad_exp_tanh_Delta_E2')

            with tf.name_scope('graphs_variable'):
                mean_delta_e = tf.reduce_mean(batch_delta_e)
                mean_delta_e2 = tf.reduce_mean(batch_delta_e2)
                energy_rmse = tf.sqrt((mean_delta_e2 - mean_delta_e**2),
                                      name='2.Energy_RMSE')

                mean_delta_e_atom = tf.reduce_mean(
                    tf.div(batch_delta_e, batch_natoms))
                mean_delta_e2_atom = tf.reduce_mean(
                    tf.div(batch_delta_e2, tf.square(batch_natoms)))
                energy_rmse_atom = tf.sqrt(
                    (mean_delta_e2_atom - mean_delta_e_atom**2),
                    name='3.Energy_RMSE_atom')

        tf.add_to_collection('losses', tot_loss)
        tf.add_to_collection('losses', energy_rmse_atom)
        tf.add_to_collection('losses', energy_rmse)

        return tot_loss, batch_delta_e

    def tf_single_force_evaluate_dense(self, x):
        """ Function operating on each element of the batch.
        x contains: (Natoms, gsize, dEdg, dGdx, F_ref)
        Elements are sliced (if padded) and reshaped to compute
        F_k = \sum_{ij} dE/dg_{ij} dg{ij}/dx_k
        Then sum of square differences is computed so we return a number

        """
        Natoms = tf.cast(x[0], tf.int64)
        gsize = x[1]
        dGdx = tf.reshape(x[3][:Natoms**2 * gsize * 3],
                          [Natoms * gsize, Natoms * 3])

        dEdg = tf.reshape(x[2][:Natoms, :], [1, -1])
        NNF = tf.reshape(tf.matmul(dEdg, dGdx), [-1])
        # Version with matvec:
        # dEdg = tf.reshape(x[2][:Natoms, :], [-1])
        # NNF = tf.linalg.matvec(dGdx, dEdg, transpose_a=True)

        # NNF is -force
        Fdiff = NNF + x[4][:Natoms * 3]
        Floss = tf.reduce_sum(tf.square(Fdiff))
        return Floss

    def tf_single_force_evaluate_sparse(self, x):
        """ Function operating on each element of the batch.
        x contains: (Natoms, gsize, dEdg, dGdx_size, dGdx_values, dGdx_ind1, dGdx_ind2, F_ref)
        Elements are sliced (if padded) and sparse matrix is reconstructed to compute
        F_k = \sum_{ij} dE/dg_{ij} dg{ij}/dx_k
        Then sum of square differences is computed so we return a number

        """
        Natoms = tf.cast(x[0], tf.int64)
        gsize = x[1]
        dGdxsize = x[3]
        dEdg = tf.reshape(x[2][:Natoms, :], [-1, 1])
        dGdx = tf.SparseTensor(indices=tf.cast(tf.stack(
            [x[5][:dGdxsize], x[6][:dGdxsize]], axis=1),
                                               dtype=tf.int64),
                               values=x[4][:dGdxsize],
                               dense_shape=[Natoms * gsize, Natoms * 3])
        NNF = tf.sparse.matmul(dGdx, dEdg, adjoint_a=True)

        # Alternative code with sparse_to_dense (seems to be slower)
        # dGdxDense = tf.sparse_to_dense(
        #             sparse_indices=tf.cast(
        #                 tf.stack([x[5][:dGdxsize], x[6][:dGdxsize]], axis=1), dtype=tf.int64),
        #             output_shape=[Natoms*gsize, Natoms*3],
        #             sparse_values=x[4][:dGdxsize])
        # NNF = tf.matmul(dGdxDense, dEdg, transpose_a=True, a_is_sparse=True)

        # NNF is -force
        Fdiff = NNF[:, 0] + x[7][:Natoms * 3]
        Floss = tf.reduce_sum(tf.square(Fdiff))
        return Floss

    def tf_force_evaluate(self,
                          batch_size,
                          batch_dEdG,
                          batch_forces_ref,
                          batch_natoms,
                          sparse_dgvect=False,
                          batch_dgvect=None,
                          batch_dgvect_size=None,
                          batch_dgvect_values=None,
                          batch_dgvect_indices1=None,
                          batch_dgvect_indices2=None):
        g_size = self[self.atomic_sequence[0]].feature_size
        batch_gsize = g_size * tf.ones(batch_size, dtype=tf.int64)

        # Executing single force calculation for each element of the batch
        if not sparse_dgvect:
            sumFdiff = tf.map_fn(self.tf_single_force_evaluate_dense,
                                 (batch_natoms, batch_gsize, batch_dEdG,
                                  batch_dgvect, batch_forces_ref),
                                 dtype=tf.float32,
                                 parallel_iterations=batch_size,
                                 infer_shape=False)
        else:
            sumFdiff = tf.map_fn(
                self.tf_single_force_evaluate_sparse,
                (batch_natoms, batch_gsize, batch_dEdG, batch_dgvect_size,
                 batch_dgvect_values, batch_dgvect_indices1,
                 batch_dgvect_indices2, batch_forces_ref),
                dtype=tf.float32,
                parallel_iterations=batch_size,
                infer_shape=False)

        return sumFdiff

    def tf_force_loss(self, sumFdiff, batch_natoms):
        """this is the cost function contribution from forces

        Parameters
        ----------
            sumFdiff (sum of square difference of forces per example),
            batch_natoms,

        Returns
        -------
            the loss value for the forces term

        """

        if self._floss_func == "quad":
            force_loss = tf.reduce_sum(sumFdiff, name='5.F_Loss')
        elif self._floss_func == "exp_quad":
            force_loss = 0.5 * tf.exp(2.0 * tf.reduce_sum(sumFdiff),
                                      name='5.F_Loss')
        elif self._floss_func == "quad_atom":
            force_loss = tf.reduce_sum(tf.div(sumFdiff,
                                              tf.square(batch_natoms)),
                                       name='5.F_Loss')
        elif self._floss_func == "exp_quad_atom":
            force_loss = 0.5 * tf.exp(
                2.0 * tf.reduce_sum(tf.div(sumFdiff, tf.square(batch_natoms))),
                name='5.F_Loss')

        tf.add_to_collection('losses', force_loss)

        return force_loss

    def tf_network(self, batch_size, batch_dict):
        """ helper to combine all the evaluation

        Parameters
        ----------
            batch_size: number of elements in the batch
            batch_dict: containing:
                batch_of_species:
                batch_of_gvects:
                batch_size:
                batch_energy_ref:
                [opt] batch_of_dgvect: (or sparse variant)
                [opt] batch_forces_ref:
        Return
        ------

        """
        # Parse input dictionary
        batch_of_species = batch_dict['species']
        batch_of_gvects = batch_dict['gvects']
        batch_energies_ref = batch_dict['energy']
        if self._tf_require_gradients:
            batch_forces_ref = batch_dict['forces']
            if not self._sparse_derivatives:
                batch_of_dgvect = batch_dict['dgvects']
            else:
                batch_dgvect_size = batch_dict['dgvect_size']
                batch_dgvect_values = batch_dict['dgvect_values']
                batch_dgvect_indices1 = batch_dict['dgvect_indices1']
                batch_dgvect_indices2 = batch_dict['dgvect_indices2']

        if self._tf_require_gradients:
            prediction_matrix, batch_n_atoms, des_dgs = self.tf_evaluate(
                batch_of_species, batch_of_gvects, batch_size)
        else:
            prediction_matrix, batch_n_atoms = self.tf_evaluate(
                batch_of_species, batch_of_gvects, batch_size)

        prediction = tf.reduce_sum(prediction_matrix, axis=1)
        with tf.name_scope('energy_loss_calculation'):
            # reshape for convenience[100,1]->[100]
            batch_energies_ref = tf.reshape(batch_energies_ref, [-1],
                                            name='reshape_ref_en')
            batch_delta_e = prediction - batch_energies_ref

        energy_cost, _batch_deltae = self.tf_energy_loss(
            batch_delta_e, batch_n_atoms)

        if self._tf_require_gradients:
            if not self._sparse_derivatives:
                sumFdiff = self.tf_force_evaluate(batch_size,
                                                  des_dgs,
                                                  batch_forces_ref,
                                                  batch_n_atoms,
                                                  sparse_dgvect=False,
                                                  batch_dgvect=batch_of_dgvect)
            else:
                sumFdiff = self.tf_force_evaluate(
                    batch_size,
                    des_dgs,
                    batch_forces_ref,
                    batch_n_atoms,
                    sparse_dgvect=True,
                    batch_dgvect_size=batch_dgvect_size,
                    batch_dgvect_values=batch_dgvect_values,
                    batch_dgvect_indices1=batch_dgvect_indices1,
                    batch_dgvect_indices2=batch_dgvect_indices2)
            force_cost = self.tf_force_loss(sumFdiff, batch_n_atoms)
            energy_cost += self._forces_cost * force_cost

        return energy_cost, batch_delta_e

    @property
    def tfr_parse_function(self):
        # same trick done in tf_evaluate
        if not self._usable_for_training:
            raise ValueError('this scaffold is missing training parameters')
        g_size = self[self.atomic_sequence[0]].feature_size
        zeros = np.asarray(
            [self._zeros[species] for species in self.atomic_sequence],
            dtype=np.float32)

        return partial(parse_fn,
                       g_size=g_size,
                       zeros=zeros,
                       n_species=self.n_species,
                       forces=self._tf_require_gradients,
                       sparse_dgvect=self._sparse_derivatives,
                       energy_rescale=self._en_rescale)

    def _base_metadata(self):
        """ metadata that are always available
        """
        metadata = {}
        metadata['verison'] = 'v0'
        metadata['scaffold_type'] = self._scaffolf_type
        metadata['networks_type'] = [
            self[x].network_type for x in self.atomic_sequence
        ]
        metadata['networks_layers_size'] = [
            self[x].layers_size for x in self.atomic_sequence
        ]
        metadata['networks_feature_size'] = [
            self[x].feature_size for x in self.atomic_sequence
        ]
        metadata['networks_layers_activation'] = [
            self[x].layers_activation for x in self.atomic_sequence
        ]
        metadata['networks_layers_trainable'] = [
            self[x].layers_trainable for x in self.atomic_sequence
        ]
        metadata['networks_species'] = self.atomic_sequence
        metadata['networks_offset'] = [
            self[x].offset for x in self.atomic_sequence
        ]
        return metadata

    def _networks_file_metadata(self):
        # FIXME, I moved this code here for now, but this can be
        # done in a better way
        networks_files_name = []
        networks_tensors = []

        for species in self.atomic_sequence:
            network_files_name = []
            network_tensors = []
            network = self[species]
            for idx_l, layer in enumerate(network):
                w_size = layer.wb_shape
                b_size = layer.b_shape
                w_file_name = ('species_{}_layer_{}_weights_'
                               '{}x{}.npy'.format(
                                   species,
                                   idx_l,
                                   *w_size,
                               ))
                b_file_name = ('species_{}_layer_{}_biases_'
                               '{}.npy'.format(
                                   species,
                                   idx_l,
                                   *b_size,
                               ))
                network_files_name.append((w_file_name, b_file_name))
                network_tensors.append((layer.w_value, layer.b_value))

            networks_files_name.append(network_files_name)
            networks_tensors.append(network_tensors)
        return networks_files_name, networks_tensors

    @property
    def metadata(self):
        metadata = self._base_metadata()
        return metadata

    @property
    def ckpt_metadata(self):
        metadata = self._base_metadata()
        networks_files_name, networks_tensors = self._networks_file_metadata()
        metadata['networks_files'] = networks_files_name
        return metadata, networks_tensors

    @property
    def name(self):
        return self._name

    @property
    def atomic_sequence(self):
        return tuple(self._atomic_sequence)

    @property
    def n_species(self):
        return len(self._atomic_sequence)

    @property
    def default_network(self):
        return deepcopy(self._default_network)

    @default_network.setter
    def default_network(self, value):
        if not self._default_network:
            self._default_network = value
        else:
            raise ValueError('default network can not be changed')

    @property
    def sparse_derivative(self):
        if self._usable_for_training:
            return self._sparse_derivatives
        else:
            raise ValueError('Network not usable for training')

    @property
    def tf_ckpt_elements(self):
        """ regexps for TF checkpoint

        Return
        ------
        stirngs  + objects
        """
        names = []
        objects = []
        for species_symbol in self.atomic_sequence:
            names.append('')
            network = self[species_symbol]
            objects.append(network)
        return names, objects

""" Code use to evaluate a dataset on a checkpoint/s
note:
    if batchsize == -1 then use the whole dataset
"""
import os
import logging
import argparse
import configparser
import random as rnd
import numpy as np
import multiprocessing as mp

import neuralnet as net
from neuralnet.systemscaffold import SystemScaffold
import neuralnet.parser_callable as parser_callable

# logger
logger = logging.getLogger('logfile')
formatter = logging.Formatter('%(asctime)s - %(name)s - \
    %(levelname)s - %(message)s')

# console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def _eval_function(example, scaffold, add_offset):
    """Helper for parallelization

    Args:
        example: a example obj
        scaffold: a scaffold obj
    Return:
        a string: filename, n_natoms, dft, prediction

    """
    en_prediction = scaffold.evaluate(example, add_offset=add_offset)
    en_dft = example.true_energy
    string = '{} {} {} {}'.format(example.name, example.n_atoms, en_dft,
                                  en_prediction)
    return string


def _eval_function_forces(example, scaffold, add_offset):
    """Helper for parallelization

    Args:
        example: a example obj
        scaffold: a scaffold obj
    Return:
        a string: n_natoms, dft, prediction

    """
    en_prediction, forces = scaffold.evaluate(example, True, add_offset)
    en_dft = example.true_energy
    string = '{} {} {} {}'.format(example.name, example.n_atoms, en_dft,
                                  en_prediction)
    forces_reshape = forces.reshape(int(len(forces) / 3), 3)
    string2_list = [
        example.name + ' {} '.format(idx) + ' '.join([str(f) for f in line])
        for idx, line in enumerate(forces_reshape)
    ]
    string2 = '\n'.join(string2_list) + '\n'
    return string, string2


def parse_file(conf_file):
    """ Parse validation config file
    """
    config = configparser.ConfigParser(
        converters={
            '_comma_list': parser_callable.get_list_from_comma_sep_strings,
            '_comma_list_floats': parser_callable.\
                get_list_floats_from_comma_sep_strings,
            '_network_architecture': parser_callable.get_network_architecture,
            '_network_trainable': parser_callable.get_network_trainable
        }
    )
    config.read(conf_file)

    class IOParameters():
        pass

    io_parameters = IOParameters()
    # recover parameters
    io_info = config['IO_INFORMATION']
    io_parameters.train_dir = io_info.get('train_dir')
    io_parameters.eval_dir = io_info.get('eval_dir')
    io_parameters.data_dir = io_info.get('data_dir')

    # if network is PANNA there are no problem,
    io_parameters.networks_format = io_info.get('networks_format',
                                                'checkpoint')
    if io_parameters.networks_format == 'PANNA':
        io_parameters.networks_folder = io_info.get('networks_folder',
                                                    './saved_networks')

    io_parameters.example_format = io_info.get('example_format', 'TFR')

    io_parameters.number_of_process = 4
    if config.has_section('PARALLELIZATION'):
        parallel = config['PARALLELIZATION']
        io_parameters.number_of_process = parallelization.getint(
            'number_of_process', number_of_process)

    class DataParameters():
        pass

    data_parameters = DataParameters()

    data_information = config['DATA_INFORMATION']
    data_parameters.atomic_sequence = data_information.get_comma_list(
        'atomic_sequence')

    data_parameters.n_species = len(data_parameters.atomic_sequence)

    data_parameters.g_size = data_information.getint('g_size')
    zeros = data_information.get_comma_list_floats(
        'zeros', [0.0 for x in data_parameters.atomic_sequence])
    zeros = np.asarray(zeros)

    class ValidationParameters():
        pass

    validation_parameters = ValidationParameters()
    validation_options = config['VALIDATION_OPTIONS']

    validation_parameters.compute_forces = validation_options.getboolean(
        'compute_forces', False)

    validation_parameters.batch_size = validation_options.getint(
        'batch_size', -1)
    if io_parameters.example_format == 'TFR':
        logger.info('batch_size set to -1,'
                    ' TFR does not support this option')
        validation_parameters.batch_size = -1
    validation_parameters.single_step = validation_options.getboolean(
        'single_step', False)
    validation_parameters.step_number = validation_options.getint(
        'step_number', None)
    validation_parameters.subsampling = validation_options.getint(
        'subsampling', None)
    validation_parameters.add_offset = validation_options.getboolean(
        'add_offset', True)

    networks_kind = []
    networks_metadata = []

    if io_parameters.networks_format == 'checkpoint':
        if 'DEFAULT_NETWORK' in config:
            default_net_params = config['DEFAULT_NETWORK']
            # set network type
            default_nn_type = default_net_params.get('nn_type', 'a2aff')
            if default_nn_type in ['a2aff', 'A2AFF', 'ff', 'a2a', 'FF', 'A2A']:
                default_nn_type = 'a2ff'
            else:
                raise ValueError(
                    '{} != a2aff : '
                    'Only all-to-all feed forward networks supported'.format(
                        nn_type))

            # set activations
            default_layer_act = default_net_params.get_network_act(
                'activations', None)
            if default_layer_act:
                logger.waring('Found a default activation list: '
                              '{}'.format(default_layer_act))
            else:
                # default is gaussian:gaussian: ... : linear
                logger.info('Set to default activation: '
                            'gaussians + last linear')
        else:
            default_nn_type = 'a2ff'
            default_layer_act = None

        for species, zero in zip(data_parameters.atomic_sequence, zeros):
            if species in config:
                logger.info(
                    '=={}== Found network specifications'.format(species))
                species_config = config[species]

                spe = '=={}== '.format(species)
                nn_type = default_net_params.get('nn_type', 'a2aff')
                if nn_type in ['a2aff', 'A2AFF', 'ff', 'a2a', 'FF', 'A2A']:
                    nn_type = 'a2ff'
                else:
                    raise ValueError(
                        '{} != a2aff : '
                        'Only all-to-all feed forward networks supported'.
                        format(nn_type))
                # activation
                layers_act = species_config.get_network_act(
                    'activations', None)
                if layers_act:
                    logger.info(spe +
                                'New activations : {}'.format(layers_act))

                networks_kind.append(nn_type)
                networks_metadata.append({
                    'species': species,
                    'activations': layer_act,
                    'offset': zero
                })

            else:
                networks_kind.append(default_nn_type)

                networks_metadata.append({
                    'species': species,
                    'activations': default_layer_act,
                    'offset': zero
                })
    return (io_parameters, data_parameters, validation_parameters,
            networks_kind, networks_metadata)


def main(parameters):
    (io_parameters, data_parameters, validation_parameters, networks_kind,
     networks_metadata) = parameters

    if not os.path.isdir(io_parameters.eval_dir):
        os.mkdir(io_parameters.eval_dir)

    dataset_files = [
        os.path.join(io_parameters.data_dir, x)
        for x in os.listdir(io_parameters.data_dir)
    ]

    logger.info('files in the dataset: {}'.format(len(dataset_files)))

    if io_parameters.networks_format == 'checkpoint':
        ck_files = net.Checkpoint.checkpoint_file_list(io_parameters.train_dir)
        ck_steps = net.Checkpoint.checkpoint_step_list(io_parameters.train_dir)

        if (validation_parameters.single_step
                and not validation_parameters.step_number):
            logger.info('evaluating last checkpoint')
            ck_files = [ck_files[-1]]
            ck_steps = [ck_steps[-1]]
        elif (validation_parameters.single_step
              and validation_parameters.step_number):
            logger.info('evaluating step number {}'.format(
                parameters.step_number))
            logger.info(
                'not implemented, but one has just to look in steps finder')
        elif (not validation_parameters.single_step
              and validation_parameters.subsampling):
            logger.info(
                'evaluation of all the checkpoints at steps of {}'.format(
                    validation_parameters.subsampling))
            ck_files = ck_files[::validation_parameters.subsampling]
            ck_steps = ck_steps[::validation_parameters.subsampling]
        else:
            logger.info('evaluation of all the checkpoints')

        cks = [
            net.Checkpoint(
                os.path.join(io_parameters.train_dir,
                             x), data_parameters.atomic_sequence,
                networks_kind, networks_metadata) for x in ck_files
        ]
        nns = [x.get_scaffold for x in cks]

        writers = [
            open(
                os.path.join(io_parameters.eval_dir, '{}.dat'.format(x)), 'w')
            for x in ck_steps
        ]
        if validation_parameters.compute_forces:
            f_writers = [
                open(
                    os.path.join(parameters.eval_dir,
                                 '{}_forces.dat'.format(x)), 'w')
                for x in ck_steps
            ]

    elif io_parameters.networks_format == 'PANNA':
        arch_file = os.path.join(io_parameters.networks_folder,
                                 'networks_metadata.json')
        if os.path.isfile(arch_file):
            system_scaffold = SystemScaffold.load_PANNA_checkpoint_folder(
                './saved_networks')
            nns = [system_scaffold]
            writers = [
                open(
                    os.path.join(io_parameters.eval_dir, 'energies.dat'), 'w')
            ]
            if validation_parameters.compute_forces:
                f_writers = [
                    open(
                        os.path.join(io_parameters.eval_dir, 'forces.dat'),
                        'w')
                ]
        else:
            logger.info('Please specify a valid json file and path.')
            exit()

    else:
        logger.info('Unknown network format.')
        exit()

    logger.info('----start----')

    [x.write('#filename n_atoms e_dft e_nn\n') for x in writers]

    if validation_parameters.compute_forces:
        [x.write('#filename atom_id fx_ref fx_nn, fy_ref, fy_nn fz_ref, fz_nn\n') for x in f_writers]

    examples = []
    pool = mp.Pool(processes=io_parameters.number_of_process)

    for scaffold, wi in zip(nns, range(len(writers))):
        logger.info('validating network: {}/{}'.format(scaffold.name,
                                                       nns[-1].name))

        if io_parameters.example_format == 'single_files':
            # when the examples are all separated in different files
            if validation_parameters.batch_size == -1 and not examples:
                # we want to load all the example in memory
                # this is done only once
                example_files = dataset_files
                logger.info('loading all the example in the folder,'
                            'this operation may take a while')
                examples_parameters = [(x, data_parameters.n_species,
                                        validation_parameters.compute_forces)
                                       for x in example_files]
                examples = pool.starmap(net.load_example, examples_parameters)

            elif validation_parameters.batch_size > 0:
                # if the validation dataset is too big can be useful
                # to only randomly sample it
                rnd.shuffle(dataset_files)
                example_files = dataset_files[:batch_size]
                logger.info('loading a subset of all the available '
                            'examples in the folder')
                examples_parameters = [(x, data_prameters.n_species,
                                        validation_parameters.compute_forces)
                                       for x in example_files]
                examples = pool.starmap(net.load_example, examples_parameters)

        elif io_parameters.example_format == 'TFR':
            if validation_parameters.batch_size == -1 and not examples:
                logger.info('loading TFR, this may take a while')
                for example in net.iterator_over_tfdata(
                        data_parameters.g_size, *dataset_files,\
                        derivatives=validation_parameters.compute_forces):
                    examples.append(example)

        if len(examples) == 0:
            logger.info('no example found')
            break

        parallel_feed = [(example, scaffold, validation_parameters.add_offset)
                         for example in examples]

        if validation_parameters.compute_forces:
            Ef = pool.starmap(_eval_function_forces, parallel_feed)
            for energy, forces in Ef:
                string = '\n'.join([x[0] for x in Ef])
                string2 = ''.join([x[1] for x in Ef])
        else:
            string = '\n'.join(pool.starmap(_eval_function, parallel_feed))

        writers[wi].write(string)
        writers[wi].flush()
        if validation_parameters.compute_forces:
            f_writers[wi].write(string2)
            f_writers[wi].flush()

    [x.close() for x in writers]
    if validation_parameters.compute_forces:
        [x.close() for x in f_writers]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, help='configuration filename', required=True)
    parser.add_argument(
        '--debug', action='store_true', help='debug flag', required=False)
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    parameters = parse_file(args.config)
    main(parameters)

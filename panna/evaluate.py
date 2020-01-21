###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
""" Code use to evaluate a dataset on a checkpoint/s
note:
    if batchsize == -1 then use the whole dataset
"""
import argparse
import configparser
import logging
import multiprocessing as mp
import os
import random as rnd

from tensorflow.errors import NotFoundError

import neuralnet as net
from neuralnet.scaffold_selector import scaffold_selector
from lib import init_logging
from lib.parser_callable import converters

# logger
logger = logging.getLogger('panna')


class DataContainer():
    pass


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
    """Helper for parallelization when computing forces

    Parameters
    ----------
    example:
        the example to be evaluate

    scaffold:
        the scaffold with the networks that will
        be used in the evaluation

    add_offset: bool
        boolean to enable/disable the offset

    Returns
    -------
    string
        'n_atoms reference energy predicted energy'
    string
        'example_id atom_id fx fy fz'
    """

    en_prediction, forces_prediction = scaffold.evaluate(
        example, True, add_offset)

    n_atoms = example.n_atoms
    en_ref = example.true_energy
    forces_ref = example.forces

    string = '{} {} {} {}'.format(example.name, example.n_atoms, en_ref,
                                  en_prediction)

    forces_ref = forces_ref.reshape(n_atoms, 3)
    forces_prediction = forces_prediction.reshape(n_atoms, 3)
    string2_list = [
        example.name + ' {} '.format(idx) +
        ' '.join([str(f) for f in forces_prediction[idx]]) + ' ' +
        ' '.join([str(f) for f in forces_ref[idx]]) for idx in range(n_atoms)
    ]
    string2 = '\n'.join(string2_list) + '\n'
    return string, string2


def _ckpts_selector(train_dir, validation_parameters):
    ck_files = net.Checkpoint.checkpoint_file_list(train_dir)
    ck_steps = net.Checkpoint.checkpoint_step_list(train_dir)

    if (validation_parameters.single_step
            and not validation_parameters.step_number):
        logger.info('evaluating last checkpoint')
        ck_files = [ck_files[-1]]
        ck_steps = [ck_steps[-1]]
    elif (validation_parameters.single_step
          and validation_parameters.step_number):
        logger.info('evaluating step number %d',
                    validation_parameters.step_number)
        logger.info('not implemented, '
                    'but one has just to look in steps finder')
        raise NotImplementedError()
    elif (not validation_parameters.single_step
          and validation_parameters.subsampling):
        logger.info('evaluation of all the checkpoints at steps of %d ',
                    validation_parameters.subsampling)
        ck_files = ck_files[::validation_parameters.subsampling]
        ck_steps = ck_steps[::validation_parameters.subsampling]
    else:
        logger.info('evaluation of all the checkpoints')
    return ck_files, ck_steps


def _parse_io(io_info):
    io_parameters = DataContainer()
    # recover parameters
    io_parameters.train_dir = io_info.get('train_dir')
    io_parameters.eval_dir = io_info.get('eval_dir')
    io_parameters.data_dir = io_info.get('data_dir')

    # if network is PANNA there are no problem,
    io_parameters.networks_format = io_info.get('networks_format',
                                                'tf_checkpoint')
    if io_parameters.networks_format == 'PANNA':
        io_parameters.networks_folder = io_info.get('networks_folder',
                                                    './saved_networks')
    elif io_parameters.networks_format == 'tf_checkpoint':
        io_parameters.networks_folder = None
    else:
        raise ValueError('this input format is no more supported')

    io_parameters.example_format = io_info.get('example_format', 'TFR')
    return io_parameters


def _parse_validation(validation_options):
    validation_parameters = DataContainer()

    validation_parameters.compute_forces = validation_options.getboolean(
        'compute_forces', False)
    validation_parameters.batch_size = validation_options.getint(
        'batch_size', -1)
    validation_parameters.single_step = validation_options.getboolean(
        'single_step', False)
    validation_parameters.step_number = validation_options.getint(
        'step_number', None)
    validation_parameters.subsampling = validation_options.getint(
        'subsampling', None)
    validation_parameters.add_offset = validation_options.getboolean(
        'add_offset', True)
    return validation_parameters


def _parse_tfr_structure(tfr_options):
    tfr_parameters = DataContainer()

    tfr_parameters.sparse_derivatives = tfr_options.getboolean(
        'sparse_derivatives', False)
    tfr_parameters.g_size = tfr_options.getint('g_size')
    return tfr_parameters


def parse_file(conf_file):
    """ Parse validation config file
    """
    config = configparser.ConfigParser(converters=converters)
    config.read(conf_file)

    io_info = config['IO_INFORMATION']
    io_parameters = _parse_io(io_info)

    parallel_param = DataContainer()
    parallel_param.number_of_process = 4
    if config.has_section('PARALLELIZATION'):
        parallelization = config['PARALLELIZATION']
        parallel_param.number_of_process = parallelization.getint(
            'number_of_process', parallel_param.number_of_process)

    validation_options = config['VALIDATION_OPTIONS']
    validation_parameters = _parse_validation(validation_options)

    # small hack for not supported op
    if io_parameters.example_format == 'TFR':
        logger.info('batch_size set to -1,'
                    ' TFR does not support this option')
        validation_parameters.batch_size = -1
        tfr_parameters = _parse_tfr_structure(config['TFR_STRUCTURE'])
    else:
        tfr_parameters = None

    return io_parameters, parallel_param, validation_parameters, tfr_parameters


def main(parameters):
    io_parameters, parallel_param, validation_parameters,\
        tfr_parameters = parameters

    if not os.path.isdir(io_parameters.eval_dir):
        os.mkdir(io_parameters.eval_dir)

    dataset_files = [
        os.path.join(io_parameters.data_dir, x)
        for x in os.listdir(io_parameters.data_dir)
    ]

    logger.info('files in the dataset: %d ', len(dataset_files))

    if io_parameters.networks_format == 'tf_checkpoint':

        ck_files, ck_steps = _ckpts_selector(io_parameters.train_dir,
                                             validation_parameters)

        # load ckpts
        ckpts = [
            net.Checkpoint(
                ckpt_file=os.path.join(io_parameters.train_dir, x),
                json_file=os.path.join(io_parameters.train_dir,
                                       'networks_metadata.json'),
            ) for x in ck_files
        ]

        # extract scaffols
        scaffolds = []
        for ckpt in ckpts:
            try:
                scaf = ckpt.get_scaffold
                scaffolds.append(scaf)
            except NotFoundError:
                logger.warning('%s not found because of TF bug', ckpt.filename)

        scaffolds_non_computed = []
        energy_writers = []

        if validation_parameters.compute_forces:
            force_writers = []

        # filter already computed scaffolds
        for ck_step, scaffold in zip(ck_steps, scaffolds):
            file_name = os.path.join(io_parameters.eval_dir,
                                     '{}.dat'.format(ck_step))

            if os.path.isfile(file_name) and os.path.getsize(file_name) > 100:
                logger.info('%s already computed', file_name)
                continue

            scaffolds_non_computed.append(scaffold)

            energy_writers.append(
                open(
                    os.path.join(io_parameters.eval_dir,
                                 '{}.dat'.format(ck_step)), 'w'))
            if validation_parameters.compute_forces:
                force_writers.append(
                    open(
                        os.path.join(io_parameters.eval_dir,
                                     '{}_forces.dat'.format(ck_step)), 'w'))

    elif io_parameters.networks_format == 'PANNA':

        scaffold = scaffold_selector('PANNA')()

        scaffold.load_panna_checkpoint_folder('./saved_networks')

        scaffolds_non_computed = [scaffold]
        energy_writers = [
            open(os.path.join(io_parameters.eval_dir, 'energies.dat'), 'w')
        ]
        if validation_parameters.compute_forces:
            force_writers = [
                open(os.path.join(io_parameters.eval_dir, 'forces.dat'), 'w')
            ]

    else:
        logger.info('Unknown network format.')
        exit()

    logger.info('----start----')

    [x.write('#filename n_atoms e_ref e_nn\n') for x in energy_writers]

    if validation_parameters.compute_forces:
        [
            x.write(
                '#filename atom_id fx_nn fy_nn fz_nn fx_ref fy_ref fz_ref\n')
            for x in force_writers
        ]

    examples = []
    pool = mp.Pool(processes=parallel_param.number_of_process)

    for scaffold, writer_idx in zip(scaffolds_non_computed,
                                    range(len(energy_writers))):

        logger.info('validating network: %s/%s', scaffold.name,
                    scaffolds_non_computed[-1].name)

        if io_parameters.example_format == 'single_files':
            # when the examples are all separated in different files
            if validation_parameters.batch_size == -1 and not examples:
                # we want to load all the example in memory
                # this is done only once
                example_files = dataset_files
                logger.info('loading all the example in the folder,'
                            'this operation may take a while')
                examples_parameters = [(x, ) for x in example_files]
                examples = pool.starmap(net.load_example, examples_parameters)

            elif validation_parameters.batch_size > 0:
                # if the validation dataset is too big can be useful
                # to only randomly sample it
                rnd.shuffle(dataset_files)
                example_files = dataset_files[:validation_parameters.
                                              batch_size]
                logger.info('loading a subset of all the available '
                            'examples in the folder')
                examples_parameters = [(x, ) for x in example_files]
                examples = pool.starmap(net.load_example, examples_parameters)

        elif io_parameters.example_format == 'TFR':
            if validation_parameters.batch_size == -1 and not examples:
                logger.info('loading TFR, this may take a while')
                for example in net.iterator_over_tfdata(
                        tfr_parameters.g_size,
                        *dataset_files,
                        derivatives=validation_parameters.compute_forces,
                        sparse_derivatives=tfr_parameters.sparse_derivatives):
                    examples.append(example)

        if not examples:
            logger.info('no example found')
            break

        parallel_feed = [(example, scaffold, validation_parameters.add_offset)
                         for example in examples]

        if validation_parameters.compute_forces:
            energy_forces = pool.starmap(_eval_function_forces, parallel_feed)
            string = '\n'.join([x[0] for x in energy_forces])
            string2 = ''.join([x[1] for x in energy_forces])
        else:
            string = '\n'.join(pool.starmap(_eval_function, parallel_feed))

        energy_writers[writer_idx].write(string)
        energy_writers[writer_idx].flush()
        if validation_parameters.compute_forces:
            force_writers[writer_idx].write(string2)
            force_writers[writer_idx].flush()

    [x.close() for x in energy_writers]
    if validation_parameters.compute_forces:
        [x.close() for x in force_writers]


if __name__ == '__main__':
    init_logging()
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-c',
                        '--config',
                        type=str,
                        help='configuration filename',
                        required=True)
    PARSER.add_argument('--debug',
                        action='store_true',
                        help='debug flag, not working for now',
                        required=False)
    ARGS = PARSER.parse_args()

    PARAMETERS = parse_file(ARGS.config)
    main(PARAMETERS)

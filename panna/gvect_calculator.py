###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import argparse
import configparser
import logging
import multiprocessing as mp
import os

import numpy as np

import gvector
from lib import ExampleJsonWrapper, init_logging, parser_callable

np.set_printoptions(16)

# logger
logger = logging.getLogger('panna')


def parse_file(conf_file):
    """ Parser helper
    """
    config = configparser.ConfigParser(
        converters={
            '_comma_list_floats':
            parser_callable.get_list_floats_from_comma_sep_strings,
        })
    config.read(conf_file)

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=missing-docstring, too-few-public-methods
    class ParameterContainer():
        pass

    # folders related parameters:
    folder_parameters = ParameterContainer()

    folder_info = config['IO_INFORMATION']
    folder_parameters.input_json_dir = folder_info.get('input_json_dir', None)
    folder_parameters.binary_out_dir = folder_info.get('output_gvect_dir',
                                                       './bin')
    folder_parameters.log_dir = folder_info.get('log_dir', './logs')

    # parallelization related parameters
    parallelization = config['PARALLELIZATION']
    number_of_process = parallelization.getint('number_of_process', 1)

    # PBC related prameters
    if config.has_section('PBC'):
        logger.warning('PBC will be assumed, json must be consistent '
                       'otherwise the code will fail')
        pbc = config['PBC']
        pbc_directions = np.asarray(
            [pbc.getboolean('pbc{}'.format(x), False) for x in range(1, 4)])
    else:
        pbc_directions = None
        logger.info('PBC will be determined by json file each time')

    # TODO define a name for this section
    misc_parameters = ParameterContainer()
    if 'MISC' in config:
        misc = config['MISC']
        long_range = misc.getboolean('long_range', False)
        per_atom_energy = misc.getboolean('per_atom_energy', False)
    else:
        long_range = False
        per_atom_energy = False

    misc_parameters.per_atom_energy = per_atom_energy
    misc_parameters.long_range = long_range

    # *gvect* related parameters
    symmetry_function = config['SYMMETRY_FUNCTION']
    compute_dgvect = symmetry_function.getboolean('include_derivatives', False)
    if compute_dgvect:
        sparse_dgvect = symmetry_function.getboolean('sparse_derivatives',
                                                     False)
    else:
        sparse_dgvect = False
    species = symmetry_function.get('species', None)

    gv_par = config['GVECT_PARAMETERS']
    # common parts to all descriptors
    param_unit = gv_par.get('gvect_parameters_unit', 'angstrom')

    if symmetry_function.get('type') == 'mBP':

        # RADIAL_COMPONENTS
        eta_rad = gv_par.get_comma_list_floats('eta_rad', None)
        if len(eta_rad) > 1:
            logger.warning('more than 1 eta_rad, but mBP required '
                           'default action: consider only the first element')
        eta_rad = eta_rad[0]

        # pylint: disable=invalid-name
        Rc_rad = gv_par.getfloat('Rc_rad')
        Rs0_rad = gv_par.getfloat('Rs0_rad')
        RsN_rad = gv_par.getint('RsN_rad')

        # infer if not present:
        Rsst_rad = gv_par.getfloat('Rsst_rad', None)

        # ANGULAR_COMPONENTS
        eta_ang = gv_par.get_comma_list_floats('eta_ang', None)
        if len(eta_ang) > 1:
            logger.warning('more than 1 eta_ang, but mBP required '
                           'default action: consider only the first element')
        eta_ang = eta_ang[0]

        ThetasN = gv_par.getint('ThetasN')

        Rc_ang = gv_par.getfloat('Rc_ang')
        Rs0_ang = gv_par.getfloat('Rs0_ang')
        RsN_ang = gv_par.getint('RsN_ang')

        # infer if not present
        Rsst_ang = gv_par.getfloat('Rsst_ang', None)

        zeta = gv_par.get_comma_list_floats('zeta', None)
        if len(zeta) > 1:
            logger.warning('more than 1 zeta, but mBP required '
                           'default action: consider only the first element')
        zeta = zeta[0]

        gvect_func = gvector.GvectmBP(compute_dgvect=compute_dgvect,
                                      sparse_dgvect=sparse_dgvect,
                                      species=species,
                                      param_unit=param_unit,
                                      pbc_directions=pbc_directions,
                                      Rc_rad=Rc_rad,
                                      Rs0_rad=Rs0_rad,
                                      RsN_rad=RsN_rad,
                                      eta_rad=eta_rad,
                                      Rc_ang=Rc_ang,
                                      ThetasN=ThetasN,
                                      Rs0_ang=Rs0_ang,
                                      RsN_ang=RsN_ang,
                                      eta_ang=eta_ang,
                                      zeta=zeta,
                                      Rsst_rad=Rsst_rad,
                                      Rsst_ang=Rsst_ang)

    elif symmetry_function.get('type') == 'BP':
        # RADIAL_COMPONENTS
        eta = gv_par.get_comma_list_floats('eta_rad')
        Rc = gv_par.getfloat('Rc_rad')
        Rs0 = gv_par.getfloat('Rs0_rad', 0.0)
        RsN = gv_par.getint('RsN_rad')

        # ANGULAR_COMPONENTS
        eta_ang = gv_par.get_comma_list_floats('eta_ang', None)
        Rc_ang = gv_par.getfloat('Rc_ang', None)
        zeta = gv_par.get_comma_list_floats('zeta')

        gvect_func = gvector.GvectBP(compute_dgvect=compute_dgvect,
                                     sparse_dgvect=sparse_dgvect,
                                     species=species,
                                     param_unit=param_unit,
                                     pbc_directions=pbc_directions,
                                     Rc=Rc,
                                     RsN=RsN,
                                     eta=eta,
                                     eta_ang=eta_ang,
                                     Rc_ang=Rc_ang,
                                     zeta=zeta,
                                     Rs0=Rs0)
    else:
        raise ValueError('Not recognized symmetry function type')

    if long_range:
        raise NotImplementedError()
        # here one can add a second function to compute like the symmetry

    # pylint: enable=attribute-defined-outside-init
    # pylint: enable=missing-docstring, too-few-public-methods
    return gvect_func, folder_parameters, number_of_process, misc_parameters


def _remove_already_computed_keys(all_example_keys, log_dir):
    try:
        with open(os.path.join(log_dir, 'gvect_already_computed.dat'))\
                               as file_stream:
            logger.info('the computation has been restarted'
                        '(a file named gvect_already_computed.dat '
                        'has been found). Already computed files '
                        'will not be recomputed.')
            keys_already_computed = file_stream.read().split(',')
    except FileNotFoundError:
        keys_already_computed = []

    keys_already_computed = set(keys_already_computed)
    all_example_keys = set(all_example_keys)

    logger.info('computed keys %d/%d',
                max(len(keys_already_computed), 1) - 1, len(all_example_keys))

    return list(all_example_keys - keys_already_computed)


def main(gvect_func, folder_parameters, number_of_process, misc):

    # fast adaptation for parameters
    input_json_dir = folder_parameters.input_json_dir
    binary_out_dir = folder_parameters.binary_out_dir
    log_dir = folder_parameters.log_dir

    compute_dgvect = gvect_func.compute_dgvect
    sparse_dgvect = gvect_func.sparse_dgvect

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # log g_size based on the kind of symmetry function type
    logger.info('g_size: %d', gvect_func.gsize)
    logger.info('--start--')

    if not os.path.exists(binary_out_dir):
        os.makedirs(binary_out_dir)

    all_example_keys = []
    for file in os.listdir(input_json_dir):
        name, ext = os.path.splitext(file)
        if ext == '.example':
            all_example_keys.append(name)
    if len(all_example_keys) == 0:
        logger.info('No example found. Stopping')
        exit(1)

    example_keys = _remove_already_computed_keys(all_example_keys, log_dir)

    pool = mp.Pool(number_of_process)

    log_for_recover = open(os.path.join(log_dir, 'gvect_already_computed.dat'),
                           'a')

    while True:
        logger.info('----run----')
        parallel_batch = []
        elements_in_buffer = 0
        while elements_in_buffer < number_of_process:
            try:
                key = example_keys.pop()
            except IndexError:
                # exit conditions form process population if
                # no more key to compute
                break

            logger.debug("loading %s:%s", elements_in_buffer, key)

            # === THE CODE WORKS IN ANGSTROM AND EV ===
            example = ExampleJsonWrapper(
                os.path.join(input_json_dir, '{}.example'.format(key)),
                gvect_func.species_idx_2str)

            # load common quantities
            example_dict = {
                'key': example.key,
                'lattice_vectors': example.angstrom_lattice_vectors,
                'species': example.species_indexes,
                'positions': example.angstrom_positions,
                'E': example.ev_energy
            }

            # load specific quantities
            if misc.per_atom_energy:
                example_dict['per_atom_quantity'] = example.per_atom_ev_energy
            elif misc.long_range:
                example_dict['per_atom_quantity'] = example.per_atom_charges

            if compute_dgvect:
                example_dict['forces'] = example.forces

            logger.debug(example_dict)
            parallel_batch.append(example_dict)
            elements_in_buffer += 1

        if not parallel_batch:
            # exit condition if the computation has ended
            break

        logger.info('start parallel computation, %d to go', len(example_keys))

        # COMPUTE gvect and other quantities
        logger.debug('compute gvectors')
        feed = []
        for example_dict in parallel_batch:
            feed.append(
                (example_dict['key'], example_dict['positions'],
                 example_dict['species'], example_dict['lattice_vectors']))

        feed = pool.starmap(gvect_func, feed)

        # reorganize the results
        for example_dict in parallel_batch:
            for element in feed:
                if example_dict['key'] == element[0]:
                    example_dict['Gvect'] = element[1]
                    if compute_dgvect:
                        if not sparse_dgvect:
                            example_dict['dGvect'] = element[2]
                        else:
                            example_dict['dGvect_val'] = element[2]
                            example_dict['dGvect_ind'] = element[3]
                    logger.debug('assigned')
                    logger.debug(element[0])

        # SAVE computed result
        logger.debug('save gvectors')
        feed = []
        for example_dict in parallel_batch:
            feed.append((example_dict, binary_out_dir))

        pool.starmap(gvector.compute_binary, feed)

        # final log for recovery
        log_for_recover.write(','.join(
            [example_dict['key'] for example_dict in parallel_batch]))
        log_for_recover.write(',')
        log_for_recover.flush()
    log_for_recover.close()


if __name__ == '__main__':
    init_logging()
    PARSER = argparse.ArgumentParser(description='Gvectors calculator')

    PARSER.add_argument('-c',
                        '--config',
                        type=str,
                        help='config file',
                        required=True)
    PARSER.add_argument('-f',
                        '--folder_info',
                        type=str,
                        help='folder_info, if supplied override config',
                        required=False)
    PARSER.add_argument('--debug',
                        action='store_true',
                        help='debug flag',
                        required=False)

    ARGS = PARSER.parse_args()

    if ARGS.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    GVECT_PARAMETERS, FOLDER_PARAMETERS, NUMBER_OF_PROCESS, MISC = parse_file(
        ARGS.config)

    if ARGS.folder_info:
        logger.info('overriding folder parameters with cli values')
        FOLDER_PARAMETERS.input_json_dir = os.path.join(
            ARGS.folder_info, 'examples')
        FOLDER_PARAMETERS.binary_out_dir = os.path.join(
            ARGS.folder_info, 'bin')
        FOLDER_PARAMETERS.log_dir = ARGS.folder_info

    main(GVECT_PARAMETERS, FOLDER_PARAMETERS, NUMBER_OF_PROCESS, MISC)

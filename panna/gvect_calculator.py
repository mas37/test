import os
import json
import logging
import argparse
import configparser

import numpy as np
import multiprocessing as mp

import gvector
from neuralnet import parser_callable

np.set_printoptions(16)
# logger
logger = logging.getLogger('logfile')
formatter = logging.Formatter('%(asctime)s - %(name)s - '
                              '%(levelname)s - %(message)s')

# console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# Conversion constants
BOHR2A = np.float32(0.52917721067)
RY2EV = 13.6056980659


def parse_file(conf_file):
    """ Parser helper
    """
    config = configparser.ConfigParser(
        converters={
            '_comma_list_floats':
            parser_callable.get_list_floats_from_comma_sep_strings,
        })
    config.read(conf_file)

    # folders related parameters:
    class FolderParameters():
        pass

    folder_parameters = FolderParameters()

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

        # TODO this is read but never used, can someone comment on it?
        # parameters.directions_strings = np.asarray(['a1', 'a2',
        #                                             'a3'])[pbc_directions]
        # if len(parameters.directions_strings) > 0:
        #     logger.info('ATTENTION:Periodic boundary conditions is imposed\
        #      along {}'.format(str(directions_strings)))
        # else:
        #     logger.info('ATTENTION: No Periodic boundary conditions')
    else:
        pbc_directions = None
        logger.info('PBC will be determined by json file each time')

    # gvect related parameters
    symmetry_function = config['SYMMETRY_FUNCTION']
    symmetry_function_type = symmetry_function.get('type')
    compute_dGvect = symmetry_function.getboolean('include_derivatives', False)
    species = symmetry_function.get('species', None)

    gv_par = config['GVECT_PARAMETERS']
    # common parts to all descriptors
    param_unit = gv_par.get('gvect_parameters_unit', 'angstrom')

    if symmetry_function_type == 'mBP':
        # RADIAL_COMPONENTS
        eta_rad = gv_par.get_comma_list_floats('eta_rad', None)
        if len(eta_rad) > 1:
            logger.warning('more than 1 eta_rad, but mBP required '
                           'default action: consider only the first element')
        eta_rad = eta_rad[0]

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

        gvect_func = gvector.GvectmBP(
            compute_dGvect=compute_dGvect,
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

    elif symmetry_function_type == 'BP':
        # RADIAL_COMPONENTS
        eta = gv_par.get_comma_list_floats('eta_rad')
        Rc = gv_par.getfloat('Rc_rad')
        Rs0 = gv_par.getfloat('Rs0_rad', 0.0)
        RsN = gv_par.getint('RsN_rad')
        # infer if not present:
        Rsst = gv_par.getfloat('Rsst', None)
        # ANGULAR_COMPONENTS
        eta_ang = gv_par.get_comma_list_floats('eta_ang', None)
        Rc_ang = gv_par.getfloat('Rc_ang', None)
        zeta = gv_par.get_comma_list_floats('zeta')

        gvect_func = gvector.GvectBP(
            compute_dGvect=compute_dGvect,
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

    return gvect_func, folder_parameters, number_of_process


def main(gvect_func, folder_parameters, number_of_process):

    # fast adaptation for parameters
    input_json_dir = folder_parameters.input_json_dir
    binary_out_dir = folder_parameters.binary_out_dir
    log_dir = folder_parameters.log_dir

    symmetry_function_type = 'mBP'
    compute_dGvect = gvect_func.compute_dGvect
    species = gvect_func.species
    species_idx_2str = gvect_func.species_idx_2str
    number_of_species = gvect_func.number_of_species
    species_str_2idx = gvect_func.species_str_2idx
    param_unit = gvect_func.param_unit
    unit2A = gvect_func.unit2A
    gvect_parameters_dict = gvect_func.gvect
    pbc_directions = gvect_func.pbc_directions

    number_of_process = number_of_process

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # setting the logger
    fh = logging.FileHandler(os.path.join(log_dir, 'logfile_gvect.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # log g_size based on the kind of symmetry function type
    logger.info('g_size: {}'.format(gvect_func.gsize))

    logger.info('--start--')
    if not os.path.exists(binary_out_dir):
        os.makedirs(binary_out_dir)

    all_example_keys = [x.split('.')[0] for x in os.listdir(input_json_dir)]

    try:
        with open(os.path.join(log_dir, 'gvect_already_computed.dat')) as f:
            logger.info(
                'the computation has been restarted (a file named gvect_already_computed.dat '
                'has been found). Already computed files will not be recomputed.'
            )
            keys_already_computed = f.read().split(',')
    except FileNotFoundError as e:
        keys_already_computed = []

    keys_already_computed = set(keys_already_computed)
    all_example_keys = set(all_example_keys)
    example_keys = list(all_example_keys - keys_already_computed)

    logger.info('computed keys {}/{}'.format(
        max(len(keys_already_computed), 1) - 1, len(all_example_keys)))

    p = mp.Pool(number_of_process)

    log_for_recover = open(
        os.path.join(log_dir, 'gvect_already_computed.dat'), 'a')
    while True:
        logger.info('----run----')
        parallel_batch = []
        x = 0
        while x < number_of_process:
            try:
                key = example_keys.pop()
            except IndexError:
                # exit conditions form process population if
                # no more key to compute
                break

            logger.debug("loading {}:{}".format(x, key))
            with open(
                    os.path.join(input_json_dir, '{}.example'.format(key)),
                    'r') as f:
                example = json.load(f)

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
                    'WARNING: unit of energy unknown, assumed eV {}'.format(
                        key))

            # lattice info
            unit_of_length = example.get('unit_of_length', None)
            lattice_vectors = example.get('lattice_vectors', None)

            if unit_of_length == 'bohr':
                alat = np.float32(1) * BOHR2A
            elif unit_of_length == 'angstrom':
                alat = np.float32(1)
            else:
                logger.warning(
                    'WARNING: unit_of_length unknown, assumed angstrom {}'.
                    format(key))
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

                if compute_dGvect and len(atom) > 3:
                    if unit_of_length == 'angstrom':
                        forces.append(
                            np.asarray(atom[3]).astype(float) * unit2eV)
                    elif unit_of_length == 'bohr':
                        forces.append(
                            np.asarray(atom[3]).astype(float) / BOHR2A *
                            unit2eV)

            species_symbols = [species_idx_2str[x] for x in species_idxs]
            example_dict = {
                'key': key,
                'positions': pos,
                'species': species_idxs,
                'lattice_vectors': lattice_vectors,
                'number_of_species': number_of_species,
                'symbols': species_symbols,
                'E': example['energy'][0] * unit2eV,
                'n_atoms': len(example['atoms']),
                'species_str_2idx': species_str_2idx,
                'pbc': pbc_directions,
                'forces': forces
            }
            logger.debug(example_dict)
            parallel_batch.append(example_dict)
            x += 1

        if len(parallel_batch) == 0:
            # exit condition if the computation has ended
            break

        logger.info('start parallel computation, {} to go'.format(
            len(example_keys)))
        feed = []

        for example_dict in parallel_batch:
            feed.append(
                (example_dict['key'], example_dict['positions'],
                 example_dict['species'], example_dict['lattice_vectors']))

        feed = p.starmap(gvect_func, feed)

        for example_dict in parallel_batch:
            for i, y in enumerate(feed):
                if example_dict['key'] == y[0]:
                    example_dict['Gvect'] = y[1]
                    if compute_dGvect:
                        example_dict['dGvect'] = y[2]
                    logger.debug('assigned')
                    logger.debug(y[0])
        feed = []
        for example_dict in parallel_batch:
            feed.append((example_dict, binary_out_dir))

        if compute_dGvect:
            p.starmap(gvector.compute_binary_dGvect, feed)
        else:
            p.starmap(gvector.compute_binary, feed)
        # final log for recovery
        log_for_recover.write(','.join(
            [example_dict['key'] for example_dict in parallel_batch]))
        log_for_recover.write(',')
        log_for_recover.flush()
    log_for_recover.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gvectors calculator')
    parser.add_argument(
        '-c', '--config', type=str, help='config file', required=True)
    parser.add_argument(
        '-f',
        '--folder_info',
        type=str,
        help='folder_info, if supplied override config',
        required=False)
    parser.add_argument(
        '--debug', action='store_true', help='debug flag', required=False)
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    gvect_parameters, folder_parameters, number_of_process = parse_file(
        args.config)

    if args.folder_info:
        logger.info('overriding folder parameters with cli values')
        folder_parameters.input_json_dir = os.path.join(
            folder_info, 'examples')
        folder_parameters.binary_out_dir = os.path.join(folder_info, 'bin')
        folder_parameters.log_dir = folder_info

    main(gvect_parameters, folder_parameters, number_of_process)

"""
  code used to extract weights from a checkpoint
"""
import os
import logging
import argparse
import configparser

import neuralnet as net
from gvect_calculator import parse_file as parse_gvect_input

# logger
logger = logging.getLogger('logfile')
formatter = logging.Formatter('%(asctime)s - %(name)s - \
    %(levelname)s - %(message)s')

# console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def main(conf_file):
    config = configparser.ConfigParser()
    config.read(conf_file)
    folder_info = config['IO_INFORMATION']
    in_dir = folder_info.get('train_dir', None)
    step_number = folder_info.getint('step_number', 0)
    out_dir = folder_info.get('output_dir', './saved_network')
    species_list_string = folder_info.get('species_list', None)
    if species_list_string:
        species_list = species_list_string.split(',')
    train_input = folder_info.get('train_input', None)
    gvector_input = folder_info.get('gvector_input', None)
    output_type = folder_info.get('output_type', 'PANNA')
    output_file = folder_info.get('output_file', 'panna.in')

    extra_data = {}

    if train_input:
        io_parameters, parallelization_parameters, train_parameters, \
            parameters_container, system_scaffold = net.parameter_file_parser(train_input)
        if species_list_string == None:
            species_list = system_scaffold.atomic_sequence

    if gvector_input:
        gvect_func, folder_parameters, number_of_process = parse_gvect_input(gvector_input)
        extra_data['gvect_params'] = gvect_func.gvect

    ck_files = net.Checkpoint.checkpoint_file_list(in_dir)
    ck_steps = net.Checkpoint.checkpoint_step_list(in_dir)
    if step_number in ck_steps:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        ck = net.Checkpoint(
            os.path.join(in_dir, 'model.ckpt-{}'.format(step_number)),
            species_list=species_list)
        if output_type=='PANNA':
            ck.dump_PANNA_checkpoint_folder(folder=out_dir, extra_data=extra_data)
        elif output_type=='LAMMPS':
            ck.dump_LAMMPS_checkpoint_folder(folder=out_dir, filename=output_file,
                                             extra_data=extra_data)
        else:
            logger.info('Unknown output type {}'.format(output_type))
        logger.info('Weights saved in {}'.format(out_dir))
    else:
        logger.info('step number {} not found'.format(step_number))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, help='config file', required=True)
    parser.add_argument(
        '--debug', action='store_true', help='debug flag', required=False)
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    main(args.config)

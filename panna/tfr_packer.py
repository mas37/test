###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import os
import argparse
import configparser
import numpy as np
import logging

import gvector
import neuralnet

# logger
logger = logging.getLogger('logfile')
formatter = logging.Formatter('%(asctime)s - %(name)s - \
    %(levelname)s - %(message)s')

# console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def parser_file(conf_file):
    """parser file

    Args:
      in_path: path to the directory to be compressed
      out_path: path to the save directory
      elements_per_file: number of element for each tfrecord file
      n_atoms: simulation info: number of atoms in each file
      num_species: simulation info: number of species
      g_size: simulation info: size of g-vector

      prefix: an extra prefix, optional, it is added in front of the file_name
              eg: train for the training set

    Return:
      None

    IMPORTANT:
      In the in_path there must be ONLY files that are binary representation
      of already calculated g-vector
    """
    config = configparser.ConfigParser()
    config.read(conf_file)

    class Parameters():
        pass

    parameters = Parameters()

    io_param = config['IO_INFORMATION']
    parameters.in_path = io_param.get('input_dir', None)
    parameters.out_path = io_param.get('output_dir', None)
    parameters.elements_per_file = io_param.getint('elements_per_file', 1000)
    parameters.prefix = io_param.get('prefix', '')

    gv_param = config['CONTENT_INFORMATION']
    parameters.num_species = gv_param.getint('n_species', None)
    parameters.derivatives = gv_param.getboolean('include_derivatives', False)

    return parameters


def main(parameters):
    """Package all the gvector in a folder to tfrecord files
    """

    in_path = parameters.in_path
    out_path = parameters.out_path
    elements_per_file = parameters.elements_per_file
    prefix = parameters.prefix
    num_species = parameters.num_species
    derivatives = parameters.derivatives

    f = [os.path.join(in_path, x) for x in os.listdir(in_path)]
    n_files = int(np.ceil(len(f) / elements_per_file))
    f = [
        f[i * elements_per_file:(i + 1) * elements_per_file]
        for i in range(n_files)
    ]

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if prefix != '':
        file_pattern = prefix + '-{}-{}'
    else:
        file_pattern = '{}-{}'

    n_files = len(f)
    for i, s in enumerate(f):
        logger.info('file {}/{}'.format(i + 1, n_files))

        filename = file_pattern.format(i + 1, n_files)
        target_file = os.path.join(out_path, '{}.tfrecord'.format(filename))

        if os.path.isfile(target_file) and (os.path.getsize(target_file) > 0):
            logger.info('file already computed')
            continue

        gvector.writer(
            filename=filename,
            path=out_path,
            data=[
                gvector.example_tf_packer(
                    neuralnet.load_example(
                        x, num_species, derivatives=derivatives), derivatives)
                for x in s
            ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFR packer')
    parser.add_argument(
        '-c', '--config', type=str, help='config file', required=True)
    args = parser.parse_args()
    logger.setLevel(logging.INFO)
    parameters = parser_file(args.config)
    main(parameters)

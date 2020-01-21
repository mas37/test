###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import glob
import json
import os
import struct
import logging

import numpy as np
from tensorflow.python import pywrap_tensorflow

from .scaffold_selector import scaffold_selector

logger = logging.getLogger('panna.neuralnet')


def _tf_ckpt_search(seed, obj, reader):
    """ LOAD the network contained in a scaffold form a tf checkpoint

    Parameters
    ----------
        seed: a string to find the name
        obj: an object that has tf_ckpt_elements, scaffold is ok
             but this function is recursive so other things can also work
        reader: a tf reader

    Return
    ------
        Nothing, the scaffold get's loaded with the required parameters
    """
    seed_extensions, sub_objs = obj.tf_ckpt_elements
    sub_seeds = []

    for seed_extension in seed_extensions:
        if seed == '':
            new_seed = seed_extension
        else:
            new_seed = seed + '/' if seed_extension != '' else seed
            new_seed += seed_extension
        sub_seeds.append(new_seed)

    if sub_objs:
        for new_seed, new_obj in zip(sub_seeds, sub_objs):
            _tf_ckpt_search(new_seed, new_obj, reader)
    else:
        tensors = []
        for new_seed in sub_seeds:
            logger.debug('loading: %s', new_seed)
            tensors.append(reader.get_tensor(new_seed))
        obj.tf_ckpt_elements = tensors


class Checkpoint(object):
    """ A wrapper around a TF checkpoint
    Parameters
    ----------
        ckpt_file: str
            complete path to a checkpoint file eg: a/path/to/model.ckpt-0
        config_file: str
            file used to train the network
    """
    def __init__(self, ckpt_file, json_file, name=None):
        super().__init__()

        self._ckpt_file = ckpt_file
        logger.debug('ckpt, train json file: %s', json_file)

        with open(json_file) as file_stream:
            self._metadata = json.load(file_stream)

        scaffold_type = self._metadata['scaffold_type']
        self._Scaffold = scaffold_selector(scaffold_type)

        self._name = name if name else ckpt_file

        reader = pywrap_tensorflow.NewCheckpointReader(self._ckpt_file)
        self._reader = reader

    @property
    def get_scaffold(self):
        """
        return
        ------
           The encapsulated scaffold filled with the ckpt
        """
        for name, shape in self._reader.get_variable_to_shape_map().items():
            logger.debug('available variables: %s ---- %s', name, str(shape))

        scaffold = self._Scaffold()
        scaffold.load_panna_metadata(self._metadata)
        # fill the scaffold
        _tf_ckpt_search('', scaffold, self._reader)

        return scaffold

    def dump_PANNA_checkpoint_folder(self,
                                     folder='dump_checkpoint',
                                     extra_data=None):
        """Dump weights and biases of a checkpoint to a directory, also metadata

        Parameters
        ----------
            folder: where to dump the files
            extra_data: extra dictionary to be added to the json

        Return
        ------
            None

        This method create a folder and put inside that folder
        - one file of weights for each layer dumped with dumping_function
        - one file of biases for each layer dumped with dumping_function
        - networks_metadata.json a json containing extra info
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        scaffold = self.get_scaffold
        metadata, networks_tensors = scaffold.ckpt_metadata

        for network_files_name, network_tensors in zip(
                metadata['networks_files'], networks_tensors):
            for names, tensors in zip(network_files_name, network_tensors):
                for name, tensor in zip(names, tensors):
                    np.save(os.path.join(folder, name), tensor)

        if extra_data:
            metadata.update(extra_data)
        with open(os.path.join(folder, 'networks_metadata.json'),
                  'w') as file_stream:
            json.dump(metadata, file_stream)

    def dump_LAMMPS_checkpoint_folder(self,
                                      folder='network_potential',
                                      filename='panna.in',
                                      extra_data={}):
        """Dump weights and biases of a checkpoint to a directory, also metadata

        Args:
            dump: where to dump the files
            extra_data: extra dictionary to be added to the json


        Return:
            None

        This method creates a folder containing
        - a text file defining a NN potential of 'pair_panna' type for LAMMPS 
        - one file with all weights for each species
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        scaffold = self.get_scaffold
        metadata, _dummy = scaffold.ckpt_metadata

        with open(os.path.join(folder, filename), 'w') as f:
            f.write("[GVECT_PARAMETERS]\n")
            f.write("Nspecies = {}\n".format(len(scaffold.atomic_sequence)))
            f.write("species = {}\n".format(",".join(
                scaffold.atomic_sequence)))
            for p in extra_data['gvect_params'].keys():
                f.write("{} = {}\n".format(p, extra_data['gvect_params'][p]))

            for idx_s, species in enumerate(scaffold.atomic_sequence):
                network = scaffold[species]
                sizes = []
                activs = []
                weights = np.array([], dtype=np.float32)
                for idx_l, layer in enumerate(network):
                    weights = np.append(weights, layer.w_value.flatten())
                    if idx_l == len(network._layers)-1:
                        biase = layer.b_value + network.offset
                        print('network_offset=',network.offset)
                        weights = np.append(weights, biase.flatten())
                    else:
                        weights = np.append(weights, layer.b_value.flatten())

                    sizes.append(layer.b_shape[0])
                    activs.append(layer.activation)
                    #weights = np.append(weights, layer.b_value.flatten())
                f.write("\n[{}]\n".format(species))
                f.write("Nlayers = {}\n".format(len(sizes)))
                f.write("sizes = {}\n".format(",".join(map(str, sizes))))
                wfname = "weights_{}.dat".format(species)
                f.write("file = {}\n".format(wfname))
                f.write("activations = {}\n".format(",".join(map(str,
                                                                 activs))))
                myfmt = 'f' * len(weights)
                binw = struct.pack(myfmt, *weights)
                wf = open(os.path.join(folder, wfname), "wb")
                wf.write(binw)
                wf.close()

            f.close()

        return None

    @property
    def filename(self):
        return self._ckpt_file.split('/')[-1]

    @property
    def step(self):
        return int(self._ckpt_file.split('model.ckpt-')[1].split('.')[0])

    @classmethod
    def laststep_finder(cls, directory):
        """ Find last checkpoint in a checkpoint directory
        """
        ckpts = Checkpoint.checkpoint_file_list(directory)
        return os.path.join(directory, ckpts[-1])

    @classmethod
    def checkpoint_file_list(cls, directory):
        """ Find all the available checkpoints in a checkpoint directory

        Args:
            directory: where to look for checkpoints

        Return:
            List of checkpoints.
        """
        ckpts = [
            x[:-6] for x in os.listdir(directory)
            if x.split('.')[-1] == 'index'
        ]

        tmp = []
        for ckp in ckpts:
            parts = glob.glob(os.path.join(directory, '{}.data*'.format(ckp)))
            if len(parts) == 0:
                logger.info(
                    'file {} failed to be loaded, no parts'.format(ckp))
                continue

            if len(parts) < int(parts[0].split('/')[-1].split('-')[4]):
                logger.info('file {} failed to be loaded, '
                            'not enough parts'.format(ckp))
                continue

            if not os.path.isfile(
                    os.path.join(directory, '{}.meta'.format(ckp))):
                logger.info('file {} failed to be loaded, no meta'.format(ckp))
                continue
            tmp.append(ckp)
        ckpts = tmp
        ckpts.sort(key=lambda x: int(x.split('-')[-1]))
        if len(ckpts) == 0:
            raise ValueError('Not ckpt available')
        return ckpts

    @classmethod
    def checkpoint_file_analizer(cls, ckpt):
        """Analyze checkpoint file.

        Args:
            ckpt: checkpoint file

        Return:
            list of steps where ckpts were created.

        Rise:
            FileNotFoundError: if file not found
            Value Error: if the checkpoint file is incoherent
        """
        ckpts = []
        try:
            with open(ckpt) as f:
                line0 = f.readline()
                line0 = line0.split('"')[1].split('-')[1]
                for x in f:
                    xt = x.split('"')[1].split('-')[1]
                    ckpts.append(int(xt))
            ckpts.sort()
            if ckpts[-1] != int(line0):
                raise ValueError('error on last line')
            return ckpts
        except FileNotFoundError as e:
            raise e

    @classmethod
    def checkpoint_step_list(cls, directory):
        """checkpoints step list
        """
        lst = Checkpoint.checkpoint_file_list(directory)
        lst = [int(x.split('-')[-1]) for x in lst]
        return lst

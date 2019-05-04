import re
import os
import glob
import json
import numpy as np
import logging
import struct

from tensorflow.python import pywrap_tensorflow
from .systemscaffold import SystemScaffold
from .systemscaffold import A2affNetwork

logger = logging.getLogger(__name__).getChild('checkpoint_ops')


class Checkpoint(object):
    """
    class to handle a Checkpoint
    """

    def __init__(self,
                 path_file,
                 species_list,
                 networks_kind=None,
                 networks_metadata=None):
        """Init checkpoint

        Args:
            path_file: eg: a/path/to/model.ckpt-0
            species_list: atom sequence eg ['H','C','N','O'],
                       if None atomic species can not be inferred by name
                       just by position
        """
        super().__init__()

        self._path_file = path_file
        self._species_list = species_list

        self._networks_kind = networks_kind or ['a2ff' for x in species_list]
        self._networks_metadata = networks_metadata or [{
            'species': x
        } for x in species_list]

        reader = pywrap_tensorflow.NewCheckpointReader(self._path_file)
        self._reader = reader

    @property
    def get_scaffold(self):
        scaffold = SystemScaffold()
        for network_kind in set(self._networks_kind):
            if network_kind == 'a2ff':
                Network = A2affNetwork

            matches = []
            for layer_name, shape in \
                self._reader.get_variable_to_shape_map().items():

                for regexp in Network.regexps():
                    match = re.findall(regexp, layer_name)
                    if len(match) >= 1:
                        matches.append((match, shape,
                                        self._reader.get_tensor(layer_name)))
            networks = Network.reconstruct_from_regexp(matches,
                                                       self._networks_metadata)

            for species_idx, network in networks:
                species = network.name
                scaffold[str(species)] = network
        scaffold.sort_atomic_sequence(self._species_list)
        return scaffold

    def dump_PANNA_checkpoint_folder(self,
                                     folder='dump_checkpoint',
                                     dumping_function=np.save,
                                     dumping_extension='npy',
                                     extra_data={}):
        """Dump weights and biases of a checkpoint to a directory, also metadata

        Args:
            dump: where to dump the files
            dumping_function: the function that will be used to dump the data,
                              the function must take 2 parameters
                              (stirng, numpy_array)
                              and must execute the saving operation.
            dumping_extension: the extension of the dumped files
            extra_data: extra dictionary to be added to the json


        Return:
            None

        This method create a folder and put inside that folder
        - one file of weights for each layer dumped with dumping_function
        - one file of biases for each layer dumped with dumping_function
        - networks_metadata.json a json containing extra info
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        scaffold = self.get_scaffold
        metadata = scaffold.metadata

        # TODO, This for now is here but it must be moved inside the
        # network class somehow, just the dump operation should be done here
        metadata['networks_files'] = []

        for idx_s, species in enumerate(scaffold.atomic_sequence):
            network_files = []
            network = scaffold[species]
            for idx_l, layer in enumerate(network):
                base_name = 'species_{}_layer_{}/'.format(idx_s, idx_l)

                w_name = base_name + 'weights'
                b_name = base_name + 'biases'
                w_size = layer.wb_shape
                b_size = layer.b_shape

                w_file_name = ('species_{}_layer_{}_weights_'
                               '{}x{}.{extension}'.format(
                                   species,
                                   idx_l,
                                   *w_size,
                                   extension=dumping_extension))
                b_file_name = ('species_{}_layer_{}_biases_'
                               '{}.{extension}'.format(
                                   species,
                                   idx_l,
                                   *b_size,
                                   extension=dumping_extension))

                network_files.append((w_file_name, b_file_name))

                dumping_function(
                    os.path.join(folder, w_file_name), layer.w_value)
                dumping_function(
                    os.path.join(folder, b_file_name), layer.b_value)

            metadata['networks_files'].append(network_files)

        metadata.update(extra_data)
        with open(os.path.join(folder, 'networks_metadata.json'), 'w') as f:
            json.dump(metadata, f)
        return None

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
        metadata = scaffold.metadata

        with open(os.path.join(folder,filename), 'w') as f:
            f.write("[GVECT_PARAMETERS]\n")
            f.write("Nspecies = {}\n".format(len(scaffold.atomic_sequence)))
            f.write("species = {}\n".format(",".join(scaffold.atomic_sequence)))
            for p in extra_data['gvect_params'].keys():
                f.write("{} = {}\n".format(p,extra_data['gvect_params'][p]))

            for idx_s, species in enumerate(scaffold.atomic_sequence):
                network = scaffold[species]
                sizes = []
                activs = []
                weights = np.array([], dtype=np.float32)
                for idx_l, layer in enumerate(network):
                    sizes.append(layer.b_shape[0])
                    activs.append(layer.activation)
                    weights = np.append(weights,layer.w_value.flatten())
                    weights = np.append(weights,layer.b_value.flatten())
                f.write("\n[{}]\n".format(species))
                f.write("Nlayers = {}\n".format(len(sizes)))
                f.write("sizes = {}\n".format(",".join(map(str, sizes))))
                wfname = "weights_{}.dat".format(species)
                f.write("file = {}\n".format(wfname))
                f.write("activations = {}\n".format(",".join(map(str, activs))))
                myfmt='f'*len(weights)
                binw=struct.pack(myfmt,*weights)
                wf=open(os.path.join(folder,wfname),"wb")
                wf.write(binw)
                wf.close()

            f.close()
        
        return None


    @property
    def filename(self):
        return self._path_file.split('/')[-1]

    @property
    def step(self):
        return int(self._path_file.split('model.ckpt-')[1].split('.')[0])

    @property
    def stats_101(self):
        """ for each tensor compute basic stats

        return:
            dict, 1 key for each tensor with a tuple (min, max, var, mean)
                  as value
        """
        res = {}
        for k, v in self._tensors_value.items():
            res[k] = (v.min(), v.max(), v.var(), v.mean())
        return res

    @property
    def norms_calculator(self):
        """Calculate the norm 1 and 2 of all the tensors

        Return:
            a dict {tensor_name:(l1, l2)}

        l1 = sum_i(abs(v_i))
        l2 = sum_i(v_i**2)/2
        """
        n = {}
        for k, v in self._tensors_value.items():
            n[k] = (np.sum(np.abs(v)), np.sum(v**2) / 2)
        return n

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
                logger.info('file {} failed to be loaded'.fromat(f))
                break

            if len(parts) < int(parts[0].split('-')[4]):
                logger.info('file {} failed to be loaded'.fromat(f))
                break

            if not os.path.isfile(
                    os.path.join(directory, '{}.meta'.format(ckp))):
                break
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

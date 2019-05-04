import os
import json
import numpy as np
from copy import copy, deepcopy

if __name__ == 'systemscaffold':
    from a2affnetwork import A2affNetwork
else:
    from .a2affnetwork import A2affNetwork


class NetworkNotAvailableError(ValueError):
    pass


class SystemScaffold():
    def __init__(self,
                 default_network=None,
                 atomic_sequence=None,
                 zeros=None,
                 name='default system'):
        super().__init__()
        self._networks = {}
        self._default_network = default_network
        self.name = name

        self._atomic_sequence = atomic_sequence or []

        if zeros and atomic_sequence:
            if len(zeros) != len(atomic_sequence):
                raise ValueError('output offset must be of '
                                 'the same size as atomic sequence')
            self._zeros = dict(zip(atomic_sequence, zeros))
        elif zeros and not atomic_sequence:
            raise ValueError('output offset can not be specified '
                             'without atomic_sequence')
        else:
            self._zeros = {}

    def __getitem__(self, value):
        tmp_network = self._networks.get(value, None)
        if tmp_network:
            return tmp_network
        elif self._default_network:
            tmp_network = deepcopy(self._default_network)
            tmp_network.name = value
            tmp_network.offset = self._zeros.get(value, 0.0)
            self._networks[value] = tmp_network
            if value not in self._atomic_sequence:
                self._atomic_sequence.append(value)
            return tmp_network
        else:
            raise NetworkNotAvailableError('default network not available')

    def __setitem__(self, index, value):
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

    def evaluate(self, example, forces_flag=False, add_offset=True):
        n_atoms = example.n_atoms
        ss, gs = example.species_vector, example.gvects
        energy = 0

        if forces_flag:
            forces = np.zeros(n_atoms * 3)
            dgs = example.dgvects

        for species_idx, species_symbol in enumerate(self.atomic_sequence):
            network = self[species_symbol]

            # select per species gvects
            species_indices = np.where(ss == species_idx)
            in_gs = gs[species_indices]
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
        else:
            return energy

    def sort_atomic_sequence(self, new_atomic_sequence):
        """

        override internal atomic sequence,
        pls avoid using if possible
        """
        for species in new_atomic_sequence:
            if species not in self._atomic_sequence:
                raise ValueError('species not already present')
        self._atomic_sequence = new_atomic_sequence

    @property
    def atomic_sequence(self):
        return tuple(self._atomic_sequence)

    @property
    def n_species(self):
        return len(self._atomic_sequence)

    @property
    def metadata(self):
        metadata = {}
        metadata['networks_type'] = [
            self[x].network_kind for x in self.atomic_sequence
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

    # back compatible property, used only in train.py
    # adaptation layer
    @property
    def old_layers_sizes(self):
        return [self[x].layers_size for x in self.atomic_sequence]

    @property
    def old_layers_trainable(self):
        return [self[x].layers_trainable for x in self.atomic_sequence]

    @property
    def old_gsize(self):
        return self[self.atomic_sequence[0]].feature_size

    @property
    def old_networks_wb(self):
        tmp = []
        for x in self.atomic_sequence:
            tmp2 = []
            for w, b in self[x].network_wb:
                if w.size == 0:
                    w = None
                if b.size == 0:
                    b = None
                tmp2.append((w, b))
            tmp.append(tmp2)
        return tmp

    @property
    def old_layers_act(self):
        return [self[x].layers_activation for x in self.atomic_sequence]

    @property
    def old_zeros(self):
        return [self[x].offset for x in self.atomic_sequence]

    @classmethod
    def load_PANNA_checkpoint_folder(cls,
                                     folder,
                                     default=None,
                                     atomic_sequence_extension=None,
                                     load_function=np.load):
        # load the data
        file_name = 'networks_metadata.json'
        with open(os.path.join(folder, file_name)) as f:
            data = json.load(f)
        networks_files = data['networks_files']
        networks_type = data['networks_type']
        networks_layers_size = data['networks_layers_size']
        networks_feature_size = data['networks_feature_size']
        networks_layers_activation = data['networks_layers_activation']
        networks_layers_trainable = data['networks_layers_trainable']
        networks_name = data['networks_species']
        networks_offset = data['networks_offset']

        if atomic_sequence_extension == None:
            atomic_sequence_extension = []

        system_scaffold = cls(default, name=folder)

        networks = []
        for i, network_type in enumerate(networks_type):
            init_wb = []
            for w_file, b_file in networks_files[i]:
                w = load_function(os.path.join(folder, w_file))
                b = load_function(os.path.join(folder, b_file))
                init_wb.append((w, b))

            if network_type == 'a2ff':
                network = A2affNetwork(
                    networks_feature_size[i], networks_layers_size[i],
                    networks_name[i], init_wb, networks_layers_trainable[i],
                    networks_layers_activation[i], networks_offset[i])
            else:
                raise ValueError('tipe not supported {}'.format(network_type))
            system_scaffold[networks_name[i]] = network
        # complete scaffold atomic sequence with new species
        # this is a hack and should not be done, if we think of using
        # this feature elsewhere would be better to think a setter for
        # atomic sequence
        old_sequence = system_scaffold.atomic_sequence
        for species in atomic_sequence_extension:
            if species not in old_sequence:
                system_scaffold._atomic_sequence.append(species)
        return system_scaffold

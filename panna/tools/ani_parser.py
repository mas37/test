###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
# expander for ani data

import os
import json
import argparse
from os.path import join

import pyanitools as pya
from keys_generator import hash_key_v2

# constants
HA2EV = 27.2113966413079


def main(args):

    print('begin')
    # Set the HDF5 file containing the data
    hdf5file = args.hdf5file
    # Set output folder
    output_folder = args.output_folder
    # not a good idea to have verbose mode on big files
    verbose = args.verbose

    # Construct the data loader class
    adl = pya.anidataloader(hdf5file)

    if verbose:
        print('group list information:')
        print(adl.get_group_list())

        # recover all the species that are present in the data set
        s = set()
        for x in adl:
            for y in x['species']:
                s.update(y)
        print('species in the dataset: {}'.format(s))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for x in adl:
        species = x['species']
       # print('species: {}'.format(species))
        molecule = ''.join(x['smiles'])
        #print('smile: ' + molecule)

        if not os.path.exists(join(output_folder, molecule)):
            os.makedirs(join(output_folder, molecule))

        if not os.path.exists(join(output_folder, molecule, 'examples')):
            os.makedirs(join(output_folder, molecule, 'examples'))

        # dump of already extracted configurations
        filedump = open(join(output_folder, molecule, 'comp_examples.dat'),
                        'a')

        with open(join(output_folder, molecule, 'comp_examples.dat'), 'r')\
                as f:
            already_computed = f.read().split(',')

        if len(already_computed) - 1 == len(x['energies']):
            print(molecule + 'already computed')
            filedump.close()
            continue

        print('already computed: {}/{}'.format(len(already_computed) - 1,
                                               len(x['energies'])))

        # cycle over all the available configurations of a molecule
        for cords, e in zip(x['coordinates'], x['energies']):
            sim = {}
            for i, v in enumerate(zip(cords, species)):
                cord, kind = v
                x1, x2, x3 = cord
                cs = sim.get('atoms', [])
                cs.append((i + 1,
                           kind,
                           (float(x1), float(x2), float(x3)),
                           (0, 0, 0)
                           ))
                sim['atoms'] = cs
            sim['energy'] = (e * HA2EV, 'ev')
            sim['lattice_vectors'] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            sim['atomic_position_unit'] = 'cartesian'
            sim['unit_of_length'] = 'angstrom'
            sim['name'] = molecule
            key = hash_key_v2(sim)

            if key in already_computed:
                print('computed {}'.format(key))
                continue

            with open(os.path.join(output_folder, molecule, 'examples',
                                   key + '.example'), 'w') as f:
                json.dump(sim, f)
            filedump.write(key + ',')
            filedump.flush()
        filedump.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ANI data expander ')
    parser.add_argument('-in', '--hdf5file', type=str,
                        help='file to decompress', required=True)
    parser.add_argument('-out', '--output_folder', type=str,
                        help='output main folder', required=True)
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help='verbose mode', required=False)
    args = parser.parse_args()
    main(args)

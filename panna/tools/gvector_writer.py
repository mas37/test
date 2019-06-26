###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import os
import argparse
import numpy as np


def main(source, navg, **kvargs):
    '''
    Parses binary gvector files and writes out in plain text.

    If source is a directory, it converts all files in the dir,
    and writes out filename_plain.dat
    In each filename_plain.dat, Gvectors are written in column per each atom,
    i.e. H2O file has three columns H1 H2 O - the species number is listed in the first line.

    If flag -a navg is passed, it computes and writes only the avg per
       species, for the first navg species, in a file named average_g.dat,
       one column per species. The species order is the same as the Gvector.
    '''

    if os.path.isdir(source):
        files = [
            os.path.join(source, f) for f in os.listdir(source)
            if os.path.isfile(os.path.join(source, f))
        ]
        nfiles = len(files)
    else:
        files = [source]
        source = os.path.dirname(source)
        nfiles = 1

    for nsf, single_file in enumerate(files):
        f = np.fromfile(single_file, dtype=np.float32)
        n_atoms = int(f[0])
        g_size = int(f[1])
        en = f[2]
        spec_size = n_atoms
        gvect_size = n_atoms * g_size

        if len(f) != (3 + spec_size + gvect_size):
            print("file {} size inconsistent".format(single_file))
            exit(1)

        spec_tensor = np.reshape((f[3:spec_size + 3]).astype(np.int32),
                                 [1, n_atoms])
        gvect_tensor = np.reshape(f[3 + spec_size:3 + gvect_size + spec_size],
                                  [n_atoms, g_size])

        if navg == 0:
            outname = single_file + "_plain.dat"
            with open(outname, "w") as out:
                out.write("# ")
                np.savetxt(out, spec_tensor, fmt="%d", delimiter=" ")
                np.savetxt(
                    out,
                    np.transpose(gvect_tensor),
                    fmt="%1.6e",
                    delimiter=" ")
        else:
            if nsf == 0:
                avg_g = np.zeros([navg, g_size], dtype=np.float32)
                num = np.zeros(navg, dtype=np.int32)
                gref = g_size
            if g_size != gref:
                print("gvectors have incompatible sizes")
                exit(1)
            for aa in range(n_atoms):
                sp = spec_tensor[0, aa]
                if (sp < navg):
                    avg_g[sp] += gvect_tensor[aa]
                    num[sp] += 1
            if nsf == nfiles - 1:
                for at in range(navg):
                    if num[at] != 0:
                        avg_g[at] /= num[at]
                with open(os.path.join(source, "average_g.dat"), "w") as out:
                    np.savetxt(
                        out, np.transpose(avg_g), fmt="%1.6e", delimiter=" ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gvector plain text converter')
    parser.add_argument(
        '-s',
        '--source',
        type=str,
        help='source file or folder',
        required=True)
    parser.add_argument('-a', '--average', type=int, default=0, required=False)
    args = parser.parse_args()
    main(args.source, args.average)

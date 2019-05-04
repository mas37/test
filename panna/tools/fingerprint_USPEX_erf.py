import numpy as np
import itertools
from gvector.pbc import replicas_max_idx
import json
import os
import argparse
import multiprocessing as mp
from functools import partial
from scipy.special import erf

number_of_process = 20


def fingprint(Rmax, delta, sigma, outdir, filename):
    '''This module compute fingerprint for a given configuration in json
       The fingerprint are saved in json for each pair and the associated weights
       the total energy of configuration.
       e.g: for a system with H and C: {"HH":[....],"wHH": real-number, "HC":[....].. etc}

       Rmax: cutoff
       delta: descritization steps
       sigma: gaussian width
       outdir: output folder where fingerprint is written
       filename is the input json (currectly in .simulation extension)
    '''

    df = open(filename, 'r')
    data = json.load(df)
    outfile = filename.split('/')[-1]
    figpr = open(os.path.join(outdir, outfile.split('.')[0]), 'w+')

    vol = np.abs(np.dot(data['lattice_vectors'][0],\
                 np.cross(data['lattice_vectors'][1],\
                 data['lattice_vectors'][2])))
    atomic_position_unit = data['atomic_position_unit']

    lattice_vectors = np.asarray(data['lattice_vectors']).astype(float)
    max_indices = replicas_max_idx(lattice_vectors, Rmax)
    l_max, m_max, n_max = max_indices
    l_list = range(-l_max, l_max + 1)
    m_list = range(-m_max, m_max + 1)
    n_list = range(-n_max, n_max + 1)

    energy = data['energy'][0]
    pos = []
    atype = []
    all_symbols = []
    for atom in data['atoms']:
        idx, symbol, position, force = atom
        if atomic_position_unit == "crystal":

            pos.append([
                idx, symbol,
                np.dot(
                    np.transpose(lattice_vectors),
                    np.asarray(position).astype(float))
            ])
        else:
            pos.append([idx, symbol, np.asarray(position).astype(float)])

        all_symbols.append(symbol)

        if symbol not in atype:
            atype.append(symbol)
    data['atoms'] = pos
    Natoms = len(pos)
    Nspecies = len(atype)
    # compute total weight
    weightab = 0
    for i in range(Nspecies):
        Na = all_symbols.count(atype[i])
        for j in range(i, Nspecies):
            Nb = all_symbols.count(atype[j])
            weightab += Na * Nb
    #################################

    F_size = int(Nspecies * (Nspecies + 1) * 0.5)
    Nr = int((Rmax) / delta)
    F_vector = -np.ones((F_size, Nr), dtype=float)

    for idx_i in range(Nspecies):
        typa = atype[idx_i]
        Na = all_symbols.count(typa)
        for atom1 in data['atoms']:

            idxi, symboli, pos_i = atom1
            pos_i = np.asarray(pos_i).astype(float)
            if symboli == typa:
                for idx_j in range(idx_i, Nspecies):
                    typb = atype[idx_j]

                    Nb = 0
                    idx_row = int(idx_i * (Nspecies - (idx_i + 1) * 0.5) +
                                  idx_j)
                    Nb = all_symbols.count(typb)
                    for atom2 in data['atoms']:
                        idxj, symbolj, posj = atom2
                        # loop over all cells around
                        if symbolj == typb:
                            for l, m, n in itertools.product(
                                    l_list, m_list, n_list):
                                pos_j = np.asarray(posj).astype(float) +\
                                            l * lattice_vectors[0] +\
                                            m * lattice_vectors[1] +\
                                            n * lattice_vectors[2]

                                Rij = np.linalg.norm(pos_j - pos_i)
                                if Rij <= Rmax and Rij > 1e-5:

                                    idx_bin = int(Rij / delta)
                                    #  bins_away = range(idx_bin-int(delta/sigma),\
                                    #              int(delta/sigma)+idx_bin)
                                    deltafunct = 1.0e10
                                    idxfict = idx_bin
                                    while deltafunct > 0.000001:
                                        Rk_down = (idxfict) * delta
                                        Rk_up = (idxfict + 1) * delta
                                        R1 = (Rk_down - Rij) / np.sqrt(
                                            2.0 * sigma**2)
                                        R2 = (Rk_up - Rij) / np.sqrt(
                                            2.0 * sigma**2)
                                        deltafunct = 0.5 * (erf(R2) - erf(R1))
                                        #print(deltafunct,idxfict,idx_bin)
                                        idxfict += 1
                                    bins_away = range(2 * idx_bin - idxfict,
                                                      idxfict)

                                    for idx_column in bins_away:
                                        if 0 <= idx_column <= int(
                                                Rmax / delta) - 1:
                                            Rk_down = (idx_column) * delta
                                            Rk_up = (idx_column + 1) * delta
                                            R1 = (Rk_down - Rij) / np.sqrt(
                                                2.0 * sigma**2)
                                            R2 = (Rk_up - Rij) / np.sqrt(
                                                2.0 * sigma**2)
                                            deltafunc = 0.5 * (
                                                erf(R2) - erf(R1))
                                            #   print(idx_column,int(Rmax/delta),deltafunc)

                                            F_vector[idx_row][idx_column] += deltafunc*vol/\
                                                              (4.*np.pi*Rij**2*Na*Nb*delta)

    F_vector_dict = {}
    for idx_i in range(Nspecies):
        typa = atype[idx_i]
        Na = all_symbols.count(typa)
        for idx_j in range(idx_i, Nspecies):
            typb = atype[idx_j]
            Nb = all_symbols.count(typb)
            idx_row = int(idx_i * (Nspecies - (idx_i + 1) * 0.5) + idx_j)
            F_vector_dict[typa + typb] = F_vector[idx_row].tolist()
            F_vector_dict['w' + typa + typb] = Na * Nb / weightab
    F_vector_dict['energy'] = energy
    figpr.write(json.dumps(F_vector_dict))
    figpr.close()
    return


def main(indir, outdir):
    #compute many configuration at a time

    p = mp.Pool(number_of_process)

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    files = [os.path.join(indir,x) for x in os.listdir(indir) \
            if x.split('.')[-1]=='simulation' and \
            os.stat(os.path.join(indir,x)).st_size != 0]

    Rmax = 10.0  #Ang
    delta = 0.08  #Ang
    sigma = 0.03  #Ang
    #sigma = sigma/np.sqrt(2 * np.log(2))
    ###############################################################
    Fingprint = partial(fingprint, Rmax, delta, sigma, outdir)
    i = 0
    while i <= int(len(files) / number_of_process):
        ll = i * number_of_process
        lu = (i + 1) * number_of_process
        try:
            fi = files[ll:lu]
        except IndexError:
            fi = files[ll, len(files)]
        data = p.map(Fingprint, fi)
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='making FP')
    parser.add_argument(
        '-i', '--indir', type=str, help='in path', required=True)

    parser.add_argument(
        '-o', '--outdir', type=str, help='out path', required=True)

    args = parser.parse_args()
    if args.indir and args.outdir:
        main(indir=args.indir, outdir=args.outdir)

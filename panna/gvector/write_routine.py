import numpy as np

from os.path import join


def compute_binary(example_dict, binary_out=None):
    """Transform a example_dict to a binary output

    Args:
        example_dict: dictionary for a simulation
        binary_out: where to save, if None save is not performed
    Return:
        binary data
    """
    n = np.asarray(len(example_dict['species']), dtype=np.float32).flatten()
    g = np.asarray(len(example_dict['Gvect'][0]), dtype=np.float32).flatten()
    e = np.asarray(example_dict['E'], dtype=np.float32).flatten()
    t1 = np.asarray(example_dict['species'], dtype=np.float32).flatten()
    t2 = np.asarray(example_dict['Gvect'], dtype=np.float32).flatten()

    contents = b"".join(np.concatenate((n, g, e, t1, t2), axis=0))

    if binary_out:
        with open(join(binary_out, example_dict['key']), 'wb') as f:
            f.write(contents)

    return contents


def compute_binary_dGvect(example_dict, binary_out=None):
    """Transform a example_dict to a binary output

    Args:
        example_dict: dictionary for a simulation
        binary_out: where to save, if None save is not performed
    Return:
        binary data
    """
    n = np.asarray(len(example_dict['species']), dtype=np.float32).flatten()
    g = np.asarray(len(example_dict['Gvect'][0]), dtype=np.float32).flatten()
    e = np.asarray(example_dict['E'], dtype=np.float32).flatten()
    t1 = np.asarray(example_dict['species'], dtype=np.float32).flatten()
    t2 = np.asarray(example_dict['Gvect'], dtype=np.float32).flatten()
    t3 = np.asarray(example_dict['dGvect'], dtype=np.float32).flatten()
    if len(example_dict['forces']) > 0:
        t4 = np.asarray(example_dict['forces'], dtype=np.float32).flatten()
    else:
        t4 = np.asarray([], dtype=np.float32)

    contents = b"".join(np.concatenate((n, g, e, t1, t2, t3, t4), axis=0))

    if binary_out:
        with open(join(binary_out, example_dict['key']), 'wb') as f:
            f.write(contents)

    return contents


def compute_binary_oldformat(example_dict, binary_out=None):
    """Transform a example_dict to a binary output

    Args:
        example_dict: dictionary for a simulation
        binary_out: where to save, if None save is not performed
    Return:
        binary data
    """
    g_size = len(example_dict['Gvect'][0])
    natoms = example_dict['max_n_of_atoms']
    Nspecies = example_dict['number_of_species']

    t1 = [[0 for j in range(Nspecies)] for i in range(natoms)]
    t2 = [[[0 for k in range(g_size)] for j in range(Nspecies)]
          for i in range(natoms)]
    for i, v in enumerate(example_dict['species']):
        t1[i][v] = np.float32(1)
        t2[i][v] = example_dict['Gvect'][i]
    e1 = np.asarray(example_dict['E'], dtype=np.float32).flatten()
    t1 = np.asarray(t1, dtype=np.float32).flatten()
    t2 = np.asarray(t2, dtype=np.float32).flatten()

    contents = b"".join(np.concatenate((e1, t1, t2), axis=0))

    if binary_out:
        with open(join(binary_out, example_dict['key']), 'wb') as f:
            f.write(contents)

    return contents

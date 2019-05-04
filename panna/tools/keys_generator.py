"""Containg different key generators

hash_key_v2: slighly improved key
hash_key_v1: old key, not reimplemented
"""
import hashlib


def hash_key_v2(example_dict):
    """Return an unique key for every example
       IT IS NOT RANDOM

    Args:
        example_dict: dictionary representig a example
    Retun:
        an unique string for the simulator
    """

    # creation of the string
    strings = []
    for atom in example_dict['atoms']:
        idx, kind, position, force = atom
        s = str(idx)
        s += str(kind)
        position = [float(x).hex() for x in position]
        force = [float(x).hex() for x in force]
        s += ''.join(map(str, position))
        s += ''.join(map(str, force))
        strings.append(s)

    strings.append(str(float(example_dict['energy'][0]).hex()))
    strings.append(example_dict['energy'][1])
    strings.append(str(example_dict['lattice_vectors']))
    strings.append(example_dict['atomic_position_unit'])
    strings.append(example_dict['unit_of_length'])
    strings.append(example_dict['name'])

    strings = ''.join(strings)
    key = hashlib.sha224(strings.encode('utf-8')).hexdigest()
    return key

###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import tensorflow as tf
import numpy as np
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def writer(filename, data, path='.'):
    with tf.python_io.TFRecordWriter(
            os.path.join(path, '{}.tfrecord'.format(filename)))\
            as record_writer:
        for x in data:
            record_writer.write(x.SerializeToString())


def example_tf_packer(example, forces):
    """Create an example with only the g-vectors without padding

    Args:
       example: instance of Example
       forces: flag to add derivative an forces

    Return:
       tf.train.Example with the following keys:
         + gvects: (number of atoms, g-vector size) vector of float
         + species: (number of atoms) vector of integer
                    in range 0, number of species - 1
         + energy: (1) vector of float
         --- if forces
         + dgvects : (number of atoms, g-vecotr size, 3 * number of atom )
         + forces : (3 * number of atoms )
    """
    feature = {
        'gvects': _floats_feature(example.gvects.flatten()),
        'species': _int64_feature(example.species_vector),
        'energy': _floats_feature([example.true_energy]),
        'name': _bytes_feature([example.name.encode()])
    }
    if forces:
        if example.dgvects.size > 0:
            feature['dgvects'] = _floats_feature(example.dgvects.flatten())
        if example.forces.size > 0:
            feature['forces'] = _floats_feature(example.forces)
    return tf.train.Example(features=tf.train.Features(feature=feature))

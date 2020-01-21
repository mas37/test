###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logger

from lib.errors import ShapeError

from .layers import Layer


class A2affNetwork():
    """ Implementation of a all to all connected feed forward network
    A2affNetwork (feature_size, layers_size, name, network_wb=None,
                  trainables=None, activations=None, offset=None)

    Returns an all to all connected feed forward network that is a
    basic  **general purpose** neural network

    Parameters
    ----------
    feature_size: int
        Size of the feature vector, or Input vector, or Gvector
        depending on notations
    layers_size: list of integers
        List of integers that describe the layers shape:
        for example a list like [128, 64, ..., 4, 1] will create a network
        like
        in layer: feature_size: 128
        first layer: 128:64
        second layer: 64:..
        .....
        last layer: ..:4
        out layer:  4:1
    name: str
        name of the network
    network_wb: list of tuples
          one tuple for each layer, a tuple is composed by 2 numpy_array
          [(w_values, b_values),......, (w_values, b_values)]
          an empty value is passed as np.empty(0)
          default is: creates all layer with Gaussian distribution
    trainables: list of bools, optional
          one for each layer to set a layer as trainable or not
          default is all layer trainables
    activations: list of integer, optional
          one for each layer to set the activation kind of the layer
          default is: gaussian:gaussian:....:linear
    offset: float, optional
          network offset value
          default is: 0.0
    Raises
    ------
    ValueError if:
        - network_wb is too small
        - trainables is too small
        - activations is too small
    """

    _reg_exps = {
        'network': r'species_(\d+)',
    }  # quantity used to recover tensors from TF checkpoint

    def __init__(self,
                 feature_size,
                 layers_size,
                 name,
                 network_wb=None,
                 trainables=None,
                 activations=None,
                 offset=None):

        super().__init__()

        if not network_wb:
            # default behavior: creates all layer with gaussian distribution
            network_wb = [(np.empty(0), np.empty(0))
                          for i in range(len(layers_size))]
        if not trainables:
            # default behavior: all trainable
            trainables = [True for i in range(len(layers_size))]
        if not activations:
            # default behavior: gaussian:gaussian:....:linear
            activations = [1 for i in range(len(layers_size) - 1)] + [0]

        if len(layers_size) != len(network_wb):
            raise ValueError('network wb parameters are not enough')
        if len(layers_size) != len(trainables):
            raise ValueError('trainables parameters are not enough')
        if len(layers_size) != len(activations):
            raise ValueError('activations parameters are not enough')

        self.feature_size = int(feature_size)
        self.layers_size = [int(x) for x in layers_size]
        self.offset = offset or 0.0
        self.name = name

        layers = []
        for i in range(len(layers_size)):
            w_value, b_value = network_wb[i]
            layer = Layer(self.layers_shaping[i], trainables[i],
                          activations[i], w_value, b_value)
            layers.append(layer)

        self._layers = layers

    @property
    def layers_shaping(self):
        """ list of tuple
        two elements for each layer
        [(shape_w, shape_b),.....]
        """
        layer_shaping = [(self.layers_size[i], self.layers_size[i + 1])
                         for i in range(len(self.layers_size) - 1)]
        return [(self.feature_size, self.layers_size[0])] + layer_shaping

    @property
    def layers_trainable(self):
        """ tuple of Boolean
        one elements for each layer

        Raises
        ------
        ValueError
            if number of elements differ from number of layer
        """
        trainability = []
        for layer in self._layers:
            trainability.append(layer.trainable)
        return tuple(trainability)

    @layers_trainable.setter
    def layers_trainable(self, value):
        if len(value) != len(self.layers_size):
            raise ValueError('passed trainable vector is '
                             'not compatible with current layer size')
        for layer, trainable in zip(self._layers, value):
            layer.trainable = trainable

    @property
    def layers_activation(self):
        """ tuple of int
        one elements for each layer

        Raises
        ------
        ValueError
            if number of elements differ from number of layer
        """
        activations = []
        for layer in self._layers:
            activations.append(layer.activation)
        return tuple(activations)

    @layers_activation.setter
    def layers_activation(self, value):
        if len(value) != len(self.layers_size):
            raise ValueError('passed act vector is '
                             'not compatible with current layer size')
        for layer, activation in zip(self._layers, value):
            layer.activation = activation

    @property
    def network_wb(self):
        """ list of tuple
        two numpy_array for each layer, weights and biases values
        """
        wb = []
        for layer in self._layers:
            wb.append(layer.wb_value)
        return tuple(wb)

    @property
    def network_type(self):
        """ string
        kind of network
        """
        return 'a2ff'

    def __getitem__(self, index):
        result = self._layers[index]
        return deepcopy(result)

    def __setitem__(self, index, value):
        old_layer = self._layers[index]
        if value.wb_shape != old_layer.wb_shape:
            raise ShapeError('request to substitute a leayer of shape '
                             '{} with one of shape {}'.format(
                                 old_layer.wb_shape, value.wb_shape))
        self._layers[index] = value

    def evaluate(self,
                 features_vector,
                 dfeatures_vectors=np.empty(0),
                 add_offset=True):
        """ Evaluate the layer for a given set of inputs

        Parameters
        ----------
        features_vector: numpy_array(?, feature_size)
            ? is the number of features that the network has to estimate
              can be read as the number of atoms of the kind the
              network should process
        dfeatures_vectors: numpy_array(?, 3 * total ?, features_size), optional
            TODO: check the sizes, maybe 1,2 are inverted
            3 * total ? is the total number of atoms in the system
            if not passed forces are not evaluated
        add_offset: Boolean, optional
            if adding the offset or not

        Returns
        -------
        numpy_array (?, last_layer_shape)
             ? is the number of elements that have been be processed

        numpy_array(?, 3 * total ?)
            Derivatives, aka forces.
            this is returned ONLY if dfeatures_vectors.size > 0
        """
        in_vectors = features_vector

        if dfeatures_vectors.size > 0:
            din_vectors = np.transpose(dfeatures_vectors, (0, 2, 1))
            for layer in self._layers:
                out_vectors, dout_vectors = layer.evaluate(
                    in_vectors, din_vectors)
                in_vectors, din_vectors = out_vectors, dout_vectors
            dout_vectors = np.squeeze(dout_vectors, axis=(2))
            if add_offset:
                out_vectors += self.offset
            return out_vectors, dout_vectors

        for layer in self._layers:
            out_vectors = layer.evaluate(in_vectors)
            in_vectors = out_vectors
        if add_offset:
            out_vectors += self.offset
        return out_vectors

    def tf_evaluate(self, features_vector, compute_gradients=False):
        in_vectors = features_vector

        for l_idx, layer in enumerate(self._layers):
            logger.debug('inserting layer - %d', l_idx)
            with tf.variable_scope("species_{}_layer_{}".format(
                    self.name, l_idx)):
                out_vectors = layer.tf_evaluate(in_vectors)
                in_vectors = out_vectors

        if compute_gradients:
            return out_vectors, tf.gradients(out_vectors, features_vector)

        return out_vectors

    def customize_network(self,
                          feature_size=None,
                          layers_size=None,
                          trainables=None,
                          behaviors=None,
                          activations=None,
                          offset=None,
                          override_wb=None):
        """Inplace customization of an already defined netwrok

        Parameters
        ----------
        feature_size: int, optional
            Size of the feature vector, or Input vector, or Gvector
            depending on notations
        layers_size: list of integers, optional
            List of integers that describe the layers shape:
            for example a list like [128, 64, ..., 4, 1] will create a network
            like
            in layer: feature_size: 128
            first layer: 128:64
            second layer: 64:..
            .....
            last layer: ..:4
            out layer:  4:1
        trainables: list of bools, optional
            one for each layer to set a layer as trainable or not
            default is all layer trainables
        behaviors: list of string, optional

        activations: list of integer, optional
            one for each layer to set the activation kind of the layer
        offset: float, optional
            network offset value
        override_wb: list of tuples, optional
            one tuple for each layer, a tuple is composed by 2 numpy_array
            [(w_values, b_values),......, (w_values, b_values)]
            an empty value is passed as np.empty(0)

        Note
        ----
        The behavior can be summarized as follow:
        ....
        For now use unittest as reference
        ....

        Return
        ------
        None

        Raises
        ------
        ValueError if:
            -.....
        """

        if layers_size:
            if len(layers_size) != len(self.layers_size):
                new_structure = True
            else:
                for new, old in zip(layers_size, self.layers_size):
                    if old != new:
                        new_structure = True
                        break
                else:
                    new_structure = False
        else:
            new_structure = False

        if feature_size:
            if feature_size != self.feature_size:
                new_structure = True

        if offset:
            self.offset = offset

        if new_structure:
            # is easier to recreate the network
            # trainable and activation falgs are layer property
            # not network property, so overriding a layer means inherit
            # his trainability and activation
            # keep_wb is putted here to emphasize this behavior
            new_net = A2affNetwork(
                feature_size if feature_size else self.feature_size,
                layers_size if layers_size else self.layers_size, self.name,
                None, trainables, activations)
            old_layers = self._layers
            old_layers_size = self.layers_size

            self.feature_size = feature_size if feature_size\
                else self.feature_size
            self.layers_size = layers_size if layers_size else self.layers_size
            self._layers = new_net._layers

            if behaviors:
                if len(behaviors) != len(self.layers_size):
                    raise ValueError('{} layer_behavior != layer sizes'.format(
                        self.name))
                for i, behavior in enumerate(behaviors):
                    if behavior == 'keep':
                        if i >= len(old_layers_size):
                            raise ValueError('requested to default a not '
                                             'available layer')
                        self._layers[i].w_value = old_layers[i].w_value
                        self._layers[i].b_value = old_layers[i].b_value

                    elif behavior == 'load':
                        tmp_layer = self._layers[i]
                        tmp_layer.w_value, tmp_layer.b_value = override_wb[i]
                    elif behavior == 'new':
                        # nothing to do, the layer is already the new version
                        pass
                    else:
                        raise ValueError('undefined behavior for layer')
        else:
            if behaviors:
                if len(behaviors) != len(self.layers_size):
                    raise ValueError('{} layer_behavior != layer size'.format(
                        self.name))
                for i, behavior in enumerate(behaviors):
                    if behavior == 'keep':
                        # do nothing, the layer is already in place
                        pass
                    elif behavior == 'load':
                        tmp_layer = self._layers[i]
                        tmp_layer.w_value, tmp_layer.b_value = override_wb[i]
                    elif behavior == 'new':
                        tmp_layer = self._layers[i]
                        tmp_layer.w_value, tmp_layer.b_value = (np.empty(0),
                                                                np.empty(0))

                    else:
                        raise ValueError('undefined behavior for layer')

            if trainables:
                self.layers_trainable = trainables
            if activations:
                self.layers_activation = activations

    @property
    def tf_ckpt_elements(self):
        """ regexps for TF checkpoint

        Return
        ------
        stirngs  + objects
        """
        names = []
        objects = []

        for l_idx, layer in enumerate(self._layers):
            names.append("species_{}_layer_{}".format(self.name, l_idx))
            objects.append(layer)
        return names, objects

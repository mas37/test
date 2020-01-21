###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from lib.errors import ShapeError

from .variable_helpers import _variable_on_cpu, _variable_random_uniform


class Layer():
    """All to all connected layer to be used to create networks
    Layer(wb_shape, trainable, activation, w_value=np.empty(0),
          b_value=np.empty(0))

    Returns a layer.
    A layer is a a generic all to all connected layer with several
    activation functions.

    Parameters
    ----------
    wb_shape: sequence of 2 numbers
        First number is interpreted as the number of weights per node
        Second number is interpreted as the number of nodes
    trainable: integer of bool
        Specify if the layer is trainable or not
    activation: integer
        Specify the activation type
            - 0:: linear
            - 1:: Gaussian
            - 2:: rbf
                  o_i = \\exp(-\\sum_{j=0}^{in_size}(in_j - w_i)^2)
                  basically each neuron represent a basis element
                  (Gaussian centered around w)
            - 3:: relu
    w_value: numpy array, optional
        Tensor with weights value, shape must be
        (number_of_weights, number_of_biases)
        if not provided the layer can not be evaluated and it will be
        randomly initialized if required in TF
    b_value: numpy array, optional
        Tensor with biases value, shape must be
        if not provided the layer can not be evaluated and it will be
        randomly initialized if required in TF
    """
    def __init__(self,
                 wb_shape,
                 trainable,
                 activation,
                 w_value=np.empty(0),
                 b_value=np.empty(0)):
        super().__init__()

        for i in wb_shape:
            if not i:
                raise ShapeError('wb shape can not be undefined nor zero')

        self.wb_shape = wb_shape
        self.b_shape = (wb_shape[1], )
        self.trainable = trainable
        self.activation = activation

        self.w_value = w_value
        self.b_value = b_value

    @property
    def wb_value(self):
        """ tuple(numpy_array): value of weights and biases
        """
        return self.w_value, self.b_value

    @property
    def w_value(self):
        """ numpy_array: value of weights

        The setter also check for shape consistency with what
        stored in wb_shape that is the leading quantity

        Raises
        ------
        ShapeError:
            if the shape is not consistent with what is stored in Wb_shape
        """
        return self._w_value

    @w_value.setter
    def w_value(self, value):
        if value.size == 0:
            self._w_value = value
            return None

        if value.shape != self.wb_shape:
            raise ShapeError('request to setup a leayer of shape '
                             '{} with parameters of shape {}'.format(
                                 self.wb_shape, value.shape))
        self._w_value = value

    @property
    def b_value(self):
        """ numpy_array: value of biases

        The setter also check for shape consistency with what
        stored in b_shape that is the leading quantity

        Raises
        ------
        ShapeError:
            if the shape is not consistent with what is stored in b_shape
        """
        return self._b_value

    @b_value.setter
    def b_value(self, value):
        if value.size == 0:
            self._b_value = value
            return None

        if value.shape != self.b_shape:
            raise ShapeError('request to setup a leayer of shape '
                             '{} with parameters of shape {}'.format(
                                 self.b_shape, value.shape))
        self._b_value = value

    def evaluate(self, in_vectors, din_vectors=np.empty(0)):
        """ Evaluate the layer for a given set of inputs

        Note
        ----
        activation can be
          - 0:: linear
          - 1:: gaussian
          - 2:: rbf o_i = \exp(-\sum_{j=0}^{in_size}(in_j - w_i)^2)
          - 3:: relu

        Parameters
        ----------
        in_vectors: numpy_array (?, wb_shape[0])
            First dimension is the number of vector that must be evaluate by
            the layer
        din_vectors: numpy_array(?, wb_shape[0]), optional
            Derivatives of the layer to be computed, used in to evaluate forces

            First dimension is the number of vector that must be evaluate by
            the layer

        Returns
        -------
        numpy_array(?, wb_shape[1])
            Result of the activation

        numpy_array(?, wb_shape[1])
            Derivative of the activation.
            this is returned ONLY if din_vectors.size > 0

        Raises
        ------
        ValueError:
            if weights or biases are not available and the computation
            can not be performed

        ValueError:
            For some non yet implemented derivatives
        """
        if (self._w_value.size == 0
                or self._b_value.size == 0) and self.activation != 0:
            raise ValueError('weights and bias not availables')
        elif self._w_value.size == 0 and self.activation == 0:
            raise ValueError('weights not availables')

        if self.activation == 0:
            tmp = np.matmul(in_vectors, self._w_value)
            act = tmp + self._b_value
            if din_vectors.size > 0:
                d_act = np.matmul(din_vectors, self._w_value)
        elif self.activation == 1:
            tmp = np.matmul(in_vectors, self._w_value) + self._b_value
            act = np.exp(np.negative(np.square(tmp)))
            if din_vectors.size > 0:
                tmp2 = np.matmul(din_vectors, self._w_value)
                # vulgar shaping:
                # in_vectors.shape[0]: number of atoms of this species
                # wb_shape[1]: out size
                # din_vectors.shape[-1]: N_atoms * 3
                d_act = -2.0 * np.reshape(
                    tmp * act,
                    (in_vectors.shape[0], 1, self.wb_shape[1])) * tmp2

        elif self.activation == 2:
            tmp = np.tile(np.expand_dims(in_vectors, -1), self.wb_shape[1])
            tmp = tmp - self._w_value
            act = np.exp(-np.sum(tmp**2, 1))
            if din_vectors.size > 0:
                d_act = - 2.0 * np.reshape(act,
                    (in_vectors.shape[0], 1, self.wb_shape[1])) * \
                    np.matmul(din_vectors, tmp)
        elif self.activation == 3:
            tmp = np.matmul(in_vectors, self._w_value) + self._b_value
            act = np.maximum(0, tmp)
            if din_vectors.size > 0:
                tmp2 = np.matmul(din_vectors, self._w_value)
                d_act = np.reshape(
                    np.where(tmp > 0.0, 1.0, 0.0),
                    (in_vectors.shape[0], 1, self.wb_shape[1])) * tmp2
        else:
            raise ValueError('activation function not implemented')
        if din_vectors.size > 0:
            return act, d_act
        return act

    def tf_evaluate(self, in_tensor):
        """Define an all to all connected layer with species division.

        Parameters
        ----------
        in_tensor: input to be computed, (n_atoms of this species x in_size)

        Returns
        -------
        Output of the layer
        """
        dist_parameter = 1.0 / self.b_shape[0]

        if self.w_value.size > 0:
            weights = _variable_on_cpu('weights',
                                       self.wb_shape,
                                       tf.constant_initializer(self.w_value),
                                       trainable=self.trainable)
        else:
            weights = _variable_random_uniform('weights',
                                               self.wb_shape,
                                               dist_parameter,
                                               trainable=self.trainable)

        if self.b_value.size > 0:
            biases_init = tf.constant_initializer(self.b_value)
        else:
            biases_init = tf.constant_initializer(0.0)

        biases = _variable_on_cpu('biases',
                                  self.b_shape,
                                  biases_init,
                                  trainable=self.trainable)

        tf.add_to_collection(ops.GraphKeys.WEIGHTS, weights)
        tf.add_to_collection(ops.GraphKeys.BIASES, biases)

        if self.activation == 1:  # gaussian
            w_contrib = tf.matmul(in_tensor, weights,
                                  name='w_contrib') + biases
            exp_arg = tf.square(w_contrib, name='exp_arg')
            output = tf.exp(tf.negative(exp_arg), name='activation')
        elif self.activation == 2:  # rbf
            reshaped_in = tf.tile(tf.expand_dims(in_tensor, -1),
                                  [1, 1, self.b_shape[0]])
            w_contrib = tf.subtract(reshaped_in, weights, name='w_contrib')
            exp_arg = tf.reduce_sum(tf.square(w_contrib, name='exp_arg'), 1)
            output = tf.exp(tf.negative(exp_arg), name='activation')
        elif self.activation == 3:  # relu
            w_contrib = tf.matmul(in_tensor, weights,
                                  name='w_contrib') + biases
            output = tf.nn.relu(w_contrib, name='my_relu')
        elif self.activation == 4:  # tanh
            w_contrib = tf.matmul(in_tensor, weights,
                                  name='w_contrib') + biases
            output = tf.nn.tanh(w_contrib, name='activation')
        elif self.activation == 0:  # linear
            w_contrib = tf.matmul(in_tensor, weights, name='w_contrib')
            output = tf.add(w_contrib, biases, name='activation')
        else:
            raise ValueError('activation not implemented', self.activation)

        return output

    @property
    def tf_ckpt_elements(self):
        names = ['weights', 'biases']
        objects = None
        return names, objects

    @tf_ckpt_elements.setter
    def tf_ckpt_elements(self, value):
        self.w_value, self.b_value = value

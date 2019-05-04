import unittest
import numpy as np

if __name__ == '__main__':
    from example import Example
    from layers import ShapeError
    from a2affnetwork import A2affNetwork
    from systemscaffold import SystemScaffold
    from systemscaffold import NetworkNotAvailableError
else:
    from .example import Example
    from .layers import ShapeError
    from .a2affnetwork import A2affNetwork
    from .systemscaffold import SystemScaffold
    from .systemscaffold import NetworkNotAvailableError


class Test_SystemScaffold(unittest.TestCase):
    def setUp(self):
        feature_size = 16
        layer_sizes = [8, 4, 2, 1]
        name = 'default'
        self.layer_sizes = layer_sizes
        self.default_network = A2affNetwork(feature_size, layer_sizes, name)

    def test_get_network(self):
        system_scaffold = SystemScaffold(self.default_network)
        c_network = system_scaffold['C']
        self.assertEqual(self.default_network.layers_shaping,
                         c_network.layers_shaping)
        self.assertEqual(self.default_network.layers_trainable,
                         c_network.layers_trainable)
        self.assertEqual(self.default_network.layers_activation,
                         c_network.layers_activation)
        for ref, test in zip(self.default_network.network_wb,
                             c_network.network_wb):
            np.testing.assert_array_equal(ref, test)
        self.assertEqual(self.default_network.offset, c_network.offset)

        system_scaffold = SystemScaffold()
        with self.assertRaises(NetworkNotAvailableError):
            c_network = system_scaffold['C']

    def test_creation_atomic_sequence(self):
        system_scaffold = SystemScaffold(self.default_network)
        sequence = ('C', 'H', 'N')
        [system_scaffold[x] for x in sequence]
        self.assertEqual(system_scaffold.atomic_sequence, sequence)
        self.assertEqual(system_scaffold.n_species, len(sequence))
        with self.assertRaises(ValueError):
            system_scaffold[sequence[0]] = A2affNetwork(24, [1, 2, 4], 'C')
        with self.assertRaises(ValueError):
            system_scaffold['K'] = A2affNetwork(24, [1, 2, 4], 'C')

        system_scaffold['K'] = A2affNetwork(24, [1, 2, 4], 'K')
        new_sequence = tuple(list(sequence) + ['K'])
        self.assertEqual(system_scaffold.atomic_sequence, new_sequence)

    def test_creation_given_atomic_sequence(self):
        sequence = ('C', 'H', 'N')
        zeros = (1.0, 2.0, 3.0)

        system_scaffold = SystemScaffold(self.default_network, sequence, zeros)
        for species, zero in zip(sequence, zeros):
            self.assertEqual(system_scaffold[species].offset, zero)

        system_scaffold = SystemScaffold(self.default_network, sequence)
        for species, zero in zip(sequence, zeros):
            self.assertEqual(system_scaffold[species].offset, 0.0)

        with self.assertRaises(ValueError):
            system_scaffold = SystemScaffold(self.default_network, zeros=zeros)

        with self.assertRaises(ValueError):
            system_scaffold = SystemScaffold(self.default_network, sequence,
                                             zeros[:2])

    def test_evaluation_of_a_system(self):
        # taking the default network
        network = self.default_network

        # give value to wb
        wb = [(np.arange(x * y).reshape(x, y), np.arange(y).reshape(y))
              for x, y in network.layers_shaping]
        behaviors = ['load' for x in self.layer_sizes]
        network.customize_network(behaviors=behaviors, override_wb=wb)

        # extract the scaffold
        sequence = ('C', 'H', 'N')
        zeros = (1.0, 2.0, 3.0)
        system_scaffold = SystemScaffold(self.default_network, sequence, zeros)

        # evaluate
        example = Example(
            np.arange(48).reshape(3, 16), np.arange(3), 0,
            np.arange(432).reshape(3, 16, 3 * 3),
            np.arange(9).reshape(3, 3))

        energy, forces = system_scaffold.evaluate(example, True)

        self.assertEqual(energy.shape, ())
        self.assertEqual(forces.shape, (9, ))

    def test_failed_evaluation_of_a_system(self):
        """
        if not weight and biases error must be raised
        """
        # taking the default network
        network = self.default_network

        # extract the scaffold
        sequence = ('C', 'H', 'N')
        zeros = (1.0, 2.0, 3.0)
        system_scaffold = SystemScaffold(self.default_network, sequence, zeros)

        # evaluate
        example = Example(
            np.arange(48).reshape(3, 16), np.arange(3), 0,
            np.arange(432).reshape(3, 16, 3 * 3),
            np.arange(9).reshape(3, 3))

        with self.assertRaises(ValueError):
            energy, forces = system_scaffold.evaluate(example, True)


if __name__ == '__main__':
    unittest.main()

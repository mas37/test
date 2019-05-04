import os
import train
import shutil
import unittest
import numpy as np

from neuralnet.trainparameters import TrainParameters
from neuralnet.trainparameters import IOParameters
from neuralnet.trainparameters import ParallelizationParameters
from neuralnet.trainparameters import ParametersContainer1AM
from neuralnet.systemscaffold import SystemScaffold
from neuralnet.systemscaffold import A2affNetwork


class Test_train(unittest.TestCase):
    def setUp(self):
        class Flags():
            def __init__(self, config, **kvarg):
                self.config = config
                self.list_of_nodes = kvarg.get('list_of_nodes', '')
                self.task_index_variable = kvarg.get('task_index_variable', '')
                self.debug = kvarg.get('debug', None)
                self.debug_parallel = kvarg.get('debug_parallel', None)
                self.debug_parallel_index = kvarg.get('debug_parallel_index',
                                                      None)
                self.communication_port = kvarg.get('communication_port',
                                                    22222)

            def __getattr__(self, attr_name):
                """
                return empty string for non defined attributes
                """
                return ''

        self.flags = Flags('pippo')
        self.data_source_folder = './tests/data/'
        self.parallelization_parameters = ParallelizationParameters()
        self.cwd = os.getcwd()

    def tearDown(self):
        train.tf.reset_default_graph()
        os.chdir(self.cwd)
        # comment this line to not delete the outputs!!
        try:
            shutil.rmtree(os.path.join(self.cwd, self.test_data_dir))
        except AttributeError:
            pass

    def test_1(self):
        """ Testing for training features
        """
        test_data_dir = './tests/test_train_1'
        self.test_data_dir = test_data_dir
        if not os.path.isdir(test_data_dir):
            os.makedirs(test_data_dir)
        else:
            shutil.rmtree(os.path.join(self.cwd, self.test_data_dir))
            os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink('../data/train/test_1/train_data', 'train_data')
        os.symlink('../data/train/test_1/saved_networks', 'saved_networks')

        # buiding the input
        train_parameters = TrainParameters(
            batch_size=2,
            learning_rate=1e-3,
            learning_rate_constant=True,
            learning_rate_decay_factor=None,
            learning_rate_decay_step=None,
            beta1=0.9,
            beta2=0.999,
            adam_eps=1e-8,
            max_steps=1000,
            wscale_l1=0.0,
            wscale_l2=0.0,
            bscale_l1=0.0,
            bscale_l2=0.0,
            forces_cost=0.0,
            clip_value=0.0,
            loss_func='quad',
            floss_func='quad')

        io_parameters = IOParameters(
            train_dir='./train_output',
            data_dir='./train_data',
            input_format='TFR',
            save_checkpoint_steps=500,
            max_ckpt_to_keep=100000,
            log_frequency=1,
            tensorboard_log_frequency=1,
            avg_speed_steps=1,
            log_device_placement=False)

        parameters_container = ParametersContainer1AM(en_rescale=1.0)

        system_scaffold = SystemScaffold.load_PANNA_checkpoint_folder(
            './saved_networks')
        system_scaffold['H'].customize_network(trainables=[1, 1, 0, 0])

        parameters = (io_parameters, self.parallelization_parameters,
                      train_parameters, parameters_container, system_scaffold)

        # Run the training
        train.train(self.flags, parameters)

        # Read the output
        outE = []
        with open("./delta_e.dat") as f:
            for Es in f:
                tmp = [float(x) for x in Es.split()]
                tmp.sort()
                outE.append(np.asarray(tmp))
        # Checking against reference
        # values and threshold hardcoded for now...
        refE_0 = np.asarray([-0.00003815, 0.00082397])
        np.testing.assert_allclose(outE[0], refE_0)
        np.testing.assert_allclose(outE[-1], np.zeros(2), rtol=1e-5)

    def test_2(self):
        """ Testing for training features
        """
        test_data_dir = './tests/test_train_2'
        self.test_data_dir = test_data_dir
        if not os.path.isdir(test_data_dir):
            os.makedirs(test_data_dir)
        else:
            shutil.rmtree(os.path.join(self.cwd, self.test_data_dir))
            os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink('../data/train/test_2/train_data', 'train_data')
        os.symlink('../data/train/test_2/saved_networks', 'saved_networks')

        # hack to allow multiple test during the same unit test run
        self.flags.communication_port = 2223
        # buiding the input
        train_parameters = TrainParameters(
            batch_size=3,
            learning_rate=1e-3,
            learning_rate_constant=True,
            learning_rate_decay_factor=None,
            learning_rate_decay_step=None,
            beta1=0.9,
            beta2=0.999,
            adam_eps=1e-8,
            max_steps=1000,
            wscale_l1=0.0,
            wscale_l2=0.0,
            bscale_l1=0.0,
            bscale_l2=0.0,
            forces_cost=0.0,
            clip_value=0.0,
            loss_func='quad',
            floss_func='quad')

        io_parameters = IOParameters(
            train_dir='./train_output',
            data_dir='./train_data',
            input_format='TFR',
            save_checkpoint_steps=500,
            max_ckpt_to_keep=100000,
            log_frequency=1,
            tensorboard_log_frequency=1,
            avg_speed_steps=1,
            log_device_placement=False)

        parameters_container = ParametersContainer1AM(en_rescale=1.0)

        system_scaffold = SystemScaffold.load_PANNA_checkpoint_folder(
            './saved_networks')

        system_scaffold['H'].customize_network(
            behaviors=['keep', 'keep', 'new', 'new'])
        system_scaffold['C'].customize_network(
            behaviors=['keep', 'keep', 'new', 'new'])
        system_scaffold['N'].customize_network(
            behaviors=['keep', 'keep', 'new', 'new'])
        system_scaffold['O'].customize_network(
            behaviors=['keep', 'keep', 'new', 'new'])
        parameters = (io_parameters, self.parallelization_parameters,
                      train_parameters, parameters_container, system_scaffold)

        # Run the training
        train.train(self.flags, parameters)

        # Read the output
        outE = []
        with open("./delta_e.dat") as f:
            for Es in f:
                outE.append(np.asarray(Es.split(), dtype=np.float))
        #starting error
        std_1 = np.std(outE[0])

        #error after a 1000 epoch
        std_2 = np.std(outE[-1])

        self.assertEqual(std_1 > std_2 * 10, True)


if __name__ == '__main__':
    unittest.main()

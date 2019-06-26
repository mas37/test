###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import os
import shutil
import unittest
import tfr_packer
import numpy as np
import gvect_calculator
from gvector import GvectmBP


class Test_Tft_Packer(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(os.path.join(self.cwd, self.test_data_dir),ignore_errors=True)

    def test_tfr_packer(self):
        '''
          test: creating binary
        '''
        # TESTING FOR TFRS PREPARATION
        test_data_dir = './tests/test_tfr_packer'
        self.test_data_dir = test_data_dir
        if not os.path.isdir(test_data_dir):
            os.makedirs(test_data_dir)
        else:
            shutil.rmtree(os.path.join(self.cwd, self.test_data_dir))
            os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink('../data/gvector_calculator/examples', 'examples')
        os.symlink('../data/gvector_calculator/bin_references',
                   'bin_references')
        gvect_parameters = GvectmBP(
            compute_dGvect=True,
            species='H, C, N, O',
            param_unit='angstrom',
            pbc_directions=None,
            Rc_rad=4.6,
            Rs0_rad=0.5,
            RsN_rad=16,
            eta_rad=16,
            Rc_ang=3.1,
            ThetasN=8,
            Rs0_ang=0.5,
            RsN_ang=4,
            eta_ang=16,
            zeta=50)

        folder_parameters = type(
            'FolderParameters', (object, ),
            dict(
                input_json_dir='./examples',
                binary_out_dir=os.path.join(test_data_dir, 'bin'),
                log_dir=test_data_dir,
            ))()

        gvect_calculator.main(gvect_parameters, folder_parameters, 1)

        # compacting the data
        tfrs_parameters = type(
            'Tfrs', (object, ),
            dict(
                in_path=os.path.join(test_data_dir, 'bin'),
                out_path=os.path.join(test_data_dir, 'tfrs_w_d'),
                elements_per_file=2,
                prefix='',
                num_species=4,
                derivatives=True))()

        # packing with derivatives
        tfr_packer.main(tfrs_parameters)

        tfrs_parameters.out_path = os.path.join(test_data_dir, 'tfrs_wo_d')
        tfrs_parameters.derivatives = False
        # packing without derivatives
        tfr_packer.main(tfrs_parameters)

        # TODO compute difference
        # is this necessray? maybe
        # cleanup the log files if the test went fine


if __name__ == '__main__':
    unittest.main()

###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import configparser

from tensorflow.python.platform import tf_logging as logger

from lib.parser_callable import converters
from .scaffold_selector import scaffold_selector


class TrainParameters():
    def __init__(self, batch_size, learning_rate, learning_rate_constant,
                 learning_rate_decay_factor, learning_rate_decay_step, beta1,
                 beta2, adam_eps, max_steps, wscale_l1, wscale_l2, bscale_l1,
                 bscale_l2, clip_value):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_constant = learning_rate_constant
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.learning_rate_decay_step = learning_rate_decay_step
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
        self.max_steps = max_steps
        self.wscale_l1 = wscale_l1
        self.wscale_l2 = wscale_l2
        self.bscale_l1 = bscale_l1
        self.bscale_l2 = bscale_l2
        self.clip_value = clip_value


class IOParameters():
    def __init__(self, train_dir, data_dir, input_format,
                 save_checkpoint_steps, max_ckpt_to_keep, log_frequency,
                 tensorboard_log_frequency, avg_speed_steps,
                 log_device_placement):
        super().__init__()
        self.train_dir = train_dir
        self.data_dir = data_dir
        self.input_format = input_format
        self.save_checkpoint_steps = save_checkpoint_steps
        self.max_ckpt_to_keep = max_ckpt_to_keep
        self.log_frequency = log_frequency
        self.tensorboard_log_frequency = tensorboard_log_frequency
        self.avg_speed_steps = avg_speed_steps
        self.log_device_placement = log_device_placement


class ParallelizationParameters():
    def __init__(self,
                 num_parallel_readers=1,
                 num_parallel_calls=1,
                 shuffle_buffer_size_multiplier=10,
                 prefetch_buffer_size_multiplier=10,
                 inter_op_parallelism_threads=0,
                 intra_op_parallelism_threads=0,
                 dataset_cache=False):
        super().__init__()
        self.num_parallel_readers = num_parallel_readers
        self.num_parallel_calls = num_parallel_calls
        self.shuffle_buffer_size_multiplier = shuffle_buffer_size_multiplier
        self.prefetch_buffer_size_multiplier = prefetch_buffer_size_multiplier
        self.inter_op_parallelism_threads = inter_op_parallelism_threads
        self.intra_op_parallelism_threads = intra_op_parallelism_threads
        self.dataset_cache = dataset_cache


def parameter_file_parser(filename):
    config = configparser.ConfigParser(converters=converters)
    logger.info('reading {}'.format(filename))
    config.read(filename)

    #=== IO part ===
    io_info = config['IO_INFORMATION']
    train_dir = io_info.get('train_dir')
    data_dir = io_info.get('data_dir')
    input_format = io_info.get('input_format', 'TFR')
    save_checkpoint_steps = io_info.getint('save_checkpoint_steps', 1)
    max_ckpt_to_keep = io_info.getint('max_ckpt_to_keep', 100000)
    log_frequency = io_info.getint('log_frequency', 1)
    tensorboard_log_frequency = io_info.getint('tensorboard_log_frequency',
                                               log_frequency)
    avg_speed_steps = io_info.getint('avg_speed_steps', log_frequency)
    log_device_placement = io_info.getboolean('log_device_placement', False)

    io_parameters = IOParameters(train_dir, data_dir, input_format,
                                 save_checkpoint_steps, max_ckpt_to_keep,
                                 log_frequency, tensorboard_log_frequency,
                                 avg_speed_steps, log_device_placement)

    #=== Parallel part ===
    num_parallel_readers = 1
    num_parallel_calls = 1
    shuffle_buffer_size_multiplier = 10
    prefetch_buffer_size_multiplier = 10
    inter_op_parallelism_threads = 0
    intra_op_parallelism_threads = 0
    inter_node_parallelism = False
    communication_port = 22222
    dataset_cache = False

    if config.has_section('PARALLELIZATION'):
        parallel = config['PARALLELIZATION']
        num_parallel_readers = parallel.getint('num_parallel_readers',
                                               num_parallel_readers)
        num_parallel_calls = parallel.getint('num_parallel_calls',
                                             num_parallel_calls)
        shuffle_buffer_size_multiplier = parallel.\
            getint('shuffle_buffer_size_multiplier',
                   shuffle_buffer_size_multiplier)
        prefetch_buffer_size_multiplier = parallel.\
            getint('prefetch_buffer_size_multiplier',
                   prefetch_buffer_size_multiplier)
        inter_op_parallelism_threads = parallel.\
            getint('inter_op_parallelism_threads',
                   inter_op_parallelism_threads)
        intra_op_parallelism_threads = parallel.\
            getint('intra_op_parallelism_threads',
                   intra_op_parallelism_threads)
        dataset_cache = parallel.getboolean('dataset_cache', dataset_cache)

        # log all the parameters
        logger.info('num_parallel_readers: {}'.format(num_parallel_readers))
        logger.info('num_parallel_calls: {}'.format(num_parallel_calls))
        logger.info('shuffle_buffer_size_multiplier: {}'.format(
            shuffle_buffer_size_multiplier))
        logger.info('prefetch_buffer_size_multiplier: {}'.format(
            prefetch_buffer_size_multiplier))
        logger.info('inter_op_parallelism_threads: {}'.format(
            inter_op_parallelism_threads))
        logger.info('intra_op_parallelism_threads: {}'.format(
            intra_op_parallelism_threads))
        logger.info('dataset_cache: {}'.format(dataset_cache))
    else:
        logger.info('Parallelization is not tuned'
                    ' - using defaults may result in low efficiency')

    parallelization_parameters = ParallelizationParameters(
        num_parallel_readers, num_parallel_calls,
        shuffle_buffer_size_multiplier, prefetch_buffer_size_multiplier,
        inter_op_parallelism_threads, intra_op_parallelism_threads,
        dataset_cache)

    #=== Training option part ===
    lr_parms = config['TRAINING_PARAMETERS']
    batch_size = lr_parms.getint('batch_size')
    learning_rate = lr_parms.getfloat('learning_rate')
    learning_rate_constant = lr_parms.getboolean('learning_rate_constant',
                                                 True)
    if not learning_rate_constant:
        learning_rate_decay_factor = lr_parms.getfloat(
            'learning_rate_decay_factor')
        learning_rate_decay_step = lr_parms.getfloat(
            'learning_rate_decay_step')
    else:
        learning_rate_decay_factor = None
        learning_rate_decay_step = None
    beta1 = lr_parms.getfloat('beta1', 0.9)
    beta2 = lr_parms.getfloat('beta2', 0.999)
    adam_eps = lr_parms.getfloat('adam_eps', 1e-8)
    max_steps = lr_parms.getint('max_steps')
    wscale_l1 = lr_parms.getfloat('wscale_l1', 0.0)
    wscale_l2 = lr_parms.getfloat('wscale_l2', 0.0)
    bscale_l1 = lr_parms.getfloat('bscale_l1', 0.0)
    bscale_l2 = lr_parms.getfloat('bscale_l2', 0.0)
    clip_value = lr_parms.getfloat('clip_value', 0.0)
    # possible options are quad, exp_quad, quad_atom, exp_quad_atom

    train_parameters = TrainParameters(
        batch_size, learning_rate, learning_rate_constant,
        learning_rate_decay_factor, learning_rate_decay_step, beta1, beta2,
        adam_eps, max_steps, wscale_l1, wscale_l2, bscale_l1, bscale_l2,
        clip_value)

    # === scaffold part ===
    if 'DATA_INFORMATION' in config:
        scaffold_type = config['DATA_INFORMATION'].get(
            'scaffold_type', 'PANNA')
    else:
        scaffold_type = 'PANNA'

    logger.info('scaffold type: %s', scaffold_type)
    Scaffold = scaffold_selector(scaffold_type)  # pylint: disable=invalid-name
    scaffold = Scaffold(config)

    return io_parameters, parallelization_parameters,\
        train_parameters, scaffold

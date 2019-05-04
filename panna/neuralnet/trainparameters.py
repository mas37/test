import numpy as np
import configparser

if __name__ == 'trainparameters':
    import parser_callable
    from a2affnetwork import A2affNetwork
    from systemscaffold import SystemScaffold
    from systemscaffold import NetworkNotAvailableError
else:
    from . import parser_callable
    from .a2affnetwork import A2affNetwork
    from .systemscaffold import SystemScaffold
    from .systemscaffold import NetworkNotAvailableError

import logging
# logger
logger = logging.getLogger(__name__)

#formatter = logging.formatter('%(asctime)s - %(name)s - '
#                              '%(levelname)s - %(message)s')
#formatter = logging.Formatter('%(levelname)s - %(message)s')
# console handler
#ch = logging.StreamHandler()
#ch.setFormatter(formatter)
#logger.addHandler(ch)
#logger.setLevel(logging.INFO)


class TrainParameters():
    def __init__(self, batch_size, learning_rate, learning_rate_constant,
                 learning_rate_decay_factor, learning_rate_decay_step, beta1,
                 beta2, adam_eps, max_steps, wscale_l1, wscale_l2, bscale_l1,
                 bscale_l2, forces_cost, clip_value, loss_func, floss_func):
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
        self.forces_cost = forces_cost
        self.clip_value = clip_value
        self.loss_func = loss_func
        self.floss_func = floss_func

    @property
    def train_on_forces(self):
        if self.forces_cost > 0.0:
            return True
        else:
            return False


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


class ParametersContainer1AM():
    def __init__(self, en_rescale):
        self.en_rescale = en_rescale


def parameter_file_parser(filename):
    config = configparser.ConfigParser(
        converters={
            '_comma_list': parser_callable.get_list_from_comma_sep_strings,
            '_comma_list_floats': parser_callable.\
                get_list_floats_from_comma_sep_strings,
            '_network_architecture': parser_callable.get_network_architecture,
            '_network_trainable': parser_callable.get_network_trainable,
            '_network_act': parser_callable.get_network_act,
            '_network_behavior': parser_callable.get_network_behavior
        }
    )
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
    forces_cost = lr_parms.getfloat('forces_cost', 0.0)
    clip_value = lr_parms.getfloat('clip_value', 0.0)
    loss_func = lr_parms.get('loss_function', 'quad')
    floss_func = lr_parms.get('floss_function', 'quad')
    # possible options are quad, exp_quad, quad_atom, exp_quad_atom

    train_parameters = TrainParameters(
        batch_size, learning_rate, learning_rate_constant,
        learning_rate_decay_factor, learning_rate_decay_step, beta1, beta2,
        adam_eps, max_steps, wscale_l1, wscale_l2, bscale_l1, bscale_l2,
        forces_cost, clip_value, loss_func, floss_func)

    #=== Data option part ===
    data_params = config['DATA_INFORMATION']
    atomic_sequence = data_params.get_comma_list('atomic_sequence', [])
    en_rescale = data_params.getfloat('energy_rescale', 1.0)
    zeros = data_params.get_comma_list_floats('output_offset', [])

    # search for a default network stored in file
    default_networks_metadata_folder = data_params.get('networks_metadata',
                                                       None)
    if default_networks_metadata_folder:
        logger.info('Networks in {} will be loaded'.format(
            default_networks_metadata_folder))

    parameters_container = ParametersContainer1AM(en_rescale)
    # === NETWORK option part ===
    # default network to be loaded
    if 'DEFAULT_NETWORK' in config:
        default_net_params = config['DEFAULT_NETWORK']
        # set network type
        nn_type = default_net_params.get('nn_type', 'a2aff')
        if nn_type in ['a2aff', 'A2AFF', 'ff', 'a2a', 'FF', 'A2A']:
            nn_type = 'a2ff'
        else:
            raise ValueError(
                '{} != a2aff : '
                'Only all-to-all feed forward networks supported'.format(
                    nn_type))

        # set architecture
        default_g_size = default_net_params.getint('g_size')
        default_layer_sizes = default_net_params.get_network_architecture(
            'architecture')
        if default_layer_sizes:
            logger.info(
                'Found a default network: {}'.format([default_g_size] +
                                                     default_layer_sizes))
            logger.info('This network size will be used as '
                        'default for all species unless specified otherwise')

        # set trainability
        default_layer_trainable = default_net_params.get_network_trainable(
            'trainable', None)

        if default_layer_trainable:
            logger.info('Found a default trainability: {}'.format(
                default_layer_trainable))
        else:
            logging.warining('Set to default trainability: all trainable')

        # set activations
        default_layer_act = default_net_params.get_network_act(
            'activations', None)
        if default_layer_act:
            logger.waring('Found a default activation list: '
                          '{}'.format(default_layer_act))
        else:
            # default is gaussian:gaussian: ... : linear
            logger.info('Set to default activation: '
                        'gaussians + last linear')

        default_net = A2affNetwork(default_g_size, default_layer_sizes,
                                   'default', None, default_layer_trainable,
                                   default_layer_act)

    else:
        default_net = None
        logger.info('No default network is found, '
                    'all species must be fully specified.')

    if default_networks_metadata_folder:
        system_scaffold = SystemScaffold.load_PANNA_checkpoint_folder(
            default_networks_metadata_folder, default_net)
        # update atomic sequence
        old_atomic_sequence = atomic_sequence
        atomic_sequence = system_scaffold.atomic_sequence
        for i, j in zip(atomic_sequence, old_atomic_sequence):
            if i != j:
                logger.info('new atomic sequence {}'.format(atomic_sequence))
                break
    else:
        system_scaffold = SystemScaffold(default_net, atomic_sequence, zeros)

    # parsing single atomic species
    # if a default is specified every species behave as the default
    # if no default is specified than every species must be fully described
    for species in atomic_sequence:
        if species in config:
            logger.info('=={}== Found network specifications'.format(species))
            species_config = config[species]

            # architecture
            g_size = species_config.getint('g_size')
            layers_size = species_config.get_network_architecture(
                'architecture', None)
            # trainable flag
            layers_trainable = species_config.get_network_trainable(
                'trainable', None)
            # layers behavior flag:
            layers_behavior = species_config.get_network_behavior(
                'behavior', None)
            # activation
            layers_act = species_config.get_network_act('activations', None)
            # output_offset
            output_offset = species_config.getfloat('output_offset', None)

            # network_wb
            override_wb = {}
            if layers_behavior:
                for layer_number, behavior in enumerate(layers_behavior):
                    if behavior == 'load':
                        w_file_name = species_config.get(
                            'layer{}_w_file'.format(layer_number))
                        b_file_name = species_config.get(
                            'layer{}_b_file'.format(layer_number))
                        if not (w_file_name and b_file_name):
                            raise ValueError('not passed file names')
                        w_values = np.load(w_file_name)
                        b_values = np.load(b_file_name)
                        override_wb[layer_number] = (w_values, b_values)

            def _local_log_helper(network):
                spe = '=={}== '.format(species)
                if layers_size:
                    logger.info(spe + 'New network architecture: {}'.format(
                        network.layers_size))
                else:
                    logger.warning(spe + 'Architecture flag was not specified '
                                   '- set to {}'.format([network.feature_size]
                                                        + network.layers_size))

                if layers_trainable:
                    logger.info(spe + 'New trainable flags: {}'.format(
                        network.layers_trainable))
                else:
                    logger.warning(
                        spe + 'Trainable flag was not specified '
                        '- set to {}'.format(network.layers_trainable))

                if layers_behavior:
                    logger.info(spe + 'New layer behavior flags: {}'.format(
                        layers_behavior))

                if layers_act:
                    logger.info(spe + 'New activations : {}'.format(
                        network.layers_activation))
                else:
                    logger.warning(
                        spe + 'Activations flag is not specified '
                        '- set to {}'.format(network.layers_activation))
                if override_wb:
                    for k in override_wb:
                        logger.warning(spe + 'override layer {}'.format(k))
                else:
                    logger.info(spe + 'No override asked')

                if output_offset:
                    logger.info(spe +
                                'New output offset: {}'.format(network.offset))
                else:
                    logger.warning(spe +
                                   'Output offset flag was not specified '
                                   '- set to {}'.format(network.offset))

            # real setting up:
            try:
                species_network = system_scaffold[species]
                species_network.customize_network(
                    g_size, layers_size, layers_trainable, layers_behavior,
                    layers_act, output_offset, override_wb)
                _local_log_helper(species_network)
            except NetworkNotAvailableError:
                if not layers_size:
                    raise ValueError('not available network structure')
                # triggered only if a default fall back is not present
                network_wb = [(np.empty(0), np.empty(0))
                              for i in range(len(layers_size))]
                for k, v in override_wb.items():
                    network_wb[k] = v
                species_network = A2affNetwork(g_size, layers_size, species,
                                               network_wb, layers_trainable,
                                               layers_act, output_offset)

                system_scaffold[species] = species_network
                _local_log_helper(species_network)
    return io_parameters, parallelization_parameters, train_parameters, \
            parameters_container, system_scaffold

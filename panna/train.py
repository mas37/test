from __future__ import absolute_import, division, print_function

import argparse
import configparser
import logging as or_logging
import os
import time
from functools import partial
from itertools import zip_longest

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import control_flow_ops, resources, variables
from tensorflow.python.platform import tf_logging as logging

import neuralnet as net
from neuralnet.constants import A2BOHR
from neuralnet import parameter_file_parser

# Small snipped to mange the log
tf_logger = logging._get_logger()
formatter = or_logging.Formatter('%(levelname)s - %(message)s')
tf_logger.handlers = []


def _splash_screen(emitter):
    # Splashscreen
    record = or_logging.makeLogRecord({
        'level':
        or_logging.INFO,
        'levelname':
        'INFO',
        'msg':
        '\n'
        '    ____   _    _   _ _   _    _     \n'
        '   |  _ \ / \  | \ | | \ | |  / \    \n'
        '   | |_) / _ \ |  \| |  \| | / _ \   \n'
        '   |  __/ ___ \| |\  | |\  |/ ___ \  \n'
        '   |_| /_/   \_\_| \_|_| \_/_/   \_\ \n'
        '                                     \n'
        ' Properties from Artificial Neural Network Architectures'
    })
    emitter(record)


def train(flags, parameters, *var, **kvarg):

    # extrapolate info for parallel running
    if flags.list_of_nodes != '':
        list_of_nodes = flags.list_of_nodes.split(',')
    else:
        list_of_nodes = ['localhost']

    if flags.task_index_variable != '':
        task_index = int(os.environ[flags.task_index_variable])
    else:
        task_index = 0

    if len(list_of_nodes) > 1 and flags.task_index_variable != '':
        # parallel run
        parallel_environment = True
    else:
        if not flags.debug_parallel:
            #serial run
            parallel_environment = False
        else:
            # fake parallel for debug on a single machine
            parallel_environment = True
            list_of_nodes = ['one', 'two']
            task_index = flags.debug_parallel_index

    # set up file loggers
    fsh = or_logging.FileHandler('{}_{}.log'.format(list_of_nodes[task_index],
                                                    time.strftime('%H_%M_%S')))
    fsh.setFormatter(formatter)
    tf_logger.addHandler(fsh)

    # do the splash screen in every file logger
    _splash_screen(fsh.emit)

    # creation of the cluster
    if not flags.debug_parallel:
        list_of_worker_nodes = [
            '{}:{}'.format(x, flags.communication_port) for x in list_of_nodes
        ]
    else:
        list_of_worker_nodes = [
            'localhost:22222',
            'localhost:22223',
        ]

    cluster = tf.train.ClusterSpec({'worker': list_of_worker_nodes})

    # split parameters in subinstance
    (io_parameters, parallelization_parameters, train_parameters,
     parameters_container, system_scaffold) = parameters

    # extended splash screen
    if parallel_environment:
        logging.info('running in parallel environment')
        logging.info('node name: {}'.format(list_of_nodes[task_index]))
        logging.info('node: {}/{}'.format(task_index + 1, len(list_of_nodes)))
    else:
        logging.info('running in serial environment')

    # ===compatibility layer with old variable naming===
    # TODO: remove this part
    # train generic:
    batch_size = train_parameters.batch_size
    learning_rate = train_parameters.learning_rate
    learning_rate_constant = train_parameters.learning_rate_constant
    learning_rate_decay_factor = train_parameters.learning_rate_decay_factor
    learning_rate_decay_step = train_parameters.learning_rate_decay_step
    beta1 = train_parameters.beta1
    beta2 = train_parameters.beta2
    adam_eps = train_parameters.adam_eps
    max_steps = train_parameters.max_steps
    wscale_l1 = train_parameters.wscale_l1
    wscale_l2 = train_parameters.wscale_l2
    bscale_l1 = train_parameters.bscale_l1
    bscale_l2 = train_parameters.bscale_l2
    forces_cost = train_parameters.forces_cost
    clip_value = train_parameters.clip_value
    loss_func = train_parameters.loss_func
    floss_func = train_parameters.floss_func
    train_on_forces = train_parameters.train_on_forces

    # parallelization:
    num_parallel_readers = parallelization_parameters.num_parallel_readers
    num_parallel_calls = parallelization_parameters.num_parallel_calls
    shuffle_buffer_size_multiplier = parallelization_parameters.shuffle_buffer_size_multiplier
    prefetch_buffer_size_multiplier = parallelization_parameters.prefetch_buffer_size_multiplier
    inter_op_parallelism_threads = parallelization_parameters.inter_op_parallelism_threads
    intra_op_parallelism_threads = parallelization_parameters.intra_op_parallelism_threads
    dataset_cache = parallelization_parameters.dataset_cache

    # without category
    en_rescale = parameters_container.en_rescale

    # relevant network
    layers_sizes = system_scaffold.old_layers_sizes
    layers_trainable = system_scaffold.old_layers_trainable
    layers_act = system_scaffold.old_layers_act
    g_size = system_scaffold.old_gsize
    n_species = system_scaffold.n_species
    atomic_sequence = system_scaffold.atomic_sequence
    networks_wb = system_scaffold.old_networks_wb
    trainability = system_scaffold.old_layers_trainable
    zeros = system_scaffold.old_zeros
    #======================

    if tf.gfile.Exists(io_parameters.train_dir):
        for fname in tf.gfile.ListDirectory(io_parameters.train_dir):
            if (fname == 'checkpoint' or fname == 'graph.pbtxt'
                    or fname.split(".")[0] == 'event'
                    or fname.split(".")[0] == 'model'):
                # here we could just make a new directory
                # inside and continue rather than exit
                logging.info('saved checkpoint present, '
                             'networks will be restarted form here')
                break
    else:
        tf.gfile.MakeDirs(io_parameters.train_dir)

    with tf.Graph().as_default():
        # set local machine as default device for operation
        with tf.device(
                tf.train.replica_device_setter(
                    ps_device='',
                    worker_device='/job:worker/task:{}'.format(task_index),
                    cluster=cluster)):
            # aux computational variables
            global_step = tf.train.get_or_create_global_step()

            if io_parameters.input_format == "TFR":
                _parse_fn = partial(
                    net.parse_fn_v1,
                    g_size=g_size,
                    zeros=zeros,
                    n_species=n_species,
                    forces=train_on_forces,
                    energy_rescale=en_rescale)
            else:
                raise ValueError('input_format not avaiable')

            t_iterator = net.input_iterator(
                io_parameters.data_dir,
                batch_size,
                _parse_fn,
                'train',
                shuffle_buffer_size_multiplier,
                prefetch_buffer_size_multiplier,
                num_parallel_readers,
                num_parallel_calls,
                cache=dataset_cache)

            if train_on_forces:
                t_batch_s, t_batch_g, t_batch_e, t_batch_dg, t_batch_f =\
                    t_iterator.get_next()
            else:
                t_batch_s, t_batch_g, t_batch_e = t_iterator.get_next()

            if io_parameters.input_format == "TFR":
                if train_on_forces:
                    t_energies, t_batch_natoms, t_dEdG = \
                        net.network_A2A(t_batch_s, t_batch_g,
                                        layer_size=layers_sizes,
                                        trainability=layers_trainable,
                                        activations=layers_act,
                                        gvect_size = g_size,
                                        batch_size = batch_size,
                                        Nspecies=n_species,
                                        atomic_label=atomic_sequence,
                                        import_layer=networks_wb,
                                        compute_gradients=True)
                else:
                    t_energies, t_batch_natoms = \
                        net.network_A2A(t_batch_s, t_batch_g,
                                        layer_size=layers_sizes,
                                        trainability=layers_trainable,
                                        activations=layers_act,
                                        gvect_size = g_size,
                                        batch_size = batch_size,
                                        Nspecies=n_species,
                                        atomic_label=atomic_sequence,
                                        import_layer=networks_wb)

            # creation of the lr function
            if learning_rate_constant:
                lr = learning_rate
            else:
                lr = tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    learning_rate_decay_step,
                    learning_rate_decay_factor,
                    staircase=False)

            tf.summary.scalar('learning_rate', lr)

            # creation of the loss quantities
            w_l_norm_sum, b_l_norm_sum = net.l1l2_regularizations(
                wscale_l1, wscale_l2, bscale_l1, bscale_l2)

            logging.info('Loss func is {}'.format(loss_func))
            loss, deltae = net.loss_NN(
                t_energies, t_batch_e, t_batch_natoms, loss_func=loss_func)

            # Loss from forces
            if train_on_forces:
                Floss = net.loss_F(
                    t_batch_s,
                    t_dEdG,
                    t_batch_dg,
                    t_energies,
                    t_batch_f,
                    t_batch_natoms,
                    batch_size=batch_size,
                    Nspecies=n_species,
                    gvect_size=g_size,
                    floss_func=floss_func)

            minimizable_quantity = tf.add_n([loss, w_l_norm_sum, b_l_norm_sum],
                                            name='minimize_me')
            if train_on_forces:
                minimizable_quantity = minimizable_quantity + forces_cost * Floss

            train_op = net.train_NN(
                minimizable_quantity,
                global_step,
                lr,
                beta1,
                beta2,
                adam_eps,
                atomic_sequence=atomic_sequence,
                clip_value=clip_value)

            # add all the loss quantities to the graph
            losses = tf.get_collection('losses')
            for l in losses:
                name = l.op.name.split("/")[-1]
                tf.summary.scalar("1.Losses/" + name, l)

            # config the  Proto
            config_proto = tf.ConfigProto(
                log_device_placement=io_parameters.log_device_placement,
                inter_op_parallelism_threads=inter_op_parallelism_threads,
                intra_op_parallelism_threads=intra_op_parallelism_threads)

            # building of the ckpt
            saver = tf.train.Saver(
                max_to_keep=io_parameters.max_ckpt_to_keep, sharded=True)

            init_op = control_flow_ops.group(
                variables.global_variables_initializer(),
                resources.initialize_resources(resources.shared_resources()),
            )
            # init op are run only by chief
            # local init op are run by every worker
            scaffold = tf.train.Scaffold(
                init_op=init_op,
                saver=saver,
                local_init_op=[t_iterator.initializer])

            class _LoggerHook(tf.train.SessionRunHook):
                """Logs loss and runtime."""

                def __init__(self, starting_step=0, *args, **kwargs):
                    self._starting_step = starting_step

                def begin(self):
                    self._step = self._starting_step
                    self._start_time = time.time()
                    self._writer = open('delta_e.dat', 'w')
                    self._writer2 = open('delta_e_std.dat', 'w')

                def before_run(self, run_context):
                    self._step += 1
                    return tf.train.SessionRunArgs(
                        [deltae, t_batch_natoms, global_step])

                def after_run(self, run_context, run_values):
                    if self._step <= 0:
                        return
                    if self._step % io_parameters.log_frequency == 0:

                        delta_e = run_values.results[0]

                        self._writer.write(
                            ' '.join(['%.8f' % (x) for x in delta_e]) + '\n')
                        self._writer2.write(
                            '%.8f %.8f %.8f' %
                            (np.mean(delta_e), np.mean(np.abs(delta_e)),
                             np.std(delta_e)) + '\n')
                        self._writer.flush()
                        self._writer2.flush()
                        time_str = time.strftime("%m-%d %H:%M:%S",
                                                 time.gmtime())
                        logging.info('{}, global step: {}'.format(
                            time_str, run_values.results[2]))

                    if self._step % io_parameters.avg_speed_steps == 0:
                        current_time = time.time()
                        duration = current_time - self._start_time
                        if duration > 0.0:
                            self._start_time = current_time
                            examples_per_sec = io_parameters.avg_speed_steps * batch_size\
                                / duration
                            sec_per_batch = duration / io_parameters.avg_speed_steps

                            format_str = '{:4.2f} examples/sec, {:4.2f} sec/batch'
                            logging.info(
                                format_str.format(examples_per_sec,
                                                  sec_per_batch))

            server = tf.train.Server(
                cluster, job_name="worker", task_index=task_index)
            hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
            chief_hooks = [tf.train.NanTensorHook(loss), _LoggerHook()]
            with tf.train.MonitoredTrainingSession(
                    master=server.target,
                    is_chief=(task_index == 0),
                    checkpoint_dir=io_parameters.train_dir,
                    scaffold=scaffold,
                    save_summaries_steps=io_parameters.
                    tensorboard_log_frequency,
                    log_step_count_steps=io_parameters.
                    tensorboard_log_frequency * 100,
                    # note: every op trigger the hook!!
                    hooks=hooks,
                    chief_only_hooks=chief_hooks,
                    save_checkpoint_steps=io_parameters.save_checkpoint_steps,
                    config=config_proto) as mon_sess:
                if flags.debug:
                    mon_sess = tf_debug.LocalCLIDebugWrapperSession(mon_sess)
                    logging.debug('---variable of the session---')
                    logging.debug('\n'.join(
                        [' ' + var.name for var in tf.global_variables()]))
                    logging.debug('-----')
                while not mon_sess.should_stop():
                    mon_sess.run(train_op)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default='', help="config file", required=True)
    parser.add_argument(
        '--list_of_nodes',
        type=str,
        default='',
        help='comma-separated list of nodes')
    parser.add_argument(
        "--task_index_variable",
        type=str,
        default='',
        help="task index variable name, depends on workload manager")
    parser.add_argument(
        "--debug",
        action='store_true',
        help='enable debug mode, partial support only')
    parser.add_argument(
        "--debug_parallel",
        action='store_true',
        help='debug parallel flag, use to have 2 server in local config')
    parser.add_argument(
        "--debug_parallel_index", type=int, help='index 0 or 1')
    parser.add_argument(
        "--communication_port",
        type=int,
        default=22222,
        help='communication port for parallel implementation')
    flags, unparsed = parser.parse_known_args()

    if flags.debug:
        tf.logging.set_verbosity(1)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    # set up console logger only for the chief
    if flags.task_index_variable != '':
        task_index = int(os.environ[flags.task_index_variable])
    else:
        task_index = 0

    if task_index == 0:
        # add to master the console handler
        # and do the splash screen
        lsh = or_logging.StreamHandler()
        lsh.setFormatter(formatter)
        tf_logger.addHandler(lsh)
        _splash_screen(lsh.emit)
    parameters = parameter_file_parser(flags.config)
    train(flags, parameters)

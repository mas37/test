###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from __future__ import absolute_import, division, print_function

import argparse
import logging as or_logging
import os
import time
import json

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import control_flow_ops, resources, variables
from tensorflow.python.platform import tf_logging as logging

import neuralnet as net
from neuralnet import parameter_file_parser


def tf_version_tuple(version):
    "version getter fro TF"
    return tuple(map(int, (version.split('.'))))


# Small snipped to manage the log
import absl.logging
or_logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
if tf_version_tuple(tf.__version__) < tf_version_tuple('1.13.0'):
    tf_logger = logging._get_logger()
    tf_logger.info('TF 1.12.X Logger is used')
else:
    tf_logger = tf.get_logger()  #pylint: disable=no-member
    tf_logger.info('TF 1.13.X Logger is used')

# Small snipped to mange the log
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


def _graph_builder(parameters):
    # split parameters in subinstance
    io_parameters, parallelization_parameters, train_parameters,\
        system_scaffold = parameters

    # aux computational variables
    global_step = tf.train.get_or_create_global_step()

    if io_parameters.input_format == "TFR":
        _parse_fn = system_scaffold.tfr_parse_function
    else:
        raise ValueError('input_format not avaiable')

    t_iterator = net.input_iterator(
        io_parameters.data_dir,
        train_parameters.batch_size,
        _parse_fn,
        'train',
        parallelization_parameters.shuffle_buffer_size_multiplier,
        parallelization_parameters.prefetch_buffer_size_multiplier,
        parallelization_parameters.num_parallel_readers,
        parallelization_parameters.num_parallel_calls,
        cache=parallelization_parameters.dataset_cache)

    # this call recover all the batches needed for a step
    batches = t_iterator.get_next()

    # creation of the lr function
    if not train_parameters.learning_rate_constant:
        learning_rate = tf.train.exponential_decay(
            train_parameters.learning_rate,
            global_step,
            train_parameters.learning_rate_decay_step,
            train_parameters.learning_rate_decay_factor,
            staircase=False)
    else:
        learning_rate = train_parameters.learning_rate

    loss, predictions = system_scaffold.tf_network(train_parameters.batch_size,
                                                   batches)

    tf.summary.scalar('learning_rate', learning_rate)

    # creation of the loss quantities
    w_l_norm_sum, b_l_norm_sum = net.l1l2_regularizations(
        train_parameters.wscale_l1, train_parameters.wscale_l2,
        train_parameters.bscale_l1, train_parameters.bscale_l2)

    minimizable_quantity = tf.add_n([loss, w_l_norm_sum, b_l_norm_sum],
                                    name='minimize_me')
    train_op = net.train_neural_network(minimizable_quantity,
                                        global_step,
                                        learning_rate,
                                        train_parameters.beta1,
                                        train_parameters.beta2,
                                        train_parameters.adam_eps,
                                        clip_value=train_parameters.clip_value)

    # add all the loss quantities to the graph
    losses = tf.get_collection('losses')

    for loss in losses:
        name = loss.op.name.split("/")[-1]
        tf.summary.scalar("1.Losses/" + name, loss)

    return train_op, t_iterator, predictions, global_step, loss


def train(flags, parameters, *var, **kvarg):
    """Main train operation
    """

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
        if flags.parameter_servers:
            logging.info('parameter server (ps) is active: '
                         'please optimize the number of ps'
                         'wrt your model, usually 1 should fit')

            if flags.parameter_servers > len(list_of_nodes) - 1:
                logging.info('No nodes available as workers')
                exit()

            list_of_worker_nodes = [
                '{}:{}'.format(x, flags.communication_port)
                for x in list_of_nodes[:-flags.parameter_servers]
            ]
            list_of_parameters_nodes = [
                '{}:{}'.format(x, flags.communication_port)
                for x in list_of_nodes[-flags.parameter_servers:]
            ]

            logging.info('ps/nodes: {}/{}'.format(
                len(list_of_parameters_nodes), len(list_of_nodes)))
            logging.info('workers/nodes: {}/{}'.format(
                len(list_of_worker_nodes), len(list_of_nodes)))
        else:
            list_of_worker_nodes = [
                '{}:{}'.format(x, flags.communication_port)
                for x in list_of_nodes
            ]

            list_of_parameters_nodes = []
    else:
        # fake cluster for debug
        list_of_worker_nodes = ['localhost:22222']
        list_of_parameters_nodes = ['localhost:22223']

    logging.debug('list of ps: {}'.format(','.join(list_of_parameters_nodes)))
    logging.debug('list of wk: {}'.format(','.join(list_of_worker_nodes)))
    if parallel_environment:
        cluster = tf.train.ClusterSpec({
            'worker': list_of_worker_nodes,
            'ps': list_of_parameters_nodes
        })
    else:
        cluster = tf.train.ClusterSpec({
            'worker': list_of_worker_nodes,
        })
    # extended splash screen
    if parallel_environment:
        logging.info('running in parallel environment')
        logging.info('node name: {}'.format(list_of_nodes[task_index]))
        logging.info('node: {}/{}'.format(task_index + 1, len(list_of_nodes)))
        logging.info('is chief: {}'.format((task_index == 0)))
    else:
        logging.info('running in serial environment')

    # split parameters in subinstance
    io_parameters, parallelization_parameters,\
        train_parameters, system_scaffold = parameters

    # ===compatibility layer with old variable naming===
    # FIXME: remove this part
    # train generic:
    batch_size = train_parameters.batch_size
    max_steps = train_parameters.max_steps
    inter_op_parallelism_threads = \
                  parallelization_parameters.inter_op_parallelism_threads
    intra_op_parallelism_threads = \
                  parallelization_parameters.intra_op_parallelism_threads
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
    if not tf.gfile.Exists(os.path.join(io_parameters.train_dir,
                                        'networks_metadata.json')):
        with open(
                os.path.join(io_parameters.train_dir,
                             'networks_metadata.json'), 'w') as file_stream:
            json.dump(system_scaffold.metadata, file_stream)

    if task_index >= (len(list_of_nodes) - flags.parameter_servers):
        logging.info('Parameter server.')
        # realign task_index at zero
        task_index = task_index - len(list_of_nodes) + flags.parameter_servers
        server = tf.train.Server(cluster, job_name="ps", task_index=task_index)
        server.join()
    else:
        with tf.Graph().as_default():
            # set local machine as default device for operation
            # this construction require a ps, fix later to support no ps
            with tf.device(tf.train.replica_device_setter(cluster=cluster)):
                # build the graph
                train_op, t_iterator, predictions, global_step, loss =\
                    _graph_builder(parameters)

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
                    resources.initialize_resources(
                        resources.shared_resources()),
                )
                # init op are run only by chief
                # local init op are run by every worker
                scaffold = tf.train.Scaffold(
                    init_op=init_op,
                    saver=saver,
                    local_init_op=[t_iterator.initializer])

                # HOOKS
                class _LoggerHook(tf.train.SessionRunHook):
                    """Logs loss and runtime.
                    """
                    def __init__(self, *args, starting_step=0, **kwargs):
                        self._starting_step = starting_step
                        self._start_time = time.time()
                        self._step = 0
                        self._writer = None
                        self._writer2 = None

                    def begin(self):
                        self._step = self._starting_step
                        self._start_time = time.time()
                        self._writer = open('delta_e.dat', 'w')
                        self._writer2 = open('delta_e_std.dat', 'w')

                    def before_run(self, run_context):
                        self._step += 1
                        return tf.train.SessionRunArgs(
                            [predictions, global_step])

                    def after_run(self, run_context, run_values):
                        if self._step <= 0:
                            return
                        if self._step % io_parameters.log_frequency == 0:

                            delta_e = run_values.results[0]

                            self._writer.write(
                                ' '.join(['%.8f' % (x)
                                          for x in delta_e]) + '\n')
                            self._writer2.write(
                                '%.8f %.8f %.8f' %
                                (np.mean(delta_e), np.mean(np.abs(delta_e)),
                                 np.std(delta_e)) + '\n')
                            self._writer.flush()
                            self._writer2.flush()
                            time_str = time.strftime("%m-%d %H:%M:%S",
                                                     time.gmtime())
                            logging.info('{}, global step: {}'.format(
                                time_str, run_values.results[1]))

                        if self._step % io_parameters.avg_speed_steps == 0:
                            current_time = time.time()
                            duration = current_time - self._start_time
                            if duration > 0.0:
                                self._start_time = current_time
                                examples_per_sec = \
                  io_parameters.avg_speed_steps * batch_size / duration
                                sec_per_batch = \
                  duration / io_parameters.avg_speed_steps
                                format_str = '{:4.2f} examples/sec, ' \
                                             '{:4.2f} sec/batch'
                                logging.info(
                                    format_str.format(examples_per_sec,
                                                      sec_per_batch))

                hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
                chief_hooks = [tf.train.NanTensorHook(loss), _LoggerHook()]
                # end of hooks

                server = tf.train.Server(cluster,
                                         job_name="worker",
                                         task_index=task_index)

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
                        save_checkpoint_steps=io_parameters.
                        save_checkpoint_steps,
                        config=config_proto) as mon_sess:
                    if flags.debug:
                        mon_sess = tf_debug.LocalCLIDebugWrapperSession(
                            mon_sess)
                        logging.debug('---variable of the session---')
                        logging.debug('\n'.join(
                            [' ' + var.name for var in tf.global_variables()]))
                        logging.debug('-----')
                    # infinite loop, only hooks act here
                    while not mon_sess.should_stop():
                        mon_sess.run(train_op)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--config",
                        type=str,
                        default='',
                        help="config file",
                        required=True)
    PARSER.add_argument('--list_of_nodes',
                        type=str,
                        default='',
                        help='comma-separated list of nodes')
    PARSER.add_argument(
        "--task_index_variable",
        type=str,
        default='',
        help="task index variable name, depends on workload manager")
    PARSER.add_argument("--debug",
                        action='store_true',
                        help='enable debug mode, partial support only')
    PARSER.add_argument(
        "--debug_parallel",
        action='store_true',
        help='debug parallel flag, use to have 2 server in local config')
    PARSER.add_argument("--parameter_servers",
                        type=int,
                        default=0,
                        help='number of parameter server, default 0')
    PARSER.add_argument("--debug_parallel_index",
                        type=int,
                        help='index 0 or 1')
    PARSER.add_argument("--communication_port",
                        type=int,
                        default=22222,
                        help='communication port for parallel implementation')
    FLAGS, UNPARSED = PARSER.parse_known_args()

    if FLAGS.debug:
        tf.logging.set_verbosity(1)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    # set up console logger only for the chief
    if FLAGS.task_index_variable != '':
        TASK_INDEX = int(os.environ[FLAGS.task_index_variable])
    else:
        TASK_INDEX = 0

    if TASK_INDEX == 0:
        # add to master the console handler
        # and do the splash screen
        LSH = or_logging.StreamHandler()
        LSH.setFormatter(formatter)
        tf_logger.addHandler(LSH)
        _splash_screen(LSH.emit)
    PARAMETERS = parameter_file_parser(FLAGS.config)
    train(FLAGS, PARAMETERS)

import os
import random
import signal
import tensorflow as tf
import json
import sys
import logging
import re
import time

from kombu import Connection
from tensorflow.python.lib.io import file_io

import selftf.lib.common
import selftf.lib.util
from . import common
from selftf.lib.common import PSTunerTrainingDataSerializer, ComputeNode, get_optimizer_name
from selftf.lib.message import UpdateNumOfSteps, NaturalFinishML, SendRecoveryTime, \
    ReconfigScheme1, MessageSerializer, \
    FinshSubTrainingPhase, TriggerChiefCheckPoint, ReconfigScheme2
from .common_conf import MODE_VALUE_MLTUNER, MODE_VALUE_DRY_RUN
from .queue import KombuQueueManager, ConsumerMixin
import threading

import tensorflow.contrib.graph_editor as tfge

import selftf.lib.device_setter as device_setter


class TFProgramUtil:
    _job_name = common.job_name
    _job_prefix = common.job_prefix
    _collect_statistic_run = "collect_statistic_run"

    def __init__(self):
        logging.getLogger("amqp").setLevel(logging.INFO)
        self.logger = logging.getLogger(__name__)

        # input flags
        tf.app.flags.DEFINE_string("job_name", "worker", "Either 'ps' or 'worker'")
        tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
        tf.app.flags.DEFINE_float("targted_accuracy", 0.5, "targted accuracy of model")
        tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer we adopted")

        # <--  Add-on by pstuner -->
        # tf.app.flags.DEFINE_string("ps_list", "", "")
        # tf.app.flags.DEFINE_string("worker_list", "", "")
        tf.app.flags.DEFINE_string("node_list", "", "")
        tf.app.flags.DEFINE_string("working_dir", "", "")
        tf.app.flags.DEFINE_bool("is_chief", True, "")
        tf.app.flags.DEFINE_integer("max_iteration", 20, "")
        tf.app.flags.DEFINE_float("target_loss", 0.5, "")
        tf.app.flags.DEFINE_string("conf_dict", "{}", "")

        # <-- From Andy Framework -->
        tf.app.flags.DEFINE_string("ML_model", "LR", "ML model")
        tf.app.flags.DEFINE_float("targeted_loss", 0.05, "targted accuracy of model")
        tf.app.flags.DEFINE_integer("Batch_size", 1000, "Batch size")
        tf.app.flags.DEFINE_integer("num_Features", 54686452, "number of features")
        tf.app.flags.DEFINE_float("Learning_rate", 0.001, "Learning rate")
        tf.app.flags.DEFINE_integer("Epoch", 1, "Epoch")

        # <-- ampq -->
        tf.app.flags.DEFINE_string("amqp_user", "guest", "")
        tf.app.flags.DEFINE_string("amqp_password", "guest", "")
        tf.app.flags.DEFINE_string("amqp_host", os.getenv("AMPQ_MASTER_NODE"), "")
        tf.app.flags.DEFINE_integer("amqp_port", 5672, "")
        tf.app.flags.DEFINE_string("agent_id", "", "")
        tf.app.flags.DEFINE_integer("agent_config_idx", -1, "")
        tf.app.flags.DEFINE_string("agent_job_id", "", "")

        tf.app.flags.DEFINE_bool(self._collect_statistic_run, False, "")

        self.FLAGS = tf.app.flags.FLAGS
        self.conf_dict = json.loads(self.FLAGS.conf_dict)
        self.logger.debug("conf_dict: %s" % self.FLAGS.conf_dict)

        self.run_mode = self.conf_dict[common.conf_dict_mode]

        self.saver = None
        self.global_step = None

        self.ps_should_stop = False
        self.sess = None
        self.done_iteration_done = False

        self.recovery_begin_time = 0.0
        self.recovery_duration = 0.0

        self.init_global_step = 0

        self.mq_manager = KombuQueueManager()

        self.agent_id = self.FLAGS.agent_id
        self.agent_job_id = self.FLAGS.agent_job_id

        tf_hostname,tf_port = ComputeNode.get_hostname_tfport_from_id(self.agent_id)
        self.tf_hostname = tf_hostname
        self.tf_tf_port = tf_port
        self.compute_node= ComputeNode(tf_hostname, tf_port)

        self.optimizer_loss_variable = None
        self.optimizer_variable_list = []

        # reconfig scheme1
        self.flag_do_live_reconfig = False
        self.do_live_reconfig_config_dict = {}
        self.do_live_reconfig_config_idx = 0
        self.do_live_reconfig_max_iteration = 0
        self.do_live_reconfig_last_step = 0

        # reconfig scheme2
        self.flag_do_live_reconfig2 = False
        self.do_live_reconfig2_context = None # type: SelfTFOptimizerContext
        self.do_live_reconfig2_pending_ops = []
        self.do_live_reconfig2_pending_ops_local = []

        # A function (TFProgramUtil) -> None
        #TODO: init with worker cache and parameter server cache
        self.graph_init_func = None

        self.flag_do_checkpoint = False
        self.flag_end_process = False

        # init in-process message queue
        self.conn = Connection('amqp://%s:%s@%s:%s//' % (self.FLAGS.amqp_user, self.FLAGS.amqp_password,
                                                         self.FLAGS.amqp_host, self.FLAGS.amqp_port))
        mq_queue = self.mq_manager.get_compute_node_tf_queue(self.compute_node)
        self.consumer = ConsumerMixin(queue=mq_queue, message_handler=self.agent_tf_message_handler,
                                      connection=self.conn)

        #init thread for mq _listening
        self.mq_thread = threading.Thread(target=self.thread_mq_listener)
        self.mq_thread.start()

        self.train_op = None

        # current TF graph executing
        self.default_tf_graph = tf.Graph()

        self.tf_server_target = ""

        self.global_step = None

        self._list_trainable_variable_name = []
        self._list_variable_name_without_optimizer_variable = []

        self._end_by_term = False

        # sync optimizer
        self.sync_opt = None
        self._is_sync_optimizer = False

        # no use if replica=num_require_agg
        self._sync_init_op = None

        # Summary
        # self.tf_summary_writer = tf.summary.FileWriter("/tmp/tf_summary", self.get_default_tf_graph())
        self.global_var_ready_op = None
        self.global_var_ready_op_scope_name = "report_uninitialized_variables"

        self.managed_var = None

        self.last_iteration_runtime_sec = 0

        self.tf_run_options = tf.RunOptions()

        self.target_reconfig_step = 0

        self.last_updatenum_timestamp = 0
        self.last_updatenum_threshold = 2

    def get_defualt_init_ready_op(self):
        list_var = []
        with self.get_default_tf_graph().as_default():
            if self.run_mode in [MODE_VALUE_MLTUNER, MODE_VALUE_DRY_RUN]:
                return tf.report_uninitialized_variables()
            else:
                assert isinstance(self.do_live_reconfig2_context, SelfTFOptimizerContext)
                for replica in self.do_live_reconfig2_context._wrapper_replica_variables.list_variable_replica:
                    list_var.append(replica.original_v)
                return tf.report_uninitialized_variables(list_var,
                                                     name=self.global_var_ready_op_scope_name)

    def set_tf_server_target(self, str):
        self.tf_server_target = str

    @property
    def variable_map(self):
        variable_map = self.conf_dict.get(common.conf_dict_variable_map)
        if variable_map is None:
            return {}
        return variable_map

    def thread_mq_listener(self):

        self.logger.debug("Start listening mq")
        # init mq
        self.consumer.run()
        # with self.conn.Consumer(queues=mq_queue, callbacks=[self.agent_tf_message_handler]) as consumer:
        #     while not self.flag_end_process:
        #         # self.logger.debug("Background: checking mq message")
        #         try:
        #             self.conn.drain_events(timeout=0.05) # just let it
        #         except:
        #             pass
        #         time.sleep(0.5)
        #
        # # listening until dead


    def is_training_phase(self):
        if self.get_max_iteration() > 0:
            return True
        else:
            return False


    def log_flags_value(self):
        for key, value in tf.flags.FLAGS.__flags.items():
            self.logger.debug("TF Flags %s: %s" % (key, value))

    def get_working_dir(self):
        return self.FLAGS.working_dir

    def get_is_chief(self):
        return self.FLAGS.is_chief

    def get_agent_config_idx(self):
        return self.FLAGS.agent_config_idx

    def get_max_iteration(self):
        max_iteration = self.FLAGS.max_iteration
        if max_iteration == -1:
            return sys.maxsize
        else:
            return max_iteration

    def get_job_name(self):
        return self._job_name

    def get_task_index(self):
        return self.FLAGS.task_index

    # Mainly for data shard
    def get_worker_index(self):
        return self.get_task_index() - self.get_num_ps()
    #
    # def get_targted_accuracy(self):
    #     return self.FLAGS.targted_accuracy

    def get_target_loss(self):
        return self.FLAGS.target_loss

    def _get_optimizer(self, optimizer, learning_rate):
        # Debug for optimizer
        # return tf.train.AdamOptimizer(learning_rate)

        if optimizer == 0:
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer == 1:
            return tf.train.AdadeltaOptimizer(learning_rate)
        elif optimizer == 2:
            return tf.train.AdamOptimizer(learning_rate)
        elif optimizer == 3:
            return tf.train.RMSPropOptimizer(learning_rate)
        elif optimizer == 4:
            return tf.train.MomentumOptimizer(learning_rate, 0.9)
        elif optimizer == 5:
            return tf.train.RMSPropOptimizer(learning_rate, momentum=0.9,
                                             epsilon=1)

    def get_optimizer(self):
        self.logger.debug(self.conf_dict)
        # optimizer_idx = self.conf_dict[tuner._optimizer]
        # learning_rate = self.conf_dict[tuner._learning_rate]
        optimizer_idx = self.conf_dict[selftf.lib.common._optimizer]
        learning_rate = self.conf_dict[selftf.lib.common._learning_rate]
        self.logger.debug("get optimizer with optimizer: %s, learning rate:%f " % (
            get_optimizer_name(optimizer_idx),
            learning_rate))
        return self._get_optimizer(optimizer_idx, learning_rate)

    # def get_other_optimizers(self):
    #     ret = []
    #     optimizer_idx = self.conf_dict[tuner._optimizer]
    #     learning_rate = self.conf_dict[tuner._learning_rate]
    #     opt_idx = range(len(common.optimizer_list))
    #     opt_idx.remove(optimizer_idx)
    #     for idx in opt_idx:
    #         ret.append(self._get_optimizer(idx, learning_rate))
    #     return ret

    def get_intra_op_parallelism_threads(self):
        return self.conf_dict.get(
            selftf.lib.common._intra_op_parallelism_threads, 8)

    def get_inter_op_parallelism_threads(self):
        return self.conf_dict.get(
            selftf.lib.common._inter_op_parallelism_threads, 8)

    def get_tf_flag(self):
        return self.FLAGS

    def get_do_common_subexpression_elimination(self):
        return bool(self.conf_dict[common._do_common_subexpression_elimination])

    def get_max_folded_constant_in_bytes(self):
        return int(self.conf_dict[common._max_folded_constant_in_bytes])

    def get_do_function_inlining(self):
        return bool(self.conf_dict[common._do_function_inlining])

    def get_global_jit_level(self):
        level = self.conf_dict[common._global_jit_level]
        if level == 0:
            return tf.OptimizerOptions.OFF
        elif level == 1:
            return tf.OptimizerOptions.ON_1
        elif level == 2:
            return tf.OptimizerOptions.ON_2
        else:
            raise Exception()

    def get_infer_shapes(self):
        return bool(self.conf_dict[common._infer_shapes])

    def get_enable_bfloat16_sendrecv(self):
        return bool(self.conf_dict[common._enable_bfloat16_sendrecv])

    def get_place_pruned_graph(self):
        return bool(self.conf_dict[common._place_pruned_graph])

    def get_inter_op_parallelism_threads(self):
        return int(self.conf_dict[common._inter_op_parallelism_threads])

    def get_intra_op_parallelism_threads(self):
        return int(self.conf_dict[common._intra_op_parallelism_threads])
    
    def get_KMP_AFFINITY_granularity(self):
        gran = int(self.conf_dict[common._KMP_AFFINITY_granularity])
        if gran == 1:
            return "core"
        elif gran == 0:
            return "fine"
        else:
            raise Exception()
    
    def get_KMP_AFFINITY_respect(self):
        respect = int(self.conf_dict[common._KMP_AFFINITY_respect])
        if respect == 1:
            return "respect"
        elif respect == 0:
            return "norespect"
        else:
            raise Exception()
    
    def get_KMP_AFFINITY_type(self):
        typ = int(self.conf_dict[common._KMP_AFFINITY_type])
        if typ == 0:
            return "compact"
        elif typ == 1:
            return "scatter"
        else:
            raise Exception()

    def get_KMP_AFFINITY_permute(self):
        return int(self.conf_dict[common._KMP_AFFINITY_permute])

    def get_KMP_AFFINITY_offset(self):
        return int(self.conf_dict[common._KMP_AFFINITY_offset])

    def get_KMP_BLOCKTIME(self):
        return int(self.conf_dict[common._KMP_BLOCKTIME])
    
    def get_OMP_NUM_THREADS(self):
        return int(self.conf_dict[common._OMP_NUM_THREADS])
    
    def get_MKL_DYNAMIC(self):
        dyna = int(self.conf_dict[common._MKL_DYNAMIC])
        if dyna == 1:
            return "TRUE"
        elif dyna == 0:
            return "FALSE"
        else:
            raise Exception()

    def get_tf_config_proto(self):
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=self.get_inter_op_parallelism_threads(),
            intra_op_parallelism_threads=self.get_intra_op_parallelism_threads(),
            allow_soft_placement=True,
            log_device_placement=False,
            graph_options=tf.GraphOptions(
                optimizer_options=tf.OptimizerOptions(
                    do_common_subexpression_elimination=self.get_do_common_subexpression_elimination(),
                    do_constant_folding=True,
                    max_folded_constant_in_bytes=self.get_max_folded_constant_in_bytes(),
                    do_function_inlining=self.get_do_function_inlining(),
                    opt_level=tf.OptimizerOptions.L0,
                    global_jit_level=self.get_global_jit_level()
                ),
                enable_bfloat16_sendrecv=self.get_enable_bfloat16_sendrecv(),
                infer_shapes=self.get_infer_shapes(),
                place_pruned_graph=self.get_place_pruned_graph()
            )
        )

        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = 1
        return tf_config


    def get_chkpt_prefix(self):
        return self.FLAGS.working_dir + "/chkpt"

    def get_chkpt_dir(self):
        return self.FLAGS.working_dir

    def is_chkpt_exist(self):
        if not tf.train.latest_checkpoint(self.get_chkpt_dir()):
            return False
        else:
            return True

    def get_untrainable_variable(self):
        graph = self.get_default_tf_graph()
        all_variables = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        trainable_variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) + [self.global_step]

        for variable in trainable_variables:
            all_variables.remove(variable)

        self.logger.debug("Chris: untrainable variables: %s" % all_variables)

        return all_variables

    def get_monitored_training_session(self):
        """
        equal pre_do_iteration
        :param global_step:
        :param saver:
        :return:
        """
        with self.get_default_tf_graph().as_default():
            with tf.device(self.device_setter()):
                if self.run_mode == common.MODE_VALUE_SELFTF:
                    # little hack for monitoring session
                    self.saver = tf.train.Saver(var_list=[self.global_step])
                else:
                    self.saver = tf.train.Saver(sharded=True)
        init_op = tf.no_op()
        init_fn = None
        init_op_local = None
        ready_for_local_init_op = self.global_var_ready_op
        ready_op = tf.no_op()
        if self.is_chkpt_exist() and not self.flag_do_live_reconfig and not self.flag_do_live_reconfig2:
            # Path for restart reconfig only
            init_op = tf.variables_initializer(self.get_optimizer_var())
            def _init_fn(scaffold, sess):
                self.do_recover(sess)
                # init optimizer variable + other untrainable variable
            init_fn = _init_fn
        else:
            # Path for live reconfig / init start
            if self.flag_do_live_reconfig:
                init_op = tf.no_op()

                def _init_fn(scaffold, sess):
                    # init optimizer variable + other untrainable variable
                    sess.run(tf.variables_initializer(self.get_optimizer_var()))
                    # pass
                init_fn = _init_fn
                self.done_reconfig1()
            elif self.flag_do_live_reconfig2:
                if not self.do_live_reconfig2_context.is_final_phase():
                    # phase 1
                    def _init_fn(scaffold, sess):
                        for x in self.do_live_reconfig2_pending_ops:
                            sess.run(x)
                    init_fn = _init_fn

                    # create local init op
                    # local_init_op = tf.group(*self.do_live_reconfig2_pending_ops_local)
                    # init_op_local = local_init_op
                    ready_op = tf.no_op()
                    ready_for_local_init_op = tf.no_op()

                    if self._is_sync_optimizer:
                        init_op = self.sync_chief_init_op
                        init_op_local = self.sync_local_step_init_op

                    # if len(self.do_live_reconfig2_pending_ops_local) != 2:
                    #     raise Exception("Something wrong")
                    # with tf.control_dependencies(self.do_live_reconfig2_pending_ops_local[0]):
                    #     with tf.control_dependencies(self.do_live_reconfig2_pending_ops_local[1]):
                    #         tf.identity(tf.Constant("reconfig2", name="control"))
                    # def _init_fn_local(sess):
                    #     for x in self.do_live_reconfig2_pending_ops_local:
                    #         sess.run(x)
                    # init_fn_local = _init_fn_local
                    self.do_live_reconfig2_context.finish_phase1()
                    logging.debug("Finish reconfig scheme 2 phase 1")
                else:
                    with self.default_tf_graph.as_default():
                        # skip checking since we handle it by ourselves
                        # ready_op = tf.no_op()
                        # ready_for_local_init_op = self.global_var_ready_op

                        with tf.device(self.device_setter()):
                            # init_op = tf.variables_initializer(self.do_live_reconfig2_context.get_optimizer_variables())
                            # init_op = tf.group(*(self.do_live_reconfig2_pending_ops[0]+[init_op_opt_var]))
                            def _init_fn(scaffold, sess):
                                for list_op in self.do_live_reconfig2_pending_ops:
                                    sess.run(list_op)

                                # self.log_tv_device()

                            init_fn = _init_fn

                    # phase 2 finish all
                    self.clear_do_scheme2_reconfig()
                    self.flag_finish_reconfig = True

                    if self._is_sync_optimizer:
                        init_op_local = self.sync_local_step_init_op

                    logging.debug("Finish reconfig scheme 2 phase 2 (last)")
                    Reconfig2ProfilingTools.get_instance().finish_phase2_get_supervisor()

            else:
                # plain startup
                init_op = None
                init_fn = None

                # with self.get_default_tf_graph().as_default():
                #     list_var_worker_cache = self.do_live_reconfig2_context.get_worker_cache_var()
                #     logging.debug("list_var_worker_cache: %s "% list_var_worker_cache)
                #     init_op_local = tf.variables_initializer(list_var_worker_cache)

                self.logger.debug("Try to create checkpt folder")
                file_io.create_dir(self.get_chkpt_dir())
                self.logger.debug("Finish create checkpt folder")

        # if self.get_is_chief():
        #     return tf.train.Supervisor(is_chief=self.get_is_chief(),
        #                                global_step=self.global_step,
        #                                init_op=init_op,
        #                                init_fn=init_fn,
        #                                recovery_wait_secs=0.5,
        #                                graph=self.get_default_tf_graph(),
        #                                ready_op=None
        #                                )
        # else:
        scaffold = tf.train.Scaffold(saver=self.saver,
                                     init_op=init_op,
                                     local_init_op=init_op_local,
                                     ready_for_local_init_op=ready_for_local_init_op,
                                     init_fn=init_fn,
                                     ready_op=ready_op)
        sess = tf.train.MonitoredTrainingSession(
            master=self.tf_server_target,
            is_chief=self.get_is_chief(),
            scaffold=scaffold,
            save_checkpoint_secs=None,
            save_summaries_steps=None,
            save_summaries_secs=None,
            config=self.get_tf_config_proto(),
            stop_grace_period_secs=1
        )
        return sess
        # return tf.train.Supervisor(is_chief=self.get_is_chief(),
        #                            global_step=self.global_step,
        #                            init_op=init_op,
        #                            init_fn=init_fn,
        #                            local_init_op = init_op_local,
        #                            recovery_wait_secs=0.5,
        #                            graph=self.get_default_tf_graph(),
        #                            ready_op=ready_op
        #                            )

    def do_recover(self, sess):
        """
        :param tf.Session sess:
        :return:
        """
        if self.get_is_chief() and self.is_chkpt_exist():
            self.logger.debug("CheckPoint exist do it now")
            self.saver.restore(sess, tf.train.latest_checkpoint(self.get_chkpt_dir()))
        else:
            self.logger.debug("CheckPoint doesn't exist")

    def stop_set_flag(self):
        self.flag_end_process = True
        self.consumer.should_stop = True

    def should_stop_iteration(self, last_global_step, last_cost):
        self.logger.debug("last_cost:%f, target_loss:%f" % (last_cost, self.get_target_loss()))
        self.logger.debug("init global step:%f, current global step:%f, step per os:%f" % (self.init_global_step, last_global_step, self.get_max_iteration()))
        self.logger.debug("reconfig 1 flag:%s, reconfig 2 flag:%s" % (str(self.is_reconfig()), str(self.is_reconfig2())))
        # if self.is_training_phase_finish_sub_training(last_global_step) or self.is_overall_training_finish(last_cost):
        # if self.is_overall_training_finish(last_cost):
        #     self.stop_set_flag()
        #     if last_cost <= self.get_target_loss():
        #         self.send_nature_finish(last_cost)
        #         while not self._end_by_term:
        #             # waiting until it is killed by monitor
        #             time.sleep(10)
        #     return True
        # else:
        #     return False
        return False

    def is_training_phase_finish_sub_training(self, last_global_step):
        if last_global_step >= self.init_global_step + self.get_max_iteration():
            return True
        else:
            return False

    def is_overall_training_finish(self, last_cost):
        if last_cost <= self.get_target_loss():
            return True
        else:
            return False

    def print_iteration_statistic(self, step, final_accuracy, cost, begin_time):

        print("Step: %d," % step,
              " Accuracy: %.4f," % final_accuracy,
              " Loss: %f" % cost,
              " Time: %fs" % float(time.time() - begin_time),
              " Timestamp: %f" % time.time())

    def do_pre_build_graph(self, global_step):
        # add a operation to reset graph to iteration
        self.global_step = global_step

    def _is_optimal_run(self):
        if self.get_max_iteration() == sys.maxsize:
            return True
        else:
            return False

    def post_do_all_iteration(self, sess=None, cost=sys.maxsize):
        """
        :param tf.Session sess:
        :return:
        """
        self.logger.debug("do post_do_all_iteration")

        # Check do reconfiguration
        if self.get_is_chief() and self.flag_do_checkpoint and not self.is_overall_training_finish(last_cost=cost):
            self.chief_message_handler()
            os._exit(0)

    def post_do_iteration(self, steps, loss, timestamp, duration):
        """
        For Worker
        :param steps:
        :param loss:
        :param timestamp:
        :param duration:
        :return:
        """
        self.logger.debug("Do post iteration")
        if duration > self.last_iteration_runtime_sec:
            self.update_tf_op_timeout(duration * 2)
        self.last_iteration_runtime_sec = duration
        ps_tuner_training_data = [selftf.lib.common.PSTunerTrainingData(
            elapsed_time_in_ms=float(duration),
            loss=float(loss),
            local_step=int(steps),
            timestamp=float(timestamp),
            ps_config_idx=self.get_agent_config_idx())]

        if (self._is_sync_optimizer and self.get_is_chief()) or \
            not self._is_sync_optimizer:
            if (timestamp - self.last_updatenum_timestamp) > self.last_updatenum_threshold:
                msg = UpdateNumOfSteps.create(self.agent_id,
                                              self.mq_manager.get_monitor_name(), ps_tuner_training_data[-1].local_step,
                                              self.agent_job_id,
                                              ps_tuner_training_data_jsons=json.dumps(ps_tuner_training_data,
                                                                                  cls=PSTunerTrainingDataSerializer))
                self.mq_manager.send_msg_to_monitor(conn=self.conn, message_obj=msg)

                self.last_updatenum_timestamp = timestamp

        # if stop iteration
        # if self.is_training_phase():
        #     self.training_phase_check_do_reconfig(steps)
        # else:
        #     self.optimize_phase_check_do_reconfig()
        # For init phase
        if self.is_training_phase_finish_sub_training(steps):
            # send message to monitor
            self.training_phase_check_do_reconfig(steps)
            # wait for Finish or live reconfig
            while True:
                # logging.debug("Waiting here for next step. Reconfig / Kill")
                if self.flag_end_process or self.flag_do_checkpoint or self.flag_do_live_reconfig2:
                    # self.chief_print_summary()
                    break
                else:
                    time.sleep(0.5)

    def post_check_reconfig_or_finish(self):
        logging.debug("Start post_check_reconfig_or_finish")

        try:
            if self.flag_do_live_reconfig:
                # do live reconfig
                self.do_scheme_1_reconfig(new_conf_dict=self.do_live_reconfig_config_dict,
                                          conf_idx=self.do_live_reconfig_config_idx,
                                          max_iteration=self.do_live_reconfig_max_iteration,
                                          last_steps=self.do_live_reconfig_last_step)
                return True
            if self.flag_do_live_reconfig2:
                # do live reconfig2
                self.do_scheme_2_reconfig(
                    new_conf_dict=self.do_live_reconfig_config_dict,
                    conf_idx=self.do_live_reconfig_config_idx,
                    max_iteration=self.do_live_reconfig_max_iteration,
                    last_steps=self.do_live_reconfig_last_step)
                self.do_live_reconfig2_context.clear_all_ops_collocate()
                return True
            return False
        except Exception as e:
            logging.exception("error in reconfig")
            raise e




    def pre_do_all_iteration(self, sess):
        self.sess = sess

        # hook handler
        signal.signal(signal.SIGHUP, self.term_handler)
        signal.signal(signal.SIGINT, self.term_handler)
        signal.signal(signal.SIGTERM, self.term_handler)

    def term_handler(self, sig, stack):
        self.conn.close()
        self.stop_set_flag()
        self.mq_thread.join()
        self._end_by_term = True
        sys.exit(0)

    def pre_recovery(self):
        self.logger.debug("Start do recovery, conf_idx:%d" % self.get_agent_config_idx())
        self.recovery_begin_time = time.time()

    def chief_print_summary(self):
        if self.get_is_chief():
            list_summary = self.sess.run(tf.get_collection(tf.GraphKeys.SUMMARIES))
            for x in list_summary:
                self.tf_summary_writer.add_summary(x)
            self.tf_summary_writer.flush()
            # self.logger.debug("check opt value: %s" % self.sess.run(self.get_default_tf_graph().get_tensor_by_name("conv1/weights/part_0/Adam:0")))

    def post_recovery(self, sess):
        self.logger.debug("Finish recovery")
        self.sess = sess

        if self.init_global_step == 0:
            # after restart reconfiguration
            self.init_global_step = self.sess.run(self.global_step)

        #  little hack here. Some init op
        if self.flag_do_live_reconfig2:
            for x in self.do_live_reconfig2_pending_ops_local:
                logging.debug("do_live_reconfig2_pending_ops_local:%s" % (x))
                sess.run(x)

        self.recovery_duration = time.time() - self.recovery_begin_time
        if self.get_is_chief():
            reconfig_finish = True
            if self.run_mode not in [MODE_VALUE_DRY_RUN, MODE_VALUE_MLTUNER]:
                if self.do_live_reconfig2_context.is_final_phase():
                    reconfig_finish = False
            self.send_recovery_time(self.get_agent_config_idx(),
                                    self.recovery_duration,
                                    reconfig_finish=reconfig_finish)

        # self.chief_print_summary()

        Reconfig2ProfilingTools.get_instance().finish_phase1_get_supervisor()


    def send_nature_finish(self, cost):
        msg = NaturalFinishML.create(source=self.agent_id,
                                     destination=self.mq_manager.get_monitor_name(),
                                     job_id=self.agent_job_id,
                                     final_cost=float(cost))
        self.mq_manager.send_msg_to_monitor(self.conn, msg)

    def send_recovery_time(self, conf_idx, recovery_duration, reconfig_finish=False):
        # Only use by chief worker
        if not self.get_is_chief():
            return
        logging.debug("Send Recovery conf_idx:%s time:%s" % (conf_idx,
                                                             recovery_duration))

        managed_var = self.managed_var
        msg = SendRecoveryTime.create(source=self.agent_id,
                                      destination=self.mq_manager.get_monitor_name(),
                                      job_id=self.agent_job_id,
                                      recovery_time_sec=recovery_duration,
                                      json_variable_str=json.dumps(managed_var,
                                                                   cls=selftf.lib.util.TFVariableSeralizer),
                                      conf_idx=conf_idx,
                                      reconfig_finish=reconfig_finish
                                      # json_non_static_ops_str=json.dumps(list_static_op_name)
                                      )
        self.mq_manager.send_msg_to_monitor(self.conn, msg)

    def chief_message_handler(self, body=None, mq_message=None):
        self.logger.debug("Receive TriggerCheckPointMsg")
        self.saver.save(self.sess._sess._sess._sess._sess, save_path=self.get_chkpt_prefix(), global_step=self.global_step,
                        write_meta_graph=False)
        self.stop_set_flag()
        self.logger.debug("Finish checkpoiniting ")

    def agent_tf_message_handler(self, body, mq_message):
        # construct ConfigProto
        # send to master service
        try:
            message = json.loads(body, object_hook=MessageSerializer.message_object_hook)
            mq_message.ack()
            self.logger.debug("Receive msg from %s type: %s" % (message.get_source(), message.get_message_type()))
            self.agent_tf_message_logic(message, mq_message)
        except Exception as e:
            self.logger.exception("failt to process message")

    def agent_tf_message_logic(self, message_obj, mq_message):
        if message_obj.job_id != self.agent_job_id:
            return
        if isinstance(message_obj, ReconfigScheme1):
            self.logger.debug("Receive message ReconfigScheme1, set flag")
            self.flag_do_live_reconfig = True
            self.do_live_reconfig_max_iteration = message_obj.get_max_iteration()
            self.do_live_reconfig_config_idx = message_obj.get_conf_id()
            self.do_live_reconfig_config_dict = message_obj.get_conf_dict()
            self.do_live_reconfig_last_step = message_obj.get_last_step()
        if isinstance(message_obj, ReconfigScheme2):
            self.logger.debug("Receive message ReconfigScheme2, set flag")
            self.flag_do_live_reconfig2 = True
            self.target_reconfig_step = message_obj.target_reconfig_step
            self.do_live_reconfig_max_iteration = message_obj.get_max_iteration()
            self.do_live_reconfig_config_idx = message_obj.get_conf_id()
            self.do_live_reconfig_config_dict = message_obj.get_conf_dict()
            self.do_live_reconfig_last_step = message_obj.get_last_step()
        if isinstance(message_obj, TriggerChiefCheckPoint):
            self.logger.debug("Receive message Checkpointing, set flag")
            self.flag_do_checkpoint = True

    def do_scheme_1_reconfig(self, new_conf_dict, conf_idx, max_iteration, last_steps):
        """
        :param last_steps:
        :return:
        """

        old_config_dir = self.conf_dict
        self.conf_dict = new_conf_dict

        self.FLAGS.__setattr__("conf_dict", new_conf_dict)
        new_config_proto = self.get_tf_config_proto()

        self.FLAGS.__setattr__("agent_config_idx", conf_idx)

        logging.debug("Connect to tf master service %s:%s" % (self.tf_hostname, self.tf_tf_port))
        self.logger.debug("Session is %s" % str(self.sess))
        self.sess.reconfig(new_config_proto)

        old_optimizer = old_config_dir.get(selftf.lib.common._optimizer, "Adam")
        new_optimizer = new_conf_dict.get(selftf.lib.common._optimizer, "Adam")
        if old_optimizer != new_optimizer:
            self.clear_current_graph()
            with tf.get_default_graph().as_default():
                with tf.device(self.device_setter()):
                    self.graph_init_func(self)

        # update the criteria for init phase
        self.FLAGS.__setattr__("max_iteration", max_iteration)
        self.init_global_step = last_steps

        logging.debug("Reconfig: update init_global_step: "+str(self.init_global_step))

        #clean up
        # self.flag_do_live_reconfig = False
        logging.debug("Finish reconfig scheme 1")

    def clear_do_scheme2_reconfig(self):
        self.flag_do_live_reconfig2 = False
        self.do_live_reconfig2_context.finish_phase2()

    def log_tv_device(self):
        logging.debug("log_all_var_device==========")
        with self.get_default_tf_graph().as_default():
            for tv in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                logging.debug("VAR: %s Device: %s" % (tv.op.name, tv.op.device))

    def log_variable_map(self, variable_map=None):
        if variable_map is None:
            variable_map = self.variable_map
        logging.debug("log_variable_map==========")
        for var_name, device in variable_map.items():
            logging.debug("VAR: %s Device: %s" % (var_name, device))

    def build_variable_map(self):
        ret = {}
        with self.get_default_tf_graph().as_default():
            for tv in self.managed_var:
                ret[tv.op.name] = tv.op.device
        return ret


    def do_scheme_2_reconfig(self, new_conf_dict, conf_idx, max_iteration, last_steps):
        """
        :param last_steps:
        :return:
        """
        with self.get_default_tf_graph().as_default():
            if self.do_live_reconfig2_context.is_phase_0():

                Reconfig2ProfilingTools.get_instance().start_reconfig2_phase1()
                old_config_dict = self.conf_dict
                self.conf_dict = new_conf_dict

                self.FLAGS.__setattr__("conf_dict", new_conf_dict)
                new_config_proto = self.get_tf_config_proto()


                self.FLAGS.__setattr__("agent_config_idx", conf_idx)

                # self.log_tv_device()

                GenericTimer.start("connect_session_and_reconfig")
                logging.debug("Connect to tf master service %s:%s" % (self.tf_hostname, self.tf_tf_port))
                with tf.Session(target=self.tf_server_target) as sess:
                    sess.reconfig(new_config_proto)
                GenericTimer.stop("connect_session_and_reconfig")

                GenericTimer.start("rebuild_for_optimizer")
                # old_optimizer = old_config_dir.get(tuner._optimizer, "Adam")
                # new_optimizer = new_conf_dict.get(tuner._optimizer, "Adam")
                #
                old_variable_map = dict(old_config_dict.get(common.conf_dict_variable_map, {}))
                if len(old_variable_map) == 0:
                    old_variable_map = self.build_variable_map()

                self.conf_dict = new_conf_dict

                # logging.debug("old var map ")
                # self.log_variable_map(old_variable_map)
                # logging.debug("new var map ")
                # self.log_variable_map()
                GenericTimer.stop("rebuild_for_optimizer")
                # update the criteria for init phase
                self.FLAGS.__setattr__("max_iteration", max_iteration)
                self.init_global_step = last_steps

                logging.debug("Reconfig: update init_global_step: "+str(self.init_global_step))

                # 2 phase so start change here
                # Change to migration graph
                # set a flag

                # phase 1
                GenericTimer.start("rebuild_for_modify_graph")
                self.do_live_reconfig2_context.start_reconfig()
                self.do_live_reconfig2_context.phase1_update_graph(
                    self.variable_map,
                    int(self.conf_dict[common._worker_num]))

                self.do_live_reconfig2_pending_ops_local = [[tf.group(
                    self.do_live_reconfig2_context.moved_variable_wrapper.get_assign_op_to_worker_cache()
                )],
                    [tf.group(self.do_live_reconfig2_context.moved_variable_wrapper.get_assign_op_to_dst_ps())]]
                self.do_live_reconfig2_pending_ops = []
                GenericTimer.stop("rebuild_for_modify_graph")
                Reconfig2ProfilingTools.get_instance().finish_phase1_post_iter()
            else:
                # phase 2
                Reconfig2ProfilingTools.get_instance().start_reconfig2_phase2()

                assert self.do_live_reconfig2_context.is_final_phase()
                # Next time do migration graph
                # reinit to normal graph
                self.do_live_reconfig2_context.phase2_update_graph()

                self.do_live_reconfig2_pending_ops = [tf.group(self.do_live_reconfig2_context.moved_variable_wrapper.get_assign_ops_dst_ps_to_org_v())]
                self.do_live_reconfig2_pending_ops_local = []

                Reconfig2ProfilingTools.get_instance().finish_phase2_post_iter()



    def get_optimizer_var(self):
        # graph = tf.get_default_graph()
        # trainable_variable = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # all_variable = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # for x in trainable_variable:
        #     all_variable.remove(x)
        # return all_variable
        self.logger.debug("Optimizer variable: "+ str(self.optimizer_variable_list))
        return self.optimizer_variable_list

    # def optimize_phase_check_do_reconfig(self):
    #     self._check_do_reconfig(0.001)

    def training_phase_check_do_reconfig(self, last_global_step):
        if self.is_training_phase_finish_sub_training(last_global_step):
            # Send message to monitor
            self.logger.debug("training_phase_check_do_reconfig")
            msg = FinshSubTrainingPhase.create(self.mq_manager.get_compute_node_routing_key(self.compute_node),
                                               self.mq_manager.get_monitor_name(),
                                               job_id=self.agent_job_id)
            self.mq_manager.send_msg_to_monitor(conn=self.conn, message_obj=msg)
            # self._check_do_reconfig(9999)

    #
    # def _check_do_reconfig(self, timeout):
    #     # All reconfig should be non-data reallocated
    #     # Read message from queue
    #     with self.conn.Consumer(queues=self.mq_queue, callbacks=[self.agent_tf_message_handler]) as consumer:
    #         try:
    #             self.conn.drain_events(timeout=timeout) # just let it
    #         except:
    #             try:
    #                 consumer.recover()
    #             except:
    #                 pass

    def _set_train_op(self, loss=None):
        if loss is None:
            loss = self.optimizer_loss_variable
        else:
            self.optimizer_loss_variable = loss

    def _init_train_op(self, optimizer, sync=False, sync_exp_moving_averager=None,
        variables_to_average=None):
        if sync:
            num_workers = self.get_num_worker()
            optimizer = tf.train.SyncReplicasOptimizer(optimizer,
              replicas_to_aggregate=num_workers,
              total_num_replicas=num_workers,
              variable_averages=sync_exp_moving_averager,
              variables_to_average=variables_to_average)
            self._is_sync_optimizer = True
            self.sync_opt = optimizer

        self.train_op = optimizer.minimize(loss=self.optimizer_loss_variable, global_step=self.global_step)

        if sync:
            self.sync_local_step_init_op = optimizer.local_step_init_op
            self.sync_chief_init_op = optimizer.chief_init_op

            if self.get_is_chief():
                # if previous config num_ps > current config num_ps -> dequeue exceeded token
                # vice versa, enqueue tokens
                sync_queue = optimizer._sync_token_queue
                assert isinstance(sync_queue, tf.QueueBase)
                self.dequeue_sync_queue_op = sync_queue.dequeue()
                self.enqueue_op_sync_queue_op = sync_queue.enqueue(self.global_step)
                self.sync_debug_grad_cur_item = optimizer._accumulator_list[0][0].num_accumulated()
                self.sync_debug_grad_cur_items_ops = list(map(lambda accum: (accum[0], accum[0].num_accumulated()), optimizer._accumulator_list))
                self.sync_debug_sync_cur_item = optimizer._sync_token_queue.size()
                self.sync_debug_enqueue_test_op = optimizer._accumulator_list[0][0].apply_grad(
                    tf.zeros(shape=optimizer._accumulator_list[0][0]._shape))
                tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS,
                                     optimizer.get_chief_queue_runner())
                self._sync_init_op = optimizer.get_init_tokens_op()



    def init_train_op(self, sync=False, sync_exp_moving_averager=None,variables_to_average=None):
        # init all optimizer in once.....

        optimizer = self.get_optimizer()
        # if self.run_mode == common.MODE_VALUE_SELFTF:
        #     list_other_otimizers = self.get_other_optimizers()
        #
        #     # to init the optimizer variable for later use
        #     for opt in list_other_otimizers:
        #         self._init_train_op(opt, sync, sync_exp_moving_averager)

        # init the optimizer that we currently use
        self._init_train_op(optimizer, sync, sync_exp_moving_averager,
                            variables_to_average=variables_to_average)

    def filter_optimzer_variable(self, var):
        """

        :param tf.Variable var:
        :return:
        """
        if var.op.name in self._list_variable_name_without_optimizer_variable:
            return False
        else:
            return True

    def set_train_op(self, loss=None, sync=False, sync_exp_moving_averager=None,
                        variables_to_average=None):

        if len(self._list_variable_name_without_optimizer_variable) == 0:
            self._list_variable_name_without_optimizer_variable = list(map(
                lambda x: x.op.name,
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

        # Hack here, prevent optimizer static variable (e.g beta_1_power)
        first_var = min(tf.trainable_variables(), key=lambda x: x.name)
        first_var.op._set_device(self.get_master_ps_device_name())

        logging.debug("first_var: %s " % first_var.name)

        assert isinstance(self.global_step, tf.Variable)
        self.global_step.op._set_device(self.get_master_ps_device_name())

        self._set_train_op(loss)
        self.init_train_op(sync=sync,
                           sync_exp_moving_averager=sync_exp_moving_averager,
                           variables_to_average=variables_to_average)

        self.optimizer_variable_list = filter(self.filter_optimzer_variable, tf.global_variables())

        # first_var.op._set_device(org_device)
        return self.train_op

    def get_train_op(self):
        return self.train_op

    def get_batch_size(self):
        return self.conf_dict[selftf.lib.common._batch_size]

    def is_reconfig(self):
        return self.flag_do_live_reconfig

    def is_reconfig2(self):
        return self.flag_do_live_reconfig2

    def done_reconfig1(self):
        self.flag_do_live_reconfig = False

    def get_learning_rate(self):
        return self.conf_dict[selftf.lib.common._learning_rate]

    def get_n_partition(self):
        return 20
        return self.conf_dict[selftf.lib.common._n_partition]

    # def get_parameter_servers(self):
    #     return self.FLAGS.ps_list.split(',')
    #
    # def get_workers(self):
    #     return self.FLAGS.worker_list.split(',')

    def get_nodes(self):
        return self.FLAGS.node_list.split(',')

    def get_tf_cluster_spec(self):
        return tf.train.ClusterSpec({self._job_name: self.get_nodes()})

    def get_tf_variable_global_step(self):
        return self.global_step

    def get_tf_variable_loss(self):
        return self.optimizer_loss_variable

    def device_setter(self):
        assert len(self._list_trainable_variable_name) > 0
        if len(self.variable_map) == 0:
            #  for init step just use original device setter
            return device_setter.replica_device_setter(
                    self.get_num_ps(),
                    worker_device=self._job_prefix + str(self.get_task_index()),
                    ps_device="/job:"+self._job_name,
                    master_ps_device=self.get_master_ps_device_name(),
                    list_tf_name=self._list_trainable_variable_name
                    )

        def chosser_func(op):
            ps_ops = ["Variable", "VariableV2", "VarHandleOp"]
            if op.node_def.op in ps_ops:
                # if op is a Variable
                target_device = self.variable_map.get(op.name)
                # logging.debug("Device setter: assign %s to device: %s" % (op.name, target_device))
                # logging.debug("Device setter, colocate_stack: %s" % (self.get_default_tf_graph()._colocation_stack))
                if target_device is None:
                    # e.g. Global Steps
                    return self.get_master_ps_device_name()
                else:
                    return target_device
            else:

                return self.worker_device
        return chosser_func

    @property
    def worker_device(self):
        return self._job_prefix + str(self.get_task_index())

    def is_collect_statistic_run(self):
        return self.FLAGS.collect_statistic_run

    def get_num_ps(self):
        return self.conf_dict[common._ps_num]

    def get_num_worker(self):
        num_ps = self.get_num_ps()
        return len(self.get_nodes()) - num_ps

    def is_ps(self):
        logging.debug("Check i am ps: task_index:%d, num_ps:%d, is_ps:%s" % (self.get_task_index(), self.get_num_ps(), self.get_task_index() < self.get_num_ps()))
        if self.get_task_index() < self.get_num_ps():
            return True
        else:
            return False

    def is_worker(self):
        if self.get_task_index() >= (len(self.get_nodes()) - self.get_num_worker()):
            return True
        else:
            return False

    def set_graph_init_func(self, func):
        self.graph_init_func = func

        # try here and set the list_trainable_variable
        with self.get_default_tf_graph().as_default():
            self.graph_init_func(self)
            self._list_trainable_variable_name = list(map(
                lambda tv: tv.op.name,
                tf.trainable_variables()
            ))
            self.clear_current_graph()

        # logging.debug("Chris: %s" % str(self._list_trainable_variable_name))


    def clear_current_graph(self):
        # with self.default_tf_graph.as_default():
        #     tf.reset_default_graph()
        self.default_tf_graph = tf.Graph()

    def get_default_tf_graph(self):
        return self.default_tf_graph

    def get_master_ps_device_name(self):
        return self._job_prefix + str(0)

    def get_non_static_ops_from_conf_dict(self):
        if self.conf_dict != None:
            op_names = self.conf_dict.get(common.conf_dict_non_static_ops_names)
            if op_names != None:
                return op_names
        return []

    def get_static_ops(self):
        # Hack use this class
        context = SelfTFOptimizerContext(worker_device=self.worker_device,
                                         master_ps_device=self.get_master_ps_device_name(),
                                         graph=self.default_tf_graph)
        return context.get_static_tf_ops()

    def get_non_static_ops(self):
        # Hack use this class
        context = SelfTFOptimizerContext(worker_device=self.worker_device,
                                         master_ps_device=self.get_master_ps_device_name(),
                                         graph=self.default_tf_graph)
        return context.get_non_static_tf_ops()

    # def reallocate_static_ops(self):
    #     logging.debug("Reallocate static ops to master_ps_device %s" % str(self.get_master_ps_device_name()))
    #     # Hack use this class
    #     context = SelfTFOptimizerContext(worker_device=self.worker_device,
    #                                      master_ps_device=self.get_master_ps_device_name(),
    #                                      graph=self.default_tf_graph)
    #     context.reallocate_static_ops(self.get_non_static_ops_from_conf_dict())

    def set_global_step(self, global_step):
        self.global_step = global_step

    # def set_chief_temp_static_ops(self, param):
    #     """
    #
    #     :param list[tf.Operation] param:
    #     :return:
    #     """
    #     self._temp_static_ops = param
    #
    # def set_chief_temp_nonstatic_ops(self, param):
    #     """
    #     :param list[tf.Operation] param:
    #     :return:
    #     """
    #     self._temp_non_static_op = param

    def init_graph(self):
        self.managed_var = tf.trainable_variables()
        if self.run_mode not in [MODE_VALUE_MLTUNER, MODE_VALUE_DRY_RUN]:
            self.do_live_reconfig2_context = SelfTFOptimizerContext(
                variable_map = self.build_variable_map(),
                worker_device=self._job_prefix + str(self.get_task_index()),
                master_ps_device=self._job_prefix + str(0),
                list_variable_name_without_optimizer_variable=self._list_variable_name_without_optimizer_variable,
                graph=self.default_tf_graph,
                ready_ops_scope_name = self.global_var_ready_op_scope_name,
                is_sync=self._is_sync_optimizer,
                managed_variables=self.managed_var,
                dict_grad_accum_shared_name_op=self.build_dict_grad_accum_shared_name_op())

        self.global_var_ready_op = self.get_defualt_init_ready_op()

    def build_dict_grad_accum_shared_name_op(self):
        if not self._is_sync_optimizer:
            return
        ret = {}
        regex_AccumulatorApplyGradient = re.compile(".*AccumulatorApplyGradient(_\d*)?$")
        regex_take_gradient = re.compile(".*AccumulatorTakeGradient(_\d*)?$")

        grad_accum_ops = list(map(lambda accum: accum[0]._accumulator_ref.op, self.sync_opt._accumulator_list))
        apply_gradient_ops = []
        take_gradient_ops = []
        for op in self.get_default_tf_graph().get_operations():
            op_name = op.name
            m = regex_AccumulatorApplyGradient.match(op_name)
            if m is not None:
                apply_gradient_ops.append(op)
                continue
            m = regex_take_gradient.match(op_name)
            if m is not None:
                take_gradient_ops.append(op)
                continue
        # logging.debug("apply_gradient_ops:%s" % apply_gradient_ops)
        # logging.debug("take_gradient_ops:%s" % take_gradient_ops)

        for accumulator in grad_accum_ops:
            ret[str(accumulator.get_attr("shared_name"))] = (accumulator,
                                                             self.get_corresponding_apply_or_take_ops(accumulator, apply_gradient_ops),
                                                             self.get_corresponding_apply_or_take_ops(
                                                                 accumulator,
                                                                 take_gradient_ops)
                                                             )
        # logging.debug("dict_grad_accum_shared_name_op:%s\n"
        #               % (ret))
        return ret

    def get_corresponding_apply_or_take_ops(self, accumulator, apply_gradient_ops):
        for op in apply_gradient_ops:
            if op.inputs[0].op == accumulator:
                return op
        raise Exception

    def get_tf_run_option(self):
        ret = None

        #Goal, when using seltf, in optimal run
        # if self.flag_do_live_reconfig2 and self.sync_opt and self.do_live_reconfig2_context.is_phase_0():
        #     ret = tf.RunOptions(
        #         timeout_in_ms = int(self.last_iteration_runtime_sec * 1000 * 5)
        #     )

        return ret

    def update_tf_op_timeout(self, second):
        self.tf_run_options.timeout_in_ms = int(second * 1000)

class SelfTFOptimizerContext:
    """
    lifecycle: Per reconfiguration
    Inject
    """
    selftf_keyword = "-selftf_"
    worker_cache_suffix = selftf_keyword+"worker"
    dest_ps_suffix = selftf_keyword+ "dest"
    regex_worker_cache = re.compile("(.*)" + worker_cache_suffix+"$")
    regex_dest_ps = re.compile("(.*)" + dest_ps_suffix + "$")
    regex_get_name_from_variable = re.compile("^(.*):.*$")
    regex_get_name_from_read_ts = re.compile("^(.*)/read:.*")


    def __init__(self,
        variable_map={},
        worker_device="",
        master_ps_device="",
        list_variable_name_without_optimizer_variable=[],
        graph=tf.get_default_graph(),
        ready_ops_scope_name="report_uninitialized_variables",
        is_sync=False,
        dict_grad_accum_shared_name_op={},
        managed_variables={}):

        self.dict_grad_accum_shared_name_op = dict_grad_accum_shared_name_op
        self.managed_variables = managed_variables
        self.is_sync = is_sync
        self.variable_map = variable_map
        self.old_variable_map = {}

        self.ready_ops_scope_name = ready_ops_scope_name

        """
        Reconfig phase
        0 = no reconfiguration
        1 = send ps from old ps to new ps
        2 = dst -> ps
        """
        self._graph = graph

        self.reconfig_phase = 0 # status: 0, 1, 2

        self.worker_device = worker_device

        self.master_ps_device = master_ps_device

        self._list_variable_name_without_optimizer_variable = list_variable_name_without_optimizer_variable

        self._dict_nested_var_of_tv, self.dict_reverse_nested_var_of_tv = self.build_dict_nested_var_of_tv() # this problem

        # logging.debug("dict nested var of tv: %s" % str(self._dict_nested_var_of_tv))

        # self.init_dict_for_a_graph(graph)

        self._apply_ops = tfge.select_ts(".*/update_.*/.*", graph=self._graph)

        #include all replica
        self._wrapper_replica_variables = self._init_wrapper_replica_variables(
            self.managed_variables
        )

        self._dict_var_ops = self.build_dict_var_ops()

        # changes with each reconfiguration
        # subset of self._wrapper_replica_variables
        self.moved_variable_wrapper = None

        if is_sync:
            self.sync_take_gradient_num_required_ops, \
            self.sync_token_enqueue_ops = \
                self.get_sync_ops()
            self.sync_set_global_step_ops = self.get_sync_set_global_step_ops()

            # logging.debug("sync_token_enqueue_ops:%s " % str(self.sync_token_enqueue_ops))

    def get_sync_ops(self):
        ret = []
        sync_token_enqueue_ops = []
        for op in self._graph.get_operations():
            if "TakeGradient/num_required" in op.name:
                ret.append(op)
            if "sync_token_q_EnqueueMany" in op.name:
                assert isinstance(op, tf.Operation)
                sync_token_enqueue_ops.append(op)
        assert len(sync_token_enqueue_ops) > 0
        assert len(ret) > 0
        return ret, sync_token_enqueue_ops

    def update_sync_op_worker_num(self, worker_num):

        self.update_take_graident_num_required(worker_num)

        for op in self.sync_token_enqueue_ops:
            op.inputs[1]._shape_val = tf.TensorShape([worker_num])
            # change the dim input of Fill op
            dim_ts = op.inputs[1].op.inputs[0]
            dim_ts._shape_val = tf.TensorShape([worker_num])
            dim_ts.op._set_attr("value", tf.AttrValue(
                tensor=tf.make_tensor_proto(values=worker_num,shape=[1])))

        for shared_name, ops in self.dict_grad_accum_shared_name_op.items():
            shared_name_suffix = ":0/grad_accum"
            var_name = shared_name[:-len(shared_name_suffix)]

            accumulator = ops[0]  # type: tf.Operation
            apply_op = ops[1]  # type: tf.Operation
            take_op = ops[2]  # type: tf.Operation
            num_required_op = take_op.inputs[1].op  # type: tf.Operation

            device = self.variable_map.get(var_name)
            if device is None:
                logging.debug("accumulator without device:%s ,use master device" % var_name)
                device = self.master_ps_device

            self.reallocate_sync_accum_ops(accumulator,
                                           apply_op,
                                           take_op,
                                           num_required_op,
                                           device,
                                           worker_num,
                                           var_name)


    def _init_wrapper_replica_variables(self, managed_variables):
        list_variable = managed_variables
        list_variable_replica = []
        tv_with_nested_var = self.get_variables_with_nested_variables(list_variable)

        GenericTimer.start("create_worker_cache")
        # create worker cache
        worker_cache_variables_dict = dict(self.create_worker_cache_variables(
            tv_with_nested_var))
        GenericTimer.stop("create_worker_cache")
        # create dst ps variable
        GenericTimer.start("create_dst_ps")
        dst_ps_variable_dict = dict(self.create_dst_ps_variables(tv_with_nested_var))
        GenericTimer.stop("create_dst_ps")

        # create replica objects
        GenericTimer.start("create_replica_objects")
        for v in list_variable:
            # find worker_cache
            worker_cache, v_type = worker_cache_variables_dict[v]
            # dst_ps
            dst_ps_variable, dup_v_type = dst_ps_variable_dict[v]

            list_variable_replica.append(VariableReplica(v, worker_cache, dst_ps_variable, v_type))

            # create replica tv opt
            for opt_var in self.get_variables_under_trainable_variable(v):
                # worker_cache
                worker_cache, v_type = worker_cache_variables_dict[opt_var]
                # dst_ps
                dst_ps_variable, dup_v_type = dst_ps_variable_dict[opt_var]

                list_variable_replica.append(VariableReplica(opt_var, worker_cache, dst_ps_variable, VariableReplica.V_TYPE_OPTIMIZER_VARIABLE_IN_TRAINABLE_VARIABLE))
        GenericTimer.stop("create_replica_objects")
        return VariableReplicaWrapper(list_variable_replica)

    def phase1_update_graph(self, variable_map, worker_num):
        #phase 1
        self.old_variable_map = self.variable_map
        self.variable_map = variable_map

        self._list_moved_variable = self._get_list_moved_variable()
        list_tv_with_nested_var = self.get_variables_with_nested_variables(self._list_moved_variable)

        # logging.debug("_list_moved_variable:%s " % (list_tv_with_nested_var))

        # reallocate dst_ps
        list_moved_variable_replica = []
        for var in list_tv_with_nested_var:
            assert isinstance(self._wrapper_replica_variables,
                              VariableReplicaWrapper)
            replica = self._wrapper_replica_variables.get_replica_by_org_v(var)
            device = self.get_device_name_by_v(var)

            self._reallocate_variable(replica.dst_ps_v, device)

            if self.is_sync:
                self._reallocate_variable_grad_accum(replica.original_v, device, worker_num)

            # reallocate corresponding dst_ps_assign op
            replica.assign_op_wc_dst_ps.op._set_device(device)

            list_moved_variable_replica.append(replica)

        self.moved_variable_wrapper = VariableReplicaWrapper(list_moved_variable_replica)

        self.update_input_of_gradient_op(self.moved_variable_wrapper)

        self.reallocate_apply_grad_op(self._list_moved_variable)

        self.update_apply_ops_inputs_phase1(self.moved_variable_wrapper)

        # TODO: assign op for trainable variable direct ps -> ps
        if self.is_sync:
            # TOOD: for sync opt update ConditionalAccumulator num_required
            # self.update_sync_op_worker_num(worker_num)
            self.update_sync_op_worker_num(worker_num)


    def phase2_update_graph(self):
        assert isinstance(self.moved_variable_wrapper, VariableReplicaWrapper)
        # reverse update_input_of_gradient_op
        for replica in self.moved_variable_wrapper.list_variable_replica:
            if replica.var_type == VariableReplica.V_TYPE_OPTIMIZER_VARIABLE_IN_TRAINABLE_VARIABLE:
                continue
            device = self.get_device_name_by_v(replica.original_v)
            self._reallocate_variable(replica.original_v, device)
            # self.update_input_of_gradient_op(self.moved_variable_wrapper)
            tfge.swap_outputs(self.get_read_ts_from_variable(replica.original_v),
                              self.get_read_ts_from_variable(replica.worker_cache_v))

        self.update_apply_ops_inputs_phase2(self.moved_variable_wrapper)

            # update location of is_variable_initialize
    #         self
    #
    # def update_check_init_ops(self):
    #     with tf.variable_scope(self.ready_ops_scope_name, reuse=True) as scope:
    #         scope.
    #

    @classmethod
    def get_v_name_by_v_ref_ts(cls, tf):
        """
        :param tf.Tensor tf:
        :return:
        """
        m = cls.regex_get_name_from_variable.match(tf.name)
        if m is None:
            return None
        else:
            return m.group(1)

    def get_device_name_by_variable(self, variable):
        if isinstance(variable, str):
            return self.variable_map[variable]
        else:
            return self.variable_map[variable.op.name]

    def is_reconfig(self):
        return self.reconfig_phase != 0

    def get_worker_device_name(self):
        return self.worker_device + "/cpu:0"

    def get_worker_variable_cache_name(self, v):
        """
        :param tensorflow.Variable v:
        :return:
        """
        if isinstance(v, str):
            return v + SelfTFOptimizerContext.worker_cache_suffix
        else:
            return v.op.name + SelfTFOptimizerContext.worker_cache_suffix

    def get_dest_ps_variable_name(self, v):
        """
        :param tensorflow.Variable v:
        :return:
        """
        if isinstance(v, str):
            return v + SelfTFOptimizerContext.dest_ps_suffix
        else:
            return v.op.name + SelfTFOptimizerContext.dest_ps_suffix

    def get_variable_name_from_worker_cache(self, v):
        """
        :param tensorflow.Variable v:
        :return:
        """
        m = SelfTFOptimizerContext.regex_worker_cache.match(v.op.name)
        return m.group(1)

    def get_variable_name_from_dest_ps_variable(self, v):
        """
        :param tensorflow.Variable v:
        :return:
        """
        m = SelfTFOptimizerContext.regex_dest_ps.match(v.op.name)
        return m.group(1)

    def swap_src_ps_with_worker_cache(self, replica_v):
        """
        :param list[tensorflow.Variable] replica_v:
        :return:
        """
        graph = tf.get_default_graph()
        for v in replica_v:
            src_v = tfge.select_ts(v.op.name + "/read", graph=graph)
            worker_cache_v = tfge.select_ts(self.get_worker_variable_cache_name(v)+"/read",graph=graph)
            tfge.swap_ts(src_v, worker_cache_v)

    @staticmethod
    def get_name_scope_from_variable(v):
        if not isinstance(v, tf.Variable):
            raise TypeError()
        regex = re.compile("^(.*):.*$")
        m = regex.match(v.name)
        return m.group(1)

    # @staticmethod
    # def get_view_from_variable(v, graph=tf.get_default_graph()):
    #     if not isinstance(v, tf.Variable):
    #         raise TypeError()
    #     name_scope = SelfTFOptimizerContext.get_name_scope_from_variable(v)
    #     return tfge.make_view_from_scope(name_scope, graph)
    #


    @classmethod
    def get_read_ts_from_variable(cls, v):
        graph = tf.get_default_graph()
        if not isinstance(v, tf.Variable):
            raise TypeError()
        v_name = cls.get_name_from_variable(v)
        # ret = tfge.select_ts(v_name+"/read", graph=graph)
        return tf.get_default_graph().get_tensor_by_name(v_name+"/read:0")

    @classmethod
    def get_v_name_from_read_ts(cls, v):
        m = cls.regex_get_name_from_read_ts.match(v.name)
        if m is not None:
            return m.group(1)
        else:
            return None

    def get_parent_scope_name(self, vname):
        idx = vname.rfind("/")
        if idx == -1:
            return None
        else:
            return vname[:idx]

    def get_device_name_by_v(self, search_v, reversed=False):
        """
        Get device include the nest var
        :param tf.Variable v:
        :return:
        """
        v_name = SelfTFOptimizerContext.get_name_scope_from_variable(search_v)
        v = self._recursive_find_device_name(v_name)
        if v is None and not reversed:
            return self.get_device_name_by_v(
                self.dict_reverse_nested_var_of_tv[search_v]
            )
        if v is None:
            raise Exception("can't get the name: %s" % v_name)
        return v+"/cpu:0"

    def _recursive_find_device_name(self, v_name):
        device = self.variable_map.get(v_name)
        if device is None:
            parent_name = self.get_parent_scope_name(v_name)
            if parent_name is None:
                return None
            else:
                return self._recursive_find_device_name(parent_name)
        else:
            return device

    def create_dst_ps_variables(self, var_list):
        """
        :param list[tensorflow.Variable] var_list:
        :rtype: list[tuple[tf.Variable, (replica_v, str)]]
        """
        ret = []
        for v in var_list:
            # logging.debug("Create dst ps: %s device: %s" % (v.op.name, self.get_device_name_by_v(v)))
            ret.append(self.create_variable_replica(v, self.get_device_name_by_v(v), self.get_dest_ps_variable_name(v)))
        return ret

    def create_worker_cache_variables(self, var_list):
        """
        :param list[tensorflow.Variable] var_list:
        :return:
        """
        ret = []
        # with tf.device(self.get_worker_device_name()):
        #     for v in var_list:
        #         variables_include_optimizer_variables = self.get_variables_under_trainable_variable(v)
        #         # for vars in
        #         replica_v = tf.Variable(initial_value=tf.zeros(v.shape),
        #                                 name=self.get_worker_variable_cache_name(v),
        #                                 trainable=False
        #                                 )
        #         ret.append(replica_v)
        # return ret
        for v in var_list:
            ret.append(self.create_variable_replica(v, self.get_worker_device_name(), self.get_worker_variable_cache_name(v),
                                                    collections=[tf.GraphKeys.LOCAL_VARIABLES]))
        return ret

    def create_variable_replica(self, v, device_name, replica_name,
        replica_type="", collections=[tf.GraphKeys.GLOBAL_VARIABLES]):
        """
        Chris: Checked
        :param tf.Variable v:
        :param str device_name:
        :rtype:tuple[tf.Variable, (replica_v, str)]
        """
        if replica_type == "":
            replica_type = VariableReplica.V_TYPE_TRAINABLE_VARIABLE
        # logging.debug("create_variable_replica: %s, device: %s" % (replica_name, device_name))
        with tf.device(device_name):
            replica_v = tf.Variable(initial_value=tf.zeros(v.shape),
                                    name=replica_name,
                                    trainable=False,
                                    collections=collections
                                    )

            replica_v.op._set_device(device_name)
            assert replica_v.device == device_name
            return v, (replica_v, replica_type)
            # for opt_var in optimizer_variables:
            #     # extract the variab`le name of the last part e.g hid/part-0/Adam -> Adam
            #     # concat it with the target v name e.g hid/part-0-selftf_worker_cache/Adam
            #     variable_name = self.get_name_from_variable(v)
            #     opt_var_name = self.get_name_from_variable(opt_var)[len(variable_name)+1:]
            #     new_replica_name = replica_name+"/"+opt_var_name
            #     replica_opt_var = tf.Variable(initial_value=opt_var.initial_value,
            #                                   name=new_replica_name,
            #                                   trainable=False)
            #     ret.append((opt_var, (replica_opt_var, VariableReplica.V_TYPE_OPTIMIZER_VARIABLE_IN_TRAINABLE_VARIABLE
    #
    def get_variables_with_nested_variables(self, list_variable):
        """
        :rtype: list[tf.Variable]
        :return:
        """
        ret = []
        ret.extend(list_variable)
        ret.extend(self.get_nested_variables(list_variable))

        if len(ret) > len(set(ret)):
            raise Exception("list variable is not unique")
        return ret

    def get_nested_variables(self, list_variable):
        ret = []
        for var in list_variable:
            list_opt_var = self._dict_nested_var_of_tv.get(var)
            if list_opt_var is None:
                continue
            ret.extend(list_opt_var)
        return ret

    #
    # def get_tv_with_nested_variable(self):
    #     """
    #     :rtype: list[tf.Variable]
    #     :return:
    #     """
    #     return self.get_variables_with_nested_variables(tf.trainable_variables)

    # def get_variables_with_nested_variable(self, var_list):
    #     ret = []
    #     ret.extend(var_list)
    #     for v, opt_v in self.get_nested_variables(var_list):
    #         ret.append(opt_v)
    #     return ret

    # def get_nested_variables(self, var_list):
    #     """
    #     :param var_list:
    #     :rtype: list[tuple[tf.Variable, tf.Variable]]
    #     """
    #     ret = []
    #     for v in var_list:
    #         for opt_v in self.get_variables_under_trainable_variable(v):
    #             ret.append((v, opt_v))
    #     return ret

    def _get_list_moved_variable(self):
        return self.get_list_variable_by_name(
            self.diff_variable_map(self.variable_map, self.old_variable_map))

    @property
    def list_moved_variable(self):
        return self._list_moved_variable

    # def get_optimizer_variable_in_trainable_variable(self):
    #     """
    #     :rtype: list[tf.Variable]
    #     :return:
    #     """
    #     return map(lambda (v, opt_v): opt_v, self.get_nested_variables(self.get_trainable_variables()))

    def diff_variable_map(self, new_variable_map, old_variable_map):
        assert len(new_variable_map) == len(old_variable_map)
        list_diff_variable_name = []
        for var_name in new_variable_map.keys():
            if new_variable_map[var_name] != old_variable_map[var_name]:
                list_diff_variable_name.append(var_name)

        return list_diff_variable_name

    def get_list_variable_by_name(self, list_variable_name):
        ret = []
        for v in self._graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert isinstance(v, tf.Variable)
            if v.op.name in list_variable_name:
                ret.append(v)

        return ret

    def create_moved_trainable_variable_replica_wrapper(self, list_variable):
        """
        Chris: Checked
        Return a list of variable replica include the nested variable
        i.e. the optimizer varialbe
        :return VariableReplicaWrapper:
        """
        list_variable_replica = []
        tv_with_nested_var = self.get_variables_with_nested_variables(list_variable)

        GenericTimer.start("create_worker_cache")
        # create worker cache
        worker_cache_variables_dict = dict(self.create_worker_cache_variables(
            tv_with_nested_var))
        GenericTimer.stop("create_worker_cache")
        # create dst ps variable
        GenericTimer.start("create_dst_ps")
        dst_ps_variable_dict = dict(self.create_dst_ps_variables(tv_with_nested_var))
        GenericTimer.stop("create_dst_ps")

        # create replica objects
        GenericTimer.start("create_replica_objects")
        for v in list_variable:
            # find worker_cache
            worker_cache, v_type = worker_cache_variables_dict[v]
            # dst_ps
            dst_ps_variable, dup_v_type = dst_ps_variable_dict[v]

            list_variable_replica.append(VariableReplica(v, worker_cache, dst_ps_variable, v_type))

            # create replica tv opt
            for opt_var in self.get_variables_under_trainable_variable(v):
                # worker_cache
                worker_cache, v_type = worker_cache_variables_dict[opt_var]
                # dst_ps
                dst_ps_variable, dup_v_type = dst_ps_variable_dict[opt_var]

                list_variable_replica.append(VariableReplica(opt_var, worker_cache, dst_ps_variable, VariableReplica.V_TYPE_OPTIMIZER_VARIABLE_IN_TRAINABLE_VARIABLE))
        GenericTimer.stop("create_replica_objects")
        return VariableReplicaWrapper(list_variable_replica)

    def get_trainable_variables(self):
        """
        :rtype: list[tf.Variable]
        :return:
        """
        return tf.trainable_variables()

    def get_var_names(self, var_list):
        ret = []
        for v in var_list:
            ret.append(self.get_name_from_variable(v))
        return ret

    def get_all_grad_apply_ts(self, training_vars):
        """
        checked
        :param training_vars:
        :rtype: dict[v_name, list[tf.Tensor]]
        """
        ret_ops = {}
        for var in training_vars:
            var_name = self.get_name_from_variable(var)
            ret_ops[var_name] = self.get_grad_apply_ts(var)
        return ret_ops

    def get_grad_apply_ts(self, var):
        var_name = self.get_name_from_variable(var)
        return filter(lambda op: "update_"+var_name+"/" in op.name, self._apply_ops)

    def get_all_grad_apply_ops(self, training_vars):
        """
        :param list[tf.Variable] training_vars:
        :rtype:list[tf.Operation]
        """
        ret = []
        for list_ts in self.get_all_grad_apply_ts(training_vars).values():
            ret.extend(list(map(lambda x: x._op, list_ts)))
        return ret

    def reallocate_apply_grad_op(self, training_vars):
        if self.is_sync:
            return
        list_grad_apply_ops = self.get_all_grad_apply_ts(training_vars)
        for var_name, list_grad_apply_ops in list_grad_apply_ops.items():
            for grad_apply_ops in list_grad_apply_ops:
                device_name = self.variable_map[var_name]
                grad_apply_ops._op._set_device(device_name)

    def get_variables_under_trainable_variable(self, v):
        """
        Get all variable include optimizer variables
        :param tf.Variable v:
        :rtype: list[tf.Variable]
        """
        return self._dict_nested_var_of_tv[v]

    def build_dict_nested_var_of_tv(self):
        """
        Checked
        :return:
        """
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        tvs = self.managed_variables
        ret = {}
        ret2 = {}
        for tv in tvs:
            ret[tv] = []

        for var in all_vars:
            var_name = var.op.name
            for tv in ret.keys():
                x = tv.op.name
                if x == var:
                    continue
                if x+"/" in var_name:
                    ret[tv].append(var)
                    ret2[var] = tv
        # logging.debug("build_dict_nested_var_of_tv: %s" % str(ret))
        return ret, ret2

    def build_dict_var_ops(self):
        """
        checked
        :return:
        """
        ret = {}
        graph = tf.get_default_graph()
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in all_vars:
            ret[v] = tfge.select_ops("^"+v.op.name+"/.*$", graph=graph) + [v.op]
        return ret

    # def build_dict_tv_nest_var(self):
    #     ret = {}
    #     graph = tf.get_default_graph()
    #     tvs = tf.trainable_variables()
    #
    #     for tv in tvs:
    #         ret[tv] = []
    #
    #     for op in graph.get_operations():
    #         if op.type == "Variable" or op.type == "VariableV2":
    #             var_name = op.name
    #             for tv in ret.keys():
    #                 tv_name = self.get_name_from_variable(tv)
    #                 if var_name == tv_name:
    #                     break
    #                 if tv_name in var_name:
    #                     ret[tv_name].append(op)
    #
    #     return ret

    @classmethod
    def get_name_from_variable(cls, v):
        """
        :param tf.Variable v:
        :return:
        """
        return cls.regex_get_name_from_variable.match(v.name).group(1)

    def get_variable_by_name(self, name):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in var_list:
            if self.get_name_from_variable(v) == name:
                return v

        raise Exception("Unknown variable")

    def update_apply_ops_inputs_phase1(self, replica_wrapper):
        """
        :param VariableReplicaWrapper replica_wrapper:
        :return:
        """
        list_moved_training_variable = replica_wrapper.get_list_tv()
        apply_ops = self.get_all_grad_apply_ops(list_moved_training_variable)

        for op in apply_ops:
            self.update_apply_op_phase1(op, replica_wrapper)

    def update_apply_ops_inputs_phase2(self, replica_wrapper):
        """
        :param VariableReplicaWrapper replica_wrapper:
        :return:
        """
        list_moved_training_variable = replica_wrapper.get_list_tv()
        apply_ops = self.get_all_grad_apply_ops(
            list_moved_training_variable)

        for op in apply_ops:
            self.update_apply_op_phase2(op, replica_wrapper)

    def update_apply_op_phase1(self, apply_op, replica_wrapper):
        """
        :param tf.Operation apply_op:
        :param replica_wrapper:
        :return:
        """
        list_idx_ts = []
        for idx, input in enumerate(apply_op.inputs):
            replica = replica_wrapper.get_replica_by_v_ref_tf(input)
            if replica is not None:
                # replace the input later
                list_idx_ts.append((idx, replica.dst_ps_v))

        for idx, v in list_idx_ts:
            v_ref = v._ref()
            apply_op._update_input(idx, v_ref)

    def update_apply_op_phase2(self, apply_op, replica_wrapper):
        """
        :param tf.Operation apply_op:
        :param replica_wrapper:
        :return:
        """
        list_idx_ts = []
        for idx, input in enumerate(apply_op.inputs):
            replica = replica_wrapper.get_replica_by_v_dst_ps_ref_tf(input)
            if replica is not None:
                # replace the input later
                list_idx_ts.append((idx, replica.original_v))

        for idx, v in list_idx_ts:
            v_ref = v._ref()
            apply_op._update_input(idx, v_ref)

    def update_device_for_optimizer_variables_in_tv(self):
        """
        tv mean trainable variable
        :return:
        """
        for tv, list_opt_v in self._dict_nested_var_of_tv.items():
            device = self.variable_map[self.get_name_from_variable(tv)]
            for opt_v in list_opt_v:
                self._reallocate_variable(opt_v, device)

    # def get_static_optimizer_variables(self):
    #     """
    #     :rtype: list[tf.Variable]
    #     :return:
    #     """
    #     var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #
    #     # remove trainable and opt_var_in_tv
    #     for x in self.get_variables_with_nested_variable(self.get_trainable_variables()):
    #         var_list.remove(x)
    #
    #     # remove its replica
    #     # for x in wrapper_replica_variables:
    #     #     var_list.remove(x.dst_ps_v)
    #     #     var_list.remove(x.work_cache_v)
    #
    #     return var_list

    def get_optimizer_variables(self):
        """
        :rtype: list[tf.Variable]
        :return:
        """
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        set_pure_variable_name = set(self._list_variable_name_without_optimizer_variable)
        logging.debug("set_pure_variable_name: %s " % str(set_pure_variable_name) )
        def _filter(var):
            """
            :param tf.Variable var:
            :return:
            """
            if var.op.name in set_pure_variable_name:
                return False
            if SelfTFOptimizerContext.selftf_keyword in var.op.name:
                return False
            return True

        ret = filter(_filter, var_list)
        logging.info("get_optimizer_variables: %s" % str(ret))
        return ret


    def get_tv_prefix(self):
        """
        Checked
        :rtype: list[str]
        :return:
        """
        ret = []
        for x in self.get_trainable_variables():
            ret.append(x.op.name)
        return ret

    @classmethod
    def clear_collocate(cls, op):
        """
        :param tf.Operation op:
        :return:
        """
        try:
            org_attr_value = op.get_attr("_class")
            filtered_class_attr_value = filter(lambda class_name:not class_name.startswith(b"loc:@"), org_attr_value)

            # op.node_def.attr["_class"].Clear()
            # op.node_def.attr["_class"].CopyFrom(tf.attr_value_pb2.AttrValue(
            #     list=tf.attr_value_pb2.AttrValue.ListValue(s=filtered_class_attr_value)))
            op._set_attr("_class", tf.AttrValue(
                list=tf.AttrValue.ListValue(
                    s=filtered_class_attr_value)
            ))
        except ValueError as e:
            pass
        except:
            logging.exception("Clear colocation fail")
            pass

    @classmethod
    def clear_all_ops_collocate(cls):
        for x in tf.get_default_graph().get_operations():
            cls.clear_collocate(x)

    # def reallocate_static_ops(self, non_static_op_names=[]):
    #     """
    #     Checked
    #     :return:
    #     """
    #     static_ops = []
    #     if len(non_static_op_names) > 0:
    #         set_non_static_op_names = set(non_static_op_names)
    #         for op in tfge.select_ops(".*", graph=tf.get_default_graph()):
    #             if op.name not in set_non_static_op_names:
    #                 static_ops.append(op)
    #     else:
    #         static_ops = self.get_static_tf_ops()
    #
    #     for op in static_ops:
    #         op._set_device(self.master_ps_device)
    #     # temp = tf.get_collection(tf.GraphKeys.TRAIN_OP)
    #     # var_list = self.get_static_optimizer_variables()
    #     #
    #     # for v in var_list:
    #     #     self._reallocate_variable(v, self.master_ps_device)

    def _reallocate_variable(self, v, device):
        """
        :param tf.Variable v:
        :return:
        """
        ops = self._dict_var_ops[v]
        for op in ops:
            op._set_device(device)

    def update_input_of_gradient_op(self, wrapper_replica_variables):
        """

        :param VariableReplicaWrapper wrapper_replica_variables:
        :return:
        """
        for replica in wrapper_replica_variables.list_variable_replica:
            if replica.var_type == VariableReplica.V_TYPE_OPTIMIZER_VARIABLE_IN_TRAINABLE_VARIABLE:
                continue
            v = replica.original_v
            wc_v = replica.worker_cache_v

            wc_read_ts = self.get_read_ts_from_variable(wc_v)
            org_read_ts = self.get_read_ts_from_variable(v)
            # when calculate graident, read wight values from worker local cache
            tfge.swap_outputs(wc_read_ts,
                              org_read_ts)

            # preserve assign_op
            wc_assgin_op = replica.assign_to_worker_cache().op
            assert isinstance(wc_assgin_op, tf.Operation)
            wc_assgin_op._update_input(1, org_read_ts)

    def modify_graph_for_reconfig(self):
        """
        :param SelfTFOptimizerContext context:
        :return:
        """
        # Start Reconfig
        # # Reconfig op
        list_moved_tv = self._list_moved_variable

        # For Static optimizer variables
        # self.reallocate_static_ops()
        # END For static optimizer variables

        # For Trainable Variable
        GenericTimer.start("create_trainable_variable_replica_wrapper")
        wrapper_replica_variables = self.create_moved_trainable_variable_replica_wrapper(
            list_moved_tv
        )
        self._wrapper_replica_variables = wrapper_replica_variables
        a = set(list_moved_tv)
        b = set(wrapper_replica_variables.get_list_tv())
        assert a==b

        GenericTimer.stop("create_trainable_variable_replica_wrapper")
        # replace read_op
        GenericTimer.start("update_input_of_gradient_op")
        self.update_input_of_gradient_op(wrapper_replica_variables)
        GenericTimer.stop("update_input_of_gradient_op")
        # END For Trainable Variable
        GenericTimer.start("update_apply_ops_inputs")
        self.update_apply_ops_inputs_phase1(wrapper_replica_variables)
        GenericTimer.stop("update_apply_ops_inputs")
        # change apply_op to new device
        GenericTimer.start("reallocate_apply_grad_op")
        self.reallocate_apply_grad_op(wrapper_replica_variables.get_list_tv())
        GenericTimer.stop("reallocate_apply_grad_op")
        GenericTimer.start("update_device_for_optimizer_variables_in_tv")
        self.update_device_for_optimizer_variables_in_tv()
        GenericTimer.stop("update_device_for_optimizer_variables_in_tv")
        # END For Optimizer variables in Trainable Variable

        # GenericTimer.start("clear_all_ops_collocate")
        self.clear_all_ops_collocate()
        # GenericTimer.stop("clear_all_ops_collocate")

        # GenericTimer.start("create_assign_ops")
        worker_cache_assign_ops = wrapper_replica_variables.get_assign_op_to_worker_cache()
        dst_ps_variable_assign_ops = wrapper_replica_variables.get_assign_op_to_dst_ps()
        random.shuffle(worker_cache_assign_ops)
        random.shuffle(dst_ps_variable_assign_ops)
        # GenericTimer.stop("create_assign_ops")

        # opt_init_op = tf.variables_initializer(
        #     self.get_optimizer_variables()+wrapper_replica_variables.get_init_vars())
        opt_init_op = None

        # END Reconfig
        return worker_cache_assign_ops, dst_ps_variable_assign_ops, opt_init_op



    # def replica_trainable_variables_with_optimizer(self):
    #     ret = []
    #     var_list = tf.trainable_variables()
    #     for var in var_list:
    #         ret.extend(self.replica_trainable_variable(var))
    #     return VariableReplicaWrapper(ret)
    #
    # def replica_trainable_variable(self, v):
    #     """
    #     Replica self and nested, return a list of {VariableReplica}
    #     :param self:
    #     :param tf.Variable v:
    #     :return:
    #     """
    #
    #
    # def replica_variable(self, v, device):
    def start_reconfig(self):
        self.reconfig_phase = 1

    def finish_phase1(self):
        if self.reconfig_phase == 1:
            self.reconfig_phase = 2
        else:
            raise RuntimeError("Unknow phase change")

    def finish_phase2(self):
        if self.reconfig_phase == 2:
            self.reconfig_phase = 0
        else:
            raise RuntimeError("Unknow phase change")

    def is_phase_0(self):
        return self.reconfig_phase == 0

    def is_phase_1(self):
        return self.reconfig_phase == 1

    def is_final_phase(self):
        return self.reconfig_phase == 2

    def get_static_tf_ops(self):
        ops = tfge.select_ops(".*", graph=tf.get_default_graph())
        ts_variable_op_prefixs = self.get_tv_prefix()

        def is_trainable_variable_ops(op):
            for tv_prefix in ts_variable_op_prefixs:
                if tv_prefix in op.name:
                    return True
            return False

        def ops_filter(op):
            """
            :param tf.Operation op:
            :rtype: bool
            """
            # remove all ops with worker
            if self.worker_device in op.device:
                return False
            # remove all ops related to trainable variables
            if is_trainable_variable_ops(op):
                return False
            return True

        static_ops = filter(ops_filter, ops)

        logging.debug("static ops: %s" % str(static_ops))
        return static_ops

    def get_non_static_tf_ops(self):
        ops = tfge.select_ops(".*", graph=tf.get_default_graph())
        ts_variable_op_prefixs = self.get_tv_prefix()

        def is_trainable_variable_ops(op):
            for tv_prefix in ts_variable_op_prefixs:
                if tv_prefix in op.name:
                    return True
            return False

        def ops_filter(op):
            """
            :param tf.Operation op:
            :rtype: bool
            """
            # remove all ops with worker
            if self.worker_device in op.device:
                return True
            # remove all ops related to trainable variables
            if is_trainable_variable_ops(op):
                return True
            return False

        static_ops = filter(ops_filter, ops)

        # logging.debug("non static ops: %s" % str(static_ops))
        return static_ops

    def rellocate_tv_to_old_map(self, variable_map):
        logging.debug("rellocate_tv_to_old_map")
        for x in tf.trainable_variables():
            # get all op of tv
            # logging.debug("reallocate %s" % x.op.name)
            ops = self._dict_var_ops[x]
            old_device = variable_map[x.op.name]
            # logging.debug("reallocate ops: %s device: %s" % (str(ops), old_device))
            for op in [x.op] + ops:
                op._set_device(old_device)

    def get_worker_cache_var(self):
        return list(map(lambda replica: replica.worker_cache_v,
                   self._wrapper_replica_variables.list_variable_replica))

    def reallocate_sync_accum_ops(self, accumulator, apply_op, take_op, num_required_op, device, worker_num, var_name):
        # logging.debug(
        #     "_reallocate_variable_grad_accum %s\n accumulator:%s \n apply_grad_op: %s\n take_op: %s" % (
        #     var_name,
        #     accumulator,
        #     apply_op,
        #     take_op))
        accumulator._set_device(device)
        apply_op._set_device(device)
        take_op._set_device(device)
        num_required_op._set_device(device)

        num_required_op._set_attr("value",
                                  tf.AttrValue(
                                      tensor=tf.make_tensor_proto(
                                          values=worker_num)))

        assert isinstance(take_op, tf.Operation)
        take_op._update_input(0, accumulator._outputs[0])

        # update the setGlobalStep op3
        self.update_sync_op_set_global_step_op(accumulator, device)

    def _reallocate_variable_grad_accum(self, original_v, device, worker_num):
        shared_name = "%s:0/grad_accum" % original_v.op.name
        tuple = self.dict_grad_accum_shared_name_op.get(shared_name)
        if tuple is not None:
            accumulator = tuple[0]  # type: tf.Operation
            apply_op = tuple[1]     # type: tf.Operation
            take_op = tuple[2]      # type: tf.Operation
            num_required_op = take_op.inputs[1].op  # type: tf.Operation
            self.reallocate_sync_accum_ops(accumulator,
                                           apply_op,
                                           take_op,
                                           num_required_op,
                                           device,
                                           worker_num,
                                           shared_name)



    def update_sync_op_set_global_step_op(self, accumulator, device):
        for op in self.sync_set_global_step_ops:
            if op.inputs[0].op == accumulator:
                op._set_device(device)
                return
        raise Exception

    def get_sync_set_global_step_ops(self):
        ret = []
        for op in self._graph.get_operations():
            if "SetGlobalStep" in op.name:
                ret.append(op)
        return ret

    def update_take_graident_num_required(self, worker_num):
        for t in self.dict_grad_accum_shared_name_op.values():
            take_op = t[2]
            num_required_op = take_op.inputs[1].op
            num_required_op._set_attr("value",
                                      tf.AttrValue(
                                          tensor=tf.make_tensor_proto(
                                              values=worker_num)))



class VariableReplicaWrapper(object):
    def __init__(self, list_variable_replica):
        """
        :param list[VariableReplica] list_variable_replica:
        """
        self._list_variable_replica = list_variable_replica
        self._dict_variable_replica = self.build_variable_dict(list_variable_replica)
        self.assign_op_to_worker_cache = []
        self.assign_op_to_dist_ps_v = []
        self._init_var_list = []

    def exist_var(self, v):
        """
        :param tf.Variable v:
        :return:
        """
        return self._dict_variable_replica.has_key(v)

    def get_replica_by_v_dst_ps_ref_tf(self, v_ref_ts):
        v_name = SelfTFOptimizerContext.get_v_name_by_v_ref_ts(v_ref_ts)
        if v_name is None:
            return None
        for v in self._list_variable_replica:
            if v.dst_ps_v.op.name == v_name:
                return v
        return None

    def get_replica_by_v_ref_tf(self, v_ref_ts):
        """
        Get var of read_ts
        :return:
        """
        v_name = SelfTFOptimizerContext.get_v_name_by_v_ref_ts(v_ref_ts)
        if v_name is None:
            return None
        for v in self._list_variable_replica:
            if v.original_v.op.name == v_name:
                return v
        return None

    def get_replica_by_v_ref_ts(self,v_ref_ts):
        """
        Get var of read_ts
        :return:
        """
        v_name = SelfTFOptimizerContext.regex_get_name_from_variable.match(v_ref_ts.name).group(1)
        if v_name is None:
            return None
        for v in self._list_variable_replica:
            if v.original_v.op.name == v_name:
                return v
        return None


    def build_variable_dict(self, list_variable_replica):
        """
        :param list[VariableReplica] list_variable_replica:
        :rtype: dict[tf.Variable, VariableReplica]
        """
        ret = {}
        for x in list_variable_replica:
            ret[x.original_v] = x
        return ret

    @property
    def list_variable_replica(self):
        """
        :rtype: list[VariableReplica]
        """
        return self._list_variable_replica

    def get_assign_op_to_worker_cache(self):
        return list(map(lambda a: a.assign_to_worker_cache(),
                   self._list_variable_replica))

    def get_assign_op_to_dst_ps(self):
        return list(map(lambda a:a.assign_to_dist_ps_v(), self._list_variable_replica))

    def get_assign_ops_dst_ps_to_org_v(self):
        return list(map(lambda a:a.get_dst_ps_to_org_assign_op(), self._list_variable_replica))


    def get_dst_ps_v_by_trainable_variable(self, v):
        return self._dict_variable_replica[v].dst_ps_v

    def get_worker_cache_v_by_trainable_variable(self, v):
        return self._dict_variable_replica[v].worker_cache_v

    def _create_assign_op_to_woker_cache_dst_ps(self):
        assert len(self.assign_op_to_worker_cache) == 0
        assert len(self.assign_op_to_dist_ps_v) == 0

        for x in self.list_variable_replica:
            # if x.var_type == VariableReplica.V_TYPE_OPTIMIZER_VARIABLE_IN_TRAINABLE_VARIABLE:
            #     logging.debug("create tv_ops init_op: %s " % x.original_v.op.name)
            #     self._init_var_list.append(x.dst_ps_v)
            #     continue
            op_assign_to_w_c = x.assign_to_worker_cache()
            op_assign_to_dst_ps = x.assign_to_dist_ps_v()

            self.assign_op_to_worker_cache.append(op_assign_to_w_c)
            self.assign_op_to_dist_ps_v.append(op_assign_to_dst_ps)

    def get_init_vars(self):
        return self._init_var_list

    def get_list_tv(self):
        """
        :rtype: list[tf.Variable]
        :return:
        """
        ret = []
        for replica in self._list_variable_replica:
            if replica.var_type == VariableReplica.V_TYPE_TRAINABLE_VARIABLE:
                ret.append(replica.original_v)
        return ret

    def get_replica_by_org_v(self, v):
        return self._dict_variable_replica[v]

class VariableReplica(object):

    V_TYPE_TRAINABLE_VARIABLE = "TRAINABLE_VARIABLE"
    V_TYPE_OPTIMIZER_VARIABLE_IN_TRAINABLE_VARIABLE = "OPTIMIZER_VARIABLE_IN_TRAINABLE_VARIABLE"
    V_TYPE_OPTIMIZER_VARIABLE = "OPTIMIZER_VARIABLE"
    var_type = [""]
    """
    A container to wrap a variable and the corresponding nested
    """
    def __init__(self, original_v, worker_cache_v, dst_ps_v, var_type):
        """
        :param tf.Variable original_v:
        :param tf.Variable worker_cache_v:
        :param tf.Variable dst_ps_v:
        """
        self._original_v = original_v
        self._worker_cache_v = worker_cache_v
        self._dst_ps_v = dst_ps_v
        self._var_type = var_type
        self._worker_cache_assign_op = tf.assign(self.worker_cache_v, self.original_v)


        self._dst_ps_assign_op = tf.assign(self.dst_ps_v, self.worker_cache_v)
        self._dst_ps_assign_op_with_cond = tf.cond(
            tf.is_variable_initialized(dst_ps_v),
            lambda: tf.no_op(),
            lambda: tf.group([self._dst_ps_assign_op])
        )

        self._dst_ps_to_org_assign_op = tf.assign(self.original_v, self.dst_ps_v)

    @property
    def original_v(self):
        return self._original_v

    @property
    def worker_cache_v(self):
        return self._worker_cache_v

    @property
    def dst_ps_v(self):
        return self._dst_ps_v

    @property
    def assign_op_wc_dst_ps(self):
        return self._dst_ps_assign_op

    @property
    def var_type(self):
        return self._var_type

    def assign_to_worker_cache(self):
        # if self._worker_cache_assign_op is None:
        #     with tf.colocate_with(self.worker_cache_v):
        #         self._worker_cache_assign_op = tf.assign(self.worker_cache_v, self.original_v)
        return self._worker_cache_assign_op

    def assign_to_dist_ps_v(self):
        # if self._dst_ps_assign_op is None:
        #     with tf.colocate_with(self.dst_ps_v):
        #         self._dst_ps_assign_op = tf.group([tf.assign(self.dst_ps_v, self.worker_cache_v)])
        return self._dst_ps_assign_op_with_cond

    def get_dst_ps_to_org_assign_op(self):
        return self._dst_ps_to_org_assign_op

class Reconfig2ProfilingTools:

    instance = None

    def __init__(self):
        self.timestamp_phase1_start_time = 0.0
        self.timestamp_phase1_post_iter = 0.0
        self.timestamp_phase1_get_supervisor = 0.0
        self.timestamp_phase2_start_time = 0.0
        self.timestamp_phase2_post_iter = 0.0
        self.timestamp_phase2_get_supervisor = 0.0

    def start_reconfig2_phase1(self):
        self.timestamp_phase1_start_time = time.time()

    def finish_phase1_post_iter(self):
        self.timestamp_phase1_post_iter = time.time()

    def finish_phase1_get_supervisor(self):
        self.timestamp_phase1_get_supervisor = time.time()

    def start_reconfig2_phase2(self):
        self.timestamp_phase2_start_time = time.time()

    def finish_phase2_post_iter(self):
        self.timestamp_phase2_post_iter = time.time()

    def finish_phase2_get_supervisor(self):
        self.timestamp_phase2_get_supervisor = time.time()

        logging.info(
            "phase1_post_iter: %f" % (self.timestamp_phase1_post_iter - self.timestamp_phase1_start_time))
        logging.info(
            "phase1_get_supervisor: %f" % (self.timestamp_phase1_get_supervisor - self.timestamp_phase1_post_iter))
        logging.info(
            "phase2_post_iter: %f" % (self.timestamp_phase2_post_iter - self.timestamp_phase2_start_time))
        logging.info(
            "phase2_get_supervisor : %f" % (self.timestamp_phase2_get_supervisor - self.timestamp_phase2_post_iter))

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance


class GenericTimer:

    time_map = {}

    @classmethod
    def start(cls, name):
        cls.time_map[name] = [time.time(), 0.0]

    @classmethod
    def stop(cls, name):
        cls.time_map[name][1] = time.time()

    @classmethod
    def print_all_timer(cls):
        logging.info("====Printing timer log===")
        for timer_name, start_end in cls.time_map.items():
            logging.info("%s: %f" % (timer_name, start_end[1] - start_end[0]))
        logging.info("====End printing timer log===")

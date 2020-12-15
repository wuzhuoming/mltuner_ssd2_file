import json
import logging
import math
import os
import re
import subprocess
import time
import sys
from json import JSONEncoder

import numpy

from selftf.lib.common_conf import MODE_VALUE_SELFTF, optimizer_list

TF_PROGRAM_PS_LIST = "ps_list"
TF_PROGRAM_WORKER_LIST = "worker_list"
TF_PROGRAM_NODE_LIST = "node_list"
TF_PROGRAM_WORKING_DIR = "working_dir"
TF_PROGRAM_JOB_NAME = "job_name"
TF_PROGRAM_IS_CHIEF = "is_chief"
TF_PROGRAM_MAX_ITERATION = "max_iteration"
TF_PROGRAM_CONF_DICT = "conf_dict"
TF_PROGRAM_TASK_INDEX = "task_index"
TF_PROGRAM_TARGET_LOSS = "target_loss"
TF_PROGRAM_BATCH_SIZE = "batch_size"
TF_PROGRAM_LEARNING_RATE = "learning_rate"
TF_PROGRAM_OPTIMIZER = "optimizer"


CHECK_PT_DIR_NAME = "chkpt"

TRAINING_STATUS_TRAINING = "training"
TRAINING_STATUS_OPTIMAL_RUN = "optimal_run"

MODEL_SVM_ANDY_FULL_SPACE = "SVM_andy_config" #useless
MODEL_SVM = "SVM"
MODEL_LR = "LR"
MODEL_CNN = "CNN"

job_name = "selftf"
job_prefix = "/job:%s/task:" % job_name

conf_dict_variable_map = "variable_map"
conf_dict_non_static_ops_names = "non_static_ops_names"
conf_dict_mode = "mode" # can be selftf or dry_run

ML_JOB_FINISH_STRATEGY_WORKER_REACH_TARGET_ONCE = "ML_JOB_FINISH_STRATEGY_WORKER_REACH_TARGET_ONCE"
ML_JOB_FINISH_STRATEGY_WORKER_REACH_TARGET_ONCE_THRESHOLD = "ML_JOB_FINISH_STRATEGY_WORKER_REACH_TARGET_ONCE_THRESHOLD"
ML_JOB_FINISH_STRATEGY_WORKER_ALL_UNDER_TARGET = "ML_JOB_FINISH_STRATEGY_WORKER_ALL_UNDER_TARGET"
ML_JOB_FINISH_STRATEGY_MOVING_AVG = "ML_JOB_FINISH_STRATEGY_MOVING_AVG"

_JOB_STATUS_initialize = "initializing"
_JOB_STATUS_executing = "executing"
_JOB_STATUS_checkingpointing = "checkingpointing"
_JOB_STATUS_finish = "finish"

ENV_KEY_MLTUNER_CONF_DICT="MLTUNER_CONF_DICT"
CONF_DICT_NODE_LIST="node_list"
CONF_DICT_TASK_INDEX="task_index"
CONF_DICT_JOB_ID = "JOB_ID"


def get_optimizer_name(idx):
    return optimizer_list[idx]


def get_optimizer_by_name(search_name):
    for idx, optimizer in enumerate(optimizer_list):
        if optimizer == search_name:
            return idx


class ComputeNode(object):
    def __init__(self, hostname, tfport, sshport=22, working_dir="", num_thread=16):
        self.hostname = hostname
        self.last_touch_timestamp = time.time()
        self.tfport = int(tfport)
        self.ssh_port = sshport
        self.working_dir = working_dir
        self.num_of_thread = num_thread

    def get_hostname(self):
        return self.hostname

    def touch_compute_node(self):
        self.last_touch_timestamp = int(time.time())

    def is_die(self, timeout_in_milli_sec):
        if int(time.time()) - self.last_touch_timestamp > timeout_in_milli_sec:
            return True
        else:
            return False

    def get_tfport(self):
        return self.tfport

    def get_id(self):
        return "%s_%d" % (self.hostname, self.tfport)

    def get_working_dir(self):
        return self.working_dir

    def get_num_thread(self):
        return self.num_of_thread

    @staticmethod
    def get_hostname_tfport_from_id(id):
        regex = re.compile("^(.+)_(\\d+)$")
        m = regex.match(id)
        hostname = m.group(1)
        tfport = m.group(2)
        return hostname, tfport


class Process(object):
    def __init__(self, job_id, pOpen, training_status=TRAINING_STATUS_TRAINING, is_ps=False, config_idx=0):
        self.job_id = job_id
        self.popen = pOpen
        self.training_status = training_status
        self.is_ps = is_ps
        self.config_idx = config_idx

    @staticmethod
    def create(job_id, script, args, training_status, is_ps=False, config_idx=0, env={}):
        if not isinstance(args, list):
            raise TypeError()

        cmd = list()
        cmd.append(script)
        cmd.extend(args)

        self_home = os.getenv("SELFTF_HOME")
        if self_home is None:
            logging.error('env "SELFTF_HOME" not set')
            sys.exit(-1)

        training_cmd_str = " ".join(cmd)
        training_cmd_str = "source {}/common.sh; {}".format(self_home, training_cmd_str)
        training_cmd_str = ("bash", "-c", training_cmd_str)
        logging.info("Training cmd: {}".format(training_cmd_str))
        p = subprocess.Popen(cmd, shell=False, bufsize=-1, env=env)

        o = Process(job_id, p, training_status, is_ps, config_idx=config_idx)
        return o

    def get_job_id(self):
        return self.job_id

    def get_popen(self):
        """
        :rtype:  subprocess.Popen
        :return:
        """
        return self.popen

    def is_finished(self):
        poll_ret = self.get_popen().poll()
        if poll_ret is None:
            return False
        else:
            return True

    def get_training_status(self):
        return self.training_status

    def get_config_idx(self):
        return self.config_idx


class JobTrainingStatistic(object):

    def __init__(self):
        self.training_sample = []
        self.max_local_step = 0

    def add_training_statistic(self, ps_tuner_training_data):
        """
        :param selftf.lib.common.PSTunerTrainingData ps_tuner_training_data:
        :return:
        """
        if not isinstance(ps_tuner_training_data, PSTunerTrainingData):
            raise TypeError()
        if ps_tuner_training_data.local_step > self.max_local_step:
            self.max_local_step = int(ps_tuner_training_data.local_step)
        self.training_sample.append(ps_tuner_training_data)

    def get_num_training_sample(self):
        return len(self.training_sample)

    def get_min_cost_config(self):
        min_cost = sys.float_info.max
        best_config = None
        for x in self.get():
            if x.elapsed_time_in_ms < min_cost:
                best_config = x.ps_config
        return best_config

    def get_last_iteration_idx(self):
        # return self.max_local_step
        return len(self.training_sample)

    def get(self):
        """
        :rtype:list[selftf.lib.common.PSTunerTrainingData]
        """
        ret = sorted(self.training_sample, key=lambda x: x.timestamp)
        for idx, x in enumerate(ret):
            # attach step here
            x.step = idx + 1
        return ret


    # def has_none(self, threshold=100):
    #     if len(self.training_sample) > (threshold + 10):
    #         start = len(self.training_sample)-100
    #     else:
    #         start = 0
    #     for x in range(start, len(self.training_sample)):
    #         if self.training_sample[x] is None:
    #             return True
    #     return False


# Class for ML job
class Job:
    _JOB_STATUS_RUNNONG = "running"
    _JOB_STATUS_FINISHED = "finished"
    _ROLE_PS = "ps"
    _ROLE_WORKER = "worker"

    _job_status = [_JOB_STATUS_RUNNONG, _JOB_STATUS_FINISHED]
    _role = [_ROLE_PS, _ROLE_WORKER]

    _BO_func = "bo"
    _old_func = "old"

    def __init__(self, script="", args="", num_iter_per_os=20, os_size=20,
                 training_status=TRAINING_STATUS_TRAINING, target_loss=0.5,ml_model="ML",
                   batch_size=1000, learning_rate=0.001, online_os_size=100, online_num_iter_per_os=200,
                   estimation_func=_BO_func, mode=MODE_VALUE_SELFTF, compute_node_id_list = [],
                   optimizer=0
                ):

        self.id = ml_model+"_"+str(time.strftime("%m%d_%H%M%S", time.gmtime()))
        self.script = script
        self.args = args
        self.status_list = {}

        self.last_reconfig_steps = 0
        self.num_iter_per_os = int(num_iter_per_os) # each reconfig round, how many iteration is executed
        self.os_size = int(os_size) # how many reconfig round is required.
        self.online_os_size = online_os_size
        self.online_num_iter_per_os = online_num_iter_per_os

        self.wait_for_ssh_complete = 0
        self.current_configuration_plan = None
        self.next_configuraton_plan = None

        self.training_status = training_status
        self.training_statistic = JobTrainingStatistic()

        # lock for checking online tuning
        # The lock will be acquired when some one check to do reconfiguration
        # The lock will be released when
          # 1. new config discovered and the start job finish
          # 2. no new config
        self.online_checking_last_step = num_iter_per_os

        # list of tuple with
        # 1. timestamp
        # 2. iteration
        # 3. config
        self.configuration_history = []
        self._configuration_history_timestamp = 0
        self._configuration_history_iteration = 1
        self._configuration_history_config = 2

        self.start_time = time.time()
        self.end_time = None

        self.target_loss = target_loss
        self.ml_model = ml_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.is_ps_killed = False
        self.chief_killed = False

        self._list_conf_idx_recovery_time = []

        self.counter_check_do_reconfig = 0
        self.counter_do_reconfig = 0

        self.init_duration = None

        self.estimation_func = estimation_func

        self.counter_finish_sub_training_workers = 0
        self._finish_sub_training_workers = []

        self.variable_map = {}
        self._list_non_static_ops_name = []

        self._natural_finish_compute_ids = []

        self._mode = mode

        self.compute_node_id_list = compute_node_id_list

        self.list_final_n_machine_loss = numpy.full(len(self.compute_node_id_list), numpy.inf)

        self._status = _JOB_STATUS_initialize

    @property
    def mode(self):
        return self._mode

    def increment_counter_finish_sub_training_workers(self, hostname):
        self._finish_sub_training_workers.append(hostname)
        self.counter_finish_sub_training_workers += 1

    def increment_counter_check_do_reconfig(self):
        self.counter_check_do_reconfig += 1

    def increment_counter_do_reconfig(self):
        self.counter_do_reconfig += 1

    @property
    def finish_sub_training_workers(self):
        self._finish_sub_training_workers.sort()
        return self._finish_sub_training_workers

    def get_id(self):
        return self.id

    def get_script(self):
        return self.script

    def get_args(self):
        return self.args

    def register_compute_node(self, compute_node_id, role, is_chief=False):
        self.status_list[compute_node_id] = ComputeNodeStatus(compute_node_id, role, is_chief=is_chief)

    def compute_node_finish(self, compute_node_id):
        self.status_list[compute_node_id].set_status_finished()

    def set_all_compute_node_status_running(self):
        for compute_node_id, status in self.status_list.items():
            status.set_status_running()

    def is_all_worker_finish(self):
        ret = True
        for compute_node_id, status in self.status_list.items():
            if status.is_worker() and not status.is_finished():
                ret = False
        return ret

    def is_all_non_chief_worker_finish(self):
        ret = True
        for compute_node_id, status in self.status_list.items():
            if status.is_worker() and not status.is_chief and not status.is_finished():
                ret = False
        return ret

    def is_all_worker_ps_finish(self):
        ret = True
        ret_compute_id_list = []
        for compute_node_id, status in self.status_list.items():
            if not status.is_finished():
                ret_compute_id_list.append(compute_node_id)
                ret = False
        logging.debug("Still alive compute node: "+str(ret_compute_id_list))
        return ret

    def get_compute_node_id_list(self):
        return self.status_list.keys()

    def get_ps_compute_node_id_list(self):
        ret = []
        for compute_node_id, compute_node_status in self.status_list.items():
            if not compute_node_status.is_worker():
                ret.append(compute_node_status.get_compute_node_id())
        return ret

    def update_steps(self, compute_node_id, steps, loss, target_loss):
        status = self.status_list[compute_node_id]
        status.update_steps(steps)
        status.update_loss(loss, target_loss)

        # for moving average x[:-1] = x[1:]
        self.list_final_n_machine_loss[:-1] = self.list_final_n_machine_loss[1:]
        self.list_final_n_machine_loss[-1] = loss

    def get_global_step(self):
        return self.get_training_statistic().get_last_iteration_idx()

    # old version which determine by substep
    # def check_do_reconfig(self):
    #     global_step = self.get_global_step()
    #     if (global_step - self.last_reconfig_steps) >= self.steps_for_reconfig:
    #         self.last_reconfig_steps = global_step
    #         return True
    #     else:
    #         return False
    def check_do_init_reconfig(self):
        logging.debug("total training sample: %d, steps_for_reconfig: %d, config_history %d" % (self.os_size, self.num_iter_per_os, len(self.configuration_history)))
        if len(self.configuration_history) < self.os_size:
            return True
        else:
            return False

    def check_do_online_reconfig(self):
        logging.debug("Check do online reconfig, last iter idx:%d, online checking_last_step: %d, num_iter_per_os: %d",
                      self.training_statistic.get_last_iteration_idx(), self.online_checking_last_step, self.get_online_num_iter_per_os())
        if (self.training_statistic.get_last_iteration_idx() - self.online_checking_last_step >= self.get_online_num_iter_per_os()):
            return True
        else:
            return False

    def done_check_online_reconfig(self):
        self.online_checking_last_step = self.training_statistic.get_last_iteration_idx()

    def get_chief_compute_node_id(self):
        for status in self.status_list.values():
            if status.is_chief:
                return status.get_compute_node_id()
        raise Exception("No chief !?")

    def set_next_configuration(self, config):
        if not isinstance(config, PSTunerConfiguration):
            raise TypeError()
        self.next_configuraton_plan = config

    def clear_config_round_info(self):
        self.next_configuraton_plan = None
        self.is_ps_killed = False
        self.chief_killed = False
        self.counter_finish_sub_training_workers = 0
        self._finish_sub_training_workers = []

    def has_next_configuration(self):
        if self.next_configuraton_plan is not None:
            return True
        else:
            return False

    def finish_a_ssh_task(self):
        self.wait_for_ssh_complete -= 1

    def get_wait_for_ssh_complete(self):
        return self.wait_for_ssh_complete

    def get_training_status(self):
        return self.training_status

    def finish_training(self):
        self.init_duration = time.time() - self.start_time
        self.training_status = TRAINING_STATUS_OPTIMAL_RUN

    def get_steps_for_reconfig(self):
        return self.num_iter_per_os

    def set_current_config_plan(self, ps_conf):
        if not isinstance(ps_conf, PSTunerConfiguration):
            raise TypeError()
        self.current_configuration_plan = ps_conf
        apply_iteration_idx = self.training_statistic.get_last_iteration_idx()
        if apply_iteration_idx != 0:
            apply_iteration_idx += 1
        self.configuration_history.append((time.time(), apply_iteration_idx, ps_conf))

        # update compute node status
        if len(self.status_list) > 0:
            for idx, compute_node_id in enumerate(self.compute_node_id_list):
                compute_node_status = self.status_list[compute_node_id]
                assert isinstance(compute_node_status, ComputeNodeStatus)
                if idx < ps_conf.ps_num:
                    compute_node_status.set_role_as_ps()
                else:
                    compute_node_status.set_role_as_worker()


        return len(self.configuration_history)-1

    def get_current_config_id(self):
        return len(self.configuration_history) - 1

    def get_current_config(self):
        """
        :rtype: selftf.lib.common.PSTunerConfiguration
        :return:
        """
        return self.current_configuration_plan

    def get_config_from_statistic_with_min_cost(self):
        return self.training_statistic.get_min_cost_config()

    def get_training_statistic(self):
        return self.training_statistic

    def is_different_config(self, next_conf):
        return self.current_configuration_plan != next_conf

    def is_worker(self, compute_node_id):
        return self.status_list[compute_node_id].role == Job._ROLE_WORKER

    def get_worker_compute_node_id_list(self):
        ret = []
        for compute_node_id, compute_node_status in self.status_list.items():
            if compute_node_status.is_worker():
                ret.append(compute_node_id)
        return ret

    def get_history_config_by_idx(self, ps_config_idx):
        return self.configuration_history[ps_config_idx][self._configuration_history_config]

    def is_ps_killed(self):
        return self.is_ps_killed

    def kill_ps(self):
        self.is_ps_killed = True

    def add_recovery_time_from_chief_worker(self, config_idx, recovery_time):
        if recovery_time <= 0:
            return
        while len(self._list_conf_idx_recovery_time) <= config_idx:
            self._list_conf_idx_recovery_time.append(.0)
        self._list_conf_idx_recovery_time[config_idx] += recovery_time
        logging.debug("current recovery time map:%s" % self._list_conf_idx_recovery_time)

    def get_avg_recovery_time(self):
        data = self._list_conf_idx_recovery_time[1:]
        if len(data) == 0:
            return 0
        return sum(data) / float(len(data))

    def get_list_history_conf(self):
        return list(map(lambda x: x[self._configuration_history_config], self.configuration_history))

    def get_target_loss(self):
        return self.target_loss

    def finish(self):
        self.end_time = time.time()

    def is_finish(self):
        return self.end_time is not None

    def get_duration(self):
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def get_final_loss(self):
        if self.training_statistic.get_num_training_sample()==0:
            return 0.0
        return self.training_statistic.get()[-1].loss

    def get_ml_model(self):
        return self.ml_model

    def get_batch_size(self):
        return self.batch_size

    def get_learning_rate(self):
        return self.learning_rate

    def get_os_size(self):
        return self.os_size

    def get_iter_per_os(self):
        return self.num_iter_per_os

    def get_online_os_size(self):
        return self.online_os_size

    def get_online_num_iter_per_os(self):
        return self.online_num_iter_per_os

    def get_init_duration(self):
        if self.init_duration is None:
            return 0.0
        return self.init_duration

    def get_estimation_func_name(self):
        return self.estimation_func

    def get_non_chief_worker_compute_node_id_list(self):
        ret = []
        for compute_node_id, compute_node_status in self.status_list.items():
            if compute_node_status.is_worker() and not compute_node_status.is_chief:
                ret.append(compute_node_status.get_compute_node_id())
        return ret

    def get_chief_worker_compute_id(self):
        for compute_node_id, compute_node_status in self.status_list.items():
            if compute_node_status.is_chief:
                return compute_node_status.get_compute_node_id()

    def is_chief_killed(self):
        return self.chief_killed

    def kill_chief(self):
        self.chief_killed = True

    def is_finish_sub_training(self):
        logging.debug("is finish sub trainibg "
                          "counter_finish_sub_training_workers: %d " 
                          "worker_num: %d" % (self.counter_finish_sub_training_workers,
                                              self.get_current_config().worker_num))
        return self.counter_finish_sub_training_workers >= self.get_current_config().worker_num

    def set_variable_map_from_tf_variable_containers(self, list_var_container):
        """
        :param list[TFVariableContainer] list_var_container:
        :return:
        """
        ret = {}
        for x in list_var_container:
            ret[x.name] = x.device
        self.variable_map = ret

    def set_variable_map_plain(self, dict_name_device):
        self.variable_map = dict_name_device

    def get_variable_map(self):
        return self.variable_map

    def set_non_static_ops_names(self, param):
        """
        :param list[str] param:
        :return:
        """
        self._list_non_static_ops_name = param

    @property
    def list_non_static_ops_name(self):
        return self._list_non_static_ops_name

    def compute_node_natural_finish(self, compute_id):
        self._natural_finish_compute_ids.append(compute_id)

    def is_all_worker_natural_finish(self):
        return self._natural_finish_counter >= self.current_configuration_plan.worker_num

    def is_receive_natural_finish(self):
        return self._natural_finish_counter > 0

    @property
    def _natural_finish_counter(self):
        return len(self._natural_finish_compute_ids)

    @property
    def finished_compute_id(self):
        self._natural_finish_compute_ids.sort()
        return self._natural_finish_compute_ids

    def is_all_worker_converge(self):
        for status in self.status_list.values():
            if status.get_loss() > self.target_loss:
                return False
        return True

    def is_all_worker_reach_target_loss_once(self, threshold=0.0):
        worker_counter = 0
        touch_once_counter = 0
        for status in self.status_list.values():
            if status.is_worker():
                worker_counter += 1
                if status.is_reached_target_loss:
                    touch_once_counter += 1

        if touch_once_counter > 0:
            logging.debug("worker_counter:%s, touch_once_counter:%s, threshold:%s" %
                          (str(worker_counter),
                           str(touch_once_counter),
                           str(threshold)))

        if threshold == 0.0:
            return worker_counter == touch_once_counter
        else:
            return touch_once_counter >= (worker_counter * threshold)

    def get_avg_list_final_n_machine_loss(self):
        return numpy.ma.average(self.list_final_n_machine_loss)

    def is_moving_average_smaller_than_target_loss(self):
        target_loss = self.target_loss
        loss_ma = self.get_avg_list_final_n_machine_loss()
        if not math.isnan(loss_ma):
            logging.info("current loss: %f" % loss_ma)
        if loss_ma <= target_loss:
            return True
        else:
            return False
    def get_total_reconfiguration_time(self):
        return sum(self._list_conf_idx_recovery_time)

    def change_status(self, status):
        if status == _JOB_STATUS_initialize or status == _JOB_STATUS_checkingpointing:
            self._status = status
            return True

        elif status == _JOB_STATUS_executing:
            if self._status == _JOB_STATUS_executing:
                return False
            else:
                self._status = status
                return True

    def get_status(self):
        return self._status

class TFVariableContainer(object):
    def __init__(self, tf_variable=None):
        """

        :param tf.Variable tf_variable:
        """
        if tf_variable is not None:
            self._name = tf_variable.op.name
            self._device = tf_variable.op.device

    @property
    def name(self):
        return self._name

    @property
    def device(self):
        return self._device


class ComputeNodeStatus(object):
    def __init__(self, compute_node_id="", role=Job._ROLE_WORKER, status=Job._JOB_STATUS_RUNNONG, is_chief=False):
        self.compute_node_id = compute_node_id
        self.role = role
        self.status = status
        self.steps = 0
        self.is_chief = is_chief
        self.last_loss = 0.0
        self._is_reached_target_loss = False

    def set_status_finished(self):
        self.status = Job._JOB_STATUS_FINISHED

    def set_status_running(self):
        self.status = Job._JOB_STATUS_RUNNONG

    def is_finished(self):
        if self.status == Job._JOB_STATUS_FINISHED:
            return True
        else:
            return False

    def is_worker(self):
        if self.role == Job._ROLE_WORKER:
            return True
        else:
            return False

    def get_compute_node_id(self):
        return self.compute_node_id

    def update_steps(self, steps):
        self.steps = steps

    def update_loss(self, loss, target_loss):
        self.last_loss = loss
        if loss < target_loss:
            logging.info(
                "ComputeNode: %s reach target loss" % self.compute_node_id)
            self._is_reached_target_loss = True

    def get_steps(self):
        if self.role == Job._ROLE_PS:
            return sys.maxsize
        return self.steps

    def get_loss(self):
        if self.role == Job._ROLE_PS:
            return sys.float_info.max
        return self.last_loss

    @property
    def is_reached_target_loss(self):
        return self._is_reached_target_loss

    def set_role_as_worker(self):
        self.role = Job._ROLE_WORKER

    def set_role_as_ps(self):
        self.role = Job._ROLE_PS


class PSTunerTrainingDataSerializer(JSONEncoder):
    def default(self, o):
        if isinstance(o, PSTunerTrainingData):
            obj_dict =  o.__dict__
            return obj_dict
        else:
            try:
                return super(PSTunerTrainingDataSerializer, self).default(o)
            except Exception:
                return o.__dict__

    @staticmethod
    def object_hook(obj):
        try:
            ret = PSTunerTrainingData()
            psconfig_dict = obj["ps_config"]
            obj["ps_config"] = None

            ret.__dict__ = obj
            ret.ps_config = PSTunerConfigurationSerializer.object_hook(psconfig_dict)
            return ret

        except Exception:
            return obj


class PSTunerConfigurationSerializer(JSONEncoder):
    def default(self, o):
        if isinstance(o, PSTunerConfiguration):
            obj_dict =  o.__dict__
            return obj_dict
        else:
            try:
                return super(PSTunerConfigurationSerializer, self).default(o)
            except Exception:
                return o.__dict__

    @staticmethod
    def object_hook(obj):
        try:
            ret = PSTunerConfiguration()
            ret.__dict__ = obj
            return ret
        except Exception:
            return obj


class PSTunerConfiguration(object):
    def __init__(self, num_ps=1,
                 num_worker=1,
                 intra_op_parallelism_threads=1,
                 inter_op_parallelism_threads=1,
                 n_partition=1,
                 batch_size=100,
                 optimizer=1,
                 learning_rate=0.1,
                 do_common_subexpression_elimination=1,
                 max_folded_constant_in_bytes=10485760,
                 do_function_inlining=1,
                 global_jit_level=0,
                 infer_shapes=1,
                 enable_bfloat16_sendrecv=0,
                 place_pruned_graph=0,
                 KMP_AFFINITY_granularity=1,
                 KMP_AFFINITY_respect=1,
                 KMP_AFFINITY_type=0,
                 KMP_AFFINITY_permute=0,
                 KMP_AFFINITY_offset=0,
                 KMP_BLOCKTIME=0,
                 OMP_NUM_THREADS=1,
                 MKL_DYNAMIC=0,
                 py_dict=None
                 ):
        self.ps_num = num_ps
        self.worker_num = num_worker
        self.intra_op_parallelism_threads = intra_op_parallelism_threads
        self.inter_op_parallelism_threads = inter_op_parallelism_threads
        self.n_partition=n_partition
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.optimizer=optimizer
        self.do_common_subexpression_elimination=do_common_subexpression_elimination
        self.max_folded_constant_in_bytes = max_folded_constant_in_bytes
        self.do_function_inlining = do_function_inlining
        self.global_jit_level = global_jit_level
        self.infer_shapes = infer_shapes
        self.enable_bfloat16_sendrecv = enable_bfloat16_sendrecv
        self.place_pruned_graph = place_pruned_graph
        self.KMP_AFFINITY_granularity = KMP_AFFINITY_granularity
        self.KMP_AFFINITY_respect = KMP_AFFINITY_respect
        self.KMP_AFFINITY_type = KMP_AFFINITY_type
        self.KMP_AFFINITY_permute = KMP_AFFINITY_permute
        self.KMP_AFFINITY_offset = KMP_AFFINITY_offset
        self.KMP_BLOCKTIME = KMP_BLOCKTIME
        self.OMP_NUM_THREADS = OMP_NUM_THREADS
        self.MKL_DYNAMIC = MKL_DYNAMIC

        if py_dict is not None:
            if isinstance(py_dict, dict):
                self.__dict__.update(py_dict)
            else:
                logging.error("json_dict is ignored")

    def __getitem__(self, item):
        return getattr(self, item)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__dict__ == other.__dict__

    def __hash__(self):
        try:
            return hash(frozenset(self.__dict__.items()))
        except:
            logging.debug("Exception when hash with: %s" % self.__dict__.items())

    def __str__(self):
        return str(self.__dict__)

    def is_data_reallocateion_invole(self, old_config):
        """
        :param PSTunerConfiguration old_config:
        :return:
        """
        if self.ps_num != old_config.ps_num:
            return True
        return False

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        return newone


class PSTunerTrainingData(object):
    def __init__(self, ps_config=None, elapsed_time_in_ms=0, loss=0, step=0, timestamp=0, local_step=0,
                 ps_config_idx=0):
        """
        :param selftf.lib.common.PSTunerConfiguration ps_config:
        :param elapsed_time_in_ms:
        :param loss:
        :param step: jsut attached by monitor while reading
        :param local_step: Step from worker.... may be duplicated with other worker
        """
        self.ps_config = ps_config
        self.ps_config_idx = ps_config_idx
        self.elapsed_time_in_ms = elapsed_time_in_ms
        self.loss = loss
        self.step = step
        self.local_step = local_step
        self.timestamp = timestamp

# A summary object generated from selftf
class Summary(object):
    def __init__(self, job_id = None,
            runtime_sec= None,
            avg_recovery_time_sec= None,
            target_loss= None,
            ml_model= None,
            batch_size= None,
            learning_rate= None,
            final_loss= None,
            os_size= None,
            n_iter_per_os= None,
            online_os_size= None,
            online_n_iter_per_os= None,
            check_reconfig= None,
            do_reconfig= None,
            total_iteration= None,
            init_duration= None,
            estimation_func= None,
            run_mode = None,
            total_reconfiguration_time=None):
        self.job_id = job_id
        self.runtime_sec= runtime_sec
        self.avg_recovery_time_sec= avg_recovery_time_sec
        self.target_loss= target_loss
        self.ml_model= ml_model
        self.batch_size= batch_size
        self.learning_rate= learning_rate
        self.final_loss= final_loss
        self.os_size= os_size
        self.n_iter_per_os= n_iter_per_os
        self.online_os_size= online_os_size
        self.online_n_iter_per_os= online_n_iter_per_os
        self.check_reconfig= check_reconfig
        self.do_reconfig= do_reconfig
        self.total_iteration= total_iteration
        self.init_duration= init_duration
        self.estimation_func= estimation_func
        self.run_mode = run_mode
        self.total_reconfiguration_time = total_reconfiguration_time


# A summary object generated from benchmark script
class BaselineRecord(object):
    def __init__(self,
        N_iterations,
        N_workers,
        N_intra,
        Optimizer,
        Learning_rate,
        Batch_size,
        n_partitions,
        current_loss,
        time_cost,
        model_name,
        job_id,
        first_loss,
        pstuner_config=None
    ):
        self.N_iterations = int(N_iterations)
        self.N_workers = int(N_workers)
        self.N_intra = int(N_intra)
        self.Optimizer = str(Optimizer)
        self.Learning_rate = float(Learning_rate)
        self.Batch_size = int(Batch_size)
        self.n_partitions = int(n_partitions)
        self.current_loss = float(current_loss)
        self.time_cost = float(time_cost)
        self.model_name = str(model_name)
        self.job_id = str(job_id)
        self.first_loss = float(first_loss)
        self.pstuner_config = ""

        if isinstance(pstuner_config,PSTunerConfiguration):
            self.N_workers = pstuner_config.worker_num
            self.N_intra = pstuner_config.intra_op_parallelism_threads
            self.Optimizer = get_optimizer_name(pstuner_config.optimizer)
            self.Learning_rate = pstuner_config.learning_rate
            self.Batch_size = pstuner_config.batch_size
            self.pstuner_config = json.dumps(pstuner_config.__dict__)


class TestEstimationFuncSummaryRecord(Summary):
    def __init__(self,
        job_id = None,
        runtime_sec= None,
        avg_recovery_time_sec= None,
        target_loss= None,
        ml_model= None,
        batch_size= None,
        learning_rate= None,
        final_loss= None,
        os_size= None,
        n_iter_per_os= None,
        online_os_size= None,
        online_n_iter_per_os= None,
        check_reconfig= None,
        do_reconfig= None,
        total_iteration= None,
        init_duration= None,
        estimation_func= None,
        run_mode = None,
        stop_config_idx = None,
        estimated_remaining_iterarion = None,
        estimated_remaining_time = None,
        actual_remaining_iteration = None,
        actual_remaining_time = None):
        self.job_id = job_id
        self.runtime_sec= runtime_sec
        self.avg_recovery_time_sec= avg_recovery_time_sec
        self.target_loss= target_loss
        self.ml_model= ml_model
        self.batch_size= batch_size
        self.learning_rate= learning_rate
        self.final_loss= final_loss
        self.os_size= os_size
        self.n_iter_per_os= n_iter_per_os
        self.online_os_size= online_os_size
        self.online_n_iter_per_os= online_n_iter_per_os
        self.check_reconfig= check_reconfig
        self.do_reconfig= do_reconfig
        self.total_iteration= total_iteration
        self.init_duration= init_duration
        self.estimation_func= estimation_func
        self.run_mode = run_mode
        self.stop_config_idx = stop_config_idx
        self.estimated_remaining_time = estimated_remaining_time
        self.actual_remaining_time = actual_remaining_time
        self.estimated_remaining_iterarion = estimated_remaining_iterarion
        self.actual_remaining_iteration = actual_remaining_iteration


def get_default_learning_rate_batch_size_optimizer(model):
    defalt_learning_rate = 0.1
    default_batch_size = 1
    default_optimizer = 2

    if model == "CNN":
        defalt_learning_rate = 0.0001
        default_batch_size = 100
        default_optimizer = "Adam"
    elif model == "SVM":
        defalt_learning_rate = 0.00008
        default_batch_size = 2590
        default_optimizer = "SGD"
    elif model == "SVM_BIG":
        defalt_learning_rate = 0.00008
        default_batch_size = 2590
        default_optimizer = "SGD"
    elif model == "LR":
        defalt_learning_rate = 0.000024
        default_batch_size = 4560
        default_optimizer = "RMSProp"
    elif model == 'INCEPTION':
        defalt_learning_rate = 0.00005
        default_batch_size = 10
        default_optimizer = "RMSProp"
    elif model == 'ALEXNET_IMAGENET':
        defalt_learning_rate = 0.02
        default_batch_size = 32
        default_optimizer = "RMSProp_Imagenet"
    else:
        raise Exception("unknown model")
    return defalt_learning_rate, default_batch_size, default_optimizer


class JobEstimationFuncTest(Job):
    def __init__(self, script="", args="", num_iter_per_os=20, os_size=20,
        training_status=TRAINING_STATUS_TRAINING, target_loss=0.5,
        ml_model="ML",
        batch_size=1000, learning_rate=0.001, online_os_size=100,
        online_num_iter_per_os=200,
        estimation_func=Job._BO_func, mode=MODE_VALUE_SELFTF,
        compute_node_id_list=[],
        optimizer=0, stop_config_idx=0):

        super(JobEstimationFuncTest, self).__init__(
            script, args, num_iter_per_os, os_size,
            training_status, target_loss,
            ml_model,
            batch_size, learning_rate, online_os_size,
            online_num_iter_per_os,
            estimation_func, mode,
            compute_node_id_list,
            optimizer
        )

        self._stop_config_idx = stop_config_idx
        self.timestamp_start_test = .0
        self.estimated_remaining_iteration = 0
        self.estimated_remaining_time = .0
        self.actual_remaining_iteration = 0
        self.actual_remaining_time = .0

    def should_start_estimation(self):
        if self.get_current_config_id() == self._stop_config_idx:
            return True
        else:
            return False

    def set_estimated_values(self, remaining_iteration, remaining_time):
        self.estimated_remaining_iteration = remaining_iteration
        self.estimated_remaining_time = remaining_time
        self.timestamp_start_test = time.time()


_ps_num = "ps_num"
_worker_num = "worker_num"
_intra_op_parallelism_threads = "intra_op_parallelism_threads"
_inter_op_parallelism_threads = "inter_op_parallelism_threads"
_n_partition = "n_partition"
_learning_rate = "learning_rate"
_batch_size = "batch_size"
_optimizer = "optimizer"
_sync_protocal = "sync_protocal"
_ps_stragegy = "ps_stragegy"
_session_inter_op_thread_pools = "session_inter_op_thread_pool"
_placement_period = "placement_period"
_allow_soft_placement = "allow_soft_placement"
_operation_timeout_in_ms = "operation_timeout_in_ms"
_do_common_subexpression_elimination = "do_common_subexpression_elimination"
_max_folded_constant_in_bytes = "max_folded_constant_in_bytes"
_do_function_inlining = "do_function_inlining"
_global_jit_level = "global_jit_level"
_infer_shapes = "infer_shapes"
_enable_bfloat16_sendrecv = "enable_bfloat16_sendrecv"
_place_pruned_graph = "place_pruned_graph"
_KMP_AFFINITY_granularity = "KMP_AFFINITY_granularity"
_KMP_AFFINITY_respect = "KMP_AFFINITY_respect"
_KMP_AFFINITY_type = "KMP_AFFINITY_type"
_KMP_AFFINITY_permute = "KMP_AFFINITY_permute"
_KMP_AFFINITY_offset = "KMP_AFFINITY_offset"
_KMP_BLOCKTIME = "KMP_BLOCKTIME"
_OMP_NUM_THREADS = "OMP_NUM_THREADS"
_MKL_DYNAMIC = "MKL_DYNAMIC"

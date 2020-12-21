import csv
import json
import os
import subprocess
import sys
import time
import pexpect
import re
import logging
import selftf.lib.common_conf
from selftf.lib.common import get_default_learning_rate_batch_size_optimizer
from selftf.lib import common_conf
import selftf.lib.common
from selftf.test_os_parameter_config import list_setting

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    )
_num_compute_node=int(os.getenv("SELFTF_NUM_COMPUTE_NODE"))
_num_thread=int(os.getenv("SELFTF_NUM_THREAD"))
_target_model = "SVM_BIG"
best_time = 36000.0
setting = None

logging.debug("Total number of compute node: %d\n Total number of thread:%d"
              % (_num_compute_node, _num_thread))

home_path = os.getenv("SELFTF_HOME")
python_exe = os.getenv("SCRIPT_PYTHON_EXECUTABLE")
dataset_base_path = os.getenv("DATASET_BASE_PATH")

if home_path is None:
    logging.error("env \'SELFTF_HOME\' is not set")
    sys.exit(-1)

# ML job example ("SVM", 20, 36, 20, 36, 0.00005, 2000, "bo", 10, 8, "Adam", "dry_run / selftf"),
# (
#   0. Model Name,
#   1. Number of setting in init phase,
#   2. Number of iteration of a single setting in init phase,
#   3. Number of setting in online tuning phase,
#   4. Number of iteration of a single setting in online tuning,
#   5. Learning rate,
#   6. Batch size,
#   7. Estimation function
#   8. Number of worker
#   9. Number of thread in intra thread pool
#  10. Optimizer
#  11. Run mode. dry_run: run single configuration only
# )
# example: ("SVM", 20, 36, 20, 36, 0.00005, 2000, "bo")
# (
#   Model Name = SVM ,
#   Number of setting in init phase = 20 settings ,
#   Number of batch of a single setting in init phase = 36 batch ,
#   Number of setting in online tuning phase = 20 setting ,
#   Number of batch  of a single setting in online tuning = 36 batch ,
#   Learning rate(deprecated) = 0.00005,
#   Batch size (deprecated) = 2000,
#   Estimation function = bo function (paper's one)
# )

_SVM_batch_range = (1000,5000)
_SVM_learning_rate_range = (0.00001,0.0001)
_LR_batch_range = (1000,5000)
_LR_learning_rate_range = (0.00001,0.0001)
_CNN_batch_range = (100,1000)
_CNN_learning_rate_range = (0.00001,0.0001)
_INCEPTION_batch_range = (100,1000)
_INCEPTION_learning_rate_range = (0.00001,0.0001)


def get_batch_size_range_learining_rate_rnage_by_model(model):
    return [0, 0], [0, 0] # we wont optimize the hp now
    if model == "SVM":
        batch_range = _SVM_batch_range
        learning_rate_range = _SVM_learning_rate_range
    elif model == "SVM_BIG":
        batch_range = _SVM_batch_range
        learning_rate_range = _SVM_learning_rate_range
    elif model == "LR":
        batch_range = _LR_batch_range
        learning_rate_range = _LR_learning_rate_range
    elif model == "CNN":
        batch_range = _CNN_batch_range
        learning_rate_range = _CNN_learning_rate_range
    elif model == "INCEPTION":
        batch_range = _INCEPTION_batch_range
        learning_rate_range = _INCEPTION_learning_rate_range
    elif model == "ALEXNET_IMAGENET":
        batch_range = _INCEPTION_batch_range
        learning_rate_range = _INCEPTION_learning_rate_range
    else:
        raise Exception
    return batch_range, learning_rate_range


def random_generating_config(model):
    batch_range, learning_rate_range = \
        get_batch_size_range_learining_rate_rnage_by_model(model)
    ret = []

    from selftf.lib.tuner import TensorFlowConfigurationManager
    tf_cfg_manager = TensorFlowConfigurationManager(
        lambda :_num_compute_node,
        lambda :_num_thread,
        learning_rate_range,
        batch_range
    )
    from selftf.lib.lhs import LHSAdapter
    lhs_runner = LHSAdapter(tf_cfg_manager)
    mock_job_obj = selftf.lib.common.Job()
    list_ps_tuner_config = lhs_runner.get_batch_lhs_config(100, mock_job_obj)

    # Hack to generate random setting
    # TODO: change monitor to generate setting with varying parameter only
    defalt_learning_rate, default_batch_size, default_optimizer = \
        get_default_learning_rate_batch_size_optimizer(model)

    for config in list_ps_tuner_config:
        config.learning_rate = defalt_learning_rate
        config.batch_size = default_batch_size
        config.optimizer = selftf.lib.common.get_optimizer_by_name(default_optimizer)

    for x in list_ps_tuner_config:
        ret.append(get_benchmark_config_from_list([
            model,
            1,
            1,
            1,
            1,
            defalt_learning_rate,
            default_batch_size,
            'bo',
            x.worker_num,
            x.intra_op_parallelism_threads,
            default_optimizer,
            'dry_run'],
            pstuner_config=x
        ))
    return ret

def is_diverge(monitor_out_path):
    _threshold = 20
    regex_ma_loss = re.compile("Job current ma loss: (.*)\n")
    monitor_output = subprocess.check_output(["tail", "-n","20" ,monitor_out_path])
    m=regex_ma_loss.search(monitor_output.decode('utf-8'))
    if m is not None:
        try:
            loss = float(m.group(1))
            if loss > _threshold:
                return True
        except:
            logging.debug("Fail to read loss")

    return False


def wait_finish(job_id):
    start_time = time.time()
    dir = os.path.join(home_path, "log", job_id)
    while not os.path.exists(os.path.join(dir,"monitor.out")):
        # logging.info("The job %s is not finish" % job_id)
        if (time.time() - start_time) > best_time or is_diverge(os.path.join(home_path,"monitor.out")):
            logging.info("Run too long and break")
            pexpect.run("python selftf/client.py --action dump_log %s" % job_id)
            time.sleep(10)
            break
        else:
            time.sleep(10)

    logging.info("The job %s is finish !" % job_id)
    time.sleep(5)


def execute(config):
    """
    :param BenchmarkConfig config:
    :return:
    """
    model = config.model
    init_os = config.init_os
    init_num_iter = config.init_num_iter
    online_os = config.online_os
    online_num_iter = config.online_num_iter
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    estimation_func = config.estimation_func
    num_of_workers = config.worker_num
    num_of_intra_thread = config.intra_op_parallelism_threads
    optimizer = common_conf.optimizer_list[config.optimizer]
    run_mode = config.run_mode

    assert run_mode in common_conf.MODE_VALUES

    batch_range, learning_rate_range = \
        get_batch_size_range_learining_rate_rnage_by_model(model)

    logging.info("run config: init_os:%d, init_num_iter:%d, online_os:%d, online_num_iter:%d" % (
        init_os,
        init_num_iter,
        online_os,
        online_num_iter
    ))
    base_cmd = "sh -c 'cd %s; source bin/common.sh;export BATCH_SIZE_RANGE=\"%s\"; " \
               "export LEARNING_RATE_RANGE=\"%s\";export KEY_ESTIMATION_FUNC=%s;export ONLINE_NUM_ITER_PER_OS=%d;export ONLINE_OS_SIZE=%d;export OS_SIZE=%d;" \
               "export NUM_ITER_PER_OS=%d;sh bin/stop_all_agent.sh && sh bin/stop_monitor.sh &&  " \
               "sleep 2 && bin/start_monitor.sh && sh bin/start_all_agent.sh && sleep 2 && python selftf/client.py " \
               "--action submit_job --ml_model %s --batch_size %d --learning_rate %f " \
               "--optimizer %s --num_worker %d --num_intra_thread %d --mode %s " \
               "--json_dict_pstuner_config \"%s\" "  \
               % (home_path,
                 "[%s,%s]" % (batch_range[0], batch_range[1]),
                 "[%s,%s]" % (learning_rate_range[0], learning_rate_range[1]),
                 estimation_func,
                 online_num_iter,
                 online_os,
                 init_os,
                 init_num_iter,
                 model,
                 batch_size,
                 learning_rate,
                 optimizer,
                 num_of_workers,
                 num_of_intra_thread,
                 run_mode,
                 json.dumps(config.__dict__).replace('"', '\\"'))
    if model == "SVM":
        cmd = "%s  --target_loss 0.13 " \
              "--script %s selftf/tf_job/classification/disML_new_api.py --ML_model=SVM " \
                  "--num_Features=54686452 --data_dir=%s'" % (
                                           base_cmd,
                                           python_exe,
                                           "%s/kdd12" % dataset_base_path)
    if model == "SVM_BIG":
        cmd = "%s  --target_loss 0.966 " \
              "--script %s selftf/tf_job/classification/disML_new_api.py --ML_model=SVM " \
              "--num_Features=1000000 --data_dir=%s'" % (
                                           base_cmd,
                                           python_exe,
                                           "%s/criteo_tb_split" % dataset_base_path)
    elif model == "LR":
        cmd = "%s  --target_loss 0.18 " \
              "--script %s selftf/tf_job/classification/disML_new_api.py --ML_model=LR " \
              "--num_Features=54686452 --data_dir=%s'" % (
                                            base_cmd,
                                            python_exe,
                                            "%s/kdd12" % dataset_base_path)
    elif model == "CNN":
        # cmd = "%s  --target_loss 2.0 " \
        cmd = "%s  --target_loss 0.5 " \
              "--script %s selftf/tf_job/cnn/disCNN_cifar10_new_api.py " \
              "--data_dir=%s'" % (
                  base_cmd,
                  python_exe,
                  "%s/cifar-10-batches-bin/" % dataset_base_path)
    elif model == "INCEPTION":
        cmd = "%s --target_loss 12 " \
              "--script %s selftf/tf_job/inception/inception_selftf.py " \
              "--data_dir=%s'" % (
              base_cmd,
              python_exe,
              "%s/imagenet-data" % dataset_base_path)
    elif model == "ALEXNET_IMAGENET":
        # cmd = "%s --target_loss 7.66 " \
        cmd = "%s --target_loss 0.568 " \
              "--script %s selftf/tf_job/alexnet_imagenet/alexnet_imagenet_selftf.py " \
              "--data_dir=%s --num_class=%d'" % (
                  base_cmd,
                  python_exe,
                  "%s/imagenet" % dataset_base_path,
                  8)
    elif model == "NLP_imdb":
        cmd = "%s --target_loss 0.01 " \
              "--script %s %s/selftf/tf_job/nlp/zmwu/imdb/nlp_imdb.py" % (
                  base_cmd,
                  python_exe,
                  home_path)
    elif model == "NLP_bert":
        cmd = "%s --target_loss 0.01 " \
              "--script %s %s/selftf/tf_job/nlp/zmwu/bert_tf2/tf_bert.py" % (
                  base_cmd,
                  python_exe,
                  home_path) 
    elif model == "NLP_lstm":
        cmd = "%s --target_loss 0.01 " \
              "--script %s %sselftf/tf_job/nlp/zmwu/bd_lstm/train.py" % (
                  base_cmd,
                  python_exe,
                  home_path)
    else:
        cmd = "%s --target_loss 0.2 " \
              "--script %s %s/selftf/tf_job/cnn/keras_impl/train.py --model=%s" % (
                  base_cmd,
                  python_exe,
                  home_path,
                  model)
    logging.info("run command: %s" % cmd)
    p = pexpect.spawn(cmd, timeout=65535)
    regex = re.compile("The job id is: (\\w*)")
    while(True):
        line = p.readline()
        # print(line[:-1])
        m = regex.search(line.decode('utf-8'))
        if m is not None:
            job_id = m.group(1)
            break
    return job_id


def get_log_base_path_by_job_id(job_id):
    return os.path.join(home_path, "log", job_id)


def get_summary_obj(job_id):
    summary_file_path = os.path.join(home_path, "log", job_id, "summary.csv")
    with open(summary_file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        ret_json = reader.next()
    ret = selftf.lib.common.Summary(**ret_json)
    return ret


def get_run_time(job_id):
    summary_dict = get_summary_obj(job_id)
    return summary_dict["runtime_sec"]

def set_config_intra_thread(config, num_of_intra_thread):
    """
    :param selftf.lib.common.PSTunerConfiguration config:
    :param int inter_thread:
    :return:
    """
    assert isinstance(config, selftf.lib.common.PSTunerConfiguration)
    config.intra_op_parallelism_threads = num_of_intra_thread
    config.inter_op_parallelism_threads = _num_thread - num_of_intra_thread

def set_config_num_worker(config, num_worker):
    """
    :param selftf.lib.common.PSTunerConfiguration config:
    :param int inter_thread:
    :return:
    """
    assert isinstance(config, selftf.lib.common.PSTunerConfiguration)
    config.worker_num = num_worker
    config.ps_num = _num_compute_node - num_worker

def set_config_num_ps(config, ps_num):
    """
    :param selftf.lib.common.PSTunerConfiguration config:
    :param int inter_thread:
    :return:
    """
    assert isinstance(config, selftf.lib.common.PSTunerConfiguration)
    config.ps_num = ps_num
    config.worker_num = _num_compute_node - ps_num

def analysis_selftf(job_id, benchmark_config):

    # a list of best configuration
    list_best_conf = []

    # a list of tuple, (gp_idx, config)
    list_reconfiged_conf = []

    list_list_training_data_x_config = []
    list_list_training_data_y_runtime = []

    list_config_runtime = []

    _regex_all_gf_config = "Best training conf (.*)"
    _regex_reconfiged_config = "We got a new config: (.*)"
    _regex_gp_training_data = "Config vector list\(x\):(.*) remaining time\(y\):(.*)"
    p_regex_all_gf_config = re.compile(_regex_all_gf_config)
    p_regex_changed_config = re.compile(_regex_reconfiged_config)
    p_regex_gp_training_data = re.compile(_regex_gp_training_data)

    monitor_out_path = os.path.join(get_log_base_path_by_job_id(job_id), "monitor.out")
    with open(monitor_out_path, "r") as f:
        gp_counter = -1
        for line in f:
            m = p_regex_gp_training_data.search(line)
            if m is not None:
                gp_counter += 1
                training_x = eval(m.group(1))
                training_y = eval(m.group(2))

                # constuct PStunerConfig
                # HACK if config_sequence change..... we need to change here
                list_config = []
                for config_vector in training_x:
                    config = selftf.lib.common.PSTunerConfiguration()
                    config.__dict__ = benchmark_config.__dict__.copy()
                    set_config_num_ps(config, config_vector[0])
                    set_config_intra_thread(config, config_vector[1])
                    list_config.append(config)
                list_list_training_data_x_config.append(list_config)
                list_list_training_data_y_runtime.append(training_y)

            m = p_regex_all_gf_config.search(line)
            if m is not None:
                config = selftf.lib.common.PSTunerConfiguration()
                config.__dict__ = eval(m.group(1))
                list_best_conf.append(config)

                if gp_counter == 0:
                    list_reconfiged_conf.append((gp_counter, config))

            m = p_regex_changed_config.search(line)
            if m is not None:
                config = selftf.lib.common.PSTunerConfiguration()
                config.__dict__ = eval(m.group(1))
                list_reconfiged_conf.append((gp_counter, config))

        summary_obj = get_summary_obj(job_id)


    return {
        "ret_estimation_func": (list_list_training_data_x_config, list_list_training_data_y_runtime),
        "bo_ret": list_reconfiged_conf,
        "runtime": summary_obj.runtime_sec
    }

    # for gp_idx, config_obj in list_reconfiged_conf:
    #     job_id = execute_with_config_obj(model=benchmark_config.model,
    #                                      init_os=benchmark_config.init_os,
    #                                      init_num_iter=benchmark_config.init_num_iter,
    #                                      online_os=benchmark_config.online_os,
    #                                      online_num_iter=benchmark_config.online_num_iter,
    #                                      config=config_obj)
    #     wait_finish(job_id)
    #
    #     # get the runtime
    #     runtime = get_run_time(job_id)
    #     list_config_runtime.append((config_obj, runtime))


def execute_with_config_obj(
            model,
            init_os,
            init_num_iter,
            online_os,
            online_num_iter,
            config):
    assert isinstance(config, selftf.lib.common.PSTunerConfiguration)
    benchmark_config = BenchmarkConfig(config=config,
                                       init_os = init_os,
                                       init_num_iter = init_num_iter,
                                       online_os = online_os,
                                       online_num_iter=online_num_iter,
                                       run_mode="dry_run",
                                       estimation_func="bo",
                                       model=model)

    return execute(benchmark_config)


class BenchmarkConfig(selftf.lib.common.PSTunerConfiguration):

    def __init__(self,
        config = None,
        model=None,
        init_os=None,
        init_num_iter=None,
        online_os=None,
        online_num_iter=None,
        learning_rate=None,
        batch_size=None,
        estimation_func=None,
        num_of_workers=None,
        num_of_intra_thread=None,
        optimizer=None,
        run_mode=None,
    ):
        if config is not None:
            assert isinstance(config, selftf.lib.common.PSTunerConfiguration)
            super(BenchmarkConfig, self).__init__()
            self.__dict__ = config.__dict__.copy()
        else:
            if isinstance(optimizer, int):
                optimizer = optimizer
            elif optimizer is None:
                pass
            else:
                optimizer = common_conf.optimizer_list.index(optimizer)

            super(BenchmarkConfig, self).__init__(
                learning_rate = learning_rate,
                batch_size = batch_size,
                num_worker = num_of_workers,
                intra_op_parallelism_threads = num_of_intra_thread,
                optimizer = optimizer
            )
            set_config_intra_thread(self, num_of_intra_thread)
            set_config_num_worker(self, num_of_workers)

        self.model = model
        self.init_os = init_os
        self.init_num_iter = init_num_iter
        self.online_os = online_os
        self.estimation_func = estimation_func
        self.online_num_iter = online_num_iter
        self.run_mode = run_mode

def get_benchmark_config_from_list(list_in, pstuner_config=None):
    if len(list_in) == 7:
        """
        "LR", 10, 108, 10, 108, "dry_run", {pstuner_config}
        """
        pstuner_config = selftf.lib.common.PSTunerConfiguration(py_dict=eval(list_in[-1]))
        ret = BenchmarkConfig(model=list_in[0],
                              init_os=int(list_in[1]),
                              init_num_iter=int(list_in[2]),
                              online_os=int(list_in[3]),
                              online_num_iter=int(list_in[4]),
                              learning_rate=float(0),
                              batch_size=int(0),
                              estimation_func=str("bo"),
                              num_of_workers=int(0),
                              num_of_intra_thread=int(0),
                              optimizer=str("SGD"),
                              run_mode=str(list_in[5]))
    else:
        ret = BenchmarkConfig(model=list_in[0],
                         init_os=int(list_in[1]),
                         init_num_iter=int(list_in[2]),
                         online_os=int(list_in[3]),
                         online_num_iter=int(list_in[4]),
                         learning_rate=float(list_in[5]),
                         batch_size=int(list_in[6]),
                         estimation_func=str(list_in[7]),
                         num_of_workers=int(list_in[8]),
                         num_of_intra_thread=int(list_in[9]),
                         optimizer=str(list_in[10]),
                         run_mode=str(list_in[11]))
    if pstuner_config is not None:
        ret.__dict__.update(pstuner_config.__dict__)
    return ret

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config = sys.argv[1:]
        setting = [get_benchmark_config_from_list(config)]
    elif list_setting is not None:
        setting = list(map(get_benchmark_config_from_list, list_setting))
    else:
        setting = random_generating_config(_target_model)

    for config in setting:
        logging.info("Execute job: %s" % str(config.__dict__))

        run_mode = config.run_mode
        is_selftf_analysis = False
        if run_mode == "selftf_analysis":
            is_selftf_analysis = True
            config.run_mode = "selftf"

        job_id = execute(config)
        logging.info("The job %s is started !" % job_id)
        wait_finish(job_id)

        if is_selftf_analysis:
            analysis_selftf(job_id, config)


# if __name__ == "__main__":
#     config = get_benchmark_config_from_list(("CNN", 10, 200, 10, 200, 0.00006, 100, "bo", 29, 4, "Adam", "selftf"))
#     ret = analysis_selftf("CNN_0620_222618", config)
#     assert True



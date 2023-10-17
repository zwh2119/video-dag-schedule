'''
主体来源于lat_first_pid
修改点：采用深度强化学习模型设置PID算法的初始超参数 Kp,Kd,Ki

核心：AutoPIDController类
状态
动作
奖励

'''

from logging_utils import root_logger
import pandas as pd
import os

import time
import yaml_utils
import requests

prev_video_conf = dict()

prev_flow_mapping = dict()

prev_runtime_info = dict()

available_fps = [1, 5, 10, 20, 30]
available_resolution = ["360p", "480p", "720p", "1080p"]
# available_npxpf = [480*360, 858*480, 1280*720, 1920*1080]

configs = yaml_utils.read_yaml('configure.yaml')
pid_config = configs['pid']
drl_config = configs['drl']


class AutoPIDController:
    def __init__(self, min_value, max_value):
        self.Kp = pid_config['kp']
        self.Ki = pid_config['ki']
        self.Kd = pid_config['kd']

        self.min_value = min_value
        self.max_value = max_value

        self.cur_time = time.time()
        self.last_time = self.cur_time

        self.setpoint = 0

        self.previous_error = 0
        self.integral = 0

    def get_pid_parameter(self):
        sess = requests.Session()
        r = sess.get(url=f'http://127.0.0.1:{drl_config["port"]}/drl/parameter')
        parameter = r.json()
        self.Kp = parameter['kp']
        self.Ki = parameter['ki']
        self.Kd = parameter['kd']

    def check_pid_parameter(self):
        print(f'kp:{self.Kp}, ki:{self.Ki}, kd:{self.Kd}')

    def reset_pid_parameter(self):
        self.Kp = pid_config['kp']
        self.Ki = pid_config['ki']
        self.Kd = pid_config['kd']

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def update(self, current_value):
        # 尝试更新pid参数
        self.get_pid_parameter()

        error = self.setpoint - current_value
        self.cur_time = time.time()
        dt = self.cur_time - self.last_time
        self.integral += error
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error

        # ## 控制边际，防止过度调控（自动化领域需要，这里是否需要保留？）
        # if output < self.min_value:
        #     output = self.min_value
        # elif output > self.max_value:
        #     output = self.max_value

        return output


# 给定flow_map，根据kb获取处理时延
def get_process_delay(resolution=None, flow_map=None):
    sum_delay = 0.0
    for taskname in flow_map:
        pf_filename = 'profile/{}.pf'.format(taskname)
        pf_table = None
        if os.path.exists(pf_filename):
            pf_table = pd.read_table(pf_filename, sep='\t', header=None,
                                     names=['resolution', 'node_role', 'delay'])
        else:
            root_logger.warning("using profile/face_detection.pf for taskname={}".format(taskname))
            pf_table = pd.read_table('profile/face_detection.pf', sep='\t', header=None,
                                     names=['resolution', 'node_role', 'delay'])
        # root_logger.info(pf_table)
        node_role = 'cloud' if flow_map[taskname]['node_role'] == 'cloud' else 'edge'
        pf_table['node_role'] = pf_table['node_role'].astype(str)
        matched_row = pf_table.loc[
            (pf_table['node_role'] == node_role) & \
            (pf_table['resolution'] == resolution)
            ]
        delay = matched_row['delay'].values[0]
        root_logger.info('get profiler delay={} for taskname={} node_role={}'.format(
            delay, taskname, flow_map[taskname]['node_role']
        ))

        sum_delay += delay

    root_logger.info('get sum_delay={} by knowledge base'.format(sum_delay))

    return sum_delay


# TODO：给定flow_map，获取传输时延
def get_transfer_delay(resolution=None, flow_map=None, resource_info=None):
    return 0.0


# 获取总预估的时延
def get_pred_delay(conf_fps=None, cam_fps=None, resolution=None, flow_map=None, resource_info=None):
    # 给定flow_map，
    # resolution vs process_delay：基于kb
    # resolution vs transfer_delay：基于带宽计算
    # fps vs delay：比例关系

    process_sum_delay = get_process_delay(resolution=resolution, flow_map=flow_map)
    transfer_sum_delay = get_transfer_delay(resolution=resolution, flow_map=flow_map, resource_info=resource_info)

    total_delay = (process_sum_delay + transfer_sum_delay) * conf_fps / cam_fps

    return total_delay


# TODO：给定fps和resolution，结合运行时情境，获取预测时延
def get_pred_acc(conf_fps=None, cam_fps=None, resolution=None, runtime_info=None):
    if runtime_info and 'obj_stable' in runtime_info:
        if not runtime_info['obj_stable'] and conf_fps < 20:
            return 0.6
    return 0.9


# ---------------
# ---- 冷启动 ----
def get_flow_map(dag=None, resource_info=None, offload_ptr=None):
    cold_flow_mapping = dict()
    flow = dag["flow"]

    for idx in range(len(flow)):
        taskname = flow[idx]
        if idx <= offload_ptr:
            cold_flow_mapping[taskname] = {
                "model_id": 0,
                "node_role": "host",
                "node_ip": list(resource_info["host"].keys())[0]
            }
        else:
            cold_flow_mapping[taskname] = {
                "model_id": 0,
                "node_role": "cloud",
                "node_ip": list(resource_info["cloud"].keys())[0]
            }

    return cold_flow_mapping


def get_cold_start_plan(
        job_uid=None,
        dag=None,
        resource_info=None,
        user_constraint=None,
):
    assert job_uid, "should provide job_uid"

    global prev_video_conf, prev_flow_mapping
    global available_fps, available_resolution

    # 时延优先策略：算量最小，算力最大
    cold_video_conf = {
        "resolution": "360p",
        "fps": 30,
        # "ntracking": 5,
        "encoder": "JPEG",
    }
    cold_flow_mapping = dict()
    for taskname in dag["flow"]:
        cold_flow_mapping[taskname] = {
            "model_id": 0,
            "node_role": "host",
            "node_ip": list(resource_info["host"].keys())[0]
        }

    delay_ub = user_constraint["delay"]
    delay_lb = delay_ub
    acc_ub = user_constraint["accuracy"]
    acc_lb = acc_ub

    min_delay_delta = None
    min_acc_delta = None

    # 调度维度：nproc，切分点，fps，resolution
    for fps in available_fps:
        for resol in available_resolution:
            for offload_ptr in range(0, len(dag["flow"])):
                # 枚举所有策略，根据knowledge base预测时延和精度，找出符合用户约束的。
                # 若无法同时满足，优先满足时延要求。尽量满足精度要求（不要求是最优解，所以可以提前退出）
                flow_map = get_flow_map(dag=dag,
                                        resource_info=resource_info,
                                        offload_ptr=offload_ptr)
                cam_fps = 30.0
                delay = get_pred_delay(conf_fps=fps, cam_fps=cam_fps,
                                       resolution=resol,
                                       flow_map=flow_map,
                                       resource_info=resource_info)
                acc = get_pred_acc(conf_fps=fps, cam_fps=cam_fps,
                                   resolution=resol)

                if delay < delay_ub:
                    # 若时延符合要求，找最符合精度要求的
                    # 防止符合要求的配置被替换
                    min_delay_delta = 0.0
                    if not min_acc_delta or min_acc_delta > abs(acc_lb - acc):
                        cold_video_conf["resolution"] = resol
                        cold_video_conf["fps"] = fps
                        cold_flow_mapping = flow_map
                        min_acc_delta = abs(acc_lb - acc)
                else:
                    # 若时延不符合要求，找出尽量符合的
                    if not min_delay_delta or min_delay_delta > abs(delay_ub - delay):
                        cold_video_conf["resolution"] = resol
                        cold_video_conf["fps"] = fps
                        cold_flow_mapping = flow_map
                        min_delay_delta = abs(delay_ub - delay)

    prev_video_conf[job_uid] = cold_video_conf
    prev_flow_mapping[job_uid] = cold_flow_mapping

    return prev_video_conf[job_uid], prev_flow_mapping[job_uid]


# -------------------------------------------
# ---- TODO：根据资源情境，尝试分配更多资源 ----
def try_expand_resource(next_flow_mapping=None, err_level=None, resource_info=None):
    tune_msg = None
    for taskname, task_mapping in reversed(list(next_flow_mapping.items())):
        if task_mapping["node_role"] == "host":
            print(" -------- send to cloud --------")
            next_flow_mapping[taskname]["node_role"] = "cloud"
            next_flow_mapping[taskname]["node_ip"] = list(
                resource_info["cloud"].keys())[0]
            tune_msg = "task-{} send to cloud".format(taskname)
            break

    return tune_msg, next_flow_mapping


def try_reduce_resource(next_flow_mapping=None, err_level=None, resource_info=None):
    tune_msg = None
    for taskname, task_mapping in reversed(list(next_flow_mapping.items())):
        if task_mapping["node_role"] == "cloud":
            print(" -------- send to cloud --------")
            next_flow_mapping[taskname]["node_role"] = "host"
            next_flow_mapping[taskname]["node_ip"] = list(
                resource_info["host"].keys())[0]
            tune_msg = "task-{} send to host".format(taskname)
            break

    return tune_msg, next_flow_mapping


# -----------------------------------------
# ---- TODO：根据应用情境，尝试减少计算量 ----
def try_reduce_calculation(
        next_video_conf=None,
        err_level=None,
        runtime_info=None,
        init_prior=1,
        best_effort=False
):
    global available_fps, available_resolution

    resolution_index = available_resolution.index(
        next_video_conf["resolution"])
    fps_index = available_fps.index(next_video_conf["fps"])

    tune_msg = None

    # TODO：根据运行时情境初始化优先级，实现最佳匹配
    total_prior = 2
    curr_prior = init_prior

    # 无法最佳匹配时，根据收益大小优先级调度
    while True:
        if curr_prior == 1:
            if fps_index > 0:
                print(" -------- fps lower -------- (init_prior={})".format(init_prior))
                next_video_conf["fps"] = available_fps[fps_index - 1]
                tune_msg = "fps {} -> {}".format(available_fps[fps_index],
                                                 available_fps[fps_index - 1])

        if curr_prior == 0:
            if resolution_index > 0:
                print(" -------- resolution lower -------- (init_prior={})".format(init_prior))
                next_video_conf["resolution"] = available_resolution[resolution_index - 1]
                tune_msg = "resolution {} -> {}".format(available_resolution[resolution_index],
                                                        available_resolution[resolution_index - 1])

        # 按优先级依次选择可调的配置
        if best_effort and not tune_msg:
            curr_prior = (curr_prior + 1) % total_prior
            if curr_prior == init_prior:
                break
        if best_effort and tune_msg:
            break
        if not best_effort:
            break

    return tune_msg, next_video_conf


# ----------------
# ---- 负反馈 ----
# 优化目标： 1.稳定调度（情境稳定的情况下尽量减少波动）；2.多角度复杂调度

# TODO: pid out boundry (-0.1, 0.1)
#       在边界之内的不调整
#       在边界之外的，按照超出边界的程度调整
#       不要线性调整 -- 复杂调度

def adjust_parameters(err_level=0, job_uid=None,
                      dag=None,
                      user_constraint=None,
                      resource_info=None,
                      runtime_info=None):
    assert job_uid, "should provide job_uid"

    global prev_video_conf, prev_flow_mapping, prev_runtime_info
    global available_fps, available_resolution

    next_video_conf = prev_video_conf[job_uid]
    next_flow_mapping = prev_flow_mapping[job_uid]

    # 仅支持pipeline
    flow = dag["flow"]
    assert isinstance(flow, list), "flow not list"

    resolution_index = available_resolution.index(
        next_video_conf["resolution"])
    fps_index = available_fps.index(next_video_conf["fps"])

    # err_level = round(output)
    tune_msg = None

    # TODO：参照对应的边端sniffer解析运行时情境
    print('---- runtime_info in the past time slot ----')
    print('runtime_info = {}'.format(runtime_info))
    # obj_n = runtime_info['obj_n']

    ## pid output小于阈值，不调度
    if abs(err_level) < pid_config['action_boundary']:
        return prev_video_conf[job_uid], prev_flow_mapping[job_uid]

    schedule_level = abs(err_level) / pid_config['action_boundary']

    if err_level >= 0:
        # level > 0，时延满足要求
        # TODO：结合运行时情境（应用），可以进一步优化其他目标（精度、云端开销等）：
        #              优化目标优先级：时延 > 精度 > 云端开销
        #              若优化目标为最大化精度，在达不到要求时，可以提高fps和resolution；
        #              若优化目标为最小化云端开销，可以拉回到边端计算；

        ## new add:
        '''
        时延满足要求，根据超过要求的情况分级扩大分辨率、帧率、云边切分点
        '''
        if resolution_index + 1 < len(available_resolution):
            next_video_conf["resolution"] = available_resolution[resolution_index + 1]
        else:
            if fps_index + 1 < len(available_fps):
                next_video_conf["fps"] = available_fps[fps_index + 1]
            else:
                tune_msg, next_flow_mapping = try_expand_resource(next_flow_mapping=next_flow_mapping,
                                                                  err_level=err_level,
                                                                  resource_info=resource_info)


    elif err_level < 0:
        # level < 0，时延不满足要求
        # TODO：结合运行时情境（资源），应该调整策略，以降低时延：
        #              （1）若场景稳定性，降低帧率；若场景目标较大，降低分辨率
        #              （2）分配更多资源；
        #              （3）任务卸载到空闲节点（云/边）；
        #              （4）最后考虑降低fps和resolution；
        #       结合运行时情境（应用），调整fps和resolution，比如：
        #              场景稳定则优先降低fps（对精度影响较小）
        #              物体较大则降低resolution（对精度影响较小）

        ## new add
        if not tune_msg:
            tune_msg, next_flow_mapping = try_reduce_resource(next_flow_mapping=next_flow_mapping, err_level=err_level,
                                                              resource_info=resource_info)

        if not tune_msg and fps_index - 1 > 0:
            next_video_conf["fps"] = available_fps[fps_index - 1]
            tune_msg = "fps {} -> {}".format(available_fps[fps_index],
                                             available_fps[fps_index - 1])

        if not tune_msg and resolution_index - 1 >= 0:
            next_video_conf["resolution"] = available_resolution[resolution_index - 1]
            tune_msg = "resolution {} -> {}".format(available_resolution[resolution_index],
                                                    available_resolution[resolution_index - 1])

    prev_video_conf[job_uid] = next_video_conf
    prev_flow_mapping[job_uid] = next_flow_mapping
    prev_runtime_info[job_uid] = runtime_info

    print(prev_flow_mapping[job_uid])
    print(prev_video_conf[job_uid])
    print(prev_runtime_info[job_uid])
    root_logger.info("tune_msg: {}".format(tune_msg))

    return prev_video_conf[job_uid], prev_flow_mapping[job_uid]


# -----------------
# ---- 调度入口 ----
def scheduler(
        job_uid=None,
        dag=None,
        resource_info=None,
        runtime_info=None,
        user_constraint=None,
        pid_controller=None
):
    assert job_uid, "should provide job_uid for scheduler to get prev_plan of job"

    assert pid_controller, 'PID Controller is None!'

    root_logger.info(
        "scheduling for job_uid-{}, runtime_info=\n{}".format(job_uid, runtime_info))

    global lastTime

    if not runtime_info or not user_constraint:
        root_logger.info("to get COLD start executation plan")
        return get_cold_start_plan(
            job_uid=job_uid,
            dag=dag,
            resource_info=resource_info,
            user_constraint=user_constraint
        )

    # ---- 若有负反馈结果，则进行负反馈调节 ----
    global prev_video_conf, prev_flow_mapping

    assert job_uid in prev_video_conf, \
        "job_uid not in prev_video_conf(keys={})".format(
            prev_video_conf.keys())
    assert job_uid in prev_flow_mapping, \
        "job_uid not in prev_video_conf(keys={})".format(
            prev_flow_mapping.keys())

    video_conf = None
    flow_mapping = None

    # TODO：参照对应的边端sniffer解析运行时情境
    print('---- runtime_info in the past time slot ----')
    print('runtime_info = {}'.format(runtime_info))

    if 'delay' not in runtime_info:
        return None, None
    avg_delay = runtime_info['delay']
    output = pid_controller.update(avg_delay)

    sess = requests.Session()
    sess.post(url=f'http://127.0.0.1:{drl_config["port"]}/drl/state', json={
        'pid_output': output,
        'resource_info': resource_info,
        'runtime_info': runtime_info,
        'user_constraint': user_constraint
    })

    # adjust parameters

    return adjust_parameters(output, job_uid=job_uid,
                             dag=dag,
                             user_constraint=user_constraint,
                             resource_info=resource_info,
                             runtime_info=runtime_info)

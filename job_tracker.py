import cv2
import numpy
import flask
import random
import requests
import threading
import time
import functools
import argparse

import field_codec_utils
from logging_utils import root_logger

# SingleFrameGenerator的数据生成函数
def init_task_wrapper(manager, video_id):
    # # 模拟产生数据
    # frame = list()
    # for i in range(10):
    #     frame.append(list())
    #     for j in range(3):
    #         frame[i].append(random.randint(1,1000))
    # import numpy
    # frame = numpy.array(frame)

    # 从视频流读取一帧
    video_cap_dict = manager.get_video_cap_dict()
    assert video_id in video_cap_dict.keys()

    video_cap = video_cap_dict[video_id]
    
    ret, frame = video_cap.read()
    assert ret

    input_ctx = dict()
    input_ctx['image'] = (video_cap.get(cv2.CAP_PROP_POS_FRAMES), numpy.array(frame).shape)
    # input_ctx['image'] = field_codec_utils.encode_image(frame)

    root_logger.warning("only unsupport init task with one image frame as input")
    
    return input_ctx

class Manager():
    BEGIN_TASKNAME = "Start"
    BEGIN_TOPO_STEP = 0
    END_TASKNAME = "End"
    END_TOPO_STEP = 256
    generator_func = {
        "SingleFrameGenerator": functools.partial(init_task_wrapper)
    }

    def __init__(self):
        self.cloud_addr = None
        self.local_addr = None

        # keepalive的http客户端
        self.sess = requests.Session()

        # 本地视频流
        self.video_info_list = [
            {"id": 0, "type": "student in classroom"},
            {"id": 1, "type": "people in meeting-room"}
        ]
        self.video_cap_dict = {
            0: cv2.VideoCapture("input.mov"),
            1: cv2.VideoCapture("input1.mp4")
        }

        # 模拟数据库：记录用户提交的job以及该job的执行结果
        self.global_job_count = 0
        self.job_dict = dict()
        self.job_result_dict = dict()

    def get_video_cap_dict(self):
        return self.video_cap_dict

    def set_cloud_addr(self, cloud_ip, cloud_port):
        self.cloud_addr = cloud_ip + ":" + str(cloud_port)
    
    def join_cloud(self, local_port):
        # 接入云端，汇报自身信息
        for video_info in self.video_info_list:
            r = self.sess.post(url="http://{}/node/update_status".format(self.cloud_addr),
                              json={"node_port": local_port,
                                    "video_id": video_info["id"],
                                    "video_type": video_info["type"]})
            # r = requests.post(url="http://{}/node/update_status".format(self.cloud_addr),
            #                   json={"node_port": local_port,
            #                         "video_id": video_info["id"],
            #                         "video_type": video_info["type"]})
            self.local_addr = r.json()["node_addr"]

    def generate_global_job_id(self):
        self.global_job_count += 1
        new_id = "GLOBAL_ID_" + str(self.global_job_count)
        return new_id

    def schedule_one_job(self):
        # 从队列中调度获取job
        if not self.job_dict:
            return None
        return random.choice(list(self.job_dict.values()))

    def submit_job(self, job_uid, dag_flow, dag_input, video_id, generator_func_name):
        # 启动新的job
        job = Job(job_uid=job_uid,
                  dag_flow=dag_flow, dag_input=dag_input,
                  video_id=video_id,
                  generator_func=Manager.generator_func[generator_func_name],
                  is_stream=True)
        job.set_manager(self)
        self.job_dict[job.get_job_uid()] = job

    def submit_job_result(self, job_uid, job_result, report2cloud=False):
        # 将结果保存到本地
        if job_uid not in self.job_result_dict:
            self.job_result_dict[job_uid] = list()
        self.job_result_dict[job_uid].append(job_result)

        # 将结果同步到云端（注意/node/sync_job_result的处理，要避免死循环）
        if self.cloud_addr == self.local_addr:
            root_logger.warning("{} post /node/sync_job_result to itself".format(self.local_addr))

        if report2cloud:
            r = self.sess.post(url="http://{}/node/sync_job_result".format(self.cloud_addr),
                              json={"job_uid": job_uid,
                                    "job_result": job_result})
            # r = requests.post(url="http://{}/node/sync_job_result".format(self.cloud_addr),
            #                   json={"job_uid": job_uid,
            #                         "job_result": job_result})
    
    def get_job_result(self, job_uid):
        return self.job_result_dict[job_uid]

    def restart_job(self, job):
        job.restart()

    def remove_job(self, job):
        # 根据job的id移除job
        del self.job_dict[job.get_job_uid()]

    def select_service(self, job, taskname):
        # 根据预测情况，选择task执行的节点
        serv_url = "http://192.168.56.102:9000/get_serv/{}".format(taskname)
        return serv_url

class Job():
    def __init__(self, job_uid, dag_flow, dag_input, video_id, generator_func, is_stream):
        self.job_uid = job_uid
        self.dag_flow = dag_flow
        self.dag_flow_input_deliminator = "."
        self.dag_input = dag_input
        self.video_id = video_id
        self.generator_func = generator_func
        self.topology_step = Manager.BEGIN_TOPO_STEP
        self.loop_flag = is_stream
        self.res = dict()

        self.manager = None

        # 拓扑解析dag图
        # NOTES: 目前仅支持流水线
        #        Start -> D -> C -> End
        #          0      1    2     3
        self.next_task_list = dict()
        self.prev_task_list = dict()

        assert isinstance(self.dag_flow, list)

        prev_taskname = Manager.BEGIN_TASKNAME
        curr_step = Manager.BEGIN_TOPO_STEP
        for name in self.dag_flow:
            self.next_task_list[curr_step] = [ name ]
            self.prev_task_list[name] = [ prev_taskname ]
            curr_step += 1
            prev_taskname = name
        self.next_task_list[curr_step] = [ Manager.END_TASKNAME ]
        self.prev_task_list[ Manager.END_TASKNAME ] = [ self.dag_flow[-1] ]

    def set_manager(self, manager):
        self.manager = manager
        assert isinstance(self.manager, Manager)

    def get_job_uid(self):
        return self.job_uid

    def get_loop_flag(self):
        return self.loop_flag

    def get_job_result(self):
        # return self.get_task_result(taskname=self.prev_task_list[Manager.END_TASKNAME][0],
        #                             field="head_pose")
        return self.get_task_result(taskname="SingleFrameGenerator",
                                    field="image")

    def restart(self):
        self.res = dict()
        self.topology_step = Manager.BEGIN_TOPO_STEP

    def is_end(self):
        return self.next_task_list[self.topology_step][0] == Manager.END_TASKNAME

    def store_task_result(self, done_taskname, output_ctx):
        self.res[done_taskname] = output_ctx

    def get_task_result(self, taskname, field):
        # 对数据生成的task，通过generator_func获取输入（需要实现为幂等）
        if taskname in Manager.generator_func.keys() and taskname not in self.res.keys():
            self.res[taskname] = self.generator_func(self.manager, self.video_id)
        
        # 对其他task，直接获取Job对象中缓存的中间结果
        assert taskname in self.res.keys()
        root_logger.info("task res keys: {}".format(self.res[taskname].keys()))
        assert field in self.res[taskname].keys()

        return self.res[taskname][field]

    def update_status(self):
        self.topology_step += 1

    def get_next_task_list(self):
        # taskname拓扑排序聚合
        return self.next_task_list[self.topology_step]

    def get_task_input(self, curr_taskname):
        ctx = dict()
        # 根据当前任务，寻找依赖任务
        # 将依赖任务的结果放进ctx返回
        for k, v in self.dag_input[curr_taskname].items():
            prev_taskname = v.split(self.dag_flow_input_deliminator)[0]
            prev_field = v.split(self.dag_flow_input_deliminator)[1]
            ctx[k] = self.get_task_result(taskname=prev_taskname, field=prev_field)
        return ctx




# 单例变量：主线程任务管理器，Manager
manager = Manager()
# 单例变量：后台web线程
app = flask.Flask(__name__)

# 模拟数据库
# 单例变量：接入到当前节点的节点信息
node_status = dict()




# 外部接口：从云端接受用户提交的job参数，包括指定的DAG、数据来源，将/node/submit_job的注册结果返回
@app.route("/user/submit_job", methods=["POST"])
def user_submit_job_cbk():
    # 获取用户针对视频流提交的job，转发到对应边端
    para = flask.request.json
    root_logger.info("{}".format(para))
    node_addr = para['node_addr']
    video_id = para['video_id']

    if node_addr not in node_status:
        return flask.jsonify({"status": 1, "error": "cannot found {}".format(node_addr)})
    
    # TODO：切分DAG产生多个SUB_ID
    new_req_para = dict()
    new_req_para["unique_job_id"] = manager.generate_global_job_id() + "." + "SUB_ID"
    new_req_para.update(para)
    r = manager.sess.post(url="http://{}/node/submit_job".format(node_addr),
                      json=new_req_para)
    # r = requests.post(url="http://{}/node/submit_job".format(node_addr),
    #                   json=new_req_para)

    if r.ok:
        root_logger.info("got ret: {}".format(r.json()))
        return flask.jsonify(r.json())

    return flask.jsonify(r.text)

# 外部接口：从云端获取job执行结果，需要传入job_uid
@app.route("/user/sync_job_result/<job_uid>", methods=["GET"])
def user_sync_job_result_cbk(job_uid):
    job_result = manager.get_job_result(job_uid=job_uid)
    return flask.jsonify({"status": 0,
                          "result": job_result})

# 内部接口：节点间同步job的执行结果到本地
@app.route("/node/sync_job_result", methods=["POST"])
def node_sync_job_result_cbk():
    para = flask.request.json

    # NOTES：防止再次请求/node/sync_job_result接口，否则死锁
    manager.submit_job_result(job_uid=para["job_uid"],
                              job_result=para["job_result"],
                              report2cloud=False)
    
    return flask.jsonify({"status": 200})

# 内部接口：接受其他节点传入的job初始化参数，在本地生成可以执行的job
@app.route("/node/submit_job", methods=["POST"])
def node_submit_job_cbk():
    # 获取产生job的初始化参数
    para = flask.request.json
    root_logger.info("got {}".format(para))
    generator_func_name = para["generator"]
    if generator_func_name not in Manager.generator_func:
        return flask.jsonify({"status": 1, "msg": "unsupport generator func name"})
    manager.submit_job(job_uid=para["unique_job_id"],
                       dag_flow=para["dag"]["flow"],
                       dag_input=para["dag"]["input"],
                       video_id=para["video_id"],
                       generator_func_name=generator_func_name)
    return flask.jsonify({"status": 0,
                          "msg": "submitted to manager from api: node/submit_job",
                          "job_uid": para["unique_job_id"]})

# 内部接口：其他节点接入当前节点时，需要上传节点状态
@app.route("/node/update_status", methods=["POST"])
def node_update_status_cbk():
    para = flask.request.json
    root_logger.info("from {}: got {}".format(flask.request.remote_addr, para))
    node_ip = flask.request.remote_addr
    node_port = para['node_port']
    node_addr = node_ip + ":" + str(node_port)
    video_id = para['video_id']
    video_type = para['video_type']

    if node_addr not in node_status:
        node_status[node_addr] = dict()
    if "video" not in node_status[node_addr]:
        node_status[node_addr]["video"] = dict()
    if video_id not in node_status[node_addr]["video"]:
        node_status[node_addr]["video"][video_id] = dict()
        
    node_status[node_addr]["video"][video_id].update({"type": video_type})

    return flask.jsonify({"status": 0, "node_addr": node_addr})

# 内部接口：获取当前节点所知的所有节点信息
@app.route("/node/get_all_status")
def node_get_all_status_cbk():
    return flask.jsonify({"status": 0, "data": node_status})


def start_dag_listener(serv_port=6000):
    app.run(host="0.0.0.0", port=serv_port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--side', dest='side', type=str, required=True)
    parser.add_argument('--cloud_ip', dest='cloud_ip', type=str, default='192.168.56.102')
    parser.add_argument('--cloud_port', dest='cloud_port', type=int, default=6000)
    parser.add_argument('--local_port', dest='local_port', type=int, default=6001)
    args = parser.parse_args()

    # 云端Manager的背景线程：接收节点接入、用户提交任务
    threading.Thread(target=start_dag_listener,
                     args=(args.cloud_port,),
                     daemon=True).start()

    # 工作节点Manager的背景线程：接收云端下发的job
    threading.Thread(target=start_dag_listener,
                     args=(args.local_port,),
                     daemon=True).start()

    manager.set_cloud_addr(cloud_ip=args.cloud_ip, cloud_port=args.cloud_port)
    manager.join_cloud(local_port=args.local_port)
    # if args.side == 'e':
    #     manager.join_cloud(local_port=args.local_port)

    # 工作线程：非抢占式调度job执行，每次调度后执行一个“拓扑步”
    while True:
        job = manager.schedule_one_job()
        if job is None:
            root_logger.warning("no job, sleep for 4 sec")
            time.sleep(4)
            continue

        root_logger.info("got job - {}".format(job))
        next_task_list = job.get_next_task_list()
        root_logger.info("got job next_task_list - {}".format(next_task_list))
        
        for taskname in next_task_list:
            input_ctx = job.get_task_input(taskname)
            root_logger.info("get input_ctx({}) of taskname({})".format(input_ctx, taskname))

            serv_url = manager.select_service(job, taskname)
            root_logger.info("get serv_url={}".format(serv_url))
            
            # r = requests.post(url=serv_url, json=input_ctx)
            r = manager.sess.post(url=serv_url, json=input_ctx)

            root_logger.info("got serv result: {}".format(r.text))
            job.store_task_result(taskname, r.json())
            # job.store_task_result(taskname, r.output_ctx)
        
        job.update_status()
        
        if job.is_end():
            # 当前job完成后，立刻汇报结果
            manager.submit_job_result(job_uid=job.get_job_uid(),
                                      job_result=job.get_job_result(),
                                      report2cloud=True)
            
            # 若当前job未完成对应数据流的处理，则重启当前job
            # NOTES：job的generator_func需要维护状态机，持续生成数据
            if job.get_loop_flag():
                root_logger.info("restart job: {}".format(job))
                manager.restart_job(job)
            else:
                root_logger.info("remove job: {}".format(job))
                manager.remove_job(job)

        time.sleep(1)




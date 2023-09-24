"""
采用深度强化学习算法周期性更新PID参数

"""
import flask
import flask_cors
from werkzeug.serving import WSGIRequestHandler
import threading
import yaml_utils
import time
import requests

from drl.sac_agent import SAC_Agent
from drl.Adapter import *
from drl.ReplayBuffer import RandomBuffer

configs = yaml_utils.read_yaml('configure.yaml')
drl_config = configs['drl']
pid_config = configs['pid']
cloud_configs = configs['cloud']

drl_agent_params = drl_config['agent']

WSGIRequestHandler.protocol_version = "HTTP/1.1"
drl_app = flask.Flask(__name__)
flask_cors.CORS(drl_app)

kp = pid_config['kp']
ki = pid_config['ki']
kd = pid_config['kd']

state_history = []


class EnvSimulator:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self):
        pass

    def close(self):
        pass

    def render(self):
        pass


def train_agent():
    model = SAC_Agent(**drl_agent_params)


@drl_app.route("/drl/parameter", methods=["GET"])
def get_pid_parameter():
    return flask.jsonify({'kp': kp, 'ki': ki, 'kd': kd})


@drl_app.route('/drl/state', methods=["POST"])
def post_pid_output():
    para = flask.request.json
    state_history.append(para)
    return flask.jsonify({"status": 0, "msg": "post state of pid to url /drl/output"})


def start_drl_listener(serv_port):
    drl_app.run(host="0.0.0.0", port=serv_port)


if __name__ == '__main__':
    threading.Thread(target=start_drl_listener,
                     args=(drl_config['port'],),
                     name="TrackerFlask",
                     daemon=True).start()

    time.sleep(1)

    if drl_config['mode'] == 'train':
        train_agent()
    elif drl_config['mode'] == 'test':
        pass
    elif drl_config['mode'] == 'inference':
        pass
    else:
        raise Exception(f'illegal mode of drl: {drl_config["mode"]}, please choose in [train, test, inference]')

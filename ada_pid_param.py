"""
采用深度强化学习算法周期性更新PID参数

"""
import copy

import flask
import flask_cors
import numpy as np
from werkzeug.serving import WSGIRequestHandler
import threading
import yaml_utils
import time
import requests
import os

from drl.sac_agent import SAC_Conv_Agent
from drl.Adapter import *
from drl.ReplayBuffer import RandomBuffer

configs = yaml_utils.read_yaml('configure.yaml')
drl_config = configs['drl']
pid_config = configs['pid']
cloud_configs = configs['cloud']

drl_agent_params = drl_config['agent']
drl_train_params = drl_config['train']

flask.Flask.logger_name = "listlogger"
WSGIRequestHandler.protocol_version = "HTTP/1.1"
drl_app = flask.Flask(__name__)
flask_cors.CORS(drl_app)

kp = pid_config['kp']
ki = pid_config['ki']
kd = pid_config['kd']

state_history = []

lock = threading.Lock()


class EnvSimulator:
    def __init__(self):
        self.state_buffer = [[], [], [], []]
        # self.state_buffer = []

    def reset(self):
        global kp, ki, kd
        kp = pid_config['kp']
        ki = pid_config['ki']
        kd = pid_config['kd']

        return self.get_batch_state()

    def step(self, action):
        global kp, ki, kd
        kp = action[0]
        ki = action[1]
        kd = action[2]

        time.sleep(3)

        state = self.get_batch_state()
        reward = self.cal_reward(state[1])
        done = False
        info = ''
        return state, reward, done, info

    def close(self):
        pass

    def render(self):
        pass

    def cal_reward(self, pid_output):
        reward = 0
        for i in pid_output:
            reward -= i * i
        return reward

    def get_batch_state(self):
        global state_history
        history_len = drl_agent_params['state_dim'][1]
        # state = copy.deepcopy(self.state_buffer)
        state = [[], [], [], []]
        # state = []
        while True:
            print(f'state buffer len: {len(state[0])}')
            if len(state[0]) >= history_len:
                # self.state_buffer = copy.deepcopy([s[history_len:] for s in state])
                state = copy.deepcopy([s[:history_len] for s in state])
                # state = copy.deepcopy(state[:30])
                return state
            with lock:
                history = copy.deepcopy(state_history)
                state_history.clear()

            for h in history:
                state[0].append(h['runtime_info']['delay'])
                state[1].append(h['pid_output'])
                state[2].append(h['runtime_info']['obj_n'])
                state[3].append(h['runtime_info']['obj_size'])
                print('pid_out: ', h['pid_output'])
                # state.append([h['runtime_info']['delay'], h['pid_output'], h['runtime_info']['obj_n'], h['runtime_info']['obj_size']])

            time.sleep(2)


# def evaluate_policy(env, model, render, steps_per_epoch, max_action, EnvIdex):
#     scores = 0
#     turns = opt.eval_turn
#     for j in range(turns):
#         s, done, ep_r = env.reset(), False, 0
#         while not done:
#             # Take deterministic actions at test time
#             a = model.select_action(s, deterministic=True, with_logprob=False)
#             act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
#             s_prime, r, done, info = env.step(act)
#             # r = Reward_adapter(r, EnvIdex)
#             ep_r += r
#             s = s_prime
#             if render:
#                 env.render()
#         # print(ep_r)
#         scores += ep_r
#     return scores/turns

def train_agent():
    model = SAC_Conv_Agent(**drl_agent_params)
    state_dim = drl_agent_params['state_dim']
    action_dim = drl_agent_params['action_dim']
    max_action = pid_config['parameter_bounding']
    save_interval = drl_train_params['save_interval']

    update_every = drl_train_params['update_every']
    update_after = drl_train_params['update_after']

    total_steps = drl_train_params['total_steps']

    env = EnvSimulator()

    if not os.path.exists('model'):
        os.mkdir('model')

    if drl_train_params['load_model']:
        model.load(drl_train_params['model_index'])

    replay_buffer = RandomBuffer(state_dim, action_dim, max_size=int(1e6), device=drl_agent_params['device'])

    s, done, cur_step = env.reset(), False, 0

    for t in range(total_steps):
        cur_step += 1
        '''Interact & trian'''

        s = np.asarray(s)
        a = model.select_action(s, deterministic=False, with_logprob=False)  # a∈[-1,1]
        act = Action_adapter(a, max_action)  # act∈[-max,max]

        s_prime, r, done, info = env.step(act)
        dead = Done_adapter(done, t)
        replay_buffer.add(s, a, r, s_prime, dead)
        print('cur_step:', t + 1, 'score:', r)
        s = s_prime

        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                model.train(replay_buffer)

        '''save model'''
        if (t + 1) % save_interval == 0:
            model.save(t + 1)

        # if (t + 1) % eval_interval == 0:
        #     score = evaluate_policy(eval_env, model, False, steps_per_epoch, max_action, EnvIdex)

        # print('cur_step:', t + 1, 'score:', s)

        if dead:
            s, done, current_steps = env.reset(), False, 0

    env.close()


@drl_app.route("/drl/parameter", methods=["GET"])
def get_pid_parameter():
    return flask.jsonify({'kp': float(kp), 'ki': float(ki), 'kd': float(kd)})


@drl_app.route('/drl/state', methods=["POST"])
def post_pid_output():
    para = flask.request.json
    with lock:
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
    elif drl_config['mode'] == 'fixed':
        while True:
            pass
    else:
        raise Exception(f'illegal mode of drl: {drl_config["mode"]}, please choose in [train, test, inference, fixed]')

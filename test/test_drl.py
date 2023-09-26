import time

import requests


def test_drl_capacity():
    sess = requests.Session()
    delay = 3
    pid_output = 2
    obj_size = 3
    obj_num = 10
    resource_info = {}
    user_constraint = {}
    runtime_info = {'delay':delay, 'obj_n':obj_num, 'obj_size':obj_size}


    while True:
        time.sleep(2)
        r = sess.get(url=f'http://127.0.0.1:6666/drl/parameter')
        time.sleep(1)

        sess.post(url=f'http://127.0.0.1:6666/drl/state', json={
            'pid_output': pid_output,
            'resource_info': resource_info,
            'runtime_info': runtime_info,
            'user_constraint': user_constraint
        })


if __name__ == '__main__':
    test_drl_capacity()

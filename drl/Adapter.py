'''
Adapters of different Env, Used for better training.
See https://zhuanlan.zhihu.com/p/409553262 for better understanding.
'''


def Done_adapter(done, t):
    if t % 100 == 0:
        return not done
    else:
        return done


def Action_adapter(a, max_action):
    # from [-1,1] to [-max,max]
    return abs(a) * max_action


def Action_adapter_reverse(act, max_action):
    # from [-max,max] to [-1,1]
    return act / max_action

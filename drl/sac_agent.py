import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_net(layer_shape, activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


def build_conv1d_net(input_shape, output_shape, kernel_size):
    layers = [nn.Conv1d(in_channels=input_shape,
                        out_channels=output_shape,
                        kernel_size=kernel_size), nn.ReLU()]

    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
        super(Actor, self).__init__()

        layers = [state_dim] + list(hid_shape)
        self.a_net = build_net(layers, h_acti, o_acti)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic=False, with_logprob=True):
        '''Network with Enforcing Action Bounds'''
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            u = mu
        else:
            u = dist.rsample()  # '''reparameterization trick of Gaussian'''#
        a = torch.tanh(u)

        if with_logprob:
            # get probability density of logp_pi_a from probability density of u, which is given by the original paper.
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
                axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2


class Actor_Conv(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, conv_kernel_size,
                 conv_out_dim, h_acti=nn.ReLU, o_acti=nn.ReLU):
        super(Actor_Conv, self).__init__()

        self.conv_net = build_conv1d_net(state_dim[0], conv_out_dim, conv_kernel_size)
        # for i in range(state_dim[1]):
        #     self.conv_net.append(build_conv1d_net(state_dim[0], conv_out_dim, conv_kernel_size))

        # print('conv_out:', self._get_conv_out(state_dim))
        layers = [self._get_conv_out(state_dim)] + list(hid_shape)
        self.a_net = build_net(layers, h_acti, o_acti)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic=False, with_logprob=True):
        '''Network with Enforcing Action Bounds'''
        # state_out = torch.tensor([])
        # for i in range(len(self.conv_net)):
        #     if i == 0:
        #         state_out = self.conv_net[i](state[i].view(1, 1, -1)).view(1, -1)
        #     else:
        #         state_out = torch.hstack((state_out, self.conv_net[i](state[i]).view(1, -1)))
        # print(self.conv_net(state).size())
        conv_out = self.conv_net(state)
        if len(conv_out.shape) == 2:
            state_out = conv_out.view(1, -1)
        elif len(conv_out.shape) == 3:
            state_out = conv_out.view(conv_out.size(0), -1)
        else:
            state_out = conv_out
        # print('state_out:', state_out.shape)
        net_out = self.a_net(state_out)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            u = mu
        else:
            u = dist.rsample()  # '''reparameterization trick of Gaussian'''#
        a = torch.tanh(u)

        if with_logprob:
            # get probability density of logp_pi_a from probability density of u, which is given by the original paper.
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
                axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a

    def _get_conv_out(self, shape):
        o = self.conv_net(torch.zeros(1, *shape))
        # print('conv_out:', o.view(1,-1).size())
        # print(int(np.prod(o.view(1,-1).size())))
        return int(np.prod(o.view(1,-1).size()))


class Q_Critic_Conv(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, conv_kernel_size,
                 conv_out_dim):
        super(Q_Critic_Conv, self).__init__()

        # self.conv_net = []
        # for i in range(state_dim[1]):
        #     self.conv_net.append(build_conv1d_net(state_dim[0], conv_out_dim, conv_kernel_size))

        self.conv_net = build_conv1d_net(state_dim[0], conv_out_dim, conv_kernel_size)

        layers = [self._get_conv_out(state_dim) + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        # state_out = torch.tensor([])
        # for i in range(len(self.conv_net)):
        #     if i == 0:
        #         state_out = self.conv_net[i](state[i]).view(1, -1)
        #     else:
        #         state_out = torch.hstack((state_out, self.conv_net[i](state[i]).view(1, -1)))

        conv_out = self.conv_net(state)
        if len(conv_out.shape) == 2:
            state_out = conv_out.view(1, -1)
        elif len(conv_out.shape) == 3:
            state_out = conv_out.view(conv_out.size(0), -1)
        else:
            state_out = conv_out

        # print('state:', state_out.size())
        # print('action:', action.size())

        sa = torch.cat([state_out, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2

    def _get_conv_out(self, shape):
        o = self.conv_net(torch.zeros(1, *shape))
        return int(np.prod(o.view(1, -1).size()))


class SAC_Agent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            gamma=0.99,
            hid_shape=(256, 256),
            a_lr=3e-4,
            c_lr=3e-4,
            batch_size=256,
            alpha=0.2,
            adaptive_alpha=True,
            device='cpu'
    ):

        self.actor = Actor(state_dim, action_dim, hid_shape).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.q_critic = Q_Critic(state_dim, action_dim, hid_shape).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = 0.005
        self.batch_size = batch_size

        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        if adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-action_dim, dtype=float, requires_grad=True, device=device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=c_lr)

        self.device = device

    def select_action(self, state, deterministic, with_logprob=False):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            a, _ = self.actor(state, deterministic, with_logprob)
        return a.cpu().numpy().flatten()

    def train(self, replay_buffer):
        s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)

        # ----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_prime, log_pi_a_prime = self.actor(s_prime)
            target_Q1, target_Q2 = self.q_critic_target(s_prime, a_prime)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (1 - dead_mask) * self.gamma * (
                    target_Q - self.alpha * log_pi_a_prime)  # Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.q_critic.parameters():
            params.requires_grad = False

        a, log_pi_a = self.actor(s)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = True
        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
            # if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # ----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, episode):
        torch.save(self.actor.state_dict(), "./model/sac_actor{}.pth".format(episode))
        torch.save(self.q_critic.state_dict(), "./model/sac_q_critic{}.pth".format(episode))

    def load(self, episode):
        self.actor.load_state_dict(torch.load("./model/sac_actor{}.pth".format(episode)))
        self.q_critic.load_state_dict(torch.load("./model/sac_q_critic{}.pth".format(episode)))


class SAC_Conv_Agent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            gamma=0.99,
            hid_shape=(256, 256),
            conv_kernel_size=4,
            conv_out_dim=128,
            a_lr=3e-4,
            c_lr=3e-4,
            tau=0.005,
            batch_size=256,
            alpha=0.2,
            adaptive_alpha=True,
            device='cpu'
    ):

        self.actor = Actor_Conv(state_dim, action_dim, hid_shape, conv_kernel_size,
                                conv_out_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.q_critic = Q_Critic_Conv(state_dim, action_dim, hid_shape, conv_kernel_size,
                                      conv_out_dim).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        if adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-action_dim, dtype=float, requires_grad=True, device=device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=c_lr)

        self.device = device

    def select_action(self, state, deterministic, with_logprob=False):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            a, _ = self.actor(state, deterministic, with_logprob)
        return a.cpu().numpy().flatten()

    def train(self, replay_buffer):
        s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)

        # ----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_prime, log_pi_a_prime = self.actor(s_prime)
            target_Q1, target_Q2 = self.q_critic_target(s_prime, a_prime)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (1 - dead_mask) * self.gamma * (
                    target_Q - self.alpha * log_pi_a_prime)  # Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.q_critic.parameters():
            params.requires_grad = False

        a, log_pi_a = self.actor(s)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = True
        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
            # if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # ----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, episode):
        torch.save(self.actor.state_dict(), "./model/sac_actor{}.pth".format(episode))
        torch.save(self.q_critic.state_dict(), "./model/sac_q_critic{}.pth".format(episode))

    def load(self, episode):
        self.actor.load_state_dict(torch.load("./model/sac_actor{}.pth".format(episode)))
        self.q_critic.load_state_dict(torch.load("./model/sac_q_critic{}.pth".format(episode)))

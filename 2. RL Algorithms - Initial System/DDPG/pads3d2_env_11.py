import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class PADS(gym.Env):
    def __init__(self, dt=0.5, h_max=300.0, a_max=0.6):
        self.h_speed = 6.80
        self.v_speed = 3.90
        self.finesse = 1.75
        self.k_psi = 1.70
        self.t_psi = 0.80

        self.dt = dt
        self.h_end = 0.0
        self.w1 = 0.0001
        self.w2 = 10.0
        self.w3 = 1.0
        self.wr = 0.0
        self.sz = 1

        self.j = 0
        self.k = 0

        self.act_max = a_max
        self.action_space = spaces.Box(low = -self.act_max, high = self.act_max, shape = (1,), dtype = np.float32)
        self.alt_max = h_max
        self.dist_max = self.alt_max * self.finesse * 1.2
        self.r_max = self.k_psi * 1.5
        self.low_obs = np.array([-self.dist_max, -self.dist_max,          0.0, -np.pi, -self.r_max], np.float32)
        self.high_obs = np.array([self.dist_max,  self.dist_max, self.alt_max,  np.pi,  self.r_max], np.float32)
        self.observation_space = spaces.Box(low = self.low_obs, high = self.high_obs, shape = (5,), dtype = np.float32)
        self.state = np.zeros(5)
        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def dyn(self, x, u, w):
        psi = x[3]
        r = x[4]
        x_dot = self.h_speed * np.cos(psi) + w[0]
        y_dot = self.h_speed * np.sin(psi) + w[1]
        h_dot = -self.v_speed - w[2]
        psi_dot = r
        r_dot = (self.k_psi * u - r) / self.t_psi
        return np.array([x_dot, y_dot, h_dot, psi_dot, r_dot])

    def reward(self, obs):
        done = bool(self.state[2] < self.h_end)
        rew = 0
        if done:
            cost = self.w1 * (obs[0] ** 2 + obs[1] ** 2) + self.w2 * (obs[3] ** 2) + self.w3 * (obs[4] ** 2)
            rew = 100.0 * np.exp(-cost)
            print(f"recompensa: {rew}")
            print(f"angulo: {self.state[3]}")
            print(f"vel_ang: {self.state[4]}")
            dist = np.sqrt((obs[0] - self.k) ** 2 + (obs[1] - self.j) ** 2)
            print(f"Distancia ponto-otimo: {dist}")
            if self.k == 0 and self.j == 0:
                print(f"Distancia ponto-ponto: {dist}")
            else:
                print(f"Distancia ponto-pfora: {dist}")
        return rew, done

    def step(self, act):
        new_state = self.state + self.dyn(self.state, act[0], np.zeros(3)) * self.dt
        new_state[3] = (new_state[3] + np.pi) % (2*np.pi) - np.pi
        self.state = new_state
        rew, done = self.reward(self.state)
        return self.state, rew, done, {}

    def reset(self):
        disp_z = 0.6
        low_obs = np.array([-self.dist_max, -self.dist_max, self.alt_max*disp_z, -np.pi, 0], np.float32)
        high_obs = np.array([self.dist_max,  self.dist_max, self.alt_max,         np.pi, 0], np.float32)
        state = self.np_random.uniform(low = low_obs, high = high_obs, size = (5,))
        state[3] = (state[3] + np.pi) % (2*np.pi) - np.pi
        self.state = state

        cone = self.state[2] * self.finesse
        dis_agent_cone = np.sqrt(self.state[0]**2 + self.state[1]**2)
        print(f"s_inicial = ({self.state[0]}, {self.state[1]}, {self.state[2]}, {self.state[3]}, {self.state[4]})")
        if dis_agent_cone < cone:
            self.j = 0
            self.k = 0
        else:
            self.k = ((self.state[0] * ((dis_agent_cone) - cone)) / dis_agent_cone)
            self.j = ((self.state[1] * ((dis_agent_cone) - cone)) / dis_agent_cone)

        return self.state


    def render(self, mode='human'):
        pass

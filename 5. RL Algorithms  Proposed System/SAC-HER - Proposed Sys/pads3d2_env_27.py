import gym
from gym import spaces, GoalEnv
from gym.spaces import Dict, Discrete
from gym.utils import seeding
import numpy as np
from collections import OrderedDict
from typing import Any, Dict, Optional, Union
from stable_baselines3.common.type_aliases import GymStepReturn

class PADS(GoalEnv):
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
        self.desired_goal = np.array([   0,  0, 0,        0,             0], np.float32)
        self.desired_low  = np.array([ -30,-30, 0, -np.pi/3, -self.r_max/3], np.float32)
        self.desired_high = np.array([  30, 30, 0,  np.pi/3,  self.r_max/3], np.float32)
        self.observation_space = spaces.Dict({
            'observation':      spaces.Box(low = self.low_obs,      high = self.high_obs,       shape = (5,), dtype = np.float32),
            'achieved_goal':    spaces.Box(low = self.low_obs,      high = self.high_obs,       shape = (5,), dtype = np.float32),
            'desired_goal':     spaces.Box(low = self.low_obs,      high = self.high_obs,       shape = (5,), dtype = np.float32)
        })
        self.state = None
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

    def convert_to_bit_vector(self, state: Union[int, np.ndarray], batch_size: int) -> np.ndarray:
        if isinstance(state, int):
            state = np.array(state).reshape(batch_size, -1)
            state = (((state[:, :] & (1 << np.arange(len(self.state))))) > 0).astype(int)
        else:
            state = np.array(state).reshape(batch_size, -1)
        return state

    def _get_obs(self):
        return OrderedDict(
            [
                ("observation", self.state.copy()),
                ("achieved_goal", self.state.copy()),
                ("desired_goal", self.desired_goal.copy())
            ]
        )

    def step(self, act):
        new_state = self.state + self.dyn(self.state, act[0], np.zeros(3)) * self.dt
        new_state[3] = (new_state[3] + np.pi) % (2*np.pi) - np.pi
        self.state = new_state
        obs = self._get_obs()
        done = bool(self.state[2] < self.h_end)
        rew = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None))
        return obs, rew, done, {}

    def reset(self):
        disp_z = 0.6
        altura = self.np_random.uniform(low = self.alt_max*disp_z, high = self.alt_max)
        cone = altura * self.finesse * 1.2
        x = self.np_random.uniform(low = -cone, high = cone)
        int_y = np.sqrt(cone**2 - x**2)
        y = self.np_random.uniform(low = -int_y, high = int_y)
        pipi = self.np_random.uniform(low = -np.pi, high = np.pi)
        pipi = (pipi + np.pi) % (2*np.pi) - np.pi
        quisi = self.np_random.uniform(low = 0, high = 0)
        state = np.array([x, y, altura, pipi, quisi])
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
        return self._get_obs()

    def compute_reward(self, achieved_goal, desired_goal, _info):
        if isinstance(achieved_goal, int):
            batch_size = 1
        else:
            batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
        desired_goal = self.convert_to_bit_vector(desired_goal, batch_size)
        achieved_goal = self.convert_to_bit_vector(achieved_goal, batch_size)

        cone = self.state[2] * self.finesse + 100
        dis_agent_cone = np.sqrt((self.state[0] - self.k)**2 + (self.state[1] - self.j)**2)
        if dis_agent_cone < cone:
            rew = 1
        else:
            rew = 0
        if bool(self.state[2] < self.h_end):
            cost = self.w1 * ((self.state[0] - self.k) ** 2 + (self.state[1] - self.j) ** 2) + self.w2 * (self.state[3] ** 2) + self.w3 * (self.state[4] ** 2)
            rew = 100.0 * np.exp(-cost)
            print(f"recompensa: {rew}")
            print(f"state = ({self.state[0]}, {self.state[1]}, {self.state[2]})")
            print(f"(k, j) = ({self.k}, {self.j})")
            print(f"angulo: {self.state[3]}")
            print(f"vel_ang: {self.state[4]}")
            dist = np.sqrt((self.state[0] - self.k) ** 2 + (self.state[1] - self.j) ** 2)
            print(f"Distancia ponto-otimo: {dist}")
            if self.k == 0 and self.j == 0:
                print(f"Distancia ponto-ponto: {dist}")
            else:
                print(f"Distancia ponto-pfora: {dist}")
        return rew

    def render(self, mode='human'):
        pass

    def close(self) -> None:
        pass

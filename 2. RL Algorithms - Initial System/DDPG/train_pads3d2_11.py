from pads.pads3d2_env_11 import PADS
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

if __name__ == "__main__":
    ttsteps = 250*24000
    f_out = "sac_pads3d2_11"
    pads = PADS(0.5, 300, 0.6)
    n_actions = pads.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG("MlpPolicy", pads, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=ttsteps, log_interval=10)   
    model.save(f_out)

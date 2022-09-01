from pads.pads3d2_env_21 import PADS
from stable_baselines3 import SAC

if __name__ == "__main__":
    ttsteps = 250*24000
    f_out = "sac_pads3d2_25"
    pads = PADS(0.5, 300, 0.6)
    model = SAC("MlpPolicy", pads, verbose=1)
    model.learn(total_timesteps=ttsteps, log_interval=10)   
    model.save(f_out)

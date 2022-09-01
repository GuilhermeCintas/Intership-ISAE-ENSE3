from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv 
from pads.pads3d2_env_27 import PADS
import gym.wrappers
from gym.wrappers import TimeLimit

if __name__ == "__main__":
    ttsteps = 250*24000
    f_out = "sac_pads3d2_27"
    pads = PADS(0.5, 300, 0.6)
    model = SAC(
    "MultiInputPolicy",
    pads,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
        online_sampling=True,
        max_episode_length = 100
    ),
    verbose=1,
    )
    model.learn(total_timesteps=ttsteps, log_interval=10)
    model.save(f_out)

import clip
import gymnasium as gym
import mani_skill2.envs
import numpy as np
import torch
from mani_skill2.utils.wrappers import RecordEpisode
from tqdm import tqdm

from model.mlp import MLP
from utils.data_utils import make_path
from utils.data_utils import flatten_obs
torch.manual_seed(0)
video_path = make_path('logs', 'mlp', 'videos')
model_path = make_path('logs', 'mlp','checkpoints', '25000.pt')
device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make('StackCube-v0',
               render_mode="cameras",
               enable_shadow=True,
               obs_mode="state_dict",
               control_mode="pd_joint_delta_pos", 
               max_episode_steps=300)
env = RecordEpisode(
    env,
    video_path,
    info_on_video=True,
    save_trajectory=False
)

clip_model, _ = clip.load("RN50", device=device)
text = clip.tokenize("Place the green cube on top of the red one.").to(device)
text_features = clip_model.encode_text(text).float()


mlp = MLP()
mlp.load_state_dict(torch.load(model_path))
mlp.to(device)
mlp.eval()

obs, _ = env.reset()
for i in tqdm(range(300)):
    obs = flatten_obs(obs)
    obs = torch.from_numpy(obs[None]).to(device)
    mean, sigma = mlp(text_features, obs)
    noise = torch.randn_like(sigma)
    action = mean + sigma * noise
    action = action.detach().cpu().numpy()
    obs, reward, terminated, truncated, info = env.step(action[0])
env.flush_video()  # Save the video
env.close()

import pathlib

from utils.data_utils import make_path


def make_log_dir(log_dir: str, model: str):
    checkpoint_path = make_path(log_dir, model, 'checkpoints')
    tensorboard_path = make_path(log_dir, model, 'tensorboard')
    video_path = make_path(log_dir, model, 'videos')
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(video_path).mkdir(parents=True, exist_ok=True)

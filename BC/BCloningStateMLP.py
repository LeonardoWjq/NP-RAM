import mani_skill2.envs
from torch.optim import RAdam

from model.linear import LinearRegressor
from model.mlp import MLPExtractor
from scripts.train_BC import train
from scripts.eval_BC import evaluate

if __name__ == '__main__':
    env_id = 'LiftCube-v0'
    obs_mode = 'state'
    control_mode = 'pd_ee_delta_pose'
    gpu_id = 'cuda:0'
    feature_dim = 256
    feature_extractor_kwargs = dict(feature_dim=feature_dim,
                                    layer_count=2)
    regressor_kwargs = dict(feature_dim=feature_dim,
                            layer_count=1)
    optimizer_kwargs = dict(lr=1e-3,
                            weight_decay=1e-5)

    seeds = [1999, 2023, 2028]
    for seed in seeds:
        best_ckpt, log_path = train(env_id=env_id,
                          obs_mode=obs_mode,
                          control_mode=control_mode,
                          feature_extractor_class=MLPExtractor,
                          feature_extractor_kwargs=feature_extractor_kwargs,
                          regressor_class=LinearRegressor,
                          regressor_kwargs=regressor_kwargs,
                          optimizer_class=RAdam,
                          optimizer_kwargs=optimizer_kwargs,
                          seed=seed,
                          eval_episodes=10)
        
        evaluate(env_id=env_id,
                 obs_mode=obs_mode,
                 control_mode=control_mode,
                 feature_extractor_class=MLPExtractor,
                 feature_extractor_kwargs=feature_extractor_kwargs,
                 regressor_class=LinearRegressor,
                 regressor_kwargs=regressor_kwargs,
                 epoch=best_ckpt,
                 log_path=log_path,
                 seed=seed,
                 save_video=True,
                 num_episodes=30,
                 gpu_id=gpu_id)
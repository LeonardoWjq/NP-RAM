import numpy as np
import robosuite as suite
import robomimic

# create environment instance
env = suite.make(
    env_name="PickPlaceBread", # try with other tasks like "Stack" and "Door"
    robots="Sawyer",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1):
    action = np.random.randn(env.robots[0].dof) # sample random action
    print(action.shape)
    obs, reward, done, info = env.step(action)  # take action in the environment
    for key, value in obs.items():
        print(key, value.shape)
    # env.render()  # render on display

env.close()
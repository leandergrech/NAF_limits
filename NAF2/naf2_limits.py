import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import tensorflow as tf

from envs.random_env import RandomEnv
from envs.normalize_env import NormalizeEnv
from naf2 import NAF2

random_seed = 123

if __name__ == '__main__':
    n_obs = 10
    n_act = 10
    model_name = f'NAF2_{n_obs}x{n_act}_{dt.strftime(dt.now(), "%m%d%y_%H%M")}'
    model_dir = os.path.join('models', model_name)
    log_dir = os.path.join('logs', model_name)

    rm = np.load(os.path.join('envs', 'random_env_rms', f'random_env_{n_obs}x{n_act}.npy'))

    env = NormalizeEnv(RandomEnv(n_obs, n_act, rm))
    eval_env = NormalizeEnv(RandomEnv(n_obs, n_act, rm))

    training_info = dict(polyak=0.999,
                         batch_size=100,
                         steps_per_batch=10,
                         epochs=1,
                         learning_rate=1e-3,
                         discount=0.9999)
    nafnet_info = dict(hidden_sizes=[100, 100],
                       activation=tf.nn.tanh,
                       kernel_initializer=tf.random_normal_initializer(0, 0.05, seed=random_seed))
    eval_info = dict(eval_env=eval_env,
                     frequency=100,
                     nb_episodes=10,
                     max_steps=100)

    agent = NAF2(env=env,
                 buffer_size=int(1e6),
                 train_every=1,
                 training_info=training_info,
                 eval_info=eval_info,
                 save_frequency=1000,
                 log_frequency=10,
                 directory=model_dir,
                 tb_log=log_dir,
                 q_smoothing_sigma=0.02,
                 q_smoothing_clip=0.05,
                 nafnet_info=nafnet_info)

    agent.training(warm_up_steps=1000, max_episodes=1000, max_steps=300)
import os
import numpy as np
import itertools
from tqdm import tqdm
from datetime import datetime as dt
import tensorflow as tf

from envs.random_env import RandomEnv
from envs.normalize_env import NormalizeEnv
from naf2 import NAF2

tf.get_logger().setLevel('ERROR')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

random_seed = 123
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

def train_agent(n_obs=5,
                n_act=5,
                layers=[100, 100],
                noise_decay_eps=50,
                buffer_size=(5e3),
                nb_steps=int(5e3+1),
                max_ep_steps=50,
                warm_up_steps=200,
                initial_episode_length=5,
                save_frequency=1000):
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
    nafnet_info = dict(hidden_sizes=layers,
                       activation=tf.nn.tanh,
                       kernel_initializer=tf.random_normal_initializer(0, 0.05, seed=random_seed))
    eval_info = dict(eval_env=eval_env,
                     frequency=100,
                     nb_episodes=3,
                     max_ep_steps=50)

    # linearly decaying noise function
    noise_fn = lambda act, i: act + np.random.randn(n_act) * max(1 - i/noise_decay_eps, 0)
    agent = NAF2(env=env,
                 buffer_size=buffer_size,
                 train_every=1,
                 training_info=training_info,
                 eval_info=eval_info,
                 save_frequency=save_frequency,
                 log_frequency=20,
                 directory=model_dir,
                 tb_log=log_dir,
                 # q_smoothing_sigma=0.02,
                 q_smoothing_sigma=0.02,
                 q_smoothing_clip=0.05,
                 nafnet_info=nafnet_info,
                 noise_fn=noise_fn)

    try:
        agent.training(nb_steps=nb_steps, max_ep_steps=max_ep_steps, warm_up_steps=warm_up_steps,
                       initial_episode_length=initial_episode_length)
    except KeyboardInterrupt:
        print('exiting')

    return model_name


if __name__ == '__main__':
    logfile = f'grid_search_parameters_{dt.strftime(dt.now(), "%m%d%y_%H%M")}.log'
    params = dict(n_obs=[10],
                  n_act=[10],
                  layers=[[100, 100], [200, 200]],
                  noise_decay_eps=[40, 60],
                  buffer_size=[int(5e3), int(15e3)],
                  nb_steps=[int(8e4 + 1)],
                  save_frequency=[2000])

    total = np.product([len(p) for p in params.values()])

    keys, vals = zip(*params.items())
    pbar = tqdm(total=total)

    def log_params_and_model_names(model_name, params):
        global logfile
        with open(logfile, 'a') as f:
            f.write(f'{model_name} -> {params}\n')

    for v in itertools.product(*vals):
        p = dict(zip(keys, v))
        model_name = train_agent(**p)

        log_params_and_model_names(model_name, p)
        pbar.update(1)


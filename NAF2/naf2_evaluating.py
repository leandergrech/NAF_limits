import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import tensorflow as tf

from envs.random_env import RandomEnv
from envs.normalize_env import NormalizeEnv
from naf2 import NAF2

random_seed = 123

# use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

N_EPISODES = 10
MAX_STEPS = 50
def plot_individual(agent, env, opt_env):
    global MAX_STEPS, N_EPISODES

    n_obs = env.obs_dimension
    n_act = env.act_dimension

    fig, ((ax1, ax4), (ax2, ax5)) = plt.subplots(2, 2, num=1)
    fig2, (axr, axc) = plt.subplots(2, num=2)
    o_x = range(n_obs)
    a_x = range(n_act)

    # ax1
    o_bars = ax1.bar(o_x, np.zeros(n_obs))
    ax1.axhline(0.0, color='k', ls='dashed')
    ax1.axhline(env.GOAL, color='g', ls='dashed')
    ax1.axhline(-env.GOAL, color='g', ls='dashed')
    ax1.set_title('State')
    ax1.set_ylim((-1, 1))
    # ax2
    a_bars = ax2.bar(a_x, np.zeros(n_act))
    ax2.set_title('Action')
    ax2.set_ylim((-1, 1))
    # ax4
    opt_o_bars = ax4.bar(o_x, np.zeros(n_obs))
    ax4.axhline(0.0, color='k', ls='dashed')
    ax4.axhline(env.GOAL, color='g', ls='dashed')
    ax4.axhline(-env.GOAL, color='g', ls='dashed')
    ax4.set_title('Opt State')
    ax4.set_ylim((-1, 1))
    # ax5
    opt_a_bars = ax5.bar(a_x, np.zeros(n_act))
    ax5.set_title('Opt Action')
    ax5.set_ylim((-1, 1))
    # axr
    rew_line, = axr.plot([], [], label='Agent')
    opt_rew_line, = axr.plot([], [], label='Optimal')
    axr.axhline(env.objective([env.GOAL]*2), color='g', ls='dashed', label='Reward threshold')
    axr.set_title('Reward')
    axr.set_xlabel('Steps')
    axr.set_ylabel('Reward')
    axr.legend(loc='lower right')
    # axc
    cumu_rew_line, = axc.plot([], [], label='Agent')
    cumu_opt_rew_line, = axc.plot([], [], label='Optimal')
    axc.set_title('Cumulative Reward')
    axc.set_xlabel('Steps')
    axc.set_ylabel('Reward')
    axc.legend(loc='lower right')

    def update_bars(o, a, opo, opa):
        nonlocal o_bars, a_bars, opt_o_bars, opt_a_bars, o_x, a_x
        for bar in (o_bars, a_bars, opt_o_bars, opt_a_bars):
            bar.remove()

        o_bars = ax1.bar(o_x, o, color='b')
        a_bars = ax2.bar(a_x, a, color='r')
        opt_o_bars = ax4.bar(o_x, opo, color='b')
        opt_a_bars = ax5.bar(a_x, opa, color='r')


    plt.ion()

    for ep in range(N_EPISODES):
        o = env.reset()
        opt_o = o.copy()
        opt_env.reset(opt_o)

        o_bars.remove()
        a_bars.remove()
        o_bars = ax1.bar(o_x, o, color='b')
        a_bars = ax2.bar(a_x, np.zeros(n_act))

        opt_o_bars.remove()
        opt_a_bars.remove()
        opt_o_bars = ax4.bar(o_x, opt_o, color='b')
        opt_a_bars = ax5.bar(a_x, np.zeros(n_act))

        plt.draw()
        plt.pause(2)

        rewards = []
        opt_rewards = []
        cumu_rewards = []
        cumu_opt_rewards = []
        ACTION_SCALE = 1
        for step in range(MAX_STEPS):
            # Put some obs noise to test agent
            # FInd limiting noise
            a = agent.predict(o).squeeze()
            o, r, d, _ = env.step(a)
            rewards.append(r)
            cumu_rewards.append(sum(rewards))

            opt_a = opt_env.get_optimal_action(opt_o) * ACTION_SCALE
            opt_o, opt_r, *_ = opt_env.step(opt_a)
            opt_rewards.append(opt_r)
            cumu_opt_rewards.append(sum(opt_rewards))

            fig.suptitle(f'Ep #{ep} - Step #{step} - Done {d}')
            fig2.suptitle(f'Ep #{ep} - Step #{step} - Done {d}')
            update_bars(o, a, opt_o, opt_a)

            rew_line.set_data(range(step + 1), rewards)
            cumu_rew_line.set_data(range(step + 1), cumu_rewards)
            axr.set_ylim((min(np.concatenate([rewards, opt_rewards])), 0))
            axr.set_xlim((0, step+1))

            opt_rew_line.set_data(range(step + 1), opt_rewards)
            cumu_opt_rew_line.set_data(range(step + 1), cumu_opt_rewards)
            axc.set_ylim((min(np.concatenate([cumu_rewards, cumu_opt_rewards])), 0))
            axc.set_xlim((0, step + 1))

            if plt.fignum_exists(1) and plt.fignum_exists(2):
                plt.draw()
                plt.pause(0.01)
            else:
                exit()

            if d:
                plt.pause(2)
                break

if __name__ == '__main__':
    n_obs = 5
    n_act = 5
    model_name = f'NAF2_{n_obs}x{n_act}_011821_1606'
    chkpt_step = 10000
    model_dir = os.path.join('models', model_name)
    log_dir = os.path.join('logs', model_name)

    rm = np.load(os.path.join('envs', 'random_env_rms', f'random_env_{n_obs}x{n_act}.npy'))

    env = NormalizeEnv(RandomEnv(n_obs, n_act, rm))
    opt_env = NormalizeEnv(RandomEnv(n_obs, n_act, rm))

    agent = NAF2(env=env,
                 tb_log=log_dir,
                 soft_init=True)

    # agent.training(warm_up_steps=1000, max_episodes=1000, max_steps=300)
    agent.load_checkpoint(model_dir=model_dir, chkpt_step=chkpt_step, load_buffer=False)


    plot_individual(agent, env, opt_env)

import os
import pickle
import random
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inverted_pendulum import PendulumEnv

from naf2_new import NAF

# set random seed
random_seed = 111
np.random.seed(random_seed)
random.seed(random_seed)


def plot_results(env, file_name):
    # plotting
    print('Now plotting')
    rewards = env.rewards
    initial_rewards = env.init_rewards
    # print('initial_rewards :', initial_rewards)

    iterations = []
    final_rews = []
    starts = []
    sum_rews = []
    mean_rews = []

    for i in range(len(rewards)):
        if (len(rewards[i]) > 0):
            final_rews.append(rewards[i][len(rewards[i]) - 1])

            iterations.append(len(rewards[i]))
            sum_rews.append(np.sum(rewards[i]))
            mean_rews.append(np.mean(rewards[i]))

            try:
                starts.append(initial_rewards[i])
            except:
                pass
    plot_suffix = ""  # f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'

    fig, axs = plt.subplots(2, 1)

    ax = axs[0]
    color = 'blue'
    ax.plot(iterations, c=color)
    ax.set_ylabel('steps', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax1 = plt.twinx(ax)
    color = 'k'
    ax1.plot(np.cumsum(iterations), c=color)
    ax1.set_ylabel('cumulative steps', color=color)
    ax.set_title('Iterations' + plot_suffix)
    # fig.suptitle(label, fontsize=12)

    ax = axs[1]
    color = 'red'
    ax.plot(starts, c=color)
    ax.set_ylabel('starts', color=color)
    ax.axhline(-0.05, ls=':', color='r')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title('Final reward per episode')  # + plot_suffix)
    ax.set_xlabel('# episode')

    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.set_ylabel('finals', color=color)
    ax1.axhline(-0.05, ls=':', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(final_rews, color=color)

    fig.tight_layout()
    plt.savefig(file_name + '_episodes.pdf')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    color = 'blue'
    ax.plot(sum_rews, color)
    ax.set_ylabel('cum. reward', color=color)
    ax.set_xlabel('# episode')
    ax.tick_params(axis='y', labelcolor=color)
    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.plot(mean_rews, c=color)
    ax1.set_ylabel('mean reward', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    plt.savefig(file_name + '_rewards.pdf')
    plt.show()


def plot_convergence(agent, file_name):
    losses, vs = agent.losses, agent.vs
    # losses2, vs2 = agent.losses2, agent.vs2

    fig, ax = plt.subplots()
    # ax.set_title(label)
    ax.set_xlabel('# steps')

    color = 'tab:blue'

    # ax.semilogy(losses2, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel('td_loss', color=color)
    ax.semilogy(losses, color=color)
    # ax.set_ylim(0, 1)

    ax1 = plt.twinx(ax)
    # ax1.set_ylim(-2, 1)
    color = 'lime'
    ax1.set_ylabel('V', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(vs, color=color)
    # ax1.plot(vs2, color=color)
    plt.savefig(file_name + '_convergence' + '.pdf')
    plt.show()


class MonitoringEnv(gym.Wrapper):
    '''
    Gym Wrapper show progress and visualisation
    '''

    def __init__(self, env, **kwargs):
        self.plot_label = False
        if 'plot_progress' in kwargs:
            self.plot_label = kwargs.get('plot_progress')
        if 'noise' in kwargs:
            self.noise = kwargs.get('noise')
        else:
            self.noise = 0.0

        gym.Wrapper.__init__(self, env)
        self.rewards = []
        self.init_rewards = []
        self.current_episode = -1
        self.current_step = -1

    def reset(self, **kwargs):
        self.current_step = 0
        self.current_episode += 1
        self.rewards.append([])
        obs = self.env.reset(**kwargs) + self.noise * np.random.randn(self.env.observation_space.shape[-1])
        return obs

    def step(self, action):
        self.current_step += 1
        obs, reward, done, info = self.env.step(action)
        self.rewards[self.current_episode].append(reward)

        if self.current_step >= 200:
            done = True
            if self.plot_label:
                self.plot_results(self.current_episode)
        if self.plot_label:
            # Show the environment
            self.env.showing_render()
        obs = obs + self.noise * np.random.randn(self.env.observation_space.shape[-1])
        reward = reward/10
        return obs, reward, done, info

    def plot_results(self, label):
        # plotting
        rewards = self.rewards

        iterations = []
        final_rews = []
        starts = []
        sum_rews = []
        mean_rews = []

        for i in range(len(rewards)):
            if (len(rewards[i]) > 0):
                final_rews.append(rewards[i][len(rewards[i]) - 1])

                iterations.append(len(rewards[i]))
                sum_rews.append(np.sum(rewards[i]))
                mean_rews.append(np.mean(rewards[i]))

        fig, ax = plt.subplots(1, 1)
        ax.set_title(label=label)
        color = 'blue'
        ax.plot(sum_rews, color)
        ax.set_ylabel('cum. reward', color=color)
        ax.set_xlabel('# episode')
        ax.tick_params(axis='y', labelcolor=color)
        plt.show()


if __name__ == '__main__':

    try:
        random_seed = int(sys.argv[2])
    except:
        random_seed = 25
    try:
        file_name = sys.argv[1] + '_' + str(random_seed)
    except:
        file_name = 'Data/default' + str(random_seed) + '_'
    # set random seed
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    try:
        root_dir = sys.argv[3]
    except:
        root_dir = "checkpoints/default/"

    directory = root_dir + file_name + '/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        # clipped_double_q
        index = int(sys.argv[4])
        parameter_list = [
            dict(double_clipping=True, noise=0.00, q_smoothing_sigma=0.02, q_smoothing_clip=0.05),
            dict(double_clipping=True, noise=0.00, q_smoothing_sigma=0.0, q_smoothing_clip=0.05),
            dict(double_clipping=False, noise=0.0, q_smoothing_sigma=0.02, q_smoothing_clip=0.05),
            dict(double_clipping=False, noise=0.0, q_smoothing_sigma=0.0, q_smoothing_clip=0.05),
        ]
        parameters = parameter_list[index]
        print('Running...', parameters)
    except:
        parameters = dict(double_clipping=False, noise=0.0)
        print('Running default...', parameters)

    is_continued = True  # False if is_train else True

    # We normalize in a MonitoringEnv state action and reward to [-1,1] for the agent and plot results
    env = MonitoringEnv(env=PendulumEnv(), plot_progress=False, noise=parameters.get('noise'))
    # For recording a movie of the training...
    # env = gym.wrappers.Monitor(env, "recording2", force=True)

    nafnet_kwargs = dict(hidden_sizes=[100, 100], activation=tf.nn.tanh
                         , kernel_initializer=tf.random_normal_initializer(0, .1, seed=random_seed))

    noise_info = dict(
        noise_function=lambda action, nr: action
        if np.random.uniform(min(0, -1 + nr / 50), 1) >= 0 else np.random.uniform(-1, 1, env.action_space.shape[0]))

    # the target network is updated at the end of each episode
    # the number of episodes is executed each step in the environment
    training_info = dict(polyak=0.999, batch_size=400, steps_per_batch=100, epochs=1, learning_rate=1e-3,
                         discount=0.9999)

    # init the agent
    agent = NAF(env=env, directory=directory, noise_info=noise_info,
                is_continued=is_continued, q_smoothing_clip=parameters.get('q_smoothing'),
                clipped_double_q=parameters.get('double_clipping'),
                training_info=training_info, save_frequency=5000,
                **nafnet_kwargs)

    # run the agent training
    agent.training(warm_up_steps=200, initial_episode_length=200, max_episodes=100, max_steps=500)

    # run the agent verification
    # agent.verification(max_episodes=10, max_steps=500)

    # plot the results
    files = []
    for f in os.listdir(directory):
        if 'plot_data' in f and 'pkl' in f:
            files.append(f)
    print(files)
    if len(files) > 0:
        file_name = directory + f'plot_data_{len(files)}'
    else:
        file_name = directory + 'plot_data_0'

    plot_convergence(agent=agent, file_name=file_name)
    plot_results(env, file_name=file_name)

    out_put_writer = open(file_name + '.pkl', 'wb')
    out_rewards = env.rewards
    # out_inits = env.initial_conditions
    out_losses, out_vs = agent.losses, agent.vs

    pickle.dump(out_rewards, out_put_writer, -1)
    # pickle.dump(out_inits, out_put_writer, -1)

    pickle.dump(out_losses, out_put_writer, -1)
    pickle.dump(out_vs, out_put_writer, -1)
    out_put_writer.close()

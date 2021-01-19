Based on:
https://github.com/MathPhysSim/FERMI_RL_Paper

* `envs/`
    * `random_env_rms/`: contains response matrices for RandomEnv class. Response matrices were randomly generated from np.random.uniform(-1, 1, (n_obs, n_act)) and it was ensured that the inverse model (PI) exists and that np.max(np.abs(PI)) < 1.
    * `normalize_env.py`: contains gym.Wrapper class that normalises the state and action spaces to the range [-1, 1]
    * `qfb_env.py`: contains tune feedback gym.Env class which uses the operational response matrix used by the PI controller in the LHC which controls the tune and keeps it at reference. This class only implements the tune feedback for B1 for simplicity and proof of concept.
    * `random_env.py`: contains gym.Env class which uses the random model generated and stored in `random_env_rms/`.
* `ReplayBuffer/`
    * `replay_buffer.py`: the relevant file for NAF2 which contains a standard replay buffer used in RL.
* `naf2.py`: contains QModel class which implements the quadratic form of the advantage function and the NAF2 class which uses double clipped Q technique in a NAF agent.
* `naf2_evaluating.py`: contains helper methods which load model checkpoints and evaluate model performance.
* `naf2_limits.py`: creates and trains a NAF2 agent on a NormaliseEnv(RandomEnv(...)) gym environment. Named as such as it is meant to test the limits of the largest possible size of the state and action spaces that the NAF2 can handle.

Upon starting training, `logs/` and `models/` directories will be created.
* `logs/`: contains tensorboard event files which should be viewed with the following command:
    * `tensorboard --reload_multifile True --logdir logs/`: this command should be executed within the `NAF_limits/NAF2` directory.
* `models/`: contains the saved model checkpoints. This directory is used by `naf2_evaluating.py`.
    
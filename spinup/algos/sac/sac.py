import os
import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.sac import core
from spinup.algos.sac.core import get_vars
from spinup.utils.logx import EpochLogger, restore_checkpoint_only
from itertools import islice
import multiprocessing as mp
from multiprocessing import Pool
import joblib

import matplotlib.pyplot as plt
# import seaborn as sns

def evaluation_process(
        checkpoint_path, gym_kwargs, computation_graph_kwargs
):
    """ TODO: DOCUMENTATION """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        t_construction_start = time.time()
        get_actions, _, _, _, sess, *_ = construct_computation_graph(**computation_graph_kwargs)
        print("Subprocess graph Construction time (evaluation process)", time.time()-t_construction_start)
        restore_checkpoint_only(sess, checkpoint_path)

        reward_sum = 0
        with gym.make(**gym_kwargs) as env:
            done = False

            observation = env.reset(
                start_conditions={
                    "launch":
                        [
                            {
                                "id": car_id,
                                "course_v": (env.num_cars-1 - car_id) * -20,
                                "speed_kmph": 144,
                                "angular_velocity": 0
                            }
                            for car_id in range(env.num_cars)
                        ]
                }
            )

            while not done:
                # generate action
                action = get_actions(observation)
                # take step in gym
                observation, reward, done, info = env.step(action)
                reward_sum += reward[0]
                done = done[0]

            return info[0]["previous_state"]["current_lap_time_msec"]/1000.0, reward_sum
    except Exception as e:
        print(e)
        return -1, -1


def trajectory_sample_process(
        checkpoint_path, gym_kwargs, computation_graph_kwargs,
        random_sampling=None
):
    """ TODO: DOCUMENTATION """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        t_construction_start = time.time()
        get_actions, _, _, _, sess, *_ = construct_computation_graph(**computation_graph_kwargs)
        print("Subprocess graph Construction time", time.time()-t_construction_start)
        restore_checkpoint_only(sess, checkpoint_path)

        observations = []
        actions = []
        rewards = []
        dones = []
        num_frames_for_steps = []

        with gym.make(**gym_kwargs) as env:
            num_cars = env.num_cars
            done = False

            observation_multiagent = env.reset()
            observations.append(observation_multiagent)

            while not done:

                # generate action
                action_multiagent = [random_sampling() for _ in range(num_cars)] \
                    if random_sampling \
                    else get_actions(observation_multiagent)

                # take step in gym
                observation_multiagent, reward_multiagent, done_multiagent, info = env.step(action_multiagent)

                num_frames_for_step = info[0]["state"]["frame_count"] - info[0]["previous_state"]["frame_count"]

                # store trajectory step
                observations.append(observation_multiagent)
                actions.append(action_multiagent)
                rewards.append(reward_multiagent)
                dones.append(done_multiagent)
                num_frames_for_steps.append(num_frames_for_step)

                if done_multiagent[0]:
                    break

            return {
                "multiagent_observations": observations, "multiagent_actions": actions, "multiagent_rewards": rewards,
                "multiagent_dones": dones, "num_frames_for_steps": num_frames_for_steps
            }

    except Exception as e:
        print(e)
        return None


# TODO: hardcoded: find better solution: problem reward_function can only be pickled if at top level
maf = 6
c_wall_hit = 1/(2000*10/9.3)
horizon = 100
# horizon = 5
max_eval_lap = 100


def time_done(seconds, state):
    """ Determines if game time of 'state' is bigger than 'seconds' """
    return state["frame_count"] > (60 * seconds) if state else False


def sampling_done_function(state):
    return time_done(horizon, state)


def evaluation_done_function(state):
    return state["lap_count"] > 2 or state["current_lap_time_msec"]/1000.0 > max_eval_lap if state else False


def reward_function(state, previous_state, course_length):
    if previous_state \
            and isinstance(previous_state["course_v"], float) \
            and isinstance(previous_state["lap_count"], int):

        # version robust to step length through scaling and always detecting wall contact (other than is_hit_wall)
        reward = (
                         - (
                                 (state["hit_wall_time"] - previous_state["hit_wall_time"])
                                 * 10 * state["speed_kmph"]**2 * c_wall_hit)
                         + (state["course_v"] + state["lap_count"] * course_length)
                         - (previous_state["course_v"] + previous_state["lap_count"] * course_length)
                 ) * (maf/(state["frame_count"] - previous_state["frame_count"]))  # correcting too long steps

        # older reward functions
        # reward = - (state["is_hit_wall"] * state["speed_kmph"]**2) * c_wall_hit \
        #          + (state["course_v"] + state["lap_count"] * course_length) \
        #          - (previous_state["course_v"] + previous_state["lap_count"] * course_length)

        # reward = 200*(state["delta_progress"] - previous_state["delta_progress"])\
        # - abs(state["steering"])
        # reward = -(state["steering"]**2)*16\

        return reward


def construct_computation_graph(
        observation_space, action_space, actor_critic, gamma, n_step_return, alpha, lr, polyak, ac_kwargs
):

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, std, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _, _, _, _, _, v_targ = actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n') % var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)

    # Targets for Q and V regression
    q_backup = tf.stop_gradient(r_ph + (gamma ** n_step_return) * (1 - d_ph) * v_targ)
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v) ** 2)
    value_loss = q1_loss + q2_loss + v_loss

    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q') + get_vars('main/v')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi,
                train_pi_op, train_value_op, target_update]

    # other states to show
    alpha_logpi = - alpha * logp_pi
    step_ops += [v_targ, alpha_logpi]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    inputs = {'x': x_ph, 'a': a_ph}
    outputs = {'mu': mu, 'std':std, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v}

    def get_actions(observations, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: observations})

    return get_actions, step_ops, inputs, outputs, sess, x_ph, x2_ph, a_ph, r_ph, d_ph


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    # if n == 1:
    #     print('n step is 1')
    #     return seq[:-n]

    it = iter(seq)
    result = tuple(islice(it, n))
 
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def sample_n_k(n, k):
    # source: https://github.com/chainer/chainerrl/blob/master/chainerrl/misc/random.py
    """Sample k distinct elements uniformly from range(n)"""

    if not 0 <= k <= n:
        raise ValueError("Sample larger than population or is negative")
    if k == 0:
        return np.empty((0,), dtype=np.int64)
    elif 3 * k >= n:
        return np.random.choice(n, k, replace=False)
    else:
        result = np.random.choice(n, 2 * k)
        selected = set()
        selected_add = selected.add
        j = k
        for i in range(k):
            x = result[i]
            while x in selected:
                x = result[i] = result[j]
                j += 1
                if j == 2 * k:
                    # This is slow, but it rarely happens.
                    result[k:] = np.random.choice(n, k)
                    j = k
            selected_add(x)
        return result[:k]


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = sample_n_k(self.size, batch_size)
        # idxs = np.random.randint(0, self.size, size=batch_size) (more than 10x faster but with replacement)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""


def sac(maf, ips, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1,
        step_freq=10, debug=False, relative_steering=False, n_step_return=5, evaluate=True, restore_dir=None):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. 
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        ips (list): ips of the playstations to be controlled
            Provide in the format --ips '["192.168.1.XX", "192.168.1.YY"]'

        maf (int): (see min_frames_per_action gts_gym doc) minimum number of frames that each step should take
            A maximum can not be guaranteed due to slow network communication

        step_freq (float): expected step frequency in Hz
            used for scaling the discount factor and other hyperparameters accordingly

        debug (bool): activates debug mode with shorter trajectories
        
        relative_steering (bool): see gts_gym doc

        n_step_return (int): how many sequential rewards to take into account for forming the value target
            original spinningup implementation: TD(0) (n=1)

    """
    mp.set_start_method('spawn')  # important to not fork because of potential tf session problems

    # step frequency usually used for experiment; used for easier scaling of hyperparameters to new frequencies
    base_freq = 10  # 9.3 used for paper experiments

    # the hyperparameters are scaled according to the expected frequency (step_freq)
    freq_scale = step_freq / base_freq
    gamma = gamma**(1/freq_scale)  # 0.97851091126 for mazda demio/roadster, 0.97314376879 for ttcup

    logger = EpochLogger(**logger_kwargs)
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # arguments shared between sampling gym processes
    sampling_gym_kwargs = dict(
        id="gym_gts:gts-v0",
        reward_function=reward_function,
        done_function=sampling_done_function,
        store_states=False,
        min_frames_per_action=maf,
        throttle_brake_combined=True,
        relative_steering=relative_steering,
        standardize_observations=True
    )

    # arguments for evaluation gym
    evaluation_gym_kwargs = dict(
        id="gym_gts:gts-v0",
        done_function=evaluation_done_function,
        ip=ips[-1],
        fast_mode=False,
        min_frames_per_action=1,
        store_states=True,  # TODO: how big are the resulting files?
        standardize_observations=True,
        delta_clip=True,
        throttle_brake_combined=True,
        relative_steering=relative_steering,
    )

    logger.save_config(locals())

    # ----------------------------------------- set up the computation graph -------------------------------------------

    # get gym parameters that are necessary to determine the required network dimensions
    with gym.make("gym_gts:gts-v0", ip=ips[0]) as env:
        num_cars = env.num_cars  # ASSUMING ALL PLAYSTATIONS HAVE THE SAME NUMBER OF CARS
        action_space = env.action_space
        observation_space = env.observation_space
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

    computation_kwargs = dict(
        observation_space=observation_space,
        action_space=action_space,
        actor_critic=actor_critic,
        gamma=gamma,
        n_step_return=n_step_return,
        alpha=alpha,
        lr=lr,
        polyak=polyak,
        ac_kwargs=ac_kwargs
    )

    # construct computation graph for main process
    get_actions, step_ops, inputs, outputs, sess, x_ph, x2_ph, a_ph, r_ph, d_ph = construct_computation_graph(
        **dict(
            computation_kwargs,
        )
    )

    worker_kwargs = dict(
        computation_kwargs,
    )

    if restore_dir:
        print('load previous')
        restore_checkpoint_only(sess, restore_dir)
        replay_buffer = joblib.load(os.path.join(restore_dir, "../vars.pkl"))["replay_buffer"]
    else:
        # Experience buffer
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs=inputs, outputs=outputs)

    epoch = 0
    t = 0
    start_time = time.time()

    number_of_workers = len(ips) - (1 if evaluate else 0)

    reward_list = []
    while True:
        # ----------------------------------- Save checkpoint of model to allow serving to subprocesses ----------------
        t_model_save_start = time.time()
        latest_save_path = logger.save_state({'replay_buffer': replay_buffer}, epoch)
        t_model_save = time.time() - t_model_save_start

        # ----------------------------------- Evaluate current policy --------------------------------------------------
        if epoch % 2 == 0 and evaluate:  # TODO: automatically adjust based on horizon and eval lengths
            print('on eval ')
            eval_pool = Pool(processes=1)
            evaluation_epoch = epoch
            evaluation_result = eval_pool.apply_async(
                evaluation_process,
                kwds={
                    "checkpoint_path": latest_save_path,
                    "gym_kwargs": evaluation_gym_kwargs,
                    "computation_graph_kwargs": worker_kwargs
                }
            )

        if evaluate:
            evaluation_lap_time, evaluation_reward_sum = evaluation_result.get()
            eval_pool.close()
        else:
            evaluation_lap_time, evaluation_reward_sum = -1, -1

        # ----------------------------------- sample trajectories with current policy ----------------------------------
        with Pool(processes=number_of_workers) as sampling_pool:
            t_sampling_start = time.time()

            trajectory_collection = []

            sampling_results = [
                sampling_pool.apply_async(
                    trajectory_sample_process,
                    kwds={
                        "checkpoint_path": latest_save_path,
                        "gym_kwargs": dict(sampling_gym_kwargs, ip=ip),
                        "computation_graph_kwargs": worker_kwargs,
                        "random_sampling": action_space.sample if t < start_steps else False
                    }
                )
                for ip in (ips[:-1] if evaluate else ips)
            ]

            # ----------------------------------- learn from the replay buffer -------------------------------------
            t_learning_start = time.time()
            if epoch != 0:  # only start learning once data is available
                # hacky solution such that number of updates not dependent on step frequency
                num_updates = int(64 * (0.1 if debug else 1)) * completed_trajectories
                print('num_updates is ', num_updates)
                for j in range(num_updates):
                    if j % 500 == 0:
                        print("learning step %i of %i" % (j, num_updates))
                    batch = replay_buffer.sample_batch(batch_size)
                    feed_dict = {x_ph: batch['obs1'],
                                 x2_ph: batch['obs2'],
                                 a_ph: batch['acts'],
                                 r_ph: batch['rews'],
                                 d_ph: batch['done'],
                                 }
                    outs = sess.run(step_ops, feed_dict)
                    logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],  # TODO: measure time!!!
                                 LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                                 VVals=outs[6], LogPi=outs[7])

                # print mu
                outputs_ = sess.run(outputs, feed_dict)
                mu = outputs_['mu']
                std = outputs_['std']
                print('print mu last time training')
                print('std shape', std.shape)

                x = np.arange(mu.shape[0])

                plt.figure(figsize=(13,8))
                plt.subplot(221)
                plt.plot(x, mu[:,0], 'b.')
                plt.title('mean delta')

                plt.subplot(222)
                plt.plot(x, mu[:,1], 'b.')
                plt.title('mean pedal')

                plt.subplot(223)
                plt.plot(x, std[:,0], 'b.')
                plt.title('std delta')

                plt.subplot(224)
                plt.plot(x, std[:,1], 'b.')
                plt.title('std pedal')


                (q1, q2, v, logp_pi) = outs[4:8]
                (v_targ, alpha_logpi) = outs[11:]

                plt.figure(figsize=(15,8))
                plt.subplot(2,2,1)
                plt.plot(q1, 'b.')
                plt.grid()
                plt.title('q1')

                plt.subplot(2,2,2)
                plt.plot(q2, 'b.')
                plt.grid()
                plt.title('q2')
                    
                plt.subplot(2,2,3)
                plt.plot(v, 'b.', label='v')
                plt.plot(v_targ, 'r.', label='v target')
                plt.legend()
                plt.grid()
                plt.title('v')
                    
                plt.subplot(2,2,4)
                plt.plot(alpha_logpi, 'b.')
                plt.title('- alpha_logpi')
                plt.grid()
                plt.show()
                    
            
                    
            else:
                logger.store(LossPi=-1, LossQ1=-1, LossQ2=-1, LossV=-1, Q1Vals=-1, Q2Vals=-1, VVals=-1, LogPi=-1)

            t_learning = time.time() - t_learning_start
            # ----------------------------------------------------------------------------------------------------------

            completed_trajectories = 0
            for result in sampling_results:
                multiagent_trajectory = result.get()
                if multiagent_trajectory:  # if sampling process finished successfully
                    trajectory_collection.append(multiagent_trajectory)
                    t += num_cars*len(multiagent_trajectory["multiagent_observations"])
                    completed_trajectories += num_cars

        # TODO: automatically adjust "% interval" based on horizon and eval lengths
        # if epoch % 2 == 1 and evaluate:
        #     evaluation_lap_time, evaluation_reward_sum = evaluation_result.get()
        #     eval_pool.close()
        # else:
        #     evaluation_lap_time, evaluation_reward_sum = -1, -1

        t_sampling = time.time() - t_sampling_start

        # --- concatenate the multicar trajectories from all playstations into one list of standalone trajectories -----
        t_concatenation_start = time.time()
        standalone_trajectories = []
        for multiagent_trajectory in trajectory_collection:
            # transform trajectory values from [time_step x agent-value(e.g. reward)] arrays
            # to [agent x time_step-value] arrays
            multiagent_actions = np.swapaxes(multiagent_trajectory["multiagent_actions"], 0, 1)
            multiagent_rewards = np.swapaxes(multiagent_trajectory["multiagent_rewards"], 0, 1)
            multiagent_observations = np.swapaxes(multiagent_trajectory["multiagent_observations"], 0, 1)
            multiagent_dones = np.swapaxes(multiagent_trajectory["multiagent_dones"], 0, 1)

            agent_wise_trajectories = [
                {
                    "observations": np.array(observations, dtype=np.float32),
                    "actions": np.array(actions, dtype=np.float32),
                    "rewards": np.array(rewards, dtype=np.float32),
                    "dones": np.array(dones, dtype=np.float32),
                    "num_frames_for_steps": multiagent_trajectory["num_frames_for_steps"]
                }
                for observations, actions, rewards, dones
                in zip(multiagent_observations, multiagent_actions, multiagent_rewards, multiagent_dones)
            ]

            standalone_trajectories.extend(agent_wise_trajectories)

        t_concatenation = time.time() - t_concatenation_start

        # ----------------------------------- store the sampled trajectories to the replay buffer ----------------------
        t_store_start = time.time()

        gamma_window = np.array([gamma**i for i in range(n_step_return)])  # gammas to calculate TD(n) target


        for trajectory in standalone_trajectories:
            temp_r = []
            # print('in for trajectory, n step', n_step_return)
            # print(trajectory["observations"][:-n_step_return].shape)
            # print(trajectory["actions"][:-(n_step_return-1)].shape)
            # print(window(trajectory["rewards"], n_step_return).shape)
            # print(window(trajectory["num_frames_for_steps"], n_step_return).shape)
            # print(trajectory["observations"][n_step_return:].shape)
            if n_step_return > 1:
                actions = trajectory["actions"][:-(n_step_return-1)]
            else:
                actions = trajectory["actions"]

            for idx, (observation, action, reward_window, num_frames_for_step_window, observation_n_plus) \
            in enumerate(
                zip(
                    trajectory["observations"][:-n_step_return],  # o_t
                    actions,  # a_t
                    window(trajectory["rewards"], n_step_return),  # r_t+1 ... r_t+n
                    window(trajectory["num_frames_for_steps"], n_step_return),
                    trajectory["observations"][n_step_return:]  # o_t+n
                )
            ):
                # only include tuples that didn't include any slow steps
                if all([num_frames_for_step < maf + 2 for num_frames_for_step in num_frames_for_step_window]):
                    replay_buffer.store(
                        observation, action, np.sum(np.array(reward_window)*gamma_window), observation_n_plus, False
                    )
                else:
                    print("Step %i not included since slow step is in n-step window around it" % idx)

            logger.store(EpRet=sum(trajectory["rewards"]), EpLen=len(trajectory["rewards"]))

            temp_r.append(sum(trajectory["rewards"]))
        reward_list.append(np.mean(temp_r))


        x_idx =  list(range(len(reward_list)))
        num = 100
        plt.figure(figsize=(13,5))
        plt.plot(x_idx[-num:],reward_list[-num:], 'b.-')
        plt.title('average reward at each epoch')
        plt.grid()
        plt.show()

        print('shape of obs', trajectory["observations"].shape)
        print('shape of action', trajectory["actions"].shape)
        print('average reward ', reward_list[-1])

        # print reward
        # print('print reward related ', idx)
        plt.figure(figsize=(7,4))
        plt.plot(trajectory["rewards"], 'b.')
        plt.title('trajectory["rewards"]')
        plt.grid()
        plt.show()

        # plt.figure(figsize=(7,4))
        # plt.plot(np.array(reward_window), '.b')
        # plt.title('reward_window')
        # plt.grid()
        # plt.show()

        # print action
        # print('print action ', idx)
        plt.figure(figsize=(7,4))
        plt.plot(trajectory["actions"][:,0], 'b.')
        plt.title('actions 0 delta')
        plt.grid()
        plt.show()   

        plt.figure(figsize=(7,4))
        plt.plot(trajectory["actions"][:,1], 'b.')
        plt.title('actions 1 acc')
        plt.grid()
        plt.show()  

        t_store = time.time() - t_store_start

        # ------------------------------------- epoch wrap up ----------------------------------------------------------

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', t)
        logger.log_tabular('Q1Vals', with_min_and_max=True)
        logger.log_tabular('Q2Vals', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('LogPi', with_min_and_max=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossQ1', average_only=True)
        logger.log_tabular('LossQ2', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.log_tabular('t_sampling', t_sampling)
        logger.log_tabular('t_concatenation', t_concatenation)
        logger.log_tabular('t_store', t_store)
        logger.log_tabular('t_learning', t_learning)
        logger.log_tabular('t_model_save', t_model_save)
        logger.log_tabular('completed_trajectories', completed_trajectories)
        if evaluate:
            logger.log_tabular('evaluation_epoch', evaluation_epoch)
            logger.log_tabular('evaluation_lap_time', evaluation_lap_time)
            logger.log_tabular('evaluation_reward_sum', evaluation_reward_sum)
        logger.dump_tabular()

        epoch += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--act', default=tf.nn.relu)
    parser.add_argument('ips', nargs="+")
    parser.add_argument('--step_freq', default=10)
    parser.add_argument('--maf', type=int, default=6)

    parser.add_argument('--start_steps', type=int, default=40000)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--n_step_return', type=int, default=1) # default 5, try 1?
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--evaluate', default=False, action='store_true')
    parser.add_argument('--restore_dir', type=str, default=None)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    
    # style = "whitegrid"
    # sns.set_theme(style=style) # background color
    
    sac(maf=args.maf, actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, activation=args.act),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, ips=args.ips, step_freq=args.step_freq,
        debug=args.debug, n_step_return=args.n_step_return,
        start_steps=args.start_steps, replay_size=args.replay_size, polyak=args.polyak, lr=args.lr,
        batch_size=args.batch_size, alpha=args.alpha, evaluate=args.evaluate, restore_dir=args.restore_dir
        )

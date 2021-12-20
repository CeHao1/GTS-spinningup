import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
import numpy as np


def load_policy(fpath, sess, itr='last', deterministic=False):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # ---------------------  count number of trainable parameters in network ---------------------------------------
    # source: https://stackoverflow.com/questions/38160940
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Number of total parameters: %i\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n" % total_parameters)

    # get the correct op for executing actions
    if deterministic:
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: np.transpose(x).T})

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    # try:
    #     state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
    #     env = state['env']
    # except:
    #     env = None

    return None, get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    # logger = EpochLogger()

    # initial call is always slower
    get_action([[0]*env.observation_space.shape[0]])

    o, r, d, ep_ret, ep_len, n = env.reset(  # TODO: set back to old settings
        start_conditions={
            "launch":
            [
                {
                    "id": car_id,
                    "course_v": (env.num_cars-1 - car_id) * -20,
                    "speed_kmph": 144,
                    "angular_velocity": 0
                }
                for car_id in range(20) # range(num_cars)
            ]

            # USE OTHER CARS TO BUILD OBSTACLE TRACK (ON CENTRAL OUTER LOOP)
            # "launch":
            #     [
            #         {
            #             "id": 0,
            #             "speed_kmph": 100,
            #             "pos": [-530.852905, 8.543184, 706.049072],
            #             "rot": [-0.008936, 0.625234, -0.051207]
            #         },
            #         {
            #             "id": 1,
            #             "speed_kmph": 0,
            #             "pos": [-571.36914, 8.514442, 640.18872],
            #             "rot": [-7.9e-05, 0.433823, 0.001545]
            #         },
            #         {
            #             "id": 2,
            #             "speed_kmph": 0,
            #
            #             "pos":[-592.227905, 8.483235, 586.878417],
            #             "rot": [-0.001767, 0.431587, 0.000181]
            #         },
            #         {
            #             "id": 3,
            #             "speed_kmph": 0,
            #             "pos":[-624.734008, 8.529642, 523.676208],
            #             "rot": [0.00169, 0.399741, -0.011388]
            #         },
            #         {
            #             "id": 4,
            #             "speed_kmph": 0,
            #             "pos":[-627.650329, 7.126766, 403.783386],
            #             "rot": [-0.042852, -0.199318, -0.023492]
            #         },
            #         {
            #             "id": 5,
            #             "speed_kmph": 0,
            #             "pos": [-591.090209, 2.816651, 298.729919],
            #             "rot": [-0.040147, -0.596221, -0.025032]
            #         },
            #         {
            #             "id": 6,
            #             "speed_kmph": 0,
            #             "pos": [-509.149475, 0.444449, 228.183532],
            #             "rot": [-0.001654, -0.836429, 0.000164]
            #         },
            #     ]
                }
    ), 0, False, 0, 0, 0

    # obs = []
    # try:
    while n < num_episodes:

        a = get_action(o)
        o, r, d, info = env.step(a)
        # obs.extend(o)
        ep_ret += r
        ep_len += 1

        if d[0] or (ep_len == max_ep_len):
            return info[0]["previous_state"]["current_lap_time_msec"]/1000.0

    # finally:
    #     import numpy as np
    #     print("mean", np.mean(obs, axis=0))
    #     print("std", np.std(obs, axis=0))
    #     print("min abs", np.min(abs(obs), axis=0))
    #     print("max abs", np.max(abs(obs), axis=0))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy(args.fpath, 
                                  args.itr if args.itr >=0 else 'last',
                                  args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))
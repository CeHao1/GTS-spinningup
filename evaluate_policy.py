import argparse
import os
from datetime import datetime

import gym
import pandas as pd
import tensorflow as tf

from spinup.utils.test_policy import load_policy, run_policy


""" Custom evaluation script for deterministic policy, needed due to environment pickling problems as described in:
https://spinningup.openai.com/en/latest/user/saving_and_loading.html#environment-not-found-error

Can either evaluate a series of epochs (starting from --max_itr downwards) saving the lap time of the 2nd driven round
or a single epoch (--sing_itr)
"""


parser = argparse.ArgumentParser()
parser.add_argument(
        "--model_dir",
        help="Dir of model and seed to evaluate",
        type=str
    )
parser.add_argument(
        "--max_itr",
        help="First and highest iteration to evaluate",
        type=int,
        default=None
    )
parser.add_argument(
        "--sing_itr",
        help="Single iteration to evaluate (runs for multiple laps)",
        type=int,
        default=-1
    )
parser.add_argument(
        "--ip",
        help="PlayStation ip",
        type=str,
        default=-1
    )

args = parser.parse_args()
max_itr = args.max_itr
sing_itr = args.sing_itr
model_directory = args.model_dir
ip = args.ip

with gym.make(
        "gym_gts:gts-v0",
        done_function=(
                lambda state:
                state["lap_count"] > (2 if max_itr else 10)
                or state["current_lap_time_msec"] > 1000*180 * (1 if max_itr else 10)
                if state
                else False
        ),
        ip=ip,
        fast_mode=False,  # TODO: for delta-clipping set to false
        min_frames_per_action=1,  # TODO: for delta-clipping set to 1
        store_states=False if max_itr else True,
        standardize_observations=True,
        delta_clip=True,  # TODO: set to true for fair comparison with human
        # human_controlled=[0],  # TODO: set empty
        # builtin_controlled=[2]  # TODO: set empty
) as env:

    if max_itr:  # evaluate from epoch max_itr down to epoch 0

        epochs = [{"number": epoch_number} for epoch_number in reversed(range(max_itr+1))]

    else:  # evaluate single epoch
        if sing_itr == -1:
            sing_itr = "last"

        epochs = [{"number": sing_itr}]

    for epoch in epochs:

        tf.reset_default_graph()

        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))) as sess:

            print("Epoch number:", epoch)
            _, get_action = load_policy(model_directory, sess=sess, deterministic=True, itr=epoch["number"])

            epoch["lap_time"] = run_policy(env, get_action)
            print("Lap time: %f" % epoch["lap_time"])
            save_path = os.path.join(
                    model_directory,
                    "lap_times_with_clip_%s_%s.csv" % (ip, datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            )
            pd.DataFrame(epochs).to_csv(
                save_path
            )
            print("Saving lap times to directory: %s" % save_path)



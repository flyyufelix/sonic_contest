#!/usr/bin/env python

"""
Joint Train Rainbow on 47 training levels drawn from Sonic games
"""

import csv
from functools import partial
import os

import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv, AsyncGymEnv, batched_gym_env 
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from sonic_util import AllowBacktracking, make_env

def create_envs():
    env_fns = [] 
    env_names = []
    with open('sonic-train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if 'Sonic' in row[0]:
                print('Add Environments: ', row[0] + ': ' + row[1])
                env_fns.append(partial(make_env, game=row[0], state=row[1], stack=False, scale_rew=False))
                env_names.append(row[0] + '-' + row[1])

    return env_fns, env_names

def main():
    """Run DQN until the environment throws an exception."""
    env_fns, env_names = create_envs() 
    env = BatchedFrameStack(batched_gym_env(env_fns), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4) # Use ADAM
        sess.run(tf.global_variables_initializer())

        reward_hist = []
        total_steps = 0
        def _handle_ep(steps, rew, env_rewards):
            nonlocal total_steps
            total_steps += steps
            reward_hist.append(rew)
            if total_steps % 1 == 0:
                print('%d episodes, %d steps: mean of last 100 episodes=%f' % (len(reward_hist), total_steps, sum(reward_hist[-100:]) / len(reward_hist[-100:])))

        dqn.train(num_steps=2000000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000,
                  handle_ep=_handle_ep,
                  num_envs = len(env_fns),
                  save_interval=10,
                  )

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)

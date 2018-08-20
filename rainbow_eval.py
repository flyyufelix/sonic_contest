#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN implementation.
"""

import tensorflow as tf
import datetime
import numpy as np

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from mpi4py import MPI
import csv
import os
from functools import partial

from sonic_util import AllowBacktracking, make_env

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver, RunObserver

timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')

ex = Experiment()
ex.observers.append(FileStorageObserver.create('./logs/rainbow_eval_'+timestamp))

def create_eval_envs():
    env_fns = [] 
    env_names = []
    with open('sonic-validation.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if 'Sonic' in row[0]:
                print('Add Environments: ', row[0] + ': ' + row[1])
                env_fns.append(partial(make_env, game=row[0], state=row[1]))
                env_names.append(row[0] + '-' + row[1])

    return env_fns, env_names

@ex.automain
def main():
    """Run DQN until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    comm = MPI.COMM_WORLD

    # Use MPI for parallel evaluation
    rank = comm.Get_rank()
    size = comm.Get_size()

    env_fns, env_names = create_eval_envs()

    env = AllowBacktracking(env_fns[rank](stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())

        reward_hist = []
        total_steps = 0
        def _handle_ep(steps, rew, env_rewards):
            nonlocal total_steps
            total_steps += steps
            reward_hist.append(rew)
            if total_steps % 1 == 0:
                avg_score = sum(reward_hist[-100:]) / len(reward_hist[-100:])

			# Global Score
            global_score = np.zeros(1)
            local_score = np.array(avg_score)
            print("Local Score for " + env_names[rank] + " at episode " + str(len(reward_hist)) + " with timesteps: " + str(total_steps) + ": " + str(local_score))
            comm.Allreduce(local_score, global_score, op=MPI.SUM)
            global_score /= size
            if rank == 0:
                print("Global Average Score at episode: " + str(len(reward_hist)) + ": " + str(global_score))


        dqn.train(num_steps=2000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000,
                  handle_ep=_handle_ep,
                  save_interval=None,
                  restore_path='./checkpoints_rainbow/model-10' # Model to be evaluated
                  )

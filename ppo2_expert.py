#!/usr/bin/env python

"""
Train zone specific expert models with PPO2
"""

import os
import datetime
import csv
import argparse
from functools import partial

import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

# Code for expert agents can be found in baselines/ppo2_expert
import baselines.ppo2_expert.ppo2 as ppo2
import baselines.ppo2_expert.policies as policies

import gym_remote.exceptions as gre

from sonic_util import make_env

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver, RunObserver

timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')

# Name to identify expert network
expert_name = 'greenhill'

ex = Experiment()
ex.observers.append(FileStorageObserver.create('./logs/ppo2_expert_'+expert_name+'_'+timestamp))

def create_envs():
    env_fns = [] 
    env_names = []

    # The level grouping files (in csv formats) are placed inside the "level_groups" directory 
    filename = './level_groups/' + expert_name + '.csv'

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print('Add Environments: ', row[0] + ': ' + row[1])
            env_fns.append(partial(make_env, game=row[0], state=row[1]))
            env_names.append(row[0] + '-' + row[1])

    return env_fns, env_names


@ex.automain
def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    env_fns, env_names = create_envs()
    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2.learn(policy=policies.CnnPolicy,
                   env=SubprocVecEnv(env_fns),
                   nsteps=4096, 
                   nminibatches=8, 
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=3, 
                   log_interval=1, 
                   ent_coef=0.001,
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.1, 
                   total_timesteps=int(1e9),
                   save_interval=10,
                   save_path='checkpoints_expert_'+expert_name,
                   load_path='./checkpoints_joint_ppo2/00300') # Pretrained model




# OpenAI Sonic the HedgeHog Retro Contest

This repo includes the source code I used for [OpenAI Retro Contest](https://contest.openai.com/). For a detail account of my approach, please check out my blog post at https://flyyufelix.github.io/2018/06/11/sonic-rl.html.

## Content Description

PPO Agent: `ppo2_agent.py`

PPO joint training on multiple levels of Sonic: `ppo2_joint.py`

PPO with curiosity-driven exploration: `ppo2_curiosity.py`

PPO to train expert models: `ppo2_expert.py`

Rainbow Agent: `rainbow_agent.py`

Rainbow joint training on multiple levels of Sonic: `rainbow_joint.py`

Perform local validation on validation levels with PPO model: `ppo2_eval.py`

Train classifier to classify levels into zones: `train_level_classifier.py`

CSV files containing OpenAI's recommended train/validation levels split: `sonic-train.csv` and `sonic-validation.csv`

Docker files with instructions to create docker images to submit to OpenAI server for evaluation: `ppo2.docker` and `rainbow.docker`

`baselines` and `anyrl-py` contain the core implementations of PPO and Rainbow with my customized edit for this contest.

Please note that the pretrained models are not included in this repo. You have to run the code to generate the models.
To run the code, please read the instructions [here](https://contest.openai.com/details) to install all the dependencies for the contest.

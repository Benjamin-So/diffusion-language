# coding=utf-8
# Copyright 2024 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple offline evaluation script for language table sim."""

import collections
from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging

import jax

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.oracles import push_oracle_rrt_slowdown
from language_table.environments.rewards import block2absolutelocation
from language_table.environments.rewards import block2block
from language_table.environments.rewards import block2block_relative_location
from language_table.environments.rewards import block2relativelocation
from language_table.environments.rewards import separate_blocks
from language_table.eval import wrappers as env_wrappers
from language_table.train import policy as jax_policy
from language_table.train.networks import lava

from diffusion_reward.rl.drqv2.agent import DrQV2Agent
import diffusion_reward.rl.drqv2.utils as utils

import mediapy as mediapy_lib
from ml_collections import config_flags

import tensorflow as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers as tfa_wrappers

from omegaconf import OmegaConf
import hydra
import numpy as np
import cv2
import torch


# Define configuration flags for the script
_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
_WORKDIR = flags.DEFINE_string("workdir", None, "Evaluation result directory.")
_CHECKPOINT_PATH = flags.DEFINE_string("checkpoint_path", None,
                                       "FLAX checkpoint path.")


def get_agent_config():
    # Placeholder for agent initialization
    relative_path = '../diffuson_reward_rl/configs/rl/agent/drqv2.yaml'
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))
    agent_config = OmegaConf.load(yaml_path)
    return agent_config

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec
    cfg.action_shape = action_spec

    obs_shape = cfg.obs_shape
    action_shape = cfg.action_shape
    device = cfg.device
    lr = cfg.lr
    feature_dim = cfg.feature_dim
    hidden_dim = cfg.hidden_dim
    critic_target_tau = cfg.critic_target_tau
    num_expl_steps = cfg.num_expl_steps
    update_every_steps = cfg.update_every_steps
    stddev_schedule = cfg.stddev_schedule
    stddev_clip = cfg.stddev_clip
    use_tb = cfg.use_tb

    return DrQV2Agent(obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb)

def evaluate_checkpoint(checkpoint_path, workdir, config):
    """Evaluates the given checkpoint and writes results to workdir."""
    # Create directory to store evaluation videos
    video_dir = os.path.join(workdir, "videos")
    if not tf.io.gfile.exists(video_dir):
        tf.io.gfile.makedirs(video_dir)
    
    # Define reward functions
    rewards = {
        "blocktoblock": block2block.BlockToBlockReward,
        "blocktoabsolutelocation": block2absolutelocation.BlockToAbsoluteLocationReward,
        "blocktoblockrelativelocation": block2block_relative_location.BlockToBlockRelativeLocationReward,
        "blocktorelativelocation": block2relativelocation.BlockToRelativeLocationReward,
        "separate": separate_blocks.SeparateBlocksReward,
    }

    num_evals_per_reward = 50  # Number of evaluations per reward type
    max_episode_steps = 200  # Maximum steps per episode

    policy = None
    model = lava.SequenceLAVMSE(action_size=2, **config.model)

    results = collections.defaultdict(lambda: 0)  # Initialize results dictionary

    # Iterate over each reward type and perform evaluation
    for reward_name, reward_factory in rewards.items():
        # Create environment with specified reward function
        env = language_table.LanguageTable(
            block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
            reward_factory=reward_factory,
            render_text_in_image=False,
            seed=0)
        env = gym_wrapper.GymWrapper(env)
        env = env_wrappers.ClipTokenWrapper(env)
        env = env_wrappers.CentralCropImageWrapper(
            env,
            target_width=config.data_target_width,
            target_height=config.data_target_height,
            random_crop_factor=config.random_crop_factor)
        env = tfa_wrappers.HistoryWrapper(
            env, history_length=config.sequence_length, tile_first_step_obs=True)

        # Initialize policy if not already done
        if policy is None:
            # policy = jax_policy.BCJaxPyPolicy(
            #     env.time_step_spec(),
            #     env.action_spec(),
            #     model=model,
            #     checkpoint_path=checkpoint_path,
            #     rng=jax.random.PRNGKey(0))
            
            train_env_observation_spec = (3, 64, 64)
            train_env_action_spec = (2,)
            agent_config = get_agent_config()
            policy = make_agent(train_env_observation_spec,
                                train_env_action_spec, agent_config)


        # Evaluate the policy for the specified number of episodes
        for ep_num in range(num_evals_per_reward):
            # Reset environment and ensure valid initialization using oracle
            # The oracle ensures the environment's initial state is valid by checking for a feasible motion plan.
            # If the plan is invalid, the environment is reset until a valid state is found.
            oracle_policy = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
                env, use_ee_planner=True)
            plan_success = False
            while not plan_success:
                ts = env.reset()
                raw_state = env.compute_state()
                plan_success = oracle_policy.get_plan(raw_state)
                if not plan_success:
                    logging.info(
                        "Resetting environment because the "
                        "initialization was invalid (could not find motion plan).")

            frames = [env.render()]

            episode_steps = 0
            while not ts.is_last():
                with torch.no_grad(), utils.eval_mode(policy):
                    # policy_step = policy.action(ts, ())

                    # Assuming ts.observation['rgb'] is the array with shape (1, 180, 320, 3)
                    observation = ts.observation['rgb']
                    # Remove the first dimension
                    observation = np.squeeze(observation, axis=0)  # Now the shape is (180, 320, 3)
                    # resize the image to (3, 64, 64)
                    observation = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_AREA)
                    # Transpose the dimensions to (3, 64, 64)
                    observation = np.transpose(observation, (2, 0, 1))

                    # sample an action
                    action = policy.act(observation, episode_steps, eval_mode=True)


                    # ts = env.step(policy_step.action)
                    ts = env.step(action)
                    print(f"action is: {action}")

                    frames.append(env.render())
                    episode_steps += 1

                    if episode_steps > max_episode_steps:
                        break

            # Log success or failure of the episode
            success_str = ""
            if env.succeeded:
                results[reward_name] += 1
                logging.info("Episode %d: success.", ep_num)
                success_str = "success"
            else:
                logging.info("Episode %d: failure.", ep_num)
                success_str = "failure"

            # Write out video of the episode
            video_path = os.path.join(workdir, "videos/",
                                      f"{reward_name}_{ep_num}_{success_str}.mp4")
            mediapy_lib.write_video(video_path, frames, fps=10)

        # Print cumulative results after evaluating all episodes
        print(results)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    evaluate_checkpoint(
        checkpoint_path=_CHECKPOINT_PATH.value,
        workdir=_WORKDIR.value,
        config=_CONFIG.value,
    )


if __name__ == "__main__":
    app.run(main)

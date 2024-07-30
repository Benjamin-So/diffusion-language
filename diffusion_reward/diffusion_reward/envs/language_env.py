
from collections import deque

from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.rewards import block2absolutelocation
from language_table.environments.rewards import block2block
from language_table.eval import wrappers as env_wrappers

from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers as tfa_wrappers

from diffusion_reward.envs.wrapper import ExtendedTimeStepAdroit

from .wrapper import (ExtendedTimeStepWrapper, TimeLimitWrapper)

import gym
import numpy as np
import cv2
from dm_env import StepType, specs



language_tasks = {'blocktoblock64', 'blocktoabsolute64', 'greenStarToBlueCube'}

class LanguageWrapper(gym.Env):
    """
    Language Table Wrapper for Diffusion Reward
    """
    def __init__(self, env, name, frame_stack, action_repeat, seed, num_repeats = 2, num_frames=1):
        self.name = name
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.seed = seed
        self._env = env

        self._num_repeats = num_repeats
        self._num_frames = num_frames
        self._frames = deque([], maxlen= num_frames)

        self._obs_spec = specs.BoundedArray(shape=(3, 64, 64), dtype='uint8', name='observation', minimum=0, maximum=255)
        # self._obs_sensor_spec = specs.Array(shape=(self.obs_sensor_dim,), dtype='float32', name='observation_sensor') # not sure if need
        self._action_spec = specs.BoundedArray(shape=(2,), dtype='float32', name='action', minimum=-0.1, maximum=0.1)

    
    def reset(self):
        # call language_table env's reset()
        # time_step = self._env.reset()
        # # create compatible diffusion_reward timestep
        # obs_pixels = self._format_language_pixel(time_step.observation['rgb'])
        # obs_sensor = self._format_language_sensor(time_step.observation['effector_translation'])


        observation = self._env.reset()
        obs_pixels = self._format_language_pixel(observation['rgb'])
        obs_sensor = self._format_language_sensor(observation['effector_translation'])


        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        time_step = ExtendedTimeStepAdroit(observation=obs_pixels,
                                        observation_sensor=obs_sensor,
                                        step_type=StepType.FIRST,
                                        action=action,
                                        reward=0.0,
                                        discount=1.0,
                                        n_goal_achieved=0,
                                        time_limit_reached=False,
                                        is_success=False)
        return time_step

    def render(self):
        image = self._env.render() # 180, 320, 3
        return image

    def step(self, action, force_step_type= None, debug = None):

        reward_sum = 0.0
        n_goal_achieved = 0

        for i_action in range(self._num_repeats): 

            obs, reward, done, env_info = self._env.step(action)
            succeeded = self._env.succeeded
            reward_sum += succeeded
            if succeeded:
                n_goal_achieved += 1
            
            if done:
                break

        obs_pixels = self._format_language_pixel(obs['rgb'])
        obs_sensor = self._format_language_sensor(obs['effector_translation'])
        obs_sensor.astype(np.float32)

        env_info['n_goal_achieved'] = n_goal_achieved
        discount = 1.0
        time_limit_reached = env_info['TimeLimit.truncated'] if 'TimeLimit.truncated' in env_info else False

        if done:
            steptype = StepType.LAST
        else:
            steptype = StepType.MID

        if done and not time_limit_reached:
            discount = 0.0

        
        time_step = ExtendedTimeStepAdroit(observation=obs_pixels,
                                observation_sensor=obs_sensor,
                        step_type=steptype,
                        action=action,
                        reward=reward,
                        discount=discount,
                        n_goal_achieved=n_goal_achieved,
                        time_limit_reached=time_limit_reached,
                        is_success=bool(n_goal_achieved))
    
        return time_step
        

    def observation_spec(self):
        return self._obs_spec

    def observation_sensor_spec(self):
        # return self._obs_sensor_spec
        raise NotImplementedError

    def action_spec(self):
        return self._action_spec
    
    def _format_language_pixel(self, obs_pixels):
        """
        Formats the language observation to match the required shape for Diffusion_Reward observations.

        Args:
            observation (numpy.ndarray): An array representing the observation with shape (1, 180, 320, 3).

        Returns:
            numpy.ndarray: A formatted observation array with shape (3, 64, 64).
        """
        # obs_pixels = np.squeeze(obs_pixels, axis=0)  # Now the shape is (180, 320, 3)
        obs_pixels = cv2.resize(obs_pixels, (64, 64), interpolation=cv2.INTER_AREA) # resize the image to (64, 64, 3)
        obs_pixels = np.transpose(obs_pixels, (2, 0, 1)) # Transpose the dimensions to (3, 64, 64)
        return obs_pixels
    
    def _format_language_sensor(self, obs_sensor):
        obs_sensor = np.squeeze(obs_sensor) # resize from (1,2) to (2,)
        obs_sensor = obs_sensor.astype(np.float32)
        return obs_sensor
    

def make(name, frame_stack, action_repeat, seed):
    rewards = {
        "blocktoblock64": block2block.BlockToBlockReward,
        "blocktoabsolute64": block2absolutelocation.BlockToAbsoluteLocationReward,
        "greenStarToBlueCube": block2block.BlockToBlockReward
    }

    # Hard coded values taken from language_table_resnet_sim_local.py
    random_crop_factor = 0.95
    sequence_length = 1
    data_target_width = 320
    data_target_height = 180

    # Create language_table env as in language_table/eval/main.py
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=rewards[name],
        render_text_in_image=False,
        control_frequency= 20,
        seed=seed)
    # env = gym_wrapper.GymWrapper(env)
    # env = env_wrappers.ClipTokenWrapper(env)
    # env = env_wrappers.CentralCropImageWrapper(
    #     env,
    #     target_width=data_target_width,
    #     target_height=data_target_height,
    #     random_crop_factor=random_crop_factor)
    # env = tfa_wrappers.HistoryWrapper(
    #     env, history_length=sequence_length, tile_first_step_obs=True)
    
    # Diffusion Reward Wrapper
    env = TimeLimitWrapper(env, max_episode_steps = 200)
    env = LanguageWrapper(env, name, frame_stack, action_repeat, seed)
    

    return env
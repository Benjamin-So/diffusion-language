import diffusion_reward.envs.adroit as adroit
import diffusion_reward.envs.metaworld as metaworld
import metaworld.envs.mujoco.env_dict as _mw_envs

from .adroit import _mj_envs

import diffusion_reward.envs.language_env as language
from .language_env import language_tasks


def make_env(name, frame_stack, action_repeat, seed):
    if name in _mj_envs:
        env = adroit.make(name, frame_stack, action_repeat, seed)
    elif name in _mw_envs.ALL_V2_ENVIRONMENTS.keys():
        env = metaworld.make(name, frame_stack, action_repeat, seed)
    elif name in language_tasks:
        env = language.make(name, frame_stack, action_repeat, seed)
    else:
        raise NotImplementedError
    return env
# Adapted from ManiSkill (https://github.com/haosulab/ManiSkill), Apache-2.0; see THIRD_PARTY_LICENSES.md.
from typing import Dict, Sequence

import numpy as np

from gymnasium import spaces

def get_active_joint_indices(articulation, joint_names: Sequence[str]):
    """get the indices of the provided joint names from the Articulation's list of active joints"""
    all_joint_names = [x.name for x in articulation.get_active_joints()]
    joint_indices = [all_joint_names.index(x) for x in joint_names]
    return np.array(joint_indices).astype(np.int32)


def get_joints_by_names(articulation, joint_names: Sequence[str]):
    """Gets the Joint objects by name in the Articulation's list of active joints"""
    joints = articulation.get_active_joints()
    joint_indices = get_active_joint_indices(articulation, joint_names)
    return [joints[idx] for idx in joint_indices]

def clip_and_scale_action(action, low, high):
    """Clip action to [-1, 1] and scale according to a range [low, high]."""
    action = np.clip(action, -1, 1)
    return 0.5 * (high + low) + 0.5 * (high - low) * action


def normalize_action_space(action_space: spaces.Box):
    assert isinstance(action_space, spaces.Box), type(action_space)
    return spaces.Box(-1, 1, shape=action_space.shape, dtype=action_space.dtype)

def flatten_action_spaces(action_spaces: Dict[str, spaces.Space]):
    """Flatten multiple Box action spaces into a single Box space."""
    action_dims = []
    low = []
    high = []
    action_mapping = dict()
    offset = 0

    for action_name, action_space in action_spaces.items():
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1, (action_name, action_space)
        else:
            raise TypeError(action_space)

        action_dim = action_space.shape[0]
        action_dims.append(action_dim)
        low.append(action_space.low)
        high.append(action_space.high)
        action_mapping[action_name] = (offset, offset + action_dim)
        offset += action_dim

    flat_action_space = spaces.Box(
        low=np.hstack(low),
        high=np.hstack(high),
        shape=[sum(action_dims)],
        dtype=np.float32,
    )

    return flat_action_space, action_mapping
# Adapted from ManiSkill (https://github.com/haosulab/ManiSkill), Apache-2.0; see THIRD_PARTY_LICENSES.md.
from dataclasses import dataclass, field
from typing import Dict, Sequence, Union, Literal, TypeAlias

import numpy as np
from gymnasium import spaces



from .base_controller import BaseController, ControllerConfig
from .utils import get_joints_by_names, get_active_joint_indices

DriveMode: TypeAlias = Literal["force", "acceleration"]


class PDJointPosController(BaseController):
    config: "PDJointPosControllerConfig"
    _start_qpos = None
    _target_qpos = None
    sets_target_qpos = True
    sets_target_qvel = False

    def _get_joint_limits(self):
        qlimits = (
            self.articulation.get_qlimits()[self.active_joint_indices]
        )

        # Override if specified
        if self.config.lower is not None:
            qlimits[:, 0] = self.config.lower
        if self.config.upper is not None:
            qlimits[:, 1] = self.config.upper
        return qlimits

    def _initialize_action_space(self):
        joint_limits = self._get_joint_limits()
        low, high = joint_limits[:, 0], joint_limits[:,1]
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self):
        n = len(self.joints)
        stiffness = np.broadcast_to(self.config.stiffness, n)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            drive_mode = self.config.drive_mode
            if not isinstance(drive_mode, str):
                drive_mode = drive_mode[i]
            joint.set_drive_properties(
                stiffness[i], damping[i], force_limit=force_limit[i], mode=drive_mode
            )
            joint.set_friction(friction[i])

    def reset(self):
        super().reset()
        self._step = 0  # counter of simulation steps after action is set
        if self._start_qpos is None:
            self._start_qpos = self.qpos.copy()

        if self._target_qpos is None:
            self._target_qpos = self.qpos.copy()


    def set_drive_targets(self, targets):
        for i, joint in enumerate(self.joints):
            joint.set_drive_target(targets[i])

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + action
            else:
                self._target_qpos = self._start_qpos + action
        else:
            # Compatible with mimic controllers. Need to clone here otherwise cannot do in-place replacements in the reset function
            self._target_qpos = np.broadcast_to(
                action, self._start_qpos.shape
            ).copy()
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def before_simulation_step(self):
        self._step += 1

        # Compute the next target via a linear interpolation
        if self.config.interpolate:
            targets = self._start_qpos + self._step_size * self._step
            self.set_drive_targets(targets)

@dataclass
class PDJointPosControllerConfig(ControllerConfig):
    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    use_delta: bool = False
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = PDJointPosController


class PDJointPosMimicController(PDJointPosController):
    config: "PDJointPosMimicControllerConfig"

    def _initialize_joints(self):
        super()._initialize_joints()
        # do some sanity checks and setup the mimic controller

        self.mimic_joint_indices = []
        self.mimic_control_joint_indices = []
        if len(self.config.mimic) == 0:
            if len(self.config.joint_names) == 2:
                print(
                    f"Mimic targets dictionary is missing for controller config for {self.articulation.name}. Assuming the first joint is the control joint and the second joint is the mimic joint"
                )
                self.config.mimic = {
                    self.config.joint_names[0]: {"joint": self.config.joint_names[1]}
                }
            else:
                raise ValueError(
                    "Mimic targets dictionary is missing. Please provide a mimic targets dictionary to setup mimic controllers with more than 2 joints"
                )

        self._multiplier = np.ones(
            len(self.config.mimic)
        )  # this is parallel to self.mimic_joint_indices
        self._offset = np.zeros(len(self.config.mimic))

        for i, (mimic_joint_name, mimic_data) in enumerate(self.config.mimic.items()):
            control_joint_name = mimic_data["joint"]
            multiplier = mimic_data.get("multiplier", 1.0)
            offset = mimic_data.get("offset", 0.0)

            mimic_joint = get_joints_by_names(
                self.articulation, [mimic_joint_name]
            )[0]
            control_joint = get_joints_by_names(
                self.articulation, [control_joint_name]
            )[0]

            assert (
                mimic_joint is not None
                and control_joint is not None
            ), f"Cannot find joint {mimic_joint_name} or {control_joint_name} in the articulation {self.articulation.name}"
            self.mimic_joint_indices.append(
                np.argwhere(
                    self.active_joint_indices
                    ==  get_active_joint_indices(
                        self.articulation, [mimic_joint_name]
                    )[0]
                )
            )
            self.mimic_control_joint_indices.append(
                np.argwhere(
                    self.active_joint_indices
                    == get_active_joint_indices(
                        self.articulation, [mimic_joint_name]
                    )[0]
            )
            )
            self._multiplier[i] = multiplier
            self._offset[i] = offset

        self.mimic_joint_indices = np.array(
            self.mimic_joint_indices, dtype=np.int32
        )
        self.mimic_control_joint_indices = np.array(
            self.mimic_control_joint_indices, dtype=np.int32
        )

        """list of control joint indices corresponding to mimic joint indices"""
        self.control_joint_indices = np.unique(self.mimic_control_joint_indices).astype(
            np.int32
        )
        """list of all directly controlled joint indices"""

        self.effective_dof = len(self.control_joint_indices)

    def _get_joint_limits(self):
        joint_limits = super()._get_joint_limits()
        joint_limits = joint_limits[self.control_joint_indices]
        if len(joint_limits.shape) == 1:
            joint_limits = joint_limits[None, :]
        return joint_limits

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos[self.control_joint_indices] += action
            else:
                self._target_qpos[self.control_joint_indices] = (
                    self._start_qpos[self.control_joint_indices] + action
                )
        else:
            self._target_qpos[self.control_joint_indices] = action
        self._target_qpos[self.mimic_joint_indices] = (
            self._target_qpos[self.mimic_control_joint_indices]
            * self._multiplier
            + self._offset
        )
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def __repr__(self):
        data = {k: v["joint"] for k, v in self.config.mimic.items()}
        main_str = "{\n"
        for k, v in data.items():
            main_str += f"    {k}: {v},\n"
        main_str += "}"
        data = main_str
        return f"PDJointPosMimicController(dof={self.single_action_space.shape[0]}, active_joints={len(self.joints)}, mimic_to_control_joint_map={main_str})"


@dataclass
class PDJointPosMimicControllerConfig(PDJointPosControllerConfig):
    """
    PD Joint Position mimic controller configuration. This kind of controller is used to emulate a mimic joint. For some simulation backends mimic joints are not
    well simulated and/or are hard to tune and so a mimic controller can be used for better stability as well as alignment with a real robot.

    The `joint_names` field is expected to be a list of all the joints, whether they are the joints to be controlled or the joints that are controlled via mimicking.

    `mimic` is a dictionary that maps the mimic joint name to a dictionary of the format dict(joint: str, multiplier: float, offset: float). `joint` is the controlling joint name and is required. All given joints in
    `joint_names` must be in the `mimic` dictionary as either a key or a value. The ones that are keys are referred to as the mimic joints and the ones that are values
    are referred to as the control joints. Control joints are the ones directly controlled by the agent/user and mimic joints are implicitly controlled by following the control joints via the following equation:

    q_mimic = q_controlling * multiplier + offset

    By default, the multiplier is 1.0 and the offset is 0.0.
    """

    controller_cls = PDJointPosMimicController
    mimic: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """the mimic targets. Maps the actual mimic joint name to a dictionary of the format dict(joint: str, multiplier: float, offset: float)"""
# Adapted from ManiSkill (https://github.com/haosulab/ManiSkill), Apache-2.0; see THIRD_PARTY_LICENSES.md.
import os
from typing import Dict

import numpy as np
import sapien

import env.utils.utils as utils
from env import ASSETS_DIR
from env.controllers import PDJointPosControllerConfig, PDJointPosMimicControllerConfig
from env.controllers.base_controller import CombinedController, ControllerConfig


class Panda:
    """Franka Panda arm with a parallel gripper, driven by PD joint position controllers."""

    urdf_path = os.path.join(ASSETS_DIR, "panda/panda_v2.urdf")
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            panda_leftfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            panda_rightfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    kqpos = np.array([0.0, 0.376, 0.0, -1.398, 0.00, 1.596, 0.750, 0.04, 0.04])

    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    gripper_joint_names = [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    ee_link_name = "panda_hand_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    def __init__(self, scene: sapien.Scene, control_freq: int):
        self.scene = scene
        self._control_freq = control_freq

        self._load_articulation()
        self.tcp = utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)

        # PD joint position controllers of the arm and the gripper
        self.controller = CombinedController(self._controller_configs, self.robot, self._control_freq, scene=self.scene)
        self.controller.set_drive_property()
        # NOTE: Balancing passive force is currently not supported in PhysX, so we work around by disabling gravity
        for link in self.robot.links:
            link.disable_gravity = True

    @property
    def _controller_configs(self) -> Dict[str, ControllerConfig]:
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.06,  # a trick to have force when the object is thin
            upper=0.06,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"panda_finger_joint2": {"joint": "panda_finger_joint1"}},
        )

        return dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos)

    def _load_articulation(self):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = False
        loader.disable_self_collisions = False
        urdf_config = utils.parse_urdf_config(self.urdf_config)
        utils.check_urdf_config(urdf_config)
        utils.apply_urdf_config(loader, urdf_config)

        builder = loader.parse(self.urdf_path)[0][0]
        self.robot = builder.build()
        self.robot.set_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self.robot.set_name("panda")

        self.robot.set_qf(np.zeros(self.robot.dof))

    def set_action(self, action):
        """
        Set the agent's action which is to be executed in the next environment timestep.
        This is essentially a wrapper around the controller's set_action method.
        """
        self.controller.set_action(action)

    def before_simulation_step(self):
        """Code that runs before each simulation step. By default it calls the controller's before_simulation_step method."""
        self.controller.before_simulation_step()

    def reset(self):
        """
        Reset the robot to its home pose `kqpos` with zero velocity and forces.
        """
        self.robot.set_qpos(self.kqpos)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.robot.set_qf(np.zeros(self.robot.dof))

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

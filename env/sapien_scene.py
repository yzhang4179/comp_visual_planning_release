import logging
import math
import os
from typing import Optional, Union

import mplib
import numpy as np
import sapien.core as sapien
import sapien.physx as physx
from mplib import Pose
from sapien.render import clear_cache

from env import ASSETS_DIR
from env.panda import Panda
from env.utils.create_actor import create_default_obj

OPEN = 1
CLOSED = -1


class SapienScene:
    """
    Base class of the tasks: a Panda arm on a table, one 256x256 camera and an mplib motion planner.
    An action is a target end-effector pose plus a gripper command, [x, y, z, qw, qx, qy, qz, gripper].

    Our models are trained and evaluated with the raster renderer (`enable_rt=False`, the default).
    `enable_rt=True` switches to the ray tracer and its own lights. We use it only for nicer images, and it needs a GPU
    with RT cores. The physics and the camera are the same in both modes.
    """

    def __init__(self, start: int = 0, goal: int = 0, recording: bool = False, headless: bool = True, enable_rt: bool = False):
        self._headless = headless
        self._enable_rt = enable_rt
        self._sim_freq: int = 100
        self._control_freq: int = 10
        self._control_timestep = 1 / self._control_freq
        assert self._sim_freq % self._control_freq == 0, "Simulation frequency must be divisible by control frequency"
        self._sim_steps_per_control: int = self._sim_freq // self._control_freq

        self.state_dict_registry = {'actors': {}, 'articulations': {}}
        self.steps = 0

        self.start = None
        self.goal = None
        self.set_start_goal(start, goal)
        self.recording = recording
        if self.recording:
            self.images_list = []

    def setup_scene(self):
        if self._enable_rt:
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(32)
            sapien.render.set_ray_tracing_path_depth(8)
            sapien.render.set_ray_tracing_denoiser('optix')
        else:
            # the shader is a global SAPIEN setting, reset it in case an RT scene was created before
            sapien.render.set_camera_shader_dir("default")

        physx.set_body_config(solver_position_iterations=15, solver_velocity_iterations=1, sleep_threshold=0.005)
        physx.set_scene_config(bounce_threshold=0.01, enable_pcm=True, enable_ccd=True, enable_enhanced_determinism=True, enable_friction_every_iteration=True, cpu_workers=1)
        physx_system = physx.PhysxCpuSystem()
        render_system = sapien.render.RenderSystem() if self._enable_rt else sapien.render.RenderSystem("cpu")
        self.scene = sapien.Scene([physx_system, render_system])
        self.scene.set_timestep(timestep=1 / self._sim_freq)

        if self._enable_rt:
            # dim ambient light, a strong key light with shadows, a fill light and two point lights
            self.scene.set_ambient_light([0.20, 0.20, 0.20])
            self.scene.add_directional_light([-0.6, -0.6, -1], [2.6, 2.6, 2.6], shadow=True, shadow_scale=0.2, shadow_map_size=4096)
            self.scene.add_directional_light([1, 0, -1], [1.0, 1.0, 1.0], shadow=False)
            for pos, power in [([1.0, 1.5, 1.6], [2.0, 2.0, 2.0]), ([-1.0, -1.5, 1.6], [2.0, 2.0, 2.0])]:
                self.scene.add_point_light(pos, power, shadow=False)
        else:
            self.scene.set_ambient_light([0.64, 0.64, 0.64])
            direction_lights = [[[-1, 0, -1], [0.35, 0.35, 0.35]], [[-1, -1.5, -1], [0.4, 0.35, 0.35]]]
            for direction_light in direction_lights:
                self.scene.add_directional_light(direction_light[0], direction_light[1], shadow=False)
            point_lights = [[[1, 2, 2], [1, 1, 1]], [[1, -2, 2], [1, 1, 1]]]
            for point_light in point_lights:
                self.scene.add_point_light(point_light[0], point_light[1], shadow=False)

        if not self._headless:
            # initialize viewer with camera position and orientation
            self.viewer = sapien.utils.Viewer()
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(x=1.55, y=0, z=0.5)
            self.viewer.set_camera_rpy(r=0, p=-0.15, y=math.pi)

    def load_table(self):
        table_mat = sapien.render.RenderMaterial()
        table_mat.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/light-gray-plaster.png')))
        leg_mat = sapien.render.RenderMaterial()
        leg_mat.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/steel-brushed.png')))

        visuals = [{'shape_type': 'box', 'size': [33, 33, 1e-6], 'p': [0, 0, 0], 'material': table_mat}]
        for x, y in [[0.8, 0.8], [0.8, -0.8], [-0.8, 0.8], [-0.8, -0.8]]:
            visuals.append({'shape_type': 'cylinder', 'radius': 0.05, 'length': 1.2, 'p': [x, y, -0.6], 'q': [0, 90, 0], 'material': leg_mat})
        self.state_dict_registry['actors']['table'] = \
            create_default_obj(scene=self.scene, visuals=visuals, phys_type='static', p=[0, 0, 0], q=[0, 0, 0], name='table')

    def load_agent(self):
        self.agent = Panda(self.scene, control_freq=self._control_freq)
        self.state_dict_registry['articulations']['panda'] = self.agent.robot

    def setup_planner(self):
        """
        Create an mplib planner for the robot.
        """
        link_names = [link.get_name() for link in self.agent.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.agent.robot.get_active_joints()]
        self.planner = mplib.Planner(
            urdf=os.path.join(ASSETS_DIR, 'panda/panda_v2.urdf'),
            srdf=os.path.join(ASSETS_DIR, 'panda/panda_v2.srdf'),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group='panda_hand_tcp',
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7),
        )
        self.planner.set_base_pose(sapien.Pose([0, 0, 0]))
        self.gripper_state = OPEN

    def follow_path(self, result):
        """Helper function to follow a path generated by the planner"""
        n_step = result["position"].shape[0]
        for i in range(n_step):
            action = np.hstack([result["position"][i], self.gripper_state])
            self.control_step(action)

    def _safe_plan(self, planner_fn, *args, **kwargs):
        logging.disable(logging.WARNING)
        try:
            result = planner_fn(*args, **kwargs)
        except RuntimeError as e:
            result = {"status": "FailParameterize", "error": str(e)}
        finally:
            logging.disable(logging.NOTSET)
        return result

    def move_to_pose_with_RRTConnect(self, pose: Pose):
        """
        Plan and follow a path to a pose using RRTConnect. Returns 1 on success and 0 if planning fails.
        """
        q0 = self.agent.robot.get_qpos()
        result = self._safe_plan(self.planner.plan_pose, pose, q0, time_step=self._control_timestep)
        if result["status"] != "Success":
            return 0
        self.follow_path(result)
        return 1

    def move_to_pose_with_screw(self, pose: Pose):
        """
        Interpolative planning with screw motion.
        Falls back to RRTConnect if the screw motion fails (e.g. because the path collides).
        """
        q0 = self.agent.robot.get_qpos()
        result = self._safe_plan(self.planner.plan_screw, pose, q0, time_step=self._control_timestep)
        if result["status"] == "Success":
            self.follow_path(result)
            return 1
        else:
            return self.move_to_pose_with_RRTConnect(pose)

    def open_gripper(self):
        self.gripper_state = OPEN
        qpos = np.array(self.agent.robot.get_qpos()[:-2])
        for i in range(20):
            self.control_step(np.hstack([qpos, self.gripper_state]))

    def close_gripper(self):
        self.gripper_state = CLOSED
        qpos = np.array(self.agent.robot.get_qpos()[:-2])
        for i in range(20):
            self.control_step(np.hstack([qpos, self.gripper_state]))

    def get_image(self):
        self.scene.update_render()
        self.camera.take_picture()
        color_img = self.camera.get_picture('Color')[..., :3]
        color_img = (color_img * 255).clip(0, 255).astype(np.uint8).copy()
        return color_img

    def render_state(self, state: dict, start: int, goal: int):
        """
        Render `state` as it is, without stepping the simulation (`reset` would move the arm to its home pose).
        `start` sets the parts of the scene that depend on the start, such as the pre-drawn strokes of the drawer task.
        """
        self.set_start_goal(start, goal)
        self.set_sim_state(state)
        return self.get_image()

    def check_actors_contact(self, actor1: str, actor2: str) -> bool:
        """
        Check if two actors are in contact.
        - actor1: The first actor's name.
        - actor2: The second actor's name.
        """
        contacts = self.scene.get_contacts()
        for contact in contacts:
            if (contact.bodies[0].entity.name == actor1
                    and contact.bodies[1].entity.name == actor2) or (contact.bodies[0].entity.name == actor2
                                                                     and contact.bodies[1].entity.name == actor1):
                return True
        return False

    def create_fixed_connection(
        self,
        body0: Optional[Union[sapien.Entity, physx.PhysxRigidBaseComponent]],
        pose0: sapien.Pose,
        body1: Union[sapien.Entity, physx.PhysxRigidBaseComponent],
        pose1: sapien.Pose,
        stiff: float = 1e5, damp: float = 1e3, force_limit: float = 1e7,
    ):
        """
        Fix `body1` to `body0` (or to the world if `body0` is None) with a distance joint and a stiff drive.
        """
        def _to_rigid_comp(body):
            if body is None:
                return None
            if isinstance(body, sapien.Entity):
                return next(
                    c for c in body.components
                    if isinstance(c, physx.PhysxRigidBaseComponent)
                )
            else:
                return body

        c0 = _to_rigid_comp(body0)
        if isinstance(body1, sapien.Entity):
            e1 = body1
            c1 = _to_rigid_comp(body1)
        else:
            e1 = body1.entity
            c1 = body1

        dist_joint = physx.PhysxDistanceJointComponent(c1)
        dist_joint.parent = c0
        dist_joint.pose_in_parent = pose0
        dist_joint.pose_in_child = pose1
        e1.add_component(dist_joint)
        dist_joint.set_limit(0.0, 0.0)

        drive = physx.PhysxDriveComponent(c1)
        drive.parent = c0
        drive.pose_in_parent = pose0
        drive.pose_in_child = pose1
        e1.add_component(drive)
        drive.set_drive_property_slerp(
            stiffness=stiff,
            damping=damp,
            force_limit=force_limit,
        )
        drive.set_drive_target(sapien.Pose())

        return dist_joint, drive

    def set_vel_zero(self, actor_name: str):
        rd: sapien.physx.PhysxRigidDynamicComponent = self.state_dict_registry['actors'][actor_name].find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent
        )
        assert rd is not None, f"{actor_name} must have a PhysxRigidDynamicComponent"
        rd.set_linear_velocity([0, 0, 0])
        rd.set_angular_velocity([0, 0, 0])

    def get_sim_state(self):
        state_dict = dict()
        state_dict["actors"] = dict()
        state_dict["articulations"] = dict()
        for actor in self.state_dict_registry['actors'].values():
            if actor.name not in state_dict["actors"]:
                state_dict["actors"][actor.name] = {}
            pose = actor.get_pose()
            state_dict["actors"][actor.name]['pose'] = np.concatenate((pose.p, pose.q)).copy()
            rd: sapien.physx.PhysxRigidDynamicComponent = actor.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            )
            if rd is not None:
                state_dict["actors"][actor.name]['linear_velocity'] = np.array(rd.get_linear_velocity()).copy()
                state_dict["actors"][actor.name]['angular_velocity'] = np.array(rd.get_angular_velocity()).copy()

        for articulation in self.state_dict_registry["articulations"].values():
            if articulation.name not in state_dict["articulations"]:
                state_dict["articulations"][articulation.name] = {}
            pose = articulation.get_root_pose()
            state_dict["articulations"][articulation.name]['pose'] = np.concatenate((pose.p, pose.q)).copy()
            state_dict["articulations"][articulation.name]['qpos'] = np.array(articulation.get_qpos()).copy()
            state_dict["articulations"][articulation.name]['qvel'] = np.array(articulation.get_qvel()).copy()
        if len(state_dict["actors"]) == 0:
            del state_dict["actors"]
        if len(state_dict["articulations"]) == 0:
            del state_dict["articulations"]

        return state_dict

    def set_sim_state(self, state: dict):
        if "actors" in state.keys():
            for actor_id, actor_state in state["actors"].items():
                actor = self.state_dict_registry["actors"][actor_id]

                pose_arr = actor_state["pose"]
                actor.set_pose(sapien.Pose(p=pose_arr[:3], q=pose_arr[3:]))

                if "linear_velocity" in actor_state or "angular_velocity" in actor_state:
                    rd = actor.find_component_by_type(
                        sapien.physx.PhysxRigidDynamicComponent
                    )
                    if rd is not None:
                        if "linear_velocity" in actor_state:
                            rd.set_linear_velocity(actor_state["linear_velocity"])
                        if "angular_velocity" in actor_state:
                            rd.set_angular_velocity(actor_state["angular_velocity"])

        if "articulations" in state.keys():
            for art_id, art_state in state["articulations"].items():
                self.state_dict_registry["articulations"][art_id].set_root_pose(sapien.Pose(p=art_state["pose"][:3], q=art_state["pose"][3:]))
                self.state_dict_registry["articulations"][art_id].set_qpos(art_state["qpos"])
                self.state_dict_registry["articulations"][art_id].set_qvel(art_state["qvel"])

    def set_start_goal(self, start: int, goal: int):
        self.start = start
        self.goal = goal

    def get_observations(self):
        obs = dict()
        obs['states'] = self.get_sim_state()
        obs['images'] = self.get_image()
        return obs

    def reset(self, start: int, goal: int, initial_states: Optional[dict] = None):
        """
        Reset to `initial_states`, or to a random scene of `start` if it is None,
        then hold the arm at its home pose for 50 control steps to let the scene settle.
        """
        self.set_start_goal(start, goal)

        self.agent.reset()
        self.agent.controller.reset()
        self.gripper_state = OPEN
        if initial_states is not None:
            self.set_sim_state(initial_states)
        else:
            self._initialize_episode()
        qpos = np.array(self.agent.kqpos[:-2])
        for _ in range(50):
            self.control_step(np.hstack([qpos, self.gripper_state]))
        self.steps = 0

        if self.recording:
            self.images_list = []
            self.images_list.append(self.get_image())
        return self.get_observations()

    def _initialize_episode(self):
        '''
        Set a random initial scene of `self.start`.
        '''

    def simulate(self):
        """
        Simulate the scene for one step.
        """
        self.scene.step()
        self._after_simulation_step()
        self.steps += 1
        if not self._headless:
            if self.steps % 10 == 0:
                self.scene.update_render()
                self.viewer.render()

    def _after_simulation_step(self):
        pass

    def _after_control_step(self):
        pass

    def control_step(self, action: np.ndarray):
        self.agent.set_action(action)
        for _ in range(self._sim_steps_per_control):
            self.agent.before_simulation_step()
            self.simulate()

        self._after_control_step()
        if self.recording:
            self.images_list.append(self.get_image())

    def get_recordings(self):
        """
        Get and clear the recorded images.
        """
        if not self.recording:
            print("Recording is not enabled.")
            return None

        images = self.images_list
        self.images_list = []
        return images

    def compute_reward(self):
        reward = 0
        info = dict()
        return reward, info

    def step(self, action: np.ndarray):
        '''
        Step the environment with the given action.
        Args:
            action: np.ndarray of shape (8,), target end-effector pose (p, q) and gripper command (1 open, -1 closed)
        Returns:
            obs: dict, the observations of the environment
            reward: float, 1 if the task is solved
            done: bool, whether the motion planning failed
            info: dict, task-specific progress and plan_failed
        '''
        res = self.move_to_pose_with_screw(sapien.Pose(p=action[:3], q=action[3:-1]))
        if abs(action[-1] - self.gripper_state) < 1e-3:
            pass
        else:
            if abs(action[-1] - OPEN) < 1e-3:
                self.open_gripper()
            elif abs(action[-1] - CLOSED) < 1e-3:
                self.close_gripper()

        reward, info = self.compute_reward()
        done = True if res == 0 else False
        info['plan_failed'] = 1 if res == 0 else 0

        return self.get_observations(), reward, done, info

    def clear_cache(self, clear: bool = True):
        if clear:
            clear_cache()

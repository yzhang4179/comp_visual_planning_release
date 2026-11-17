import os

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from env import ASSETS_DIR
from env.sapien_scene import OPEN, SapienScene
from env.utils.create_actor import create_actor, create_default_obj
from env.utils.utils import look_at


class Puzzle(SapienScene):
    """
    Four puzzle pieces start on the corner landmarks a, b, c, d in one of 8 orders (`start`). The robot first moves
    each piece to its own landmark, then assembles them into one of 8 shapes (`goal`) at the center target e.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_scene()
        self.load_agent()
        self.setup_planner()
        self.add_camera()
        self.load_table()

        self.load_pieces()
        self.load_goals()
        self.load_landmarks()
        self.load_target()

    def add_camera(self):
        self.camera = self.scene.add_camera(
            name="camera",
            width=256,
            height=256,
            fovy=0.5,
            near=0.01,
            far=100,
        )
        self.camera.set_entity_pose(
            look_at(eye=np.array([1.5, 0., 0.96]), target=np.array([0, 0, -0.3]))
        )
        self.scene.update_render()
        self.camera.take_picture()

    def load_pieces(self):
        for i in range(1, 5):
            self.state_dict_registry["actors"][f'piece_{i}'] = create_actor(
                    scene=self.scene,
                    modelname='piece',
                    pose=sapien.Pose(p=[(i-4) * 0.3, -10, 0.05], q=euler2quat(*np.deg2rad([0, 0, 0]))),
                    scale=(0.068, 0.068, 0.037),
                    visual_scale=(0.0695, 0.0695, 0.037),  # make the visual slightly larger to cover the gap
                    model_id=i,
                    convex=True,
                )
            self.state_dict_registry["actors"][f'piece_{i}'].set_mass(0.3)
            self.state_dict_registry["actors"][f'piece_{i}'].set_physics_properties(
                static_friction=0.6,
                dynamic_friction=0.3,
                restitution=1e-4
            )
            self.state_dict_registry["actors"][f'piece_{i}'].actor.set_name(f'piece_{i}')

    def load_landmarks(self):
        for i in range(4):
            mat = sapien.render.RenderMaterial()
            mat.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/ceramic.png')))

            mat2 = sapien.render.RenderMaterial()
            mat2.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/light-wood-v2.png')))
            visuals = [
                {
                    "shape_type": "box",
                    "size": [0.2, 0.2, 0.0005],
                    "material": mat2,
                    "ignore_collision": True
                },
                {
                    "shape_type": "box",
                    "size": [0.16, 0.17, 0.001],
                    "material": mat,
                    "ignore_collision": True
                },
            ]

            self.state_dict_registry['actors'][f'landmark_{i+1}'] = \
                create_default_obj(scene=self.scene, visuals=visuals, phys_type="static", p=[0.25, -0.25, 0.001], q=[0, 0, 0], name=f"landmark_{i+1}")

    def load_target(self):
        radius = 0.08
        mat = sapien.render.RenderMaterial()
        mat.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/ceramic.png')))
        mat2 = sapien.render.RenderMaterial()
        mat2.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/ceramic.png')))
        # five concentric rings
        radii = [radius, radius * 4 / 5, radius * 3 / 5, radius * 2 / 5, radius * 1 / 5]
        offsets = [0, 1e-5, 2e-5, 3e-5, 4e-5]
        visuals = [
            {
                "shape_type": "cylinder",
                "radius": radii[k],
                "half_length": 1e-5 / 2 + offsets[k],
                "material": mat if k % 2 == 0 else mat2,
                "ignore_collision": True
            }
            for k in range(5)
        ]

        position = [0.75, 0.0, 1e-3]
        orientation = [0, -90.0, 0]

        self.state_dict_registry['actors']['target'] = \
            create_default_obj(scene=self.scene, visuals=visuals, phys_type="static", p=position, q=orientation, name="target")

    def load_goals(self):
        """The 8 goal shapes, only used for their piece layouts and kept far outside the camera view."""
        for i in range(8):
            self.state_dict_registry["actors"][f'goal_{i}'] = create_actor(
                    scene=self.scene,
                    modelname='goal',
                    pose=sapien.Pose(p=[(i-4) * 0.3, 10, 0.05], q=euler2quat(*np.deg2rad([0, 0, 0]))),
                    scale=(0.07, 0.07, 0.04),
                    model_id=i,
                    convex=True,
                )
            self.state_dict_registry["actors"][f'goal_{i}'].set_mass(0.3)
            self.state_dict_registry["actors"][f'goal_{i}'].set_physics_properties(
                static_friction=0.6,
                dynamic_friction=0.3,
                restitution=1e-3
            )
            self.state_dict_registry["actors"][f'goal_{i}'].actor.set_name(f'goal_{i}')
        for i in range(8):
            self.state_dict_registry["actors"][f'goal_{i}'].actor.set_pose(sapien.Pose(p=[(i-4) * 0.5, 10, 0.01], q=euler2quat(*np.deg2rad([0, 0, 0]))))

    def get_sim_state(self):
        state_dict = super().get_sim_state()
        state_dict['abcde'] = {key: np.concatenate([pose.p, pose.q]) for key, pose in self.abcde_poses.items()}
        return state_dict

    def set_sim_state(self, state_dict):
        super().set_sim_state(state_dict)
        self.abcde_poses = {key: sapien.Pose(p=pose[:3], q=pose[3:]) for key, pose in state_dict['abcde'].items()}

    def generate_abcde_poses(self):
        # ---------------#
        # a   c
        #   e
        # b   d
        # ---------------#
        init_shift = {"y": np.random.uniform(-0.065, 0.065),
                      "x": np.random.uniform(-0.07, 0.0)}

        left = -0.21 + init_shift["y"]
        right = 0.21 + init_shift["y"]
        up = 0.35 + init_shift["x"]
        down = 0.68 + init_shift["x"]
        a = sapien.Pose(p=[up, left, 0.033], q=euler2quat(*np.deg2rad([0, 0, 0])))
        b = sapien.Pose(p=[down, left, 0.033], q=euler2quat(*np.deg2rad([0, 0, 0])))
        c = sapien.Pose(p=[up, right, 0.033], q=euler2quat(*np.deg2rad([0, 0, 0])))
        d = sapien.Pose(p=[down, right, 0.033], q=euler2quat(*np.deg2rad([0, 0, 0])))
        e = sapien.Pose(p=[(up+down)/2.+0.125, (left+right)/2., 0.033], q=euler2quat(*np.deg2rad([0, 0, 0])))
        self.abcde_poses = {"a": a, "b": b, "c": c, "d": d, "e": e}

    @property
    def rearrangement_goals(self):
        goals = []
        for i, key in enumerate(['a', 'b', 'c', 'd']):
            target_pose = self.abcde_poses[key]
            init_pose = self.init_pieces_abcd[f's{self.start}'][str(i+1)]
            target_pose = sapien.Pose(target_pose.p, init_pose.q)
            goals.append(target_pose)
        return goals

    @property
    def puzzle_goals(self):
        goals = []
        for i in range(1, 5):
            goal_pose = self.state_dict_registry["actors"][f'goal_{self.goal}'].get_goal_pose(goal_idx=self.goal, model_idx=i, ret='pose')
            e_pose = self.abcde_poses['e']
            goal_pose = sapien.Pose(p=[goal_pose.p[0] + e_pose.p[0], goal_pose.p[1] + e_pose.p[1], e_pose.p[2]], q=goal_pose.q)
            goals.append(goal_pose)
        return goals

    @property
    def init_pieces_abcd(self):
        def get_s_dict(order: list[str]):
            return {str(i+1): self.abcde_poses[order[i]] for i in range(len(order))}

        orders = [
            ['a', 'd', 'b', 'c'],
            ['c', 'a', 'b', 'd'],
            ['b', 'd', 'c', 'a'],
            ['b', 'c', 'a', 'd'],
            ['d', 'a', 'c', 'b'],
            ['d', 'b', 'a', 'c'],
            ['c', 'b', 'd', 'a'],
            ['a', 'c', 'd', 'b']
        ]
        abcd = {}
        for i, order in enumerate(orders):
            abcd[f"s{i}"] = get_s_dict(order)
            abcd[f"s{i}"]['2'] = abcd[f"s{i}"]['2'] * sapien.Pose(p=[0, 0, 0], q=euler2quat(*np.deg2rad([0, 0, 180])))
            abcd[f"s{i}"]['3'] = abcd[f"s{i}"]['3'] * sapien.Pose(p=[0, 0, 0], q=euler2quat(*np.deg2rad([0, 0, -135])))

        return abcd

    def ignore_collisions_among(self, actors, bit_idx=0):
        mask = 1 << bit_idx
        for actor in actors:
            if isinstance(actor, sapien.physx.PhysxArticulationLinkComponent):
                obj = actor.entity
            else:
                obj = actor
            rigid_body_component: sapien.physx.PhysxRigidBodyComponent = obj.find_component_by_type(sapien.physx.PhysxRigidBodyComponent)
            if rigid_body_component is not None:
                for shape in rigid_body_component.collision_shapes:
                    g0, g1, g2, g3 = shape.get_collision_groups()
                    g2 |= mask
                    shape.set_collision_groups([g0, g1, g2, g3])

    def clear_ignore_collisions_among(self, actors):
        for actor in actors:
            if isinstance(actor, sapien.physx.PhysxArticulationLinkComponent):
                obj = actor.entity
            else:
                obj = actor
            rigid_body_component: sapien.physx.PhysxRigidBodyComponent = obj.find_component_by_type(
                sapien.physx.PhysxRigidBodyComponent
            )
            if rigid_body_component is not None:
                for shape in rigid_body_component.collision_shapes:
                    g0, g1, g2, g3 = shape.get_collision_groups()
                    g2 = 0
                    shape.set_collision_groups([g0, g1, g2, g3])

    def reset(self, *args, **kwargs):
        self.clear_ignore_collisions_among([self.state_dict_registry["actors"][f'piece_{i}'].actor for i in range(1, 5)] + self.agent.robot.get_links())
        self.ignore_collisions_among([self.state_dict_registry["actors"][f'piece_{i}'].actor for i in range(1, 5)])
        self.pieces_rearranged = [False] * 4
        self.pieces_puzzled = [False] * 4
        return super().reset(*args, **kwargs)

    def _initialize_episode(self):
        self.generate_abcde_poses()
        init_abcd_poses = self.init_pieces_abcd[f's{self.start}']
        for i in range(1, 5):
            self.state_dict_registry["actors"][f'piece_{i}'].actor.set_pose(init_abcd_poses[str(i)])

        for i, key in enumerate(['a', 'b', 'c', 'd']):
            landmark_pose = self.abcde_poses[key]
            self.state_dict_registry['actors'][f'landmark_{i+1}'].set_pose(sapien.Pose(p=[landmark_pose.p[0], landmark_pose.p[1], 0.001], q=euler2quat(*np.deg2rad([0, 0, 0]))))

        e_pose = self.abcde_poses['e']
        self.state_dict_registry['actors']['target'].set_pose(sapien.Pose(p=[e_pose.p[0], e_pose.p[1], 0.005], q=euler2quat(*np.deg2rad([0, 90, 0]))))

    def check_reached(self, current_pose: sapien.Pose, target_pose: sapien.Pose, pos_threshold=0.05, rot_threshold=np.deg2rad(5)):
        error_pose = target_pose.inv() * current_pose
        p_error = np.linalg.norm(error_pose.p)
        R = error_pose.to_transformation_matrix()[:3, :3]
        cos_theta = np.clip((np.trace(R) - 1) / 2., -1, 1)
        error_angle = np.arccos(cos_theta)
        return p_error < pos_threshold and error_angle < rot_threshold

    def check_success_rearrange(self):
        for i in range(1, 5):
            if self.pieces_rearranged[i-1]:
                continue
            else:
                piece_pose = self.state_dict_registry["actors"][f'piece_{i}'].actor.get_pose()
                target_pose = self.rearrangement_goals[i-1]
                reached = self.check_reached(piece_pose, target_pose, pos_threshold=0.08 if i == 4 else 0.05, rot_threshold=np.deg2rad(10) if i == 4 else np.deg2rad(5))
                if reached and self.gripper_state == OPEN and self.agent.tcp_pos[2] > 0.06:
                    self.pieces_rearranged[i-1] = True
                    self.state_dict_registry["actors"][f'piece_{i}'].actor.set_pose(target_pose)
                    self.set_vel_zero(f'piece_{i}')

    def check_success_puzzle(self):
        for i in range(1, 5):
            if self.pieces_puzzled[i-1]:
                continue
            else:
                piece_pose = self.state_dict_registry["actors"][f'piece_{i}'].actor.get_pose()
                target_pose = self.puzzle_goals[i-1]
                reached = self.check_reached(piece_pose, target_pose, pos_threshold=0.08 if i == 4 or i == 3 else 0.05, rot_threshold=np.deg2rad(10) if i == 4 or i == 3 else np.deg2rad(5))
                if reached:
                    self.pieces_puzzled[i-1] = True
                    self.state_dict_registry["actors"][f'piece_{i}'].actor.set_pose(target_pose)
                    self.set_vel_zero(f'piece_{i}')
                    self.ignore_collisions_among([self.state_dict_registry["actors"][f'piece_{i}'].actor] + self.agent.robot.get_links(), bit_idx=i)

    def compute_reward(self):
        if not all(self.pieces_rearranged):
            self.check_success_rearrange()
        if all(self.pieces_rearranged):
            self.check_success_puzzle()
        if all(self.pieces_puzzled) and all(self.pieces_rearranged):
            reward = 1.0
        else:
            reward = 0
        info = {'pieces_rearranged': self.pieces_rearranged, 'pieces_puzzled': self.pieces_puzzled}
        return reward, info

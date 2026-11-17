import os

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from env import ASSETS_DIR
from env.sapien_scene import OPEN, SapienScene
from env.utils.create_actor import create_default_obj
from env.utils.utils import look_at


class Cube(SapienScene):
    """
    Four cubes start in one of 4 layouts (`start`) around the corner landmarks a, b, c, d. The robot first moves
    each cube onto its landmark, then arranges them into one of 4 shapes (`goal`) in the middle.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_scene()
        self.load_agent()
        self.setup_planner()
        self.add_camera()
        self.load_table()

        self.load_cubes()
        self.load_target()
        self.load_landmarks()

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
            look_at(eye=np.array([1.43, 0., 0.95]), target=np.array([0, 0, -0.33]))
        )
        self.scene.update_render()
        self.camera.take_picture()

    def load_cubes(self):
        for i, (texture, p) in enumerate([('red-wood', [0.5, -0.25, 0.03]), ('lemon', [0.5, 0.25, 0.03]),
                                          ('green-wood', [0.3, -0.25, 0.03]), ('blue-wood', [0.3, 0.25, 0.03])]):
            mat = sapien.render.RenderMaterial()
            mat.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, f'textures/{texture}.png')))
            visuals = [
                {
                    'shape_type': 'box',
                    'size': [0.06, 0.06, 0.058],
                    'p': [0, 0, 0],
                    'material': mat,
                },
            ]
            self.state_dict_registry['actors'][f'cube_{i+1}'] = \
                create_default_obj(scene=self.scene, visuals=visuals, phys_type=None, p=p, q=[0, 0, 0], name=f'cube_{i+1}')

    def load_landmarks(self):
        for i in range(4):
            mat = sapien.render.RenderMaterial()
            mat.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/ceramic.png')))
            visuals = [
                {
                    "shape_type": "box",
                    "size": [0.133, 0.133, 0.001],
                    "material": mat,
                    "ignore_collision": True
                },
            ]

            self.state_dict_registry['actors'][f'landmark_{i+1}'] = \
                create_default_obj(scene=self.scene, visuals=visuals, phys_type="static", p=[0.25, -0.25, 0.001], q=[0, 0, 0], name=f"landmark_{i+1}")

    def load_target(self):
        radius = 0.093
        mat = sapien.render.RenderMaterial()
        mat.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/light-wood-v2.png')))

        mat2 = sapien.render.RenderMaterial()
        mat2.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/ceramic.png')))
        # five concentric rings, alternating the two materials
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
        init_shift = {"y": np.random.uniform(-0.08, 0.08),
                      "x": np.random.uniform(-0.03, 0.0)}

        left = -0.2 + init_shift["y"]
        right = 0.2 + init_shift["y"]
        up = 0.35 + init_shift["x"]
        down = 0.65 + init_shift["x"]
        a = sapien.Pose(p=[up, left, 0.033], q=euler2quat(*np.deg2rad([0, 0, 0])))
        b = sapien.Pose(p=[down, left, 0.033], q=euler2quat(*np.deg2rad([0, 0, 0])))
        c = sapien.Pose(p=[up, right, 0.033], q=euler2quat(*np.deg2rad([0, 0, 0])))
        d = sapien.Pose(p=[down, right, 0.033], q=euler2quat(*np.deg2rad([0, 0, 0])))
        e = sapien.Pose(p=[(up + down) / 2. + 0.03, (left + right) / 2., 0.033], q=euler2quat(*np.deg2rad([0, 0, 0])))
        self.abcde_poses = {"a": a, "b": b, "c": c, "d": d, "e": e}

    @property
    def rearrangement_goals(self):
        goals = []
        for i, key in enumerate(['a', 'b', 'c', 'd']):
            target_pose = self.abcde_poses[key]
            goals.append(target_pose)
        return goals

    @property
    def final_goals(self):
        a_pose = self.abcde_poses['a']
        d_pose = self.abcde_poses['d']
        center_x = (a_pose.p[0] + d_pose.p[0]) / 2. + 0.06
        center_y = (a_pose.p[1] + d_pose.p[1]) / 2.
        goal0 = [
            sapien.Pose(p=[center_x, center_y, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))),
            sapien.Pose(p=[center_x - 0.06, center_y, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))),
            sapien.Pose(p=[center_x - 0.06, center_y + 0.068*2, 0.035], q=euler2quat(*np.deg2rad([0, 0, 90]))),
            sapien.Pose(p=[center_x - 0.06, center_y + 0.068, 0.035], q=euler2quat(*np.deg2rad([0, 0, 90]))),
        ]
        goal1 = [
            sapien.Pose(p=[center_x, center_y, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))),
            sapien.Pose(p=[center_x - 0.068, center_y, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))),
            sapien.Pose(p=[center_x - 0.068*2, center_y, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))),
            sapien.Pose(p=[center_x + 0.068, center_y, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))),
        ]
        goal2 = [
            sapien.Pose(p=[center_x, center_y, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))),
            sapien.Pose(p=[center_x, center_y + 0.068*2, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))) * sapien.Pose(q=euler2quat(*np.deg2rad([0, 0, 90]))),
            sapien.Pose(p=[center_x, center_y - 0.068, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))) * sapien.Pose(q=euler2quat(*np.deg2rad([0, 0, 90]))),
            sapien.Pose(p=[center_x, center_y + 0.068, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))) * sapien.Pose(q=euler2quat(*np.deg2rad([0, 0, 90]))),
        ]
        goal3 = [
            sapien.Pose(p=[center_x, center_y, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))),
            sapien.Pose(p=[center_x, center_y + 0.068, 0.043 * 2], q=euler2quat(*np.deg2rad([0, 0, 0]))) * sapien.Pose(q=euler2quat(*np.deg2rad([0, 0, 0]))),
            sapien.Pose(p=[center_x, center_y + 0.068 * 2, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))) * sapien.Pose(q=euler2quat(*np.deg2rad([0, 0, 90]))),
            sapien.Pose(p=[center_x, center_y + 0.068, 0.035], q=euler2quat(*np.deg2rad([0, 0, 0]))) * sapien.Pose(q=euler2quat(*np.deg2rad([0, 0, 90]))),
        ]
        goal_poses = [goal0, goal1, goal2, goal3]
        return goal_poses[self.goal]

    @property
    def init_cubes_abcd(self):
        # ---------------#
        # a  ac  c
        # ab  e  cd
        # b  bd  d
        # ---------------#
        a = self.abcde_poses['a']
        b = self.abcde_poses['b']
        c = self.abcde_poses['c']
        d = self.abcde_poses['d']
        e = self.abcde_poses['e']
        e = sapien.Pose(p=[e.p[0], e.p[1], e.p[2]], q=euler2quat(*np.deg2rad([0, 0, 0])))

        ac = sapien.Pose(p=[a.p[0], (a.p[1] + c.p[1]) / 2., a.p[2]], q=euler2quat(*np.deg2rad([0, 0, 0])))
        ab = sapien.Pose(p=[(a.p[0] + b.p[0]) / 2., a.p[1], a.p[2]], q=euler2quat(*np.deg2rad([0, 0, 0])))
        bd = sapien.Pose(p=[b.p[0], (b.p[1] + d.p[1]) / 2., b.p[2]], q=euler2quat(*np.deg2rad([0, 0, 0])))
        cd = sapien.Pose(p=[(c.p[0] + d.p[0]) / 2., c.p[1], c.p[2]], q=euler2quat(*np.deg2rad([0, 0, 0])))

        abcd = {
            's0': {'1': ac, '2': ab, '3': bd, '4': e},
            's1': {'1': ac, '2': ab, '3': bd, '4': cd},
            's2': {'1': ac, '2': ab, '3': e, '4': cd},
            's3': {'1': bd, '2': ab, '3': e, '4': cd},
        }
        return abcd

    def reset(self, *args, **kwargs):
        self.cubes_rearranged = [False] * 4
        self.cubes_puzzled = [False] * 4
        return super().reset(*args, **kwargs)

    def _initialize_episode(self):
        self.generate_abcde_poses()
        init_abcd_poses = self.init_cubes_abcd[f's{self.start}']
        for i in range(1, 5):
            self.state_dict_registry["actors"][f'cube_{i}'].set_pose(init_abcd_poses[str(i)])
        e_pose = self.abcde_poses['e']
        self.state_dict_registry["actors"]['target'].set_pose(sapien.Pose(p=[e_pose.p[0], e_pose.p[1], 0.005], q=euler2quat(*np.deg2rad([0, -90.0, 0]))))

        for i, char in enumerate(['a', 'b', 'c', 'd']):
            char_pose = self.abcde_poses[char]
            self.state_dict_registry["actors"][f'landmark_{i+1}'].set_pose(sapien.Pose(p=[char_pose.p[0], char_pose.p[1], 0.001], q=euler2quat(*np.deg2rad([0, 0, 90*i]))))

    def check_reached(self, current_pose: sapien.Pose, target_pose: sapien.Pose, pos_threshold=0.05, rot_threshold=np.deg2rad(5)):
        error_pose = target_pose.inv() * current_pose
        p_error = np.linalg.norm(error_pose.p)
        R = error_pose.to_transformation_matrix()[:3, :3]
        cos_theta = np.clip((np.trace(R) - 1) / 2., -1, 1)
        error_angle = np.arccos(cos_theta)
        return p_error < pos_threshold and error_angle < rot_threshold

    def check_success_rearrange(self):
        for i in range(1, 5):
            if self.cubes_rearranged[i-1]:
                continue
            else:
                cube_pose = self.state_dict_registry["actors"][f'cube_{i}'].get_pose()
                target_pose = self.rearrangement_goals[i-1]
                reached = self.check_reached(cube_pose, target_pose, pos_threshold=0.05, rot_threshold=np.deg2rad(5))

                if reached and self.gripper_state == OPEN and self.agent.tcp_pos[2] > 0.06:
                    self.cubes_rearranged[i-1] = True
                    self.state_dict_registry["actors"][f'cube_{i}'].set_pose(target_pose)
                    self.set_vel_zero(f'cube_{i}')

    def check_success_puzzle(self):
        for i in range(1, 5):
            if self.cubes_puzzled[i-1]:
                continue
            else:
                cube_pose = self.state_dict_registry["actors"][f'cube_{i}'].get_pose()
                target_pose = self.final_goals[i-1]
                reached = self.check_reached(cube_pose, target_pose, pos_threshold=0.05, rot_threshold=np.deg2rad(5))
                if reached:
                    self.cubes_puzzled[i-1] = True
                    self.state_dict_registry["actors"][f'cube_{i}'].set_pose(target_pose)
                    self.set_vel_zero(f'cube_{i}')

    def compute_reward(self):
        if not all(self.cubes_rearranged):
            self.check_success_rearrange()
        if all(self.cubes_rearranged):
            self.check_success_puzzle()
        if all(self.cubes_puzzled) and all(self.cubes_rearranged):
            reward = 1.0
        else:
            reward = 0
        info = {'cubes_rearranged': self.cubes_rearranged, 'cubes_puzzled': self.cubes_puzzled}
        return reward, info

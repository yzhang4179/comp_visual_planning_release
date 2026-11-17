import math
import os

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from env import ASSETS_DIR
from env.sapien_scene import SapienScene
from env.utils.create_actor import create_default_obj


class Tool(SapienScene):
    """
    A cube, an L-shaped hook and a rack. The cube starts on the rack (start 0) or on the table (start 1)
    and has to end on the target (goal 0) or under the rack (goal 1).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_scene()
        self.load_agent()
        self.setup_planner()
        self.add_camera()

        self.load_table()
        self.load_target()
        self.load_rack()
        self.load_hook()
        self.load_cube()

    def add_camera(self):
        self.camera = self.scene.add_camera(
            name="camera",
            width=256,
            height=256,
            fovy=0.6,
            near=0.01,
            far=100,
        )
        self.camera.set_entity_pose(
            sapien.Pose(p=1.6*np.array([1.1053, -0.02, 0.699361]), q=euler2quat(*np.deg2rad([0, 40, 180]))),
        )
        self.scene.update_render()
        self.camera.take_picture()

    def load_cube(self):
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(np.array([12, 42, 255, 255]) / 255)
        visuals = [
            {
                'shape_type': 'box',
                'size': [0.05, 0.05, 0.05],
                'p': [0, 0, 0],
                'material': mat
            },
        ]
        self.state_dict_registry['actors']['cube'] = \
            create_default_obj(scene=self.scene, visuals=visuals, phys_type=None, p=[0.5, -0.35, 0.22], q=[0, 0, 0], name='cube')

    def load_hook(self):
        mat = sapien.render.RenderMaterial()
        mat.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/metal.png')))
        visuals = [
            {
                'shape_type': 'box',
                'size': [0.25, 0.04, 0.04],
                'p': [0, 0, 0],
                'material': mat,
                'density': 100,
            },
            {
                'shape_type': 'box',
                'size': [0.04, 0.2, 0.04],
                'p': [0.125, 0, 0],
                'material': mat,
                'density': 20,
            },
        ]

        p = [0.4, 0.3, 0.05]
        q = [0, 0, 0]
        self.state_dict_registry['actors']['hook'] = \
            create_default_obj(scene=self.scene, visuals=visuals, phys_type=None, p=p, q=q, name="hook")

    def load_rack(self):
        mat = sapien.render.RenderMaterial()
        mat.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/light-wood.png')))
        # board and four legs
        visuals = [{'shape_type': 'box', 'size': [0.2, 0.3, 0.03], 'p': [0, 0, 0], 'material': mat}]
        for x, y in [[0.09, 0.14], [-0.09, 0.14], [0.09, -0.14], [-0.09, -0.14]]:
            visuals.append({'shape_type': 'box', 'size': [0.02, 0.02, 0.15], 'p': [x, y, -0.075], 'material': mat})
        p = [0.5, -0.4, 0.155]
        q = [0, 0, 90]
        self.state_dict_registry['actors']['rack'] = \
            create_default_obj(scene=self.scene, visuals=visuals, phys_type="static", p=p, q=q, name="rack")

    def load_target(self):
        radius = 0.1
        RED_BASE_COLOR = np.array([194, 19, 22, 255]) / 255.0
        # five concentric rings, alternating red and white
        radii = [radius, radius * 4 / 5, radius * 3 / 5, radius * 2 / 5, radius * 1 / 5]
        offsets = [0, 1e-5, 2e-5, 3e-5, 4e-5]
        visuals = [
            {
                "shape_type": "cylinder",
                "radius": radii[k],
                "half_length": 1e-5 / 2 + offsets[k],
                "material": sapien.render.RenderMaterial(base_color=RED_BASE_COLOR if k % 2 == 0 else [1, 1, 1, 1]),
                "ignore_collision": True
            }
            for k in range(5)
        ]

        position = [0.75, 0.0, 1e-3]
        orientation = [0, -90.0, 0]

        self.state_dict_registry['actors']['target'] = \
            create_default_obj(scene=self.scene, visuals=visuals, phys_type="static", p=position, q=orientation, name="target")

    def _initialize_episode(self):
        '''
        Randomize the cube position, and the hook position and orientation
        '''
        self.simulate()
        # Randomize the cube position
        if self.start == 0:  # cube on rack
            cube_xyz = np.array([0.5, -0.4, 0.22])
            cube_xyz[0] = cube_xyz[0] + np.random.uniform(-0.05, 0.05)
            cube_xyz[1] = cube_xyz[1] + np.random.uniform(-0.05, 0.05)
            self.state_dict_registry['actors']['cube'].set_pose(sapien.Pose(p=cube_xyz))
        elif self.start == 1:  # cube on table
            cube_xyz = np.array([0.4, 0.5, 0.08])
            cube_xyz[0] = cube_xyz[0] + np.random.uniform(-0.15, 0)
            cube_xyz[1] = cube_xyz[1] + np.random.uniform(-0.1, 0)
            self.state_dict_registry['actors']['cube'].set_pose(sapien.Pose(p=cube_xyz))

        # Randomize the hook position and orientation
        hook_xyz = np.array([0.4, 0.3, 0.05])
        hook_xyz[0] = hook_xyz[0] + np.random.uniform(-0.05, 0.05)
        yaw = np.random.uniform(-math.pi/6., math.pi/6.)
        self.state_dict_registry['actors']['hook'].set_pose(sapien.Pose(p=hook_xyz, q=euler2quat(0, 0, yaw)))

    def compute_reward(self):
        if self.goal == 0:  # cube on the target
            cube_xy = self.state_dict_registry['actors']['cube'].pose.p[:2]
            target_xy = self.state_dict_registry['actors']['target'].pose.p[:2]
            distance_cube_target = np.linalg.norm(np.array(cube_xy) - np.array(target_xy))
            cube_is_on_target = 1 if distance_cube_target < 0.1 else 0
            reward = cube_is_on_target
            info = {
                "cube_is_on_target": cube_is_on_target,
            }
        elif self.goal == 1:  # cube under the rack
            cube_pose = self.state_dict_registry['actors']['cube'].pose
            rack_pose = self.state_dict_registry['actors']['rack'].pose
            cube_xy = cube_pose.p[:2]
            rack_xy = rack_pose.p[:2]
            cube_z = cube_pose.p[2]
            rack_z = rack_pose.p[2]
            distance_cube_rack = np.linalg.norm(np.array(cube_xy) - np.array(rack_xy))
            cube_is_under_rack = 1 if distance_cube_rack < 0.1 and cube_z < rack_z else 0
            reward = cube_is_under_rack
            info = {
                "cube_is_under_rack": cube_is_under_rack,
            }
        return reward, info

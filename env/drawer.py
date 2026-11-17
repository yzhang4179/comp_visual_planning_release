import math
import os
from typing import List

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from env import ASSETS_DIR
from env.sapien_scene import SapienScene
from env.utils.create_actor import create_default_obj, create_urdf_obj


class Drawer(SapienScene):
    """
    A cabinet with two drawers, a canvas and a brush. The robot draws one of 4 characters (`goal`) on the canvas
    by connecting its 3x3 vertices with the brush, then closes both drawers. Stroke 4-7 is always pre-drawn. In
    start 0 and 1 the brush lies in the cabinet and stroke 1-4 is pre-drawn too; in start 2 and 3 the brush lies
    on the table and the robot also draws stroke 1-4.
    """
    CANVAS_WIDTH = 0.4
    CANVAS_LENGTH = 0.3
    CANVAS_THICKNESS = 0.005

    BRUSH_WIDTH = 0.02  # The width of the brush
    BRUSH_LENGTH = BRUSH_WIDTH * (1 + math.sqrt(2))
    BRUSH_HEIGHT = 0.07
    BRUSH_INITIAL_Z = 0.083

    CABINET_INITIAL_Z = 0.17  # The initial z position of the cabinet

    NUM_DOTS = 300  # The number of dots to be drawn
    VALID_DOTS_Z_OFFSET = 0.0205  # The z position of the dots on the canvas
    VALID_DOTS_TOLERANCE = 0.05  # The tolerance for the dots to be considered valid
    DOT_COLORS = [0.8, 0.2, 0.2, 1]
    VERTICE_COLORS = [1, 1, 1, 0]  # Not Visible

    GOALS = {"0": [[0, 1], [3, 4], [6, 7]],
             "1": [[0, 1], [3, 4], [0, 3]],
             "2": [[1, 2], [4, 5], [7, 8]],
             "3": [[1, 2], [4, 5], [2, 5]]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_scene()
        self.load_agent()
        self.setup_planner()
        self.add_camera()

        self.load_table()
        self.load_cabinet()
        self.load_canvas()
        self.load_vertices()
        self.load_brush()
        self.load_dots()

        self.draw_step = 0
        self.target_xys = None

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

    def load_cabinet(self):
        p = [0.45, -0.4, self.CABINET_INITIAL_Z]
        q = euler2quat(*np.deg2rad([0, 0, -90]))
        pose = sapien.Pose(p, q)

        self.state_dict_registry["articulations"]['cabinet'] = create_urdf_obj(
            scene=self.scene,
            pose=pose,
            modelname="cabinet",
            scale=0.25,
            fix_root_link=True
        )
        self.state_dict_registry['articulations']['cabinet'].set_qpos([0, 0])
        self.state_dict_registry['articulations']['cabinet'].set_properties(
            damping=30,
            stiffness=0.0,
            friction=15.0,
            force_limit=None
        )
        for link in self.state_dict_registry['articulations']['cabinet'].actor.get_links():
            for shape in link.get_collision_shapes():
                shape.set_contact_offset(0.0)
                shape.set_rest_offset(0.0)

    def load_canvas(self):
        mat0 = sapien.render.RenderMaterial()
        mat0.set_base_color_texture(sapien.render.RenderTexture2D(os.path.join(ASSETS_DIR, 'textures/light-wood-v2.png')))
        mat1 = sapien.render.RenderMaterial()
        mat1.set_base_color(np.array([1, 1, 1, 1]))
        visuals = [
            {
                "shape_type": "box",
                "size": [0.025, 0.15, 0.05],
                'p': [0, 0, 0.04/2.],
                "material": mat0,
                "density": 1500,
                "ignore_collision": False
            },
            {
                "shape_type": "box",
                "size": [self.CANVAS_WIDTH, self.CANVAS_LENGTH, self.CANVAS_THICKNESS * 2],
                'p': [(self.CANVAS_WIDTH/2. - 0.03), 0, 0.],
                "material": mat1,
                "ignore_collision": True
            },
        ]
        self.state_dict_registry['actors']['canvas'] = \
            create_default_obj(scene=self.scene,
                               visuals=visuals, phys_type=None,
                               p=[0.255, 0.15, self.CANVAS_THICKNESS], q=[0, 0, 0], name='canvas')

    def load_brush(self):
        RED_BASE_COLOR = np.array([194, 19, 22, 255]) / 255.0
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(RED_BASE_COLOR)
        mat.set_roughness(0.05)
        mat.set_metallic(0.0)
        visuals = []
        for i in range(4):
            visuals.append(
                {
                    'shape_type': 'box',
                    'size': [self.BRUSH_WIDTH, self.BRUSH_LENGTH, self.BRUSH_HEIGHT],
                    'p': [0, 0, 0],
                    'q': [0, 0, i * 45],
                    'material': mat,
                    'density': 1000,
                }
            )

        self.state_dict_registry['actors']['brush'] = \
            create_default_obj(scene=self.scene,
                               visuals=visuals, phys_type=None,
                               p=[-3, -3, self.BRUSH_HEIGHT/2.], q=[0, 0, 0], name='brush')

    def load_vertices(self):
        """
        The 3x3 vertices of the characters, fixed to the canvas:
            0  1  2
            3  4  5
            6  7  8
        4 is in the center of the canvas. The pre-drawn segments 1-4 and 4-7 are loaded here too;
        `set_start_goal` hides segment 1-4 and vertex 1 for start 2 and 3.
        """
        EDGE_LENGTH = 0.1

        relative_xy_0 = np.array([self.CANVAS_WIDTH/2.-0.03-EDGE_LENGTH+0.025,
                                  -EDGE_LENGTH])

        for i in range(9):
            mat = sapien.render.RenderMaterial()
            mat.set_base_color(self.VERTICE_COLORS if i in (1, 4, 7) else self.DOT_COLORS)
            visuals = [
                {
                    "shape_type": "cylinder",
                    "radius": self.BRUSH_LENGTH / 5.,
                    "half_length": 0.0015,
                    "material": mat,
                    "ignore_collision": True,
                }
            ]

            self.state_dict_registry['actors'][f'vertice_{i}'] = \
                create_default_obj(scene=self.scene,
                                   visuals=visuals, phys_type=None,
                                   p=[-1, -1, 0.2], q=[0, -90, 0], name=f'vertice_{i}')
            x = relative_xy_0[0] + (i // 3) * EDGE_LENGTH
            y = relative_xy_0[1] + (i % 3) * EDGE_LENGTH
            p = [x, y, -0.05/2.+self.VALID_DOTS_Z_OFFSET+0.009] if i in (1, 4, 7) else [x, y, -0.05/2.]
            self.create_fixed_connection(
                body0=self.state_dict_registry['actors']['canvas'],
                pose0=sapien.Pose(p=p),
                body1=self.state_dict_registry['actors'][f'vertice_{i}'],
                pose1=sapien.Pose(q=euler2quat(*np.deg2rad([0, 90, 0]))),
            )

        # pre-drawn segments 1-4 and 4-7
        for name, x_offset in [('pre_segment_14', EDGE_LENGTH/2.), ('pre_segment_47', EDGE_LENGTH/2.*3.)]:
            mat = sapien.render.RenderMaterial()
            mat.set_base_color(self.DOT_COLORS)
            visuals = [
                {
                    "shape_type": "box",
                    "size": [EDGE_LENGTH-self.BRUSH_LENGTH / 4.8, self.BRUSH_LENGTH/5.*1.9, 0.008],
                    "material": mat,
                    "ignore_collision": True,
                }
            ]
            self.state_dict_registry['actors'][name] = \
                create_default_obj(scene=self.scene,
                                   visuals=visuals, phys_type=None,
                                   p=[0, 0, 0], q=[0, 0, 0], name=name)

            self.create_fixed_connection(
                body0=self.state_dict_registry['actors']['canvas'],
                pose0=sapien.Pose(p=[relative_xy_0[0] + x_offset, relative_xy_0[1] + EDGE_LENGTH, -0.05/2.+self.VALID_DOTS_Z_OFFSET+0.0068]),
                body1=self.state_dict_registry['actors'][name],
                pose1=sapien.Pose(p=[0, 0, 0]),
            )

    def load_dots(self):
        """The dots left by the brush on the canvas, hidden under the table until drawn."""
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(self.DOT_COLORS)
        for i in range(self.NUM_DOTS):
            visuals = [
                {
                    "shape_type": "cylinder",
                    "radius": self.BRUSH_LENGTH / 5.,
                    "half_length": 0.0015,
                    "material": mat,
                    "ignore_collision": True,
                }
            ]
            p = [0.4, 0.1, -0.1]
            self.state_dict_registry['actors'][f'dot_{i}'] = \
                create_default_obj(scene=self.scene,
                                   visuals=visuals, phys_type="static",
                                   p=p, q=[0, -90, 0], name=f'dot_{i}')

    def set_start_goal(self, start: int, goal: int):
        def change_color(rgba, actor):
            render_body_component: sapien.render.RenderBodyComponent = \
                actor.find_component_by_type(sapien.render.RenderBodyComponent)
            for render_shape in render_body_component.render_shapes:
                for part in render_shape.parts:
                    part.material.set_base_color(rgba)

        # `SapienScene.__init__` calls this before the vertices are loaded
        if self.state_dict_registry['actors'].get('pre_segment_14') is None \
                or self.state_dict_registry['actors'].get('vertice_1') is None:
            pass
        else:
            if start == 0 or start == 1:
                change_color(self.DOT_COLORS, self.state_dict_registry['actors']['pre_segment_14'])
                change_color(self.DOT_COLORS, self.state_dict_registry['actors']['pre_segment_47'])
                change_color(self.DOT_COLORS, self.state_dict_registry['actors']['vertice_1'])
                change_color(self.DOT_COLORS, self.state_dict_registry['actors']['vertice_4'])
                change_color(self.DOT_COLORS, self.state_dict_registry['actors']['vertice_7'])

            elif start == 2 or start == 3:
                change_color(self.VERTICE_COLORS, self.state_dict_registry['actors']['pre_segment_14'])
                change_color(self.DOT_COLORS, self.state_dict_registry['actors']['pre_segment_47'])
                change_color(self.VERTICE_COLORS, self.state_dict_registry['actors']['vertice_1'])
                change_color(self.DOT_COLORS, self.state_dict_registry['actors']['vertice_4'])
                change_color(self.DOT_COLORS, self.state_dict_registry['actors']['vertice_7'])
            else:
                raise ValueError(f"Invalid start value: {start}. Must be 0, 1, 2, or 3.")
        return super().set_start_goal(start, goal)

    def reset(self, *args, **kwargs):
        self.draw_step = 0
        self.target_xys = None
        return super().reset(*args, **kwargs)

    def _initialize_episode(self):
        # cabinet
        angle = np.random.uniform(-105, -55)
        distance = np.interp(angle, [-105, -45], [0.6, 0.4])

        x = 0.4 + distance * np.cos(np.deg2rad(angle))
        y = 0.1 + distance * np.sin(np.deg2rad(angle))
        self.state_dict_registry['articulations']['cabinet'].actor.set_pose(
            sapien.Pose(p=[x, y, self.CABINET_INITIAL_Z], q=euler2quat(*np.deg2rad([0, 0, angle])))
        )

        # canvas
        canvas_x = 0.28 + np.random.uniform(0.05, 0.15)
        canvas_y = 0.25 + np.random.uniform(-0.1, -0.05)
        canvas_yaw = np.random.uniform(-1, 8)
        self.state_dict_registry['actors']['canvas'].set_pose(
            sapien.Pose(p=[canvas_x, canvas_y, 0.05/2.+0.01], q=euler2quat(*np.deg2rad([0, 0, canvas_yaw])))
        )

        self.draw_step = 0
        self.target_xys = None
        for i in range(self.NUM_DOTS):
            p = [0.4, 0.1, -0.1]
            self.state_dict_registry['actors'][f'dot_{i}'].set_pose(
                sapien.Pose(p=p, q=euler2quat(*np.deg2rad([0, -90, 0])))
            )

        # cabinet joint positions of each start
        cabinet_qpos = {0: [1, 1], 1: [0, 0], 2: [1.0, 0], 3: [0, 1]}[self.start]
        self.simulate()
        self.state_dict_registry['articulations']['cabinet'].set_qpos(cabinet_qpos)
        self.state_dict_registry['articulations']['cabinet'].set_qvel([0, 0])
        self.simulate()

        # brush: in the cabinet for start 0 and 1, on the table for start 2 and 3
        if self.start in (0, 1):
            cabinet_target_point = self.state_dict_registry['articulations']['cabinet'].get_target_point(0, "pose")
            cabinet_pose = self.state_dict_registry['articulations']['cabinet'].get_pose()
            cabinet_p = [cabinet_target_point.p[0], cabinet_target_point.p[1], self.BRUSH_INITIAL_Z]
            brush_pose = sapien.Pose(p=cabinet_p, q=cabinet_pose.q)
        else:
            brush_x = np.random.uniform(0.20, 0.40)
            brush_y = np.random.uniform(0.43, 0.45)
            brush_z = self.BRUSH_HEIGHT / 2.
            brush_pose = sapien.Pose(p=[brush_x, brush_y, brush_z])
        self.state_dict_registry['actors']['brush'].set_pose(brush_pose)
        self.set_vel_zero('brush')

    def _after_simulation_step(self):
        """For start 0 and 1, keep the brush at the cabinet's target point while neither finger touches it."""
        panda_leftfinger_contact = self.check_actors_contact('panda_leftfinger', 'brush')
        panda_rightfinger_contact = self.check_actors_contact('panda_rightfinger', 'brush')
        if self.start == 2 or self.start == 3:
            pass
        elif self.start == 0 or self.start == 1:
            if (not panda_leftfinger_contact) and (not panda_rightfinger_contact):
                cabinet_target_point = self.state_dict_registry['articulations']['cabinet'].get_target_point(0, "pose")  # {world}_T_{target}
                cabinet_pose = self.state_dict_registry['articulations']['cabinet'].get_pose()
                cabinet_p = [cabinet_target_point.p[0], cabinet_target_point.p[1], self.BRUSH_INITIAL_Z]
                self.state_dict_registry['actors']['brush'].set_pose(
                    sapien.Pose(p=cabinet_p, q=cabinet_pose.q)
                )

    def _after_control_step(self):
        """Draw a dot if the brush is touching the canvas."""
        brush_xyz = self.state_dict_registry['actors']['brush'].pose.p
        canvas_xyz = self.state_dict_registry['actors']['canvas'].pose.p
        canvas_xyz[0] = canvas_xyz[0] + self.CANVAS_WIDTH/2. - 0.03
        canvas_xyz[2] = 0.00
        delta_xyz = np.abs(brush_xyz - canvas_xyz)
        dx, dy, dz = delta_xyz
        is_brush_touching_canvas = (dx < self.CANVAS_WIDTH/2.) and (dy < self.CANVAS_LENGTH/2.) and (dz < ((self.CANVAS_THICKNESS + self.BRUSH_HEIGHT) / 2. + self.VALID_DOTS_TOLERANCE))

        if is_brush_touching_canvas:
            p = self.state_dict_registry['actors']['brush'].pose.p[:2].tolist() + [0.009]
            q = euler2quat(*np.deg2rad([0, -90, 0]))
            self.state_dict_registry['actors'][f'dot_{self.draw_step}'].set_pose(sapien.Pose(p=p, q=q))
            self.draw_step += 1

    def get_target_dots(self, dot_0_id: int, dot_1_id: int, n: int = 5) -> np.ndarray:
        t = np.linspace(0, 1, n+2)
        start_vertice = self.state_dict_registry['actors'][f'vertice_{dot_0_id}'].pose.p
        end_vertice = self.state_dict_registry['actors'][f'vertice_{dot_1_id}'].pose.p
        start = np.array(start_vertice[:2])
        end = np.array(end_vertice[:2])
        points = np.array([start + (end - start) * t[i] for i in range(n+2)])
        return points

    def get_current_dots(self) -> np.ndarray:
        dots = []
        for i in range(self.draw_step):
            dot = self.state_dict_registry['actors'][f'dot_{i}'].pose.p[:2]
            dots.append(dot)
        return np.array(dots)

    def compute_reward(self):
        """
        Goal 0: character 3 [[0, 1], [3, 4], [6, 7]]
        Goal 1: character 9 [[0, 1], [3, 4], [0, 3]]
        Goal 2: character E [[1, 2], [4, 5], [7, 8]]
        Goal 3: character P [[1, 2], [4, 5], [2, 5]]
        """

        def get_all_target_dots(vertices_list: List[List[int]]) -> List[np.ndarray]:
            target_xys = []
            for vertices in vertices_list:
                target_xy = self.get_target_dots(vertices[0], vertices[1], n=7)
                target_xys.append(target_xy)
            if self.start == 2 or self.start == 3:
                target_xys.append(self.get_target_dots(1, 4, n=7))  # add additional target dots for start 2 and 3
            target_xys = np.vstack(target_xys)
            target_xys_unique = np.unique(target_xys, axis=0)
            return target_xys_unique

        if self.draw_step >= 1 and self.target_xys is None:
            self.target_xys = get_all_target_dots(self.GOALS[str(self.goal)])
        if self.draw_step >= 1 and self.target_xys is not None:
            current_dots = self.get_current_dots()  # shape (n, 2)
            target_dots = self.target_xys  # shape (m, 2)

            diff = np.linalg.norm(current_dots[:, None, :] - target_dots[None, :, :], axis=-1)  # shape (n, m)
            hitmat = diff < (0.1/4)  # threshold for hit

            # 1. check that every current dot hits at least one target dot
            dot_hit = hitmat.any(axis=1)  # shape (n,)
            cond1 = dot_hit.all()  # all current dots hit at least one target dot
            # 2. check if every target dot is hit by at least one current dot
            ref_hit = hitmat.any(axis=0)  # shape (m,)
            cond2 = ref_hit.all()  # all target dots are hit by at least one
            # 3. check if all cabinet links are closed
            cabinet = self.state_dict_registry['articulations']['cabinet'].get_qpos()
            cond3 = np.all(np.isclose(cabinet, [0, 0], atol=0.03))  # cabinet is closed

            reward = float(cond1 & cond2 & cond3)
            info = {'successfully_drawn': cond1 & cond2,
                    'cabinet_closed': cond3}
            return reward, info
        else:
            return 0.0, {'successfully_drawn': False, 'cabinet_closed': False}

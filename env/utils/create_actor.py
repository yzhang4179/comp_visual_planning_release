# Adapted from RoboTwin (https://github.com/RoboTwin-Platform/RoboTwin), MIT; see THIRD_PARTY_LICENSES.md.
import json
from typing import List, Optional

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from env import ASSETS_DIR

from .actor_utils import Actor_w_goal, ArticulationActor


def create_default_obj(
        scene: sapien.Scene,
        visuals: List[dict],
        phys_type: str,
        p: List[float],
        q: Optional[List[float]] = None,
        name: str = "multi_obj"
    ) -> sapien.Entity:
    """
    Create an object that contains multiple visual components with optional physical materials.

    Args:
        scene: the SAPIEN scene
        visuals: A list of dictionaries, each describing a component with keys:
            - 'shape_type': "box", "sphere", "cylinder", or "capsule"
            - 'material': a RenderMaterial for visualization
            - 'collision_material': (optional) a PhysicalMaterial for physics
            - 'size'/'radius'/'length' etc.: shape-specific parameters
            - 'density': (optional) mass density for collision
            - 'ignore_collision': (optional) skip adding collision
            - 'p', 'q': local pose (position, and Euler angles in degrees)
        phys_type: "static", "kinematic", or None for a dynamic actor
        p: [x, y, z] world position
        q: Euler angles in degrees
        name: actor name
    Returns:
        A sapien.Entity with configured visuals and collision.
    """
    builder = scene.create_actor_builder()
    for visual in visuals:
        shape_type = visual.get("shape_type")
        vis_mat = visual.get("material")
        phys_mat = visual.get("collision_material")
        ignore_collision = visual.get("ignore_collision", False)
        if vis_mat is None:
            raise ValueError("Material parameter must be provided for visual")

        # Determine local pose
        p_, q_ = visual.get("p"), visual.get("q")
        if p_ is not None and q_ is not None:
            pose_ = sapien.Pose(p_, euler2quat(*np.deg2rad(q_)))
        elif p_ is not None:
            pose_ = sapien.Pose(p_)
        else:
            pose_ = sapien.Pose()

        # Add visual and collision based on shape
        if shape_type == "box":
            size = visual.get("size")
            if size is None:
                raise ValueError("'size' must be provided for box")
            half = np.array(size) / 2.0
            builder.add_box_visual(pose=pose_, half_size=half, material=vis_mat, name=name)
            if not ignore_collision:
                density = visual.get("density", 1000)
                if phys_mat:
                    builder.add_box_collision(pose=pose_, half_size=half, material=phys_mat, density=density)
                else:
                    builder.add_box_collision(pose=pose_, half_size=half, density=density)
        elif shape_type == "sphere":
            radius = visual.get("radius")
            if radius is None:
                raise ValueError("'radius' must be provided for sphere")
            builder.add_sphere_visual(pose=pose_, radius=radius, material=vis_mat, name=name)
            if not ignore_collision:
                if phys_mat:
                    builder.add_sphere_collision(pose=pose_, radius=radius, material=phys_mat)
                else:
                    builder.add_sphere_collision(pose=pose_, radius=radius)
        elif shape_type == "cylinder":
            radius = visual.get("radius")
            half_length = visual.get("half_length")
            if half_length is None:
                length = visual.get("length")
                if length is None:
                    raise ValueError("'half_length' or 'length' required for cylinder")
                half_length = length / 2.0
            builder.add_cylinder_visual(pose=pose_, radius=radius, half_length=half_length, material=vis_mat, name=name)
            if not ignore_collision:
                if phys_mat:
                    builder.add_cylinder_collision(pose=pose_, radius=radius, half_length=half_length, material=phys_mat)
                else:
                    builder.add_cylinder_collision(pose=pose_, radius=radius, half_length=half_length)
        elif shape_type == "capsule":
            radius = visual.get("radius")
            length = visual.get("length")
            if radius is None or length is None:
                raise ValueError("'radius' and 'length' required for capsule")
            half_length = length / 2.0
            builder.add_capsule_visual(pose=pose_, radius=radius, half_length=half_length, material=vis_mat, name=name)
            if not ignore_collision:
                if phys_mat:
                    builder.add_capsule_collision(pose=pose_, radius=radius, half_height=half_length, material=phys_mat)
                else:
                    builder.add_capsule_collision(pose=pose_, radius=radius, half_height=half_length)
        else:
            raise ValueError(f"Unknown shape_type: {shape_type}")

    # Set the root pose
    root_pose = sapien.Pose(p, euler2quat(*np.deg2rad(q))) if q is not None else sapien.Pose(p)
    builder.set_initial_pose(root_pose)

    # Build actor
    if phys_type == "static":
        return builder.build_static(name=name)
    elif phys_type == "kinematic":
        return builder.build_kinematic(name=name)
    else:
        return builder.build(name=name)

def create_actor(
        scene,
        pose: sapien.Pose,
        modelname: str,
        convex=False,
        is_static=False,
        model_id=0,
        scale: Optional[tuple] = None,
        visual_scale: Optional[tuple] = None,
):
    """
    Load `assets/objects/<modelname>/base<model_id>.glb` (for both collision and visual) as an `Actor_w_goal`,
    which holds the goal layouts of the puzzle pieces from `assets/objects/goal/model_data.json`.
    """
    model_file = ASSETS_DIR / 'objects' / modelname / f"base{model_id}.glb"
    with open(ASSETS_DIR / "objects" / "goal" / "model_data.json", "r") as file:
        model_data = json.load(file)
    if scale is not None:
        model_data["scale"] = scale
    else:
        scale = model_data["scale"]

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex:
        builder.add_multiple_convex_collisions_from_file(filename=str(model_file), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(filename=str(model_file), scale=scale)

    builder.add_visual_from_file(filename=str(model_file), scale=visual_scale if visual_scale is not None else scale)
    mesh = builder.build(name=modelname)
    mesh.set_name(modelname)
    mesh.set_pose(pose)
    return Actor_w_goal(mesh, model_data)


def create_urdf_obj(scene, pose: sapien.Pose, modelname: str, scale=None, fix_root_link=True, load_multiple_collisions_from_file=True) -> ArticulationActor:
    """Load the articulated object `assets/objects/<modelname>/0/mobility.urdf` with its annotated points."""
    modeldir = ASSETS_DIR / 'objects' / modelname / '0'
    json_file_path = modeldir / "model_data.json"
    loader: sapien.URDFLoader = scene.create_urdf_loader()

    with open(json_file_path, "r") as file:
        model_data = json.load(file)
    if scale is not None:
        loader.scale = scale
        model_data["scale"] = scale
    else:
        scale = model_data["scale"]
        loader.scale = model_data["scale"]

    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = load_multiple_collisions_from_file
    object: sapien.Articulation = loader.load(str(modeldir / "mobility.urdf"))

    object.set_root_pose(pose)
    object.set_name(modelname)

    return ArticulationActor(object, model_data)

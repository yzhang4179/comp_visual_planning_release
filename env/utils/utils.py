# Adapted from ManiSkill (https://github.com/haosulab/ManiSkill), Apache-2.0; see THIRD_PARTY_LICENSES.md.
from typing import Dict, List, TypeVar

import numpy as np
import sapien
import sapien.physx as physx
import sapien.wrapper.urdf_loader
from transforms3d.quaternions import mat2quat

T = TypeVar("T")


def get_obj_by_name(objs: List[T], name: str, is_unique=True):
    """Get an object given the name.

    Args:
        objs (List[T]): objs to query. Expect these objects to have a get_name function. These may be sapien.Entity, physx.PhysxArticulationLink etc.
        name (str): name for query.
        is_unique (bool, optional):
            whether the name should be unique. Defaults to True.

    Raises:
        RuntimeError: The name is not unique when @is_unique is True.

    Returns:
        T or List[T]:
            matched T or Ts. None if no matches.
    """
    matched_objects = [x for x in objs if x.get_name() == name]
    if len(matched_objects) > 1:
        if not is_unique:
            return matched_objects
        else:
            raise RuntimeError(f"Multiple objects with the same name {name}.")
    elif len(matched_objects) == 1:
        return matched_objects[0]
    else:
        return None


def check_urdf_config(urdf_config: dict):
    """Check whether the urdf config is valid for SAPIEN.

    Args:
        urdf_config (dict): dict passed to `sapien.URDFLoader.load`.
    """
    allowed_keys = ["material", "density", "link"]
    for k in urdf_config.keys():
        if k not in allowed_keys:
            raise KeyError(
                f"Not allowed key ({k}) for `sapien.URDFLoader.load`. Allowed keys are {allowed_keys}"
            )

    allowed_keys = ["material", "density", "patch_radius", "min_patch_radius"]
    for k, v in urdf_config.get("link", {}).items():
        for kk in v.keys():
            # In fact, it should support specifying collision-shape-level materials.
            if kk not in allowed_keys:
                raise KeyError(
                    f"Not allowed key ({kk}) for `sapien.URDFLoader.load`. Allowed keys are {allowed_keys}"
                )


def parse_urdf_config(config_dict: dict) -> Dict:
    """Parse config from dict for SAPIEN URDF loader.

    Args:
        config_dict (dict): a dict containing link physical properties.

    Returns:
        Dict: urdf config passed to `sapien.URDFLoader.load`.
    """
    urdf_config = dict()

    # Create the global physical material for all links
    if "material" in config_dict:
        urdf_config["material"] = physx.PhysxMaterial(**config_dict["material"])

    # Create link-specific physical materials
    materials = {}
    if "_materials" in config_dict:
        for k, v in config_dict["_materials"].items():
            materials[k] = physx.PhysxMaterial(**v)

    # Specify properties for links
    if "link" in config_dict:
        urdf_config["link"] = dict()
        for k, link_config in config_dict["link"].items():
            urdf_config["link"][k] = link_config.copy()
            # substitute with actual material
            urdf_config["link"][k]["material"] = materials[link_config["material"]]
    return urdf_config


def apply_urdf_config(loader: sapien.wrapper.urdf_loader.URDFLoader, urdf_config: dict):
    if "link" in urdf_config:
        for name, link_config in urdf_config["link"].items():
            if "material" in link_config:
                mat: physx.PhysxMaterial = link_config["material"]
                loader.set_link_material(
                    name, mat.static_friction, mat.dynamic_friction, mat.restitution
                )
            if "patch_radius" in link_config:
                loader.set_link_patch_radius(name, link_config["patch_radius"])
            if "min_patch_radius" in link_config:
                loader.set_link_min_patch_radius(name, link_config["min_patch_radius"])
            if "density" in link_config:
                loader.set_link_density(name, link_config["density"])
    if "material" in urdf_config:
        mat: physx.PhysxMaterial = urdf_config["material"]
        loader.set_material(mat.static_friction, mat.dynamic_friction, mat.restitution)
    if "patch_radius" in urdf_config:
        loader.set_patch_radius(urdf_config["patch_radius"])
    if "min_patch_radius" in urdf_config:
        loader.set_min_patch_radius(urdf_config["min_patch_radius"])
    if "density" in urdf_config:
        loader.set_density(urdf_config["density"])


def look_at(eye: np.ndarray, target: np.ndarray, up=(0, 0, 1)) -> sapien.Pose:
    """Camera pose at `eye` looking at `target` (SAPIEN cameras look along +x with +z up)."""

    def normalize_tensor(x, eps=1e-6):
        x = x.reshape(-1, 3)
        norm = np.linalg.norm(x, axis=-1)
        zero_vectors = norm < eps
        x[zero_vectors] = np.zeros(3, dtype=x.dtype)
        x[~zero_vectors] /= norm[~zero_vectors].reshape(-1, 1)
        return x

    forward = normalize_tensor(target - eye)

    up = normalize_tensor(np.array(up, dtype=np.float32))
    left = np.cross(up, forward)
    left = normalize_tensor(left)
    up = np.cross(forward, left)
    rotation = np.stack([forward, left, up], axis=-1)
    return sapien.Pose(p=eye, q=mat2quat(rotation))

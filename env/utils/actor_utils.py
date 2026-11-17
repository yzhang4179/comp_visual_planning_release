# Adapted from RoboTwin (https://github.com/RoboTwin-Platform/RoboTwin), MIT; see THIRD_PARTY_LICENSES.md.
from typing import Literal

import numpy as np
import sapien
import transforms3d as t3d
from sapien import Entity
from sapien.physx import PhysxArticulation, PhysxArticulationLinkComponent


class Actor_w_goal:
    """Actor that also stores the goal layouts of the puzzle pieces, see `get_goal_pose`."""

    def __init__(self, actor: Entity, actor_data: dict, mass=0.01):
        """
        `actor_data[str(goal)][str(piece)]` is the 4x4 pose of piece 1-4 in goal shape `goal`
        (piece 5 is the goal shape itself), plus `actor_data["scale"]`.
        """
        self.actor = actor
        self.config = actor_data
        self.set_mass(mass)

    @property
    def name(self):
        return self.actor.get_name()

    def get_goal_pose(
        self,
        goal_idx: int,
        model_idx: int,
        ret: Literal["matrix", "list", "pose"],
    ) -> np.ndarray | list | sapien.Pose:
        matrix = np.array(self.config[str(goal_idx)][str(model_idx)])
        matrix[:3, 3] *= np.array(self.config["scale"])

        if ret == "matrix":
            return matrix
        elif ret == "list":
            return (matrix[:3, 3].tolist() + t3d.quaternions.mat2quat(matrix[:3, :3]).tolist())
        else:
            return sapien.Pose(matrix[:3, 3], t3d.quaternions.mat2quat(matrix[:3, :3]))

    def get_pose(self) -> sapien.Pose:
        """Get the sapien.Pose of the actor."""
        return self.actor.get_pose()

    def set_pose(self, pose: sapien.Pose):
        """Set the sapien.Pose of the actor."""
        self.actor.set_pose(pose)

    def set_mass(self, mass):
        for component in self.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                component.mass = mass

    def set_physics_properties(self, dynamic_friction: float = 0.3, static_friction: float = 0.3, restitution: float = 0.3):
        for component in self.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                for shape in component.collision_shapes:
                    shape.physical_material.set_static_friction(static_friction)
                    shape.physical_material.set_dynamic_friction(dynamic_friction)
                    shape.physical_material.set_restitution(restitution)

    def find_component_by_type(self, cls):
        """Find a component by its type."""
        return self.actor.find_component_by_type(cls)


class Actor_w_function:
    """Actor with annotated points (e.g. target points) stored in its `model_data.json`."""

    POINTS = {
        "contact": "contact_points_pose",
        "target": "target_pose",
        "functional": "functional_matrix",
        "orientation": "orientation_point",
    }

    def __init__(self, actor: Entity, actor_data: dict, mass=0.01):
        self.actor = actor
        self.config = actor_data
        self.set_mass(mass)

    @property
    def name(self):
        return self.actor.get_name()

    def set_pose(self, pose: sapien.Pose):
        """Set the sapien.Pose of the actor."""
        self.actor.set_pose(pose)

    def get_point(
        self,
        type: Literal["contact", "target", "functional", "orientation"],
        idx: int,
        ret: Literal["matrix", "list", "pose"],
    ) -> np.ndarray | list | sapien.Pose:
        """Get the point of the entity actor."""
        type = self.POINTS[type]

        actor_matrix = self.actor.get_pose().to_transformation_matrix()
        local_matrix = np.array(self.config[type][idx])
        local_matrix[:3, 3] *= np.array(self.config["scale"])

        world_matrix = actor_matrix @ local_matrix

        if ret == "matrix":
            return world_matrix
        elif ret == "list":
            return (world_matrix[:3, 3].tolist() + t3d.quaternions.mat2quat(world_matrix[:3, :3]).tolist())
        else:
            return sapien.Pose(world_matrix[:3, 3], t3d.quaternions.mat2quat(world_matrix[:3, :3]))

    def get_pose(self) -> sapien.Pose:
        """Get the sapien.Pose of the actor."""
        return self.actor.get_pose()

    def get_target_point(self,
                         idx: int,
                         ret: Literal["matrix", "list", "pose"] = "list") -> np.ndarray | list | sapien.Pose:
        """Get the transformation matrix of given target point of the actor."""
        return self.get_point("target", idx, ret)

    def set_mass(self, mass):
        for component in self.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                component.mass = mass

    def find_component_by_type(self, cls):
        """Find a component by its type."""
        return self.actor.find_component_by_type(cls)


class ArticulationActor(Actor_w_function):
    """Articulated object (the drawer's cabinet) with annotated points defined on its links."""

    POINTS = {
        "contact": "contact_points",
        "target": "target_points",
        "functional": "functional_points",
        "orientation": "orientation_point",
    }

    def __init__(self, actor: PhysxArticulation, actor_data: dict, mass=0.01):
        assert isinstance(actor, PhysxArticulation), "ArticulationActor must be an Articulation"

        self.actor = actor
        self.config = actor_data

        self.link_dict = self.get_link_dict()
        self.set_mass(mass)

    @property
    def name(self):
        return self.actor.get_name()

    def get_link_dict(self) -> dict[str, PhysxArticulationLinkComponent]:
        link_dict = {}
        for link in self.actor.get_links():
            link_dict[link.get_name()] = link
        return link_dict

    def get_point(
        self,
        type: Literal["contact", "target", "functional", "orientation"],
        idx: int,
        ret: Literal["matrix", "list", "pose"],
    ) -> np.ndarray | list | sapien.Pose:
        """Get the point of the articulation actor."""
        type = self.POINTS[type]
        local_matrix = np.array(self.config[type][idx]["matrix"])
        local_matrix[:3, 3] *= self.config["scale"]
        link = self.link_dict[self.config[type][idx]["base"]]
        link_matrix = link.get_pose().to_transformation_matrix()
        # {world}_T_{point} = {world}_T_{link} @ {link}_T_{point}
        world_matrix = link_matrix @ local_matrix
        if ret == "matrix":
            return world_matrix
        elif ret == "list":
            return (world_matrix[:3, 3].tolist() + t3d.quaternions.mat2quat(world_matrix[:3, :3]).tolist())
        else:
            return sapien.Pose(world_matrix[:3, 3], t3d.quaternions.mat2quat(world_matrix[:3, :3]))

    def set_mass(self, mass, links_name: list[str] = None):
        for link in self.actor.get_links():
            if links_name is None or link.get_name() in links_name:
                link.set_mass(mass)

    def set_properties(self, damping, stiffness, friction=None, force_limit=None):
        for joint in self.actor.get_joints():
            if force_limit is not None:
                joint.set_drive_properties(damping=damping, stiffness=stiffness, force_limit=force_limit)
            else:
                joint.set_drive_properties(
                    damping=damping,
                    stiffness=stiffness,
                )
            if friction is not None:
                joint.set_friction(friction)

    def set_root_pose(self, pose):
        self.actor.set_root_pose(pose)

    def set_qpos(self, qpos):
        self.actor.set_qpos(qpos)

    def set_qvel(self, qvel):
        self.actor.set_qvel(qvel)

    def get_root_pose(self):
        return self.actor.get_root_pose()

    def get_qpos(self):
        return self.actor.get_qpos()

    def get_qvel(self):
        return self.actor.get_qvel()

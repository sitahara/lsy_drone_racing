"""The Vicon module provides an interface to the Vicon motion capture system for position tracking.

It defines the Vicon class, which handles communication with the Vicon system through ROS messages.
The Vicon class is responsible for:

* Tracking the drone and other objects (gates, obstacles) in the racing environment.
* Providing real-time pose (position and orientation) data for tracked objects.
* Calculating velocities and angular velocities based on pose changes.

This module is necessary to provide the real-world positioning data for the drone and race track
elements.
"""

from __future__ import annotations

import time

import numpy as np
import rospy
import yaml
from rosgraph import Master
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage

from lsy_drone_racing.utils import map2pi
from lsy_drone_racing.utils.import_utils import get_ros_package_path


class Vicon:
    """Vicon interface for the pose estimation data for the drone and any other tracked objects.

    Vicon sends a stream of ROS messages containing the current pose data. We subscribe to these
    messages and save the pose data for each object in dictionaries. Users can then retrieve the
    latest pose data directly from these dictionaries.
    """

    def __init__(
        self, track_names: list[str] = [], auto_track_drone: bool = True, timeout: float = 0.0
    ):
        """Load the crazyflies.yaml file and register the subscribers for the Vicon pose data.

        Args:
            track_names: The names of any additional objects besides the drone to track.
            auto_track_drone: Infer the drone name and add it to the positions if True.
            timeout: If greater than 0, Vicon waits for position updates of all tracked objects
                before returning.
        """
        assert Master("/rosnode").is_online(), "ROS is not running. Please run hover.launch first!"
        try:
            rospy.init_node("playback_node")
        except rospy.exceptions.ROSException:
            ...  # ROS node is already running which is fine for us
        self.drone_name = None
        if auto_track_drone:
            with open(get_ros_package_path("crazyswarm") / "launch/crazyflies.yaml", "r") as f:
                config = yaml.load(f, yaml.SafeLoader)
            assert len(config["crazyflies"]) == 1, "Only one crazyfly allowed at a time!"
            self.drone_name = f"cf{config['crazyflies'][0]['id']}"
            track_names.insert(0, self.drone_name)
        self.track_names = track_names
        # Register the Vicon subscribers for the drone and any other tracked object
        self.pos: dict[str, np.ndarray] = {}
        self.rpy: dict[str, np.ndarray] = {}
        self.vel: dict[str, np.ndarray] = {}
        self.ang_vel: dict[str, np.ndarray] = {}
        self.time: dict[str, float] = {}

        self.sub = rospy.Subscriber("/tf", TFMessage, self.save_pose)
        if timeout:
            tstart = time.time()
            while not self.active and time.time() - tstart < timeout:
                time.sleep(0.01)
            if not self.active:
                raise TimeoutError(
                    "Timeout while fetching initial position updates for all tracked objects. "
                    f"Missing objects: {[k for k in self.track_names if k not in self.ang_vel]}"
                )

    def save_pose(self, data: TFMessage):
        """Save the position and orientation of all transforms.

        Args:
            data: The TF message containing the objects' pose.
        """
        for tf in data.transforms:
            name = tf.child_frame_id.split("/")[-1]
            if name not in self.track_names:
                continue
            T, R = tf.transform.translation, tf.transform.rotation
            pos = np.array([T.x, T.y, T.z])
            rpy = Rotation.from_quat([R.x, R.y, R.z, R.w]).as_euler("xyz")
            if self.pos.get(name) is not None:
                self.vel[name] = (pos - self.pos[name]) / (time.time() - self.time[name])
                self.ang_vel[name] = map2pi(rpy - self.rpy[name]) / (time.time() - self.time[name])
            self.time[name] = time.time()
            self.pos[name] = pos
            self.rpy[name] = rpy

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the latest pose of a tracked object.

        Args:
            name: The name of the object.

        Returns:
            The position and rotation of the object. The rotation is in roll-pitch-yaw format.
        """
        return self.pos[name], self.rpy[name]

    @property
    def poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the latest poses of all objects."""
        return np.stack(self.pos.values()), np.stack(self.rpy.values())

    @property
    def names(self) -> list[str]:
        """Get a list of actively tracked names."""
        return list(self.pos.keys())

    @property
    def active(self) -> bool:
        """Check if Vicon has sent data for each object."""
        return all([name in self.ang_vel for name in self.track_names])

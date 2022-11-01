#!/usr/bin/python3

import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs

from ar_track_alvar_msgs.msg import AlvarMarkers
from std_msgs.msg import Header
from ros_stuff.msg import SingleState, ProcessedStates

from tf.transformations import euler_from_quaternion

WORLD_FRAME = "usb_cam"

class StatePublisher:
    def __init__(self, robot_id, object_id, base_id, corner_id):
        """
        Usage: robot_state = self.id_to_state[self.name_to_id["robot"]]
                           = self.id_to_state[robot_id]
        """

        self.base_id = base_id
        self.base_frame = f"ar_marker_{base_id}"
        self.robot_frame = f"ar_marker_{robot_id}"
        self.name_to_id = {"robot": robot_id,
                           "object": object_id,
                           "base": base_id,
                           "corner": corner_id}

        # compute velocity based on most recent 3 states
        self.id_to_state = {id: np.zeros(3) for id in self.name_to_id.values()}
        self.id_to_past_states_stamped = {id: np.empty((3, 4)) for id in self.name_to_id.values()}

        rospy.init_node("state_publisher")

        print("setting up tf buffer/listener")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        print("finished setting up tf buffer/listener")

        print("waiting for /ar_pose_marker rostopic")
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.update, queue_size=1)
        print("subscribed to /ar_pose_marker")

        self.publisher = rospy.Publisher("/processed_state", ProcessedStates, queue_size=1)
        rospy.spin()

    def update(self, msg):
        found_ids = [marker.id for marker in msg.markers]
        if not self.base_id in found_ids:
            print("\nBASE MARKER NOT FOUND\n")
            return

        for marker in msg.markers:
            if marker.id in self.id_to_state and marker.id != self.base_id:
                state = self.id_to_state[marker.id]
                past_states = self.id_to_past_states_stamped[marker.id]
            else:
                continue

            for _ in range(10):
                try:
                    camera_frame_pose = marker.pose
                    camera_frame_pose.header.frame_id = WORLD_FRAME

                    # get robot or corner pose relative to base frame
                    base_transform = self.tf_buffer.lookup_transform(self.base_frame, WORLD_FRAME, rospy.Time(0))
                    pose = tf2_geometry_msgs.do_transform_pose(camera_frame_pose, base_transform).pose

                    break
                except (tf2_ros.LookupException,
                        tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException):
                    pass

            p, o = pose.position, pose.orientation
            quat = [o.x, o.y, o.z, o.w]
            roll, pitch, yaw = euler_from_quaternion(quat)

            state[:] = [p.x, p.y, yaw]

            past_states[1:] = past_states[:-1]
            secs, nsecs = msg.header.stamp.secs, msg.header.stamp.nsecs
            past_states[0] = np.append(state, secs + nsecs / 1e9)

        pub_msg = ProcessedStates()
        rs = pub_msg.robot_state = SingleState()
        os = pub_msg.object_state = SingleState()
        cs = pub_msg.corner_state = SingleState()

        robot_pos = self.id_to_state[self.name_to_id["robot"]].copy()
        object_pos = self.id_to_state[self.name_to_id["object"]].copy()
        corner_pos = self.id_to_state[self.name_to_id["corner"]].copy()

        rs.x, rs.y, rs.yaw = robot_pos
        os.x, os.y, os.yaw = object_pos
        cs.x, cs.y, cs.yaw = corner_pos

        robot_vel, object_vel = self.compute_vel_from_past_states()
        rs.x_vel, rs.y_vel, rs.yaw_vel = robot_vel
        os.x_vel, os.y_vel, os.yaw_vel = object_vel

        pub_msg.header = Header()
        pub_msg.header.stamp = rospy.Time.now()

        self.publisher.publish(pub_msg)

    def compute_vel_from_past_states(self):
        velocities = []
        for name in ["robot", "object"]:
            past_states = self.id_to_past_states_stamped[self.name_to_id[name]]
            v1 = past_states[0] - past_states[1]
            v2 = past_states[1] - past_states[2]

            # get signed difference for yaw
            v1[2] = (v1[2] + np.pi) % (2 * np.pi) - np.pi
            v2[2] = (v2[2] + np.pi) % (2 * np.pi) - np.pi

            # divide by time delta to get velocity
            v1 = v1[:-1] / v1[-1]
            v2 = v2[:-1] / v2[-1]
            velocities.append((v1 + v2) / 2)

        return velocities


if __name__ == "__main__":
    robot_id = rospy.get_param('/state_publisher/robot_id')
    object_id = rospy.get_param('/state_publisher/object_id')
    base_id = rospy.get_param('/state_publisher/base_id')
    corner_id = rospy.get_param('/state_publisher/corner_id')

    state_publisher = StatePublisher(robot_id, object_id, base_id, corner_id)

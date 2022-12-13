#!/usr/bin/python3

import numpy as np
import torch

from ros_stuff.msg import RobotCmd


dimensions = {
    "action_dim": 2,                            # left_pwm, right_pwm
    "state_dim": 3,                             # x, y, yaw
    "object_input_dim": 4,                      # x_to_robot, y_to_robot, sin(object_yaw), cos(object_yaw)
    "robot_output_dim": 4,                      # x_delta, y_delta, sin(robot_yaw), cos(robot_yaw)
    "object_output_dim": 4,                     # x_delta, y_delta, sin(object_yaw), cos(object_yaw)
}

### GENERAL PYTORCH UTILS ###

def to_device(*args, device=torch.device("cpu")):
    ret = []
    for arg in args:
        ret.append(arg.to(device))
    return ret if len(ret) > 1 else ret[0]

def dcn(*args):
    ret = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            ret.append(arg)
        else:
            ret.append(arg.detach().cpu().numpy())
    return ret if len(ret) > 1 else ret[0]

def as_tensor(*args):
    ret = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            ret.append(torch.as_tensor(arg, dtype=torch.float))
        else:
            ret.append(arg)
    return ret if len(ret) > 1 else ret[0]

def sin_cos(angle):
    return torch.sin(angle), torch.cos(angle)

def signed_angle_difference(angle1, angle2):
    return (angle1 - angle2 + 3 * torch.pi) % (2 * torch.pi) - torch.pi

def rotate_vectors(vectors, angle):
    sin, cos = sin_cos(angle)
    rotation = torch.stack((torch.stack((cos, -sin)),
                            torch.stack((sin, cos)))).permute(2, 0, 1)
    rotated_vector = (rotation @ vectors[:, :, None]).squeeze(dim=-1)
    return rotated_vector

def build_action_msg(action, duration, action_num):
    action_msg = RobotCmd()
    action_msg.left_pwm = action[0]
    action_msg.right_pwm = action[1]
    action_msg.duration = duration
    action_msg.action_num = action_num
    return action_msg


class DataUtils:
    def __init__(self, use_object=False):
        self.use_object = use_object

    def state_to_model_input(self, state):
        if self.use_object:
            state = as_tensor(state)

            robot_xy, robot_heading = state[:, :2], state[:, 2]
            object_xy, object_heading = state[:, 3:5], state[:, 5]

            absolute_object_to_robot_xy = robot_xy - object_xy
            relative_object_to_robot_xy = rotate_vectors(absolute_object_to_robot_xy, -robot_heading)

            relative_object_heading = signed_angle_difference(object_heading, robot_heading)
            relative_object_sc = torch.stack(sin_cos(relative_object_heading), dim=1)

            relative_state = torch.cat((relative_object_to_robot_xy, relative_object_sc), dim=1)

            return relative_state
        else:
            return None

    def compute_relative_delta_xysc(self, state, next_state):
        state, next_state = as_tensor(state, next_state)

        n = 2 if self.use_object else 1
        relative_delta_xysc = torch.empty((state.size(0), 4*n))
        robot_heading = state[:, 2]

        for i in range(n):
            cur_state, cur_next_state = state[:, 3*i:3*(i+1)], next_state[:, 3*i:3*(i+1)]

            xy, next_xy = cur_state[:, :2], cur_next_state[:, :2]
            heading, next_heading = cur_state[:, 2], cur_next_state[:, 2]

            absolute_delta_xy = next_xy - xy
            relative_delta_xy = rotate_vectors(absolute_delta_xy, -robot_heading)

            relative_heading = signed_angle_difference(heading, robot_heading)
            relative_next_heading = signed_angle_difference(next_heading, robot_heading)
            rel_sin, rel_cos = sin_cos(relative_heading)
            rel_next_sin, rel_next_cos = sin_cos(relative_next_heading)
            relative_delta_sc = torch.stack((rel_next_sin - rel_sin, rel_next_cos - rel_cos), dim=1)

            relative_delta_xysc[:, 4*i:4*(i+1)] = torch.cat((relative_delta_xy, relative_delta_sc), dim=1)

        return relative_delta_xysc

    def next_state_from_relative_delta(self, state, relative_delta):
        state, relative_delta = as_tensor(state, relative_delta)

        n = 2 if self.use_object else 1
        next_state = torch.empty_like(state)
        robot_heading = state[:, 2]

        for i in range(n):
            cur_state, cur_relative_delta = state[:, 3*i:3*(i+1)], relative_delta[:, 4*i:4*(i+1)]

            absolute_xy, absolute_heading = cur_state[:, :2], cur_state[:, 2]
            relative_delta_xy, relative_delta_sc = cur_relative_delta[:, :2], cur_relative_delta[:, 2:]

            absolute_delta_xy = rotate_vectors(relative_delta_xy, robot_heading)
            absolute_next_xy = absolute_xy + absolute_delta_xy

            relative_heading = signed_angle_difference(absolute_heading, robot_heading)
            relative_sc = torch.stack(sin_cos(relative_heading), dim=1)
            relative_next_sc = relative_sc + relative_delta_sc
            rel_next_sin, rel_next_cos = relative_next_sc.T
            relative_next_heading = torch.atan2(rel_next_sin, rel_next_cos)
            absolute_next_heading = signed_angle_difference(relative_next_heading, -robot_heading)

            next_state[:, 3*i:3*(i+1)] = torch.cat((absolute_next_xy, absolute_next_heading[:, None]), dim=1)

        return next_state
